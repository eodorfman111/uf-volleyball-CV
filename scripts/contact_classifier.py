import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

FEATURES = [
    "pre_speed", "post_speed", "speed_ratio", "angle", "vx_pre", "vx_post", "vy_pre", "vy_post", "x", "y",
    "y_rel", "local_ymax", "flip_down_up", "accel", "seg_len", "seg_pos", "gap_prev", "gap_next", "n_near",
]


def window_stats(f, x, y, i, lo, hi):
    sel = (f >= f[i] + lo) & (f <= f[i] + hi) & (np.arange(len(f)) != i)
    if sel.sum() < 2:
        return np.nan, np.nan, np.nan
    df = f[sel] - f[i]
    return float(np.polyfit(df, x[sel] - x[i], 1)[0]), float(np.polyfit(df, y[sel] - y[i], 1)[0]), float(sel.sum())


def candidate_features(cands, track):
    rows = []
    segs = {s: g.sort_values("frame") for s, g in track.groupby("seg")}
    cand_frames = np.sort(cands.frame.to_numpy())
    for c in cands.itertuples():
        g = segs[c.seg]
        f, x, y = g.frame.to_numpy(float), g.x.to_numpy(float), g.y.to_numpy(float)
        i = int(np.argmin(np.abs(f - c.frame)))
        vxa, vya, _ = window_stats(f, x, y, i, -6, -1)
        vxb, vyb, _ = window_stats(f, x, y, i, 1, 6)
        pre, post = np.hypot(vxa, vya), np.hypot(vxb, vyb)
        denom = max(pre * post, 1e-6)
        ang = float(np.degrees(np.arccos(np.clip((vxa * vxb + vya * vyb) / denom, -1, 1)))) if denom > 1e-6 else np.nan
        near = (f >= f[i] - 15) & (f <= f[i] + 15)
        yr = (y[i] - y.min()) / max(y.max() - y.min(), 1.0)
        k = int(np.searchsorted(cand_frames, c.frame))
        prev_gap = float(c.frame - cand_frames[k - 1]) if k > 0 else np.nan
        next_gap = float(cand_frames[k + 1] - c.frame) if k + 1 < len(cand_frames) else np.nan
        rows.append(
            [
                pre, post, post / max(pre, 1e-6), ang, vxa, vxb, vya, vyb, x[i], y[i], yr,
                float(y[i] >= y[near].max() - 3), float(vya > 0 and vyb < 0),
                float(np.hypot(vxb - vxa, vyb - vya)), float(len(g)), i / max(len(g) - 1, 1), prev_gap, next_gap,
                float(((cand_frames >= c.frame - 30) & (cand_frames <= c.frame + 30)).sum()),
            ]
        )
    return pd.DataFrame(rows, columns=FEATURES)


def match_touches(cand_frames, touch_frames, tol):
    label = np.zeros(len(cand_frames), int)
    used = set()
    for t in sorted(touch_frames):
        best, bd = None, tol + 1
        for j, c in enumerate(cand_frames):
            if j in used:
                continue
            d = abs(c - t)
            if d < bd:
                best, bd = j, d
        if best is not None and bd <= tol:
            used.add(best)
            label[best] = 1
    return label


def assign_sets(frames, rallies, offset, fps):
    spans = []
    for s, g in rallies.groupby("set"):
        spans.append((int(s), (g.start_time.min() - offset) * fps, (g.end_time.max() - offset) * fps))
    spans.sort()
    edges = [(spans[k][2] + spans[k + 1][1]) / 2 for k in range(len(spans) - 1)]
    idx = np.searchsorted(edges, frames)
    return np.array([spans[k][0] for k in idx])


def in_rally(frames, rallies, offset, fps, pad=1.5):
    ok = np.zeros(len(frames), bool)
    for r in rallies.itertuples():
        lo, hi = (r.start_time - offset - pad) * fps, (r.end_time - offset + pad) * fps
        ok |= (frames >= lo) & (frames <= hi)
    return ok


def summarize(keep, label, n_touches, n_cand):
    kept = int(keep.sum())
    tp = int((keep & (label == 1)).sum())
    return {
        "candidates": n_cand,
        "kept": kept,
        "scout_touches": n_touches,
        "precision": round(tp / max(kept, 1), 3),
        "recall": round(tp / max(n_touches, 1), 3),
        "kept_per_scout_touch": round(kept / max(n_touches, 1), 2),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("dvw_dir")
    p.add_argument("--offset", type=float, default=526)
    p.add_argument("--tol", type=float, default=1.5)
    p.add_argument("--fps", type=float, default=30.0)
    a = p.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from vbcv.config import load_config, run_dir
    from vbcv.stages.events import find_contacts
    from vbcv.stages.track import ball_track

    cfg = load_config(a.config)
    det = pd.read_csv(run_dir(cfg) / "detect" / "ball.csv")
    track = ball_track(det, cfg["track"]["ball"])
    cands = find_contacts(track, cfg["events"]["contact"]).sort_values("frame").reset_index(drop=True)
    touches = pd.read_csv(f"{a.dvw_dir}/touches.csv").dropna(subset=["video_time"])
    rallies = pd.read_csv(f"{a.dvw_dir}/rallies.csv").dropna(subset=["start_time", "end_time"])
    touch_frames = ((touches.video_time - a.offset) * a.fps).to_numpy()
    cf = cands.frame.to_numpy()
    mask = in_rally(cf, rallies, a.offset, a.fps)
    cands = cands[mask].reset_index(drop=True)
    cf = cands.frame.to_numpy()
    label = match_touches(cf, touch_frames, a.tol * a.fps)
    X = candidate_features(cands, track)
    folds = assign_sets(cf, rallies, a.offset, a.fps)
    prob = np.zeros(len(cands))
    for s in np.unique(folds):
        te = folds == s
        m = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, max_iter=200, random_state=0)
        m.fit(X[~te], label[~te])
        prob[te] = m.predict_proba(X[te])[:, 1]
    n_touch = len(touch_frames)
    print("touches matched by some candidate (recall ceiling):", round(label.sum() / n_touch, 3))
    print("all candidates in rallies:", json.dumps(summarize(np.ones(len(cands), bool), label, n_touch, len(cands))))
    for thr in (0.2, 0.3, 0.4, 0.5, 0.6):
        print("classifier thr", thr, json.dumps(summarize(prob >= thr, label, n_touch, len(cands))))
    for s in np.unique(folds):
        te = folds == s
        nt = int(((assign_sets(touch_frames, rallies, a.offset, a.fps)) == s).sum())
        print("set", s, "thr 0.4", json.dumps(summarize((prob >= 0.4) & te, label * te, nt, int(te.sum()))))
    out = Path(run_dir(cfg)) / "contact_cls"
    out.mkdir(parents=True, exist_ok=True)
    pd.concat([cands, pd.DataFrame({"label": label, "prob": prob})], axis=1).to_csv(out / "candidates.csv", index=False, float_format="%.3f")


if __name__ == "__main__":
    main()
