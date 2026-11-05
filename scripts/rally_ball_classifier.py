import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

FEATURES = ["det_frac", "mean_conf", "mean_speed", "med_speed", "frac_fast", "frac_jump", "accel", "x_std", "y_std", "y_min", "y_max", "n_runs"]


def second_features(ball, n_sec, fps):
    b = ball[ball.visible == 1].sort_values("frame")
    f = b.frame.to_numpy(float)
    x, y, c = b.x.to_numpy(float), b.y.to_numpy(float), b.conf.to_numpy(float)
    sec = np.floor(f / fps).astype(int)
    out = np.full((n_sec, len(FEATURES)), np.nan)
    out[:, 0] = 0.0
    starts = np.searchsorted(sec, np.arange(n_sec), side="left")
    ends = np.searchsorted(sec, np.arange(n_sec), side="right")
    for s in range(n_sec):
        i, j = starts[s], ends[s]
        if j <= i:
            continue
        ff, xx, yy = f[i:j], x[i:j], y[i:j]
        out[s, 0] = (j - i) / fps
        out[s, 1] = c[i:j].mean()
        out[s, 7], out[s, 8] = xx.std(), yy.std()
        out[s, 9], out[s, 10] = yy.min(), yy.max()
        if j - i >= 2:
            gap = np.diff(ff)
            ok = gap <= 3
            d = np.hypot(np.diff(xx), np.diff(yy))[ok] / gap[ok]
            out[s, 11] = 1 + int((~ok).sum())
            if len(d):
                out[s, 2], out[s, 3] = d.mean(), np.median(d)
                out[s, 4], out[s, 5] = (d > 3).mean(), (d > 60).mean()
        if j - i >= 3:
            out[s, 6] = np.abs(np.diff(yy, 2)).mean()
    return out


def add_context(raw):
    df = pd.DataFrame(raw, columns=FEATURES)
    parts = [df]
    for w in (5, 15):
        r = df.rolling(w, center=True, min_periods=1).mean()
        r.columns = [f"{c}_r{w}" for c in FEATURES]
        parts.append(r)
    return pd.concat(parts, axis=1)


def label_bins(n_sec, rallies, offset, lead, tail):
    y = np.zeros(n_sec, int)
    centers = np.arange(n_sec) + 0.5
    for r in rallies.itertuples():
        lo, hi = r.start_time - offset - lead, r.end_time - offset + tail
        y[(centers >= lo) & (centers <= hi)] = 1
    return y


def assign_sets(n_sec, rallies, offset):
    spans = []
    for s, g in rallies.groupby("set"):
        spans.append((int(s), g.start_time.min() - offset, g.end_time.max() - offset))
    spans.sort()
    edges = [(spans[k][2] + spans[k + 1][1]) / 2 for k in range(len(spans) - 1)]
    centers = np.arange(n_sec) + 0.5
    idx = np.searchsorted(edges, centers)
    return np.array([spans[k][0] for k in idx])


def to_intervals(prob, thr, min_len, merge_gap, smooth):
    if smooth > 1:
        p = np.convolve(np.pad(prob, (smooth // 2, smooth // 2), mode="edge"), np.ones(smooth) / smooth, mode="valid")
    else:
        p = prob
    on = p >= thr
    runs, i = [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]:
                j += 1
            runs.append([i, j])
            i = j
        else:
            i += 1
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= merge_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [(a, b) for a, b in merged if b - a >= min_len]


def overlap(a, b):
    return min(a[1], b[1]) - max(a[0], b[0])


def evaluate(pred, truth, min_overlap=1.0):
    hit_t = [[j for j, p in enumerate(pred) if overlap(t, p) >= min_overlap] for t in truth]
    hit_p = [[i for i, t in enumerate(truth) if overlap(t, p) >= min_overlap] for p in pred]
    covered = sum(1 for h in hit_t if h)
    matched = sum(1 for h in hit_p if h)
    merged = sum(max(len(h) - 1, 0) for h in hit_p)
    split = sum(max(len(h) - 1, 0) for h in hit_t)
    starts = [abs(pred[h[0]][0] - t[0]) for t, h in zip(truth, hit_t) if len(h) == 1 and len(hit_p[h[0]]) == 1]
    return {
        "truth": len(truth),
        "pred": len(pred),
        "recall": round(covered / max(len(truth), 1), 3),
        "precision": round(matched / max(len(pred), 1), 3),
        "merged": merged,
        "split": split,
        "median_start_err_s": round(float(np.median(starts)), 1) if starts else None,
    }


def fit_offset(det, rallies, lo, hi, lead, tail):
    sm = pd.Series(det).rolling(5, center=True, min_periods=1).mean().to_numpy()
    best = (-2.0, None)
    for off in range(int(lo), int(hi) + 1):
        y = label_bins(len(det), rallies, off, lead, tail)
        if y.std() == 0:
            continue
        c = float(np.corrcoef(sm, y)[0, 1])
        if c > best[0]:
            best = (c, off)
    return best[1], best[0]


def run(ball, rallies, fps, offset, lead=2.0, tail=1.0, thr=0.5, min_len=4, merge_gap=3, smooth=3, fit=True):
    n_sec = int(np.ceil(ball.frame.max() / fps)) + 1
    raw = second_features(ball, n_sec, fps)
    if fit:
        off, corr = fit_offset(np.nan_to_num(raw[:, 0]), rallies, offset - 30, offset + 30, lead, tail)
        print(f"clock offset refined from {offset} to {off} s (correlation {corr:.2f})")
        offset = off
    X = add_context(raw)
    y = label_bins(n_sec, rallies, offset, lead, tail)
    folds = assign_sets(n_sec, rallies, offset)
    prob = np.zeros(n_sec)
    for s in np.unique(folds):
        te = folds == s
        m = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, max_iter=200, random_state=0)
        m.fit(X[~te], y[~te])
        prob[te] = m.predict_proba(X[te])[:, 1]
    truth = [(r.start_time - offset, r.end_time - offset) for r in rallies.itertuples()]
    pred = to_intervals(prob, thr, min_len, merge_gap, smooth)
    acc = float(((prob >= thr) == y).mean())
    per_set = {}
    for s in np.unique(folds):
        sel = [(a, b) for a, b in pred if folds[min(int((a + b) / 2), n_sec - 1)] == s]
        tr = [t for t, r in zip(truth, rallies.itertuples()) if int(r.set) == s]
        per_set[int(s)] = evaluate(sel, tr)
    return {"offset": offset, "second_accuracy": round(acc, 3), "overall": evaluate(pred, truth), "per_set": per_set}, prob, y, pred


def rule_baseline(config, truth_offset, rallies):
    from vbcv.config import load_config, run_dir
    from vbcv.stages.events import segment_rallies
    from vbcv.stages.ingest import load_meta
    from vbcv.stages.track import ball_track

    cfg = load_config(config)
    meta = load_meta(cfg)
    det = pd.read_csv(run_dir(cfg) / "detect" / "ball.csv")
    cleaned = ball_track(det, cfg["track"]["ball"])
    rl = segment_rallies(cleaned, meta, cfg["events"]["rally"])
    fps = meta["fps"]
    pred = [(r.start_frame / fps, r.end_frame / fps) for r in rl.itertuples()]
    truth = [(r.start_time - truth_offset, r.end_time - truth_offset) for r in rallies.itertuples()]
    return evaluate(pred, truth)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("dvw_dir")
    p.add_argument("--offset", type=float, default=526)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--min-len", type=int, default=4)
    p.add_argument("--merge-gap", type=int, default=3)
    p.add_argument("--smooth", type=int, default=3)
    p.add_argument("--no-fit", action="store_true")
    p.add_argument("--baseline", action="store_true")
    a = p.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from vbcv.config import load_config, run_dir

    cfg = load_config(a.config)
    ball = pd.read_csv(run_dir(cfg) / "detect" / "ball.csv")
    rallies = pd.read_csv(f"{a.dvw_dir}/rallies.csv").dropna(subset=["start_time", "end_time"])
    res, prob, y, pred = run(ball, rallies, a.fps, a.offset, thr=a.thr, min_len=a.min_len, merge_gap=a.merge_gap, smooth=a.smooth, fit=not a.no_fit)
    print(json.dumps(res["overall"]))
    print("second accuracy", res["second_accuracy"], "offset", res["offset"])
    for s, r in res["per_set"].items():
        print("set", s, json.dumps(r))
    if a.baseline:
        print("rule baseline", json.dumps(rule_baseline(a.config, res["offset"], rallies)))
    out = Path(run_dir(cfg)) / "rally_cls"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sec": np.arange(len(prob)), "prob": prob, "label": y}).to_csv(out / "oof.csv", index=False, float_format="%.3f")
    pd.DataFrame(pred, columns=["start_s", "end_s"]).to_csv(out / "pred_rallies.csv", index=False)


if __name__ == "__main__":
    main()
