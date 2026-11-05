import argparse
import json

import numpy as np
import pandas as pd


def fit_offset(dvw, pipe, lo, hi):
    ds, de = dvw.start_time.to_numpy(float), dvw.end_time.to_numpy(float)
    ps, pe = pipe.start_s.to_numpy(float), pipe.end_s.to_numpy(float)
    best = (-1.0, None)
    for off in np.arange(lo, hi + 1, 1.0):
        a, b = ds[:, None] - off, de[:, None] - off
        ov = np.clip(np.minimum(b, pe[None]) - np.maximum(a, ps[None]), 0, None)
        total = float(ov.sum())
        if total > best[0]:
            best = (total, float(off))
    return best[1], best[0]


def start_counts(dvw, pipe, lo, hi, early=8.0, late=2.0):
    serves = dvw.start_time.to_numpy(float)
    starts = np.sort(pipe.start_s.to_numpy(float))
    res = []
    for off in np.arange(lo, hi + 1, 1.0):
        t = serves - off
        idx = np.searchsorted(starts, t - early)
        ok = (idx < len(starts)) & (starts[np.minimum(idx, len(starts) - 1)] <= t + late)
        inside = (t >= starts.min() - 10) & (t <= starts.max() + 10)
        res.append((int((ok & inside).sum()), int(inside.sum()), float(off)))
    return res


def top_offsets(res, n=6, min_sep=15.0):
    out = []
    for hits, inside, off in sorted(res, key=lambda r: (-r[0], r[2])):
        if all(abs(off - o) >= min_sep for _, _, o in out):
            out.append((hits, inside, off))
        if len(out) == n:
            break
    return out


def match(dvw, pipe, off):
    pairs, used = [], set()
    for i, p in pipe.iterrows():
        best, bo = None, 1.0
        for j, d in dvw.iterrows():
            if j in used:
                continue
            ov = min(d.end_time - off, p.end_s) - max(d.start_time - off, p.start_s)
            if ov > bo:
                best, bo = j, ov
        if best is not None:
            used.add(best)
            pairs.append((i, best))
    return pairs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dvw_dir")
    p.add_argument("run_dir")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--offset-range", type=float, nargs=2, default=[-1000, 9500])
    p.add_argument("--offset", type=float, default=None)
    p.add_argument("--top", action="store_true", help="list the best-fitting offsets by counting pipeline rally starts near scout serves")
    a = p.parse_args()
    dr = pd.read_csv(f"{a.dvw_dir}/rallies.csv").dropna(subset=["start_time", "end_time"])
    dt = pd.read_csv(f"{a.dvw_dir}/touches.csv")
    pr = pd.read_csv(f"{a.run_dir}/events/rallies.csv")
    pc = pd.read_csv(f"{a.run_dir}/events/contacts.csv")
    if "start_s" not in pr:
        pr["start_s"], pr["end_s"] = pr.start_frame / a.fps, pr.end_frame / a.fps
    if a.top:
        res = start_counts(dr, pr, *a.offset_range)
        print("offset_s, scout serves with a pipeline rally start within 8 s before to 2 s after, scout serves inside the section")
        for hits, inside, off_ in top_offsets(res):
            print(f"  {off_:7.0f}  {hits:3d} of {inside}")
    if a.offset is not None:
        off = a.offset
    else:
        best = top_offsets(start_counts(dr, pr, *a.offset_range), n=1)[0][2]
        off, _ = fit_offset(dr, pr, best - 10, best + 10)
        print(f"fitted offset {off:.0f} s (serve-to-start counts, refined by interval overlap)")
    lo, hi = pr.start_s.min(), pr.end_s.max()
    window = dr[(dr.end_time - off >= lo) & (dr.start_time - off <= hi)]
    pairs = match(window, pr, off)
    start_err = [pr.at[i, "start_s"] - (window.at[j, "start_time"] - off) for i, j in pairs]
    end_err = [pr.at[i, "end_s"] - (window.at[j, "end_time"] - off) for i, j in pairs]
    ratios, ratios_nb = [], []
    for i, j in pairs:
        rid = pr.at[i, "rally_id"]
        n_pipe = int((pc.rally_id == rid).sum())
        t = dt[dt.rally == window.at[j, "rally"]]
        n_all = int((t.skill != "serve").sum())
        n_nb = int((~t.skill.isin(["serve", "block"])).sum())
        if n_all:
            ratios.append(n_pipe / n_all)
        if n_nb:
            ratios_nb.append(n_pipe / n_nb)
    out = {
        "offset_s": off,
        "scout_rallies_in_window": len(window),
        "pipeline_rallies": len(pr),
        "matched": len(pairs),
        "rally_recall": round(len(pairs) / max(len(window), 1), 3),
        "rally_precision": round(len(pairs) / max(len(pr), 1), 3),
        "median_start_error_s": round(float(np.median(start_err)), 2) if start_err else None,
        "median_end_error_s": round(float(np.median(end_err)), 2) if end_err else None,
        "pipeline_touches_per_scout_touch_median": round(float(np.median(ratios)), 2) if ratios else None,
        "pipeline_touches_per_scout_touch_excl_blocks_median": round(float(np.median(ratios_nb)), 2) if ratios_nb else None,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
