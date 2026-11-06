import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rally_ball_classifier import add_context, evaluate, label_bins, second_features, to_intervals


def offset_curve(det, rallies, lo, hi, lead, tail):
    sm = pd.Series(det).rolling(5, center=True, min_periods=1).mean().to_numpy()
    out = []
    for off in range(int(lo), int(hi) + 1):
        y = label_bins(len(det), rallies, off, lead, tail)
        out.append((off, float(np.corrcoef(sm, y)[0, 1]) if y.std() > 0 else -1.0))
    return out


def sharpest_offset(curve, min_sep=60):
    ordered = sorted(curve, key=lambda t: -t[1])
    best = ordered[0]
    second = next((t for t in ordered[1:] if abs(t[0] - best[0]) >= min_sep), (None, -1.0))
    return best, second


def features(ball, fps):
    n_sec = int(np.ceil(ball.frame.max() / fps)) + 1
    raw = second_features(ball, n_sec, fps)
    return raw, add_context(raw)


def cross_match(ball_a, rallies_a, offset_a, ball_b, rallies_b, fps=30.0, fit_range=(-200, 2000), lead=2.0, tail=1.0, thr=0.5, min_len=4, merge_gap=3, smooth=3, offset_b=None):
    raw_a, x_a = features(ball_a, fps)
    y_a = label_bins(len(raw_a), rallies_a, offset_a, lead, tail)
    raw_b, x_b = features(ball_b, fps)
    curve = offset_curve(np.nan_to_num(raw_b[:, 0]), rallies_b, fit_range[0], fit_range[1], lead, tail)
    (best_off, best_corr), (second_off, second_corr) = sharpest_offset(curve)
    off_b = best_off if offset_b is None else offset_b
    m = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, max_iter=200, random_state=0)
    m.fit(x_a, y_a)
    prob = m.predict_proba(x_b)[:, 1]
    truth = [(r.start_time - off_b, r.end_time - off_b) for r in rallies_b.itertuples()]
    keep = [t for t in truth if t[1] > 0 and t[0] < len(raw_b)]
    pred = to_intervals(prob, thr, min_len, merge_gap, smooth)
    return {
        "offset_b": off_b,
        "offset_corr": round(best_corr, 3),
        "runner_up_offset": second_off,
        "runner_up_corr": round(second_corr, 3),
        "scout_rallies_in_video": len(keep),
        "result": evaluate(pred, keep),
    }, prob


def main():
    p = argparse.ArgumentParser()
    p.add_argument("train_ball_csv")
    p.add_argument("train_dvw_dir")
    p.add_argument("test_ball_csv")
    p.add_argument("test_dvw_dir")
    p.add_argument("--train-offset", type=float, default=526)
    p.add_argument("--test-offset", type=float, default=None)
    p.add_argument("--fit-range", type=float, nargs=2, default=[-200, 2000])
    p.add_argument("--fps", type=float, default=30.0)
    a = p.parse_args()
    ball_a = pd.read_csv(a.train_ball_csv)
    ball_b = pd.read_csv(a.test_ball_csv)
    ra = pd.read_csv(f"{a.train_dvw_dir}/rallies.csv").dropna(subset=["start_time", "end_time"])
    rb = pd.read_csv(f"{a.test_dvw_dir}/rallies.csv").dropna(subset=["start_time", "end_time"])
    res, _ = cross_match(ball_a, ra, a.train_offset, ball_b, rb, a.fps, tuple(a.fit_range), offset_b=a.test_offset)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
