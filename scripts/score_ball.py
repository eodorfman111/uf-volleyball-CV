import argparse
import json

import numpy as np
import pandas as pd


def score(labels, pred, radius=12.0, drop_interp=False):
    if "visible" in pred:
        pred = pred[pred.visible == 1]
    if drop_interp and "interp" in pred:
        pred = pred[pred.interp == 0]
    p = pred.drop_duplicates("frame").set_index("frame")
    tp = fp = fn = tn = 0
    errs = []
    for r in labels.itertuples():
        has = r.frame in p.index
        if r.visible == 1:
            if has:
                d = float(np.hypot(p.at[r.frame, "x"] - r.x, p.at[r.frame, "y"] - r.y))
                if d <= radius:
                    tp += 1
                    errs.append(d)
                else:
                    fp += 1
                    fn += 1
            else:
                fn += 1
        elif has:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "frames": len(labels),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "median_error_px": round(float(np.median(errs)), 2) if errs else None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("labels")
    p.add_argument("pred")
    p.add_argument("--radius", type=float, default=12.0)
    p.add_argument("--drop-interp", action="store_true")
    p.add_argument("--min-frame", type=int, default=None)
    p.add_argument("--max-frame", type=int, default=None)
    a = p.parse_args()
    labels = pd.read_csv(a.labels)
    if a.min_frame is not None:
        labels = labels[labels.frame >= a.min_frame]
    if a.max_frame is not None:
        labels = labels[labels.frame < a.max_frame]
    print(json.dumps(score(labels, pd.read_csv(a.pred), a.radius, a.drop_interp), indent=2))


if __name__ == "__main__":
    main()
