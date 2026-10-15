import argparse
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np


def load_gt(lbl, w, h):
    out = []
    if lbl.exists():
        for line in lbl.read_text().split("\n"):
            v = line.split()
            if len(v) >= 5:
                out.append((float(v[1]) * w, float(v[2]) * h, float(v[3]) * w, float(v[4]) * h))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True)
    p.add_argument("--matches", nargs="*", default=["match17", "match18"])
    p.add_argument("--model", default="grdesignbuild/volleyball-person-ball-detector")
    p.add_argument("--thresholds", nargs="+", type=float, default=[0.2, 0.3, 0.4, 0.5])
    p.add_argument("--max-dist-px", type=float, default=10.0)
    p.add_argument("--out", default="runs/eval_ball_frames")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    from vbcv.players import DFineDetector

    det = DFineDetector(a.model, conf=min(a.thresholds), device=a.device, classes=("ball",))
    imgs = sorted(Path(a.split, "images").glob("*.jpg"))
    if a.matches:
        imgs = [i for i in imgs if re.match(r"(match\d+)_", i.name) and re.match(r"(match\d+)_", i.name).group(1) in a.matches]
    t0 = time.time()
    rows = []
    for im in imgs:
        frame = cv2.imread(str(im))
        h, w = frame.shape[:2]
        gt = load_gt(Path(a.split, "labels", im.stem + ".txt"), w, h)
        d = det([frame])[0]
        rows.append({"image": im.name, "gt": gt, "pred": [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, float(b[4])) for b in d.tolist()]})
    res = {"split": a.split, "matches": a.matches, "n_images": len(rows), "n_gt": sum(len(r["gt"]) for r in rows), "max_dist_px": a.max_dist_px, "model": a.model, "by_threshold": {}}
    for t in a.thresholds:
        tp = fp = fn = 0
        errs = []
        for r in rows:
            preds = sorted([q for q in r["pred"] if q[2] >= t], key=lambda q: -q[2])
            used = set()
            for gx, gy, gw, gh in r["gt"]:
                best, bd = None, 1e9
                for k, q in enumerate(preds):
                    dd = np.hypot(q[0] - gx, q[1] - gy)
                    if k not in used and dd < bd:
                        best, bd = k, dd
                if best is not None and bd <= max(a.max_dist_px, max(gw, gh)):
                    tp += 1
                    used.add(best)
                    errs.append(bd)
                else:
                    fn += 1
            fp += len(preds) - len(used)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        res["by_threshold"][str(t)] = {"tp": tp, "fp": fp, "fn": fn, "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4), "median_center_err_px": round(float(np.median(errs)), 2) if errs else None}
    res["seconds"] = round(time.time() - t0, 1)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    (out / "predictions.json").write_text(json.dumps(rows))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
