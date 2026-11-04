import argparse
import json
import sys
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from score_ball import score


def abs_frame(j, first, every, win):
    return first + (j // win) * every + j % win


def predict(model, files, first, every, win):
    rows = []
    for w in range(0, len(files), win):
        chunk = files[w : w + win]
        im = cv2.imread(str(chunk[0]))
        h, wd = im.shape[:2]
        it = ((abs_frame(w + k, first, every, win), cv2.imread(str(f))) for k, f in enumerate(chunk))
        rows += model.run_video(it, wd, h)
    return pd.DataFrame(rows, columns=["frame", "visible", "x", "y", "conf"])


def clean_windows(pred, first, every, max_jump_px=120, max_gap=8, min_len=5):
    from vbcv.stages.track import clean_ball

    parts = []
    pred = pred.assign(win=(pred.frame - first) // every)
    for _, g in pred.groupby("win"):
        c = clean_ball(g.drop(columns="win"), max_jump_px, max_gap, min_len)
        if len(c):
            parts.append(c)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["frame", "x", "y", "conf", "interp", "seg"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("labels")
    p.add_argument("windows")
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--first", type=int, default=228780)
    p.add_argument("--every", type=int, default=5580)
    p.add_argument("--win", type=int, default=50)
    p.add_argument("--radius", type=float, default=12.0)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None)
    p.add_argument("--keep-first", action="store_true")
    p.add_argument("--clean", action="store_true", help="apply the track-stage cleaning (jump gate, min run length) per window before scoring")
    a = p.parse_args()
    from vbcv.ball import HeatmapBall

    labels = pd.read_csv(a.labels)
    labels = labels[labels.frame >= a.first]
    if not a.keep_first:
        labels = labels[(labels.frame - a.first) % a.every != 0]
    files = sorted(Path(a.windows).glob("*.jpg"))
    print(f"{len(files)} window frames, {len(labels)} test labels ({int(labels.visible.sum())} with a ball)")
    for ck in a.checkpoints:
        model = HeatmapBall(ck, a.threshold, a.device)
        pred = predict(model, files, a.first, a.every, a.win)
        if a.out:
            Path(a.out).mkdir(parents=True, exist_ok=True)
            pred.to_csv(Path(a.out) / (Path(ck).parent.name + "_pred.csv"), index=False, float_format="%.2f")
        print(ck)
        print(json.dumps(score(labels, pred, a.radius), indent=2))
        if a.clean:
            cleaned = clean_windows(pred, a.first, a.every)
            print("after track cleaning (interpolated frames not counted):")
            print(json.dumps(score(labels, cleaned, a.radius, drop_interp=True), indent=2))


if __name__ == "__main__":
    main()
