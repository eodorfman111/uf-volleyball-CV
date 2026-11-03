import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_pred(path):
    pred = pd.read_csv(path)
    if "visible" in pred:
        pred = pred[pred.visible == 1]
    return pred.drop_duplicates("frame").set_index("frame")


def encode(frame):
    return cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])[1]


def save(labels, out):
    rows = [(f, *labels[f]) for f in sorted(labels)]
    pd.DataFrame(rows, columns=["frame", "visible", "x", "y"]).to_csv(out, index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("out")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--step", type=int, default=10)
    p.add_argument("--pred", default=None)
    p.add_argument("--frames-file", default=None)
    p.add_argument("--frames-dir", default=None, help="folder of ffmpeg-extracted jpgs; frame number = start + (file index) * step")
    p.add_argument("--split", default=None)
    p.add_argument("--shard", default=None, help="K/N: label only every Nth chosen frame, starting at K (0-based)")
    a = p.parse_args()
    if a.frames_dir:
        k, n = map(int, a.shard.split("/")) if a.shard else (0, 1)
        files = sorted(Path(a.frames_dir).glob("*.jpg"))
        cache = {a.start + j * a.step: np.fromfile(str(f), np.uint8) for j, f in enumerate(files) if j % n == k}
    elif a.frames_file:
        from vbcv.video import frames
        sel = pd.read_csv(a.frames_file)
        if a.split:
            sel = sel[sel.split == a.split]
        wanted = sorted({int(f) for f in sel.frame})
        if a.shard:
            k, n = map(int, a.shard.split("/"))
            wanted = wanted[k::n]
        wanted = set(wanted)
        cache = {i: encode(fr) for i, fr in frames(a.video, 0, max(wanted) + 1) if i in wanted}
    else:
        from vbcv.video import frames

        k, n = map(int, a.shard.split("/")) if a.shard else (0, 1)
        cache = {i: encode(fr) for i, fr in frames(a.video, a.start, a.end, a.step) if ((i - a.start) // a.step) % n == k}
    todo = sorted(cache)
    labels = {}
    if Path(a.out).exists():
        old = pd.read_csv(a.out)
        labels = {int(r.frame): (int(r.visible), float(r.x), float(r.y)) for r in old.itertuples()}
    pred = load_pred(a.pred) if a.pred else None
    i = next((k for k, f in enumerate(todo) if f not in labels), 0)
    clicks = []

    def on_click(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))

    cv2.namedWindow("label", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("label", on_click)
    shown, img = None, None
    while 0 <= i < len(todo):
        f = todo[i]
        if shown != i:
            img = cv2.imdecode(cache[f], cv2.IMREAD_COLOR)
            shown = i
            clicks.clear()
            if img is None:
                i += 1
                continue
        vis = img.copy()
        if f in labels and labels[f][0] == 1:
            cv2.circle(vis, (int(labels[f][1]), int(labels[f][2])), 12, (0, 255, 0), 2)
        if pred is not None and f in pred.index:
            cv2.circle(vis, (int(pred.at[f, "x"]), int(pred.at[f, "y"])), 16, (255, 255, 0), 2)
        cv2.putText(vis, f"frame {f}  {i + 1}/{len(todo)}  click=ball  n=no ball  b=back  q=quit", (20, 40), 0, 1.0, (0, 255, 255), 2)
        cv2.imshow("label", vis)
        k = cv2.waitKey(30) & 0xFF
        if clicks:
            x, y = clicks.pop()
            labels[f] = (1, float(x), float(y))
            save(labels, a.out)
            i += 1
        elif k == ord("n"):
            labels[f] = (0, -1.0, -1.0)
            save(labels, a.out)
            i += 1
        elif k == ord("b"):
            i = max(0, i - 1)
        elif k == ord("q"):
            break
    cv2.destroyAllWindows()
    print(f"{len(labels)} frames labeled, saved to {a.out}")


if __name__ == "__main__":
    main()
