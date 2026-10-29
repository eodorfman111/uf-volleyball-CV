import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vbcv.video import read_frame


def load_pred(path):
    pred = pd.read_csv(path)
    if "visible" in pred:
        pred = pred[pred.visible == 1]
    return pred.drop_duplicates("frame").set_index("frame")


def save(labels, out):
    rows = [(f, *labels[f]) for f in sorted(labels)]
    pd.DataFrame(rows, columns=["frame", "visible", "x", "y"]).to_csv(out, index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("out")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, required=True)
    p.add_argument("--step", type=int, default=10)
    p.add_argument("--pred", default=None)
    a = p.parse_args()
    todo = list(range(a.start, a.end, a.step))
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
            img = read_frame(a.video, f)
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
