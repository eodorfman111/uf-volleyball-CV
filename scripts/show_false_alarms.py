import argparse
from pathlib import Path

import cv2
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("labels")
    p.add_argument("pred")
    p.add_argument("windows")
    p.add_argument("out")
    p.add_argument("--first", type=int, default=228780)
    p.add_argument("--radius", type=float, default=12.0)
    a = p.parse_args()
    labels = pd.read_csv(a.labels)
    labels = labels[labels.frame >= a.first]
    pred = pd.read_csv(a.pred)
    pred = pred[pred.visible == 1].drop_duplicates("frame").set_index("frame")
    base = cv2.imread(str(sorted(Path(a.windows).glob("*.jpg"))[0]))
    img = (base * 0.55).astype("uint8")
    for r in labels.itertuples():
        if r.frame not in pred.index:
            continue
        x, y = pred.at[r.frame, "x"], pred.at[r.frame, "y"]
        if r.visible == 1 and ((x - r.x) ** 2 + (y - r.y) ** 2) ** 0.5 <= a.radius:
            cv2.circle(img, (int(x), int(y)), 10, (0, 255, 0), 2)
        else:
            cv2.circle(img, (int(x), int(y)), 10, (0, 0, 255), 2)
    for r in labels[labels.visible == 1].itertuples():
        cv2.drawMarker(img, (int(r.x), int(r.y)), (255, 255, 0), cv2.MARKER_CROSS, 14, 1)
    cv2.imwrite(a.out, img)
    print(f"saved {a.out}: red circles = false alarms, green = correct hits, yellow crosses = true ball positions")


if __name__ == "__main__":
    main()
