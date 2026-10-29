import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vbcv.video import frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("labels")
    p.add_argument("out")
    p.add_argument("--max-frame", type=int, default=None)
    p.add_argument("--box", type=float, default=16.0)
    a = p.parse_args()
    labels = pd.read_csv(a.labels)
    if a.max_frame is not None:
        labels = labels[labels.frame < a.max_frame]
    by_frame = {int(r.frame): r for r in labels.itertuples()}
    images = Path(a.out, "images")
    lbl = Path(a.out, "labels")
    images.mkdir(parents=True, exist_ok=True)
    lbl.mkdir(parents=True, exist_ok=True)
    n_pos = 0
    for i, fr in frames(a.video, 0, max(by_frame) + 1):
        if i not in by_frame:
            continue
        r = by_frame[i]
        h, w = fr.shape[:2]
        name = f"uf_{i:06d}"
        cv2.imwrite(str(images / f"{name}.jpg"), fr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        text = f"0 {r.x / w:.6f} {r.y / h:.6f} {a.box / w:.6f} {a.box / h:.6f}\n" if r.visible == 1 else ""
        (lbl / f"{name}.txt").write_text(text)
        n_pos += int(r.visible == 1)
    print(f"exported {len(by_frame)} frames ({n_pos} with a ball, {len(by_frame) - n_pos} negatives) to {a.out}")


if __name__ == "__main__":
    main()
