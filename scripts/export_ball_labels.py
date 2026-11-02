import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vbcv.video import frames


def encode(frame):
    return cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("labels")
    p.add_argument("out")
    p.add_argument("--max-frame", type=int, default=None)
    p.add_argument("--box", type=float, default=16.0)
    p.add_argument("--gap", type=int, default=0)
    a = p.parse_args()
    labels = pd.read_csv(a.labels)
    if a.max_frame is not None:
        labels = labels[labels.frame < a.max_frame]
    by_frame = {int(r.frame): r for r in labels.itertuples()}
    need = {i for i in by_frame}
    if a.gap:
        need |= {max(i - a.gap, 0) for i in by_frame} | {i + a.gap for i in by_frame}
    root = Path(a.out)
    dirs = {name: root / name for name in ("images", "labels", "images_prev", "images_next")}
    for name, d in dirs.items():
        if name in ("images", "labels") or a.gap:
            d.mkdir(parents=True, exist_ok=True)
    cache = {}
    last, last_frame = -1, None
    for i, fr in frames(a.video, 0, max(need) + 1):
        last, last_frame = i, fr
        if i in need:
            cache[i] = encode(fr)
    cache.setdefault(last, encode(last_frame))
    size = last_frame.shape[:2]
    n_pos = 0
    h, w = size
    for i, r in by_frame.items():
        name = f"uf_{i:06d}.jpg"
        (dirs["images"] / name).write_bytes(cache[i].tobytes())
        if a.gap:
            (dirs["images_prev"] / name).write_bytes(cache[max(i - a.gap, 0)].tobytes())
            (dirs["images_next"] / name).write_bytes(cache[min(i + a.gap, last)].tobytes())
        text = f"0 {r.x / w:.6f} {r.y / h:.6f} {a.box / w:.6f} {a.box / h:.6f}\n" if r.visible == 1 else ""
        (dirs["labels"] / f"uf_{i:06d}.txt").write_text(text)
        n_pos += int(r.visible == 1)
    print(
        f"exported {len(by_frame)} frames ({n_pos} with a ball, {len(by_frame) - n_pos} negatives) to {a.out}"
        + (f", with neighbours at +-{a.gap} frames" if a.gap else "")
    )


if __name__ == "__main__":
    main()
