import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

labels, export, burst_dir = sys.argv[1:4]
L, gap, every = 10, 5, 5580
frames = sorted(int(f) for f in pd.read_csv(labels).frame)
random.Random(0).shuffle(frames)


def load(path):
    im = cv2.imread(str(path), 0)
    return None if im is None else im.astype(np.float32)


own, other = [], []
for f in frames[:60]:
    j = (f // every) * L + (f % every) // gap
    a = load(Path(export) / "images" / f"uf_{f:06d}.jpg")
    b = load(Path(burst_dir) / f"{j + 1:05d}.jpg")
    nb = load(Path(burst_dir) / f"{(j - 1 if j % L == L - 1 else j + 1) + 1:05d}.jpg")
    if a is None or b is None or nb is None:
        continue
    own.append(float(np.abs(a - b).mean()))
    other.append(float(np.abs(a - nb).mean()))
own, other = np.array(own), np.array(other)
share = float((own < other).mean()) if len(own) else 0.0
print(f"checked {len(own)} frames: median difference to the clicked image {np.median(own):.2f}, closer than the neighbour image in {share:.0%}")
sys.exit(0 if len(own) >= 20 and share > 0.9 else 1)
