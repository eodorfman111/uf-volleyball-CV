import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vbcv.video import frames


def thumb(frame):
    return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)


def locate(full_frames, probes):
    best = {k: (float("inf"), -1) for k in probes}
    for j, fr in full_frames:
        t = thumb(fr)
        for k, ref in probes.items():
            d = float(np.abs(t - ref).mean())
            if d < best[k][0]:
                best[k] = (d, j)
    return {k: (v[1] - k, round(v[0], 2)) for k, v in best.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("full")
    p.add_argument("clip")
    p.add_argument("--probes", type=int, nargs="+", default=[100, 2000, 4000])
    a = p.parse_args()
    probes = {i: thumb(fr) for i, fr in frames(a.clip, 0, max(a.probes) + 1) if i in set(a.probes)}
    found = locate(frames(a.full, 0, None), probes)
    for k, (offset, dist) in found.items():
        print(f"clip frame {k}: best match at offset {offset} (difference {dist})")
    offsets = {v[0] for v in found.values()}
    if len(offsets) == 1:
        print(f"the clip starts at frame {offsets.pop()} of the full video")
    else:
        print("the probes disagree, so the clip may not be a continuous piece of the full video")


if __name__ == "__main__":
    main()
