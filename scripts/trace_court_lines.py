import argparse
import json

import cv2
import numpy as np


def white_mask(img, v_min=170, s_max=60):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return (hsv[..., 2] > v_min) & (hsv[..., 1] < s_max)


def runs(col):
    idx = np.flatnonzero(col)
    if not len(idx):
        return []
    splits = np.flatnonzero(np.diff(idx) > 1) + 1
    return [(g[0], g[-1]) for g in np.split(idx, splits)]


def trace(mask, start, step, axis, n, max_dev=6, max_width=14, max_miss=6):
    pts = [tuple(start)]
    pos = np.array(start, float)
    vel = np.array(step, float)
    miss = 0
    for _ in range(n):
        guess = pos + vel
        x, y = int(round(guess[0])), int(round(guess[1]))
        if not (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]):
            break
        if axis == "x":
            lo = max(0, y - max_dev * 3)
            cands = [((a + b) / 2 + lo) for a, b in runs(mask[lo : y + max_dev * 3, x]) if b - a < max_width]
            best = min(cands, key=lambda c: abs(c - guess[1]), default=None)
            new = None if best is None or abs(best - guess[1]) > max_dev else np.array([x, best])
        else:
            lo = max(0, x - max_dev * 3)
            cands = [((a + b) / 2 + lo) for a, b in runs(mask[y, lo : x + max_dev * 3]) if b - a < max_width]
            best = min(cands, key=lambda c: abs(c - guess[0]), default=None)
            new = None if best is None or abs(best - guess[0]) > max_dev else np.array([best, y])
        if new is None:
            miss += 1
            if miss > max_miss:
                break
            pos = guess
            continue
        miss = 0
        vel = 0.7 * vel + 0.3 * (new - pos) if len(pts) > 2 else new - pos
        if axis == "x":
            vel[0] = step[0]
        else:
            vel[1] = step[1]
        pos = new
        pts.append((float(new[0]), float(new[1])))
    return pts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("spec")
    p.add_argument("--out", default="lines.json")
    p.add_argument("--vis", default="lines.jpg")
    a = p.parse_args()
    img = cv2.imread(a.image)
    mask = white_mask(img)
    spec = json.load(open(a.spec))
    out = {}
    vis = img.copy()
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
    for k, (name, s) in enumerate(spec.items()):
        pts = trace(mask, s["start"], s["step"], s["axis"], s.get("n", 400))
        out[name] = [[round(x, 1), round(y, 1)] for x, y in pts]
        for x, y in pts:
            cv2.circle(vis, (int(x), int(y)), 2, colors[k % len(colors)], -1)
        print(name, len(pts), pts[0], pts[-1])
    json.dump(out, open(a.out, "w"))
    cv2.imwrite(a.vis, vis)


if __name__ == "__main__":
    main()
