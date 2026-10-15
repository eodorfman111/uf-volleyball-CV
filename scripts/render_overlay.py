import argparse
from pathlib import Path

import cv2
import pandas as pd

from vbcv.config import load_config, run_dir
from vbcv.court import draw_court, load_homography
from vbcv.video import frames

COLORS = {"serve": (0, 0, 255), "pass": (0, 255, 0), "set": (255, 255, 0), "attack": (0, 128, 255), "block": (255, 0, 255)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--stills", nargs="*", type=int, default=[])
    p.add_argument("--out", default=None)
    a = p.parse_args()
    cfg = load_config(a.config)
    rd = run_dir(cfg)
    H, _, cam = load_homography(rd / "calibrate" / "homography.json")
    ball = pd.read_csv(rd / "track" / "ball.csv").set_index("frame")
    players = pd.read_csv(rd / "track" / "players.csv")
    contacts = pd.read_csv(rd / "events" / "contacts.csv")
    pf = {f: g for f, g in players.groupby("frame")}
    out = Path(a.out or rd / "viz")
    out.mkdir(parents=True, exist_ok=True)
    writer = None

    for i, fr in frames(cfg["video"], a.start, a.end):
        img = draw_court(fr, H, cam=cam)
        g = pf.get(i)
        if g is None:
            g = pf.get(i - 1)
        if g is not None:
            for r in g.itertuples():
                c = (0, 200, 0) if r.on_court else (128, 128, 128)
                cv2.rectangle(img, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), c, 2)
                cv2.putText(img, f"{int(r.track_id)} {r.side}", (int(r.x1), int(r.y1) - 6), 0, 0.7, c, 2)
        if i in ball.index:
            b = ball.loc[i]
            cv2.circle(img, (int(b.x), int(b.y)), 10, (0, 255, 255), 3)
        for r in contacts[(contacts.frame <= i) & (contacts.frame > i - 15)].itertuples():
            cv2.putText(img, f"{r.label} #{r.touch_n}", (int(r.x) + 14, int(r.y)), 0, 1.0, COLORS.get(r.label, (255, 255, 255)), 3)
        cv2.putText(img, f"frame {i}", (20, 40), 0, 1.0, (255, 255, 255), 2)
        img = cv2.resize(img, None, fx=a.scale, fy=a.scale)
        if writer is None:
            writer = cv2.VideoWriter(str(out / f"overlay_{a.start}_{a.end}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (img.shape[1], img.shape[0]))
        writer.write(img)
        if i in a.stills:
            cv2.imwrite(str(out / f"frame_{i:06d}.jpg"), img)
    if writer is not None:
        writer.release()
    print(out)


if __name__ == "__main__":
    main()
