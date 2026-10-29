import argparse

import cv2
import numpy as np
import yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("config")
    p.add_argument("--frame", type=int, default=0)
    a = p.parse_args()
    cap = cv2.VideoCapture(a.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.frame)
    ok, img = cap.read()
    if not ok:
        raise SystemExit("cannot read frame")
    pts = []

    def on_click(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append([int(x), int(y)])

    cv2.namedWindow("roi", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("roi", on_click)
    while True:
        vis = img.copy()
        if len(pts) > 1:
            cv2.polylines(vis, [np.array(pts, np.int32)], len(pts) > 2, (0, 255, 255), 2)
        for x, y in pts:
            cv2.circle(vis, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(vis, f"{len(pts)} points  (click=add, u=undo, q=save)", (20, 40), 0, 1.0, (0, 255, 255), 2)
        cv2.imshow("roi", vis)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("u") and pts:
            pts.pop()
        elif k == ord("q") and len(pts) >= 3:
            break
    cv2.destroyAllWindows()
    with open(a.config) as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("track", {}).setdefault("ball", {})["roi"] = pts
    with open(a.config, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"wrote track.ball.roi ({len(pts)} points) to {a.config}")


if __name__ == "__main__":
    main()
