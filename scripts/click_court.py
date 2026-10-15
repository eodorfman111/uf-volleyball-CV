import argparse
import sys
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vbcv.court import court_points

ORDER = list(court_points().keys())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("out")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--holdout", nargs="*", default=[])
    a = p.parse_args()
    cap = cv2.VideoCapture(a.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.frame)
    ok, img = cap.read()
    if not ok:
        raise SystemExit("cannot read frame")
    pts = {}
    idx = [0]

    def on_click(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and idx[0] < len(ORDER):
            pts[ORDER[idx[0]]] = [int(x), int(y)]
            idx[0] += 1

    cv2.namedWindow("court", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("court", on_click)
    while idx[0] < len(ORDER):
        vis = img.copy()
        for n, (x, y) in pts.items():
            cv2.circle(vis, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(vis, n, (x + 6, y - 6), 0, 0.6, (0, 0, 255), 2)
        cv2.putText(vis, f"click {ORDER[idx[0]]}  (s=skip, q=done)", (20, 40), 0, 1.0, (0, 255, 255), 2)
        cv2.imshow("court", vis)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("s"):
            idx[0] += 1
        elif k == ord("q"):
            break
    cv2.destroyAllWindows()
    hold = {k: pts.pop(k) for k in a.holdout if k in pts}
    spec = {"image_size": [img.shape[1], img.shape[0]], "frame": a.frame, "points": pts, "holdout": hold}
    Path(a.out).write_text(yaml.safe_dump(spec, sort_keys=False))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
