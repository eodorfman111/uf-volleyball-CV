import argparse
import os
from pathlib import Path

IMAGES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    os.environ.setdefault("QT_LOGGING_RULES", "*.warning=false")
    import cv2

    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--speed", type=float, default=0.5)
    a = p.parse_args()
    if Path(a.path).suffix.lower() in IMAGES:
        cv2.imshow("view", cv2.imread(a.path))
        cv2.waitKey(0)
        return
    cap = cv2.VideoCapture(a.path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = max(1, int(1000 / fps / a.speed))
    pos, paused, frame = -1, False, None
    while True:
        if frame is None or not paused:
            ok, frame = cap.read()
            if not ok:
                break
            pos += 1
        cv2.imshow("view", frame)
        k = cv2.waitKey(30 if paused else delay) & 0xFF
        if k == 27:
            break
        if k == ord(" "):
            paused = not paused
        elif paused and k == ord("d"):
            ok, nxt = cap.read()
            if ok:
                frame, pos = nxt, pos + 1
        elif paused and k == ord("a") and pos > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos - 1)
            ok, prev = cap.read()
            if ok:
                frame, pos = prev, pos - 1


if __name__ == "__main__":
    main()
