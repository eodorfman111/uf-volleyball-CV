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
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imshow("view", frame)
        if cv2.waitKey(delay) == 27:
            break


if __name__ == "__main__":
    main()
