import argparse
from pathlib import Path

import yaml


def build_data_yaml(data_dir, out_dir):
    data_dir = Path(data_dir).resolve()
    spec = {"path": str(data_dir), "train": "images", "val": "images", "names": {0: "ball"}}
    path = Path(out_dir) / "data.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="runs/train_ball_yolo")
    p.add_argument("--weights", default="yolo26s.pt")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="0")
    a = p.parse_args()
    from ultralytics import YOLO

    data = build_data_yaml(a.data, a.out)
    model = YOLO(a.weights)
    model.train(
        data=str(data),
        imgsz=a.imgsz,
        epochs=a.epochs,
        batch=a.batch,
        device=a.device,
        project=str(Path(a.out).resolve()),
        name="run",
        exist_ok=True,
        workers=4,
        flipud=0.0,
        degrees=0.0,
        plots=False,
    )
    print(Path(a.out).resolve() / "run" / "weights" / "best.pt")


if __name__ == "__main__":
    main()
