import sys
import types

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")

from scripts.train_ball_yolo import build_data_yaml  # noqa: E402
from vbcv.ball import UltralyticsBall  # noqa: E402


class FakeBoxes:
    def __init__(self, conf, xywh):
        self.conf = torch.tensor(conf, dtype=torch.float32)
        self.xywh = torch.tensor(xywh, dtype=torch.float32).reshape(-1, 4)

    def __len__(self):
        return len(self.conf)


class FakeYOLO:
    def __init__(self, weights):
        self.weights = weights

    def predict(self, frames, imgsz, conf, device, verbose):
        out = []
        for k, _ in enumerate(frames):
            if k % 2 == 0:
                out.append(types.SimpleNamespace(boxes=FakeBoxes([0.3, 0.9], [[10, 20, 5, 5], [100, 200, 8, 8]])))
            else:
                out.append(types.SimpleNamespace(boxes=FakeBoxes([], [])))
        return out


def test_backend_takes_the_most_confident_box_center_and_reports_misses(monkeypatch):
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    ball = UltralyticsBall("w.pt", device="cuda")
    assert ball.device == 0
    frames = [(i * 3, np.zeros((8, 8, 3), np.uint8)) for i in range(5)]
    rows = ball.run_video(iter(frames), 1920, 1080, batch=2)
    assert [r[0] for r in rows] == [0, 3, 6, 9, 12]
    assert rows[0] == (0, 1, 100.0, 200.0, pytest.approx(0.9))
    assert rows[1] == (3, 0, -1.0, -1.0, 0.0)


def test_data_yaml_points_at_the_exported_folder(tmp_path):
    path = build_data_yaml(tmp_path / "ds", tmp_path / "out")
    spec = yaml.safe_load(path.read_text())
    assert spec["train"] == "images" and spec["val"] == "images" and spec["names"] == {0: "ball"}
    assert spec["path"].endswith("ds")
