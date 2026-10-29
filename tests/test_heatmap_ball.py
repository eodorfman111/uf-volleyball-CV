import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vbcv.ball import HeatmapBall  # noqa: E402
from vbcv.heatnet import HeatNet  # noqa: E402


def frames(n, w=192, h=108):
    rng = np.random.default_rng(0)
    return [(i * 2, rng.integers(0, 255, (h, w, 3), dtype=np.uint8)) for i in range(n)]


def test_heatnet_keeps_resolution_and_loads_both_checkpoint_layouts(tmp_path):
    net = HeatNet(8)
    assert net(torch.zeros(2, 3, 64, 96)).shape == (2, 1, 64, 96)
    torch.save({"model": net.state_dict(), "opt": {}, "step": 1}, tmp_path / "a.pt")
    torch.save(net.state_dict(), tmp_path / "b.pt")
    for name in ("a.pt", "b.pt"):
        HeatmapBall(tmp_path / name, input_size=(96, 64), width=8)


def test_heatmap_ball_returns_one_row_per_frame_in_original_coordinates(tmp_path):
    net = HeatNet(8)
    torch.save({"model": net.state_dict()}, tmp_path / "m.pt")
    ball = HeatmapBall(tmp_path / "m.pt", threshold=0.0, input_size=(96, 64), width=8)
    rows = ball.run_video(iter(frames(11)), 192, 108, batch=4)
    assert [r[0] for r in rows] == [i * 2 for i in range(11)]
    assert all(r[1] == 1 and 0 <= r[2] <= 192 and 0 <= r[3] <= 108 for r in rows)
    none = HeatmapBall(tmp_path / "m.pt", threshold=1.1, input_size=(96, 64), width=8).run_video(iter(frames(3)), 192, 108)
    assert all(r[1] == 0 and r[2] == -1.0 for r in none)
