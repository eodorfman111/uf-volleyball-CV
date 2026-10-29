import numpy as np
import pytest

torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")

from scripts.train_ball_heatmap import items, load  # noqa: E402


class Centered:
    def random(self):
        return 0.0

    def uniform(self, a, b):
        return 0.5

    def randint(self, a, b):
        return a


def image(tmp_path, name="a.jpg", w=192, h=108):
    path = tmp_path / name
    cv2.imwrite(str(path), np.full((h, w, 3), 40, np.uint8))
    return path


def peak(hm):
    i = int(hm[0].flatten().argmax())
    return i % hm.shape[-1], i // hm.shape[-1]


def test_public_style_item_keeps_natural_aspect_and_ball_position(tmp_path):
    x, hm = load((image(tmp_path), (0.25, 0.5)), (96, 54), 2.0)
    assert x.shape == (3, 54, 96) and hm.shape == (1, 54, 96)
    assert peak(hm) == (24, 27)


def test_scaled_item_is_cropped_around_the_ball(tmp_path):
    x, hm = load((image(tmp_path), (0.7, 0.4), 2), (96, 54), 2.0, rng=Centered())
    assert x.shape == (3, 54, 96)
    assert peak(hm) == (48, 27)
    assert float(hm.max()) > 0.95


def test_negative_item_has_an_empty_target(tmp_path):
    _, hm = load((image(tmp_path), None, 2), (96, 54), 2.0, rng=Centered())
    assert float(hm.max()) == 0.0


def test_items_reads_missing_and_empty_label_files_as_negatives(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    cv2.imwrite(str(tmp_path / "images" / "a.jpg"), np.zeros((8, 8, 3), np.uint8))
    cv2.imwrite(str(tmp_path / "images" / "b.jpg"), np.zeros((8, 8, 3), np.uint8))
    cv2.imwrite(str(tmp_path / "images" / "c.jpg"), np.zeros((8, 8, 3), np.uint8))
    (tmp_path / "labels" / "a.txt").write_text("")
    (tmp_path / "labels" / "c.txt").write_text("0 0.5 0.25 0.1 0.1\n")
    got = {p.name: xy for p, xy in items(tmp_path)}
    assert got == {"a.jpg": None, "b.jpg": None, "c.jpg": (0.5, 0.25)}
