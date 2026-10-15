import numpy as np

from vbcv.court import camera_model, court_points, distort, estimate_k1, fit_homography, image_to_court, undistort


def synthetic_H():
    src = np.array([[0, 0], [9, 0], [9, 18], [0, 18]], float)
    dst = np.array([[200, 1000], [1700, 1000], [1250, 450], [650, 450]], float)
    return fit_homography(dst, src)


def test_homography_roundtrip():
    H = synthetic_H()
    pts = np.array(list(court_points().values()))
    img = np.array([[200, 1000], [1700, 1000], [1250, 450], [650, 450]], float)
    back = image_to_court(H, img)
    assert np.allclose(back, [[0, 0], [9, 0], [9, 18], [0, 18]], atol=1e-6)
    assert pts.shape == (10, 2)


def test_distort_undistort_inverse():
    cam = camera_model({"k1": -0.2}, (1920, 1080))
    pts = np.array([[100.0, 900.0], [960.0, 540.0], [1800.0, 200.0]])
    assert np.allclose(undistort(distort(pts, cam), cam), pts, atol=0.5)


def test_estimate_k1_recovers_barrel():
    cam = camera_model({"k1": -0.15}, (1920, 1080))
    lines = [distort(np.stack([np.linspace(100, 1800, 40), np.full(40, y)], 1), cam) for y in (150, 900, 1000)]
    k1, before, after = estimate_k1(lines, (1920, 1080))
    assert abs(k1 + 0.15) < 0.02
    assert after < before
