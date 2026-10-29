import numpy as np
import pytest

from vbcv.arcfit import arc_filter_mask


def parabola(frames, x0=100.0, vx=12.0, y0=600.0, vy=-30.0, ay=2.0):
    t = frames - frames[0]
    return x0 + vx * t, y0 + vy * t + ay * t * t / 2


def test_keeps_a_ball_flight_and_drops_scattered_noise():
    rng = np.random.default_rng(1)
    flight = np.arange(100, 130)
    fx, fy = parabola(flight)
    noise_f = np.arange(100, 130)
    nx = rng.uniform(0, 1900, len(noise_f))
    ny = rng.uniform(0, 300, len(noise_f))
    frame = np.concatenate([flight, noise_f])
    x = np.concatenate([fx + rng.normal(0, 1.0, len(fx)), nx])
    y = np.concatenate([fy + rng.normal(0, 1.0, len(fy)), ny])
    mask = arc_filter_mask(frame, x, y)
    assert mask[: len(flight)].mean() > 0.9
    assert mask[len(flight) :].mean() < 0.1


def test_keeps_both_sides_of_a_contact_direction_change():
    a = np.arange(0, 20)
    b = np.arange(20, 40)
    ax, ay = parabola(a, vx=15.0, vy=-25.0)
    bx, by = parabola(b, x0=ax[-1], vx=-18.0, y0=ay[-1], vy=10.0)
    mask = arc_filter_mask(np.concatenate([a, b]), np.concatenate([ax, bx]), np.concatenate([ay, by]))
    assert mask.mean() > 0.9


def test_isolated_points_are_dropped_and_short_input_is_safe():
    assert not arc_filter_mask([1, 2], [0, 5], [0, 5]).any()
    frame = np.arange(0, 100, 10)
    assert not arc_filter_mask(frame, np.arange(10) * 200.0 % 1900, np.arange(10) * 97.0 % 300).any()
    assert arc_filter_mask([], [], []).shape == (0,)


def test_large_gaps_between_flights_are_skipped_quickly_and_both_kept():
    a = np.arange(0, 30)
    b = np.arange(5_000_000, 5_000_030)
    ax, ay = parabola(a)
    bx, by = parabola(b)
    mask = arc_filter_mask(np.concatenate([a, b]), np.concatenate([ax, bx]), np.concatenate([ay, by]))
    assert mask.all()


def test_min_inliers_below_four_is_rejected():
    with pytest.raises(ValueError):
        arc_filter_mask([1, 2, 3, 4, 5], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], min_inliers=3)


def test_unsorted_input_and_determinism():
    frame = np.arange(50, 80)
    x, y = parabola(frame)
    perm = np.random.default_rng(2).permutation(len(frame))
    m1 = arc_filter_mask(frame[perm], x[perm], y[perm])
    m2 = arc_filter_mask(frame[perm], x[perm], y[perm])
    assert m1.all()
    assert (m1 == m2).all()
