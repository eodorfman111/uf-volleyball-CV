import numpy as np

from scripts.eval_kinematics import jump_track
from vbcv.config import DEFAULTS
from vbcv.kinematics import ballistic, fit_ballistic, project
from vbcv.stages.events import find_jumps


def test_jump_height_from_flight_time():
    j = find_jumps(jump_track(0.5, 60, 1), 60, DEFAULTS["events"]["jump"])
    assert len(j) == 1
    assert abs(j.height_cm_flight.iloc[0] - 50) < 5


def test_ballistic_fit_recovers_speed():
    K = np.array([[1000, 0, 960], [0, 1000, 540], [0, 0, 1.0]])
    R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]])
    C = np.array([4.5, -8.0, 4.0])
    P = K @ np.c_[R, -R @ C]
    t = np.arange(0, 0.8, 1 / 30)
    q = np.array([4.5, -1.0, 2.8, 0.0, 20.0, 1.0])
    uv = project(P, ballistic(q, t))
    r = fit_ballistic(P, t, uv)
    assert abs(r["v0_kmh"] - np.linalg.norm(q[3:]) * 3.6) < 1.0
