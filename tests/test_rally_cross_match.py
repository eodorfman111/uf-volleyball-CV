import numpy as np
import pandas as pd

from scripts.rally_cross_match import cross_match, sharpest_offset


def synthetic(seed=0, fps=30, total=1500):
    rng = np.random.default_rng(seed)
    starts = np.arange(40, total - 40, 30.0) + rng.integers(-3, 3, len(np.arange(40, total - 40, 30.0)))
    dur = rng.integers(6, 14, len(starts))
    sets = np.digitize(starts, [300, 600, 900, 1200]) + 1
    rallies = pd.DataFrame({"set": sets, "start_time": starts + 526, "end_time": starts + dur + 526})
    rows = []
    for f in range(total * fps):
        t = f / fps
        inside = any(s <= t <= s + d for s, d in zip(starts, dur))
        if inside and rng.random() < 0.75:
            rows.append((f, 1, 100 + 8 * (t % 10) + rng.normal(0, 2), 300 + 80 * np.sin(t * 3), 0.9))
        elif not inside and rng.random() < 0.08:
            rows.append((f, 1, rng.uniform(0, 1900), rng.uniform(100, 300), 0.9))
    return pd.DataFrame(rows, columns=["frame", "visible", "x", "y", "conf"]), rallies


def test_cross_match_finds_the_other_games_clock_offset_and_its_rallies():
    ball_a, rallies_a = synthetic(seed=1)
    ball_b, rallies_b = synthetic(seed=2)
    rallies_b = rallies_b.assign(start_time=rallies_b.start_time + 300, end_time=rallies_b.end_time + 300)
    res, prob = cross_match(ball_a, rallies_a, 526, ball_b, rallies_b, fit_range=(300, 1000))
    assert abs(res["offset_b"] - 826) <= 3
    assert res["offset_corr"] > res["runner_up_corr"] + 0.05
    assert res["result"]["recall"] > 0.8 and res["result"]["precision"] > 0.8


def test_sharpest_offset_reports_the_runner_up_away_from_the_peak():
    curve = [(o, 0.1) for o in range(0, 300)]
    curve[100] = (100, 0.6)
    curve[101] = (101, 0.55)
    curve[220] = (220, 0.3)
    best, second = sharpest_offset(curve, min_sep=60)
    assert best[0] == 100 and second[0] == 220
