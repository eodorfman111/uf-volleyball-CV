import numpy as np
import pandas as pd

from scripts.rally_ball_classifier import evaluate, run, to_intervals


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


def test_intervals_merge_short_gaps_and_drop_short_runs():
    prob = np.array([0, 0.9, 0.9, 0.9, 0.9, 0.9, 0, 0.9, 0.9, 0.9, 0.9, 0, 0, 0, 0, 0.9, 0.9, 0])
    got = to_intervals(prob, 0.5, min_len=3, merge_gap=2, smooth=1)
    assert got == [(1, 11)]


def test_evaluate_counts_merged_and_split_rallies():
    truth = [(0, 10), (12, 20), (40, 50)]
    pred = [(0, 21), (39, 45), (46, 51)]
    r = evaluate(pred, truth)
    assert r["recall"] == 1.0 and r["merged"] == 1 and r["split"] == 1


def test_classifier_finds_rallies_on_a_clean_synthetic_game():
    ball, rallies = synthetic()
    res, prob, y, pred = run(ball, rallies, 30, 520)
    assert res["offset"] == 526 or abs(res["offset"] - 526) <= 3
    assert res["second_accuracy"] > 0.9
    assert res["overall"]["recall"] > 0.8 and res["overall"]["precision"] > 0.8
