import pandas as pd

from scripts.score_ball import score


def test_score_counts_hits_misses_and_false_alarms():
    labels = pd.DataFrame(
        {
            "frame": [0, 10, 20, 30, 40],
            "visible": [1, 1, 1, 0, 0],
            "x": [100.0, 200.0, 300.0, -1.0, -1.0],
            "y": [100.0, 200.0, 300.0, -1.0, -1.0],
        }
    )
    pred = pd.DataFrame({"frame": [0, 10, 30], "x": [103.0, 400.0, 50.0], "y": [100.0, 400.0, 50.0]})
    out = score(labels, pred, radius=12.0)
    assert (out["tp"], out["fp"], out["fn"], out["tn"]) == (1, 2, 2, 1)
    assert out["precision"] == 0.333
    assert out["recall"] == 0.333
    assert out["median_error_px"] == 3.0
