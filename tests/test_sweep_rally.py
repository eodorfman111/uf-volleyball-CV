import copy

import numpy as np
import pandas as pd
import pytest

from scripts.sweep_rally import evaluate, parse_grid, sweep
from vbcv.config import DEFAULTS

META = {"fps": 10.0, "start_frame": 0, "end_frame": 400}


def ball():
    f = np.arange(0, 400)
    moving = ((f >= 20) & (f < 80)) | ((f >= 200) & (f < 260))
    x = np.where(moving, f * 10.0, 100.0)
    return pd.DataFrame({"frame": f, "x": x, "y": 300.0, "seg": 0})


def test_evaluate_counts_covered_plays_and_false_rallies():
    rallies = pd.DataFrame({"start_frame": [20, 300], "end_frame": [80, 340]})
    assert evaluate(rallies, [25, 210], tol=10) == (1, 1)
    assert evaluate(rallies, [25, 310], tol=10) == (2, 0)


def test_sweep_ranks_speed_gate_settings_against_known_plays():
    cfg = {"events": {"rally": copy.deepcopy(DEFAULTS["events"]["rally"])}}
    cfg["events"]["rally"].update({"min_rally_s": 2.0, "merge_gap_s": 1.0, "pad_s": 0.0, "window_s": 1.0})
    table = sweep(cfg, ball(), META, [30, 205], parse_grid(["events.rally.min_speed_px=0|3"]), tol=20)
    best = table.iloc[0]
    assert best.params == "min_speed_px=3"
    assert best.rallies == 2 and best.missed == 0 and best.false_rallies == 0
    assert table.iloc[1].rallies == 1


def test_sweep_rejects_keys_it_does_not_evaluate():
    cfg = {"events": {"rally": copy.deepcopy(DEFAULTS["events"]["rally"])}}
    for bad in ("track.ball.min_len=3|5", "events.rally.min_spd=1|2"):
        with pytest.raises(ValueError):
            sweep(cfg, ball(), META, [30], parse_grid([bad]), tol=20)
