import copy

import numpy as np
import pandas as pd
import pytest

from scripts.sweep_ball import apply, parse_grid, sweep
from vbcv.config import DEFAULTS
from vbcv.stages.track import ball_track, clean_ball

ARC = "{window: 15, radius: 6.0, min_inliers: 5}"


def flight(frames, x0=300.0, vx=10.0, y0=700.0, vy=-10.0, ay=0.5):
    t = frames - frames[0]
    return x0 + vx * t, y0 + vy * t + ay * t * t / 2


def detections():
    rng = np.random.default_rng(3)
    rows = []
    for lo, hi in [(0, 60), (100, 140)]:
        f = np.arange(lo, hi)
        fx, fy = flight(f, x0=300.0 if lo == 0 else 900.0)
        rows += [{"frame": int(i), "visible": 1, "x": float(a), "y": float(b), "conf": 0.9} for i, a, b in zip(f, fx, fy)]
    for i in list(range(60, 100)) + list(range(140, 200)):
        rows.append({"frame": i, "visible": 1, "x": float(rng.uniform(0, 1900)), "y": float(rng.uniform(0, 250)), "conf": 0.6})
    return pd.DataFrame(rows).sort_values("frame").reset_index(drop=True)


def labels():
    fa, fb = np.arange(0, 60), np.arange(100, 140)
    ax, ay = flight(fa)
    bx, by = flight(fb, x0=900.0)
    rows = [(f, 1, float(ax[f]), float(ay[f])) for f in (10, 30, 50)]
    rows += [(f, 0, -1.0, -1.0) for f in (70, 80)]
    rows += [(f, 1, float(bx[f - 100]), float(by[f - 100])) for f in (110, 130)]
    rows += [(f, 0, -1.0, -1.0) for f in (160, 180)]
    return pd.DataFrame(rows, columns=["frame", "visible", "x", "y"])


def test_arc_filter_in_clean_ball_removes_scattered_false_detections():
    det = detections()
    plain = clean_ball(det, 120, 8, 5)
    arc = clean_ball(det, 120, 8, 5, arc={"window": 15, "radius": 6.0, "min_inliers": 5})
    assert (arc.frame.between(60, 99) | (arc.frame >= 140)).sum() < (plain.frame.between(60, 99) | (plain.frame >= 140)).sum()
    assert (arc.frame < 60).sum() >= 55
    assert arc.frame.between(100, 139).sum() >= 35


def test_ball_track_reads_params_from_config_defaults():
    tb = copy.deepcopy(DEFAULTS["track"]["ball"])
    assert tb["arc"] is None
    assert not ball_track(detections(), tb).empty


def test_sweep_applies_overrides_and_scores_both_halves():
    cfg = {"track": {"ball": copy.deepcopy(DEFAULTS["track"]["ball"])}}
    c = apply(cfg, "track.ball.arc.radius", "4", (1920, 1080))
    assert c["track"]["ball"]["arc"] == {"radius": 4}
    assert cfg["track"]["ball"]["arc"] is None
    assert apply(cfg, "roi_top", "100", (1920, 1080))["track"]["ball"]["roi"] == [[0, 1079], [0, 100], [1919, 100], [1919, 1079]]

    grid = parse_grid([f"track.ball.arc=null|{ARC}", "roi_top=0|100"])
    table = sweep(cfg, detections(), labels(), grid, split=100, size=(1920, 1080))
    assert len(table) == 4
    arc_rows = table[table.params.str.contains("window")]
    null_rows = table[table.params.str.contains("arc=null")]
    assert arc_rows.tune_f1.min() > 0 and arc_rows.test_f1.min() > 0
    assert arc_rows.all_f1.min() >= null_rows.all_f1.max()


def test_sweep_rejects_keys_it_does_not_evaluate():
    cfg = {"track": {"ball": copy.deepcopy(DEFAULTS["track"]["ball"])}}
    for bad in (["events.rally.min_speed_px=0|3"], ["track.ball.max_jmp_px=60|120"]):
        with pytest.raises(ValueError):
            sweep(cfg, detections(), labels(), parse_grid(bad), split=100, size=(1920, 1080))
