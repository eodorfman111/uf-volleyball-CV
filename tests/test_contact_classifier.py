import numpy as np
import pandas as pd

from scripts.contact_classifier import FEATURES, candidate_features, match_touches


def test_match_touches_is_one_to_one_and_within_tolerance():
    cand = [100, 130, 400]
    label = match_touches(cand, [105, 128, 133, 1000], tol=20)
    assert label.tolist() == [1, 1, 0]


def bounce_track():
    f = np.arange(0, 80)
    x = 100 + f * 5.0
    y = np.where(f < 40, 100 + f * 8.0, 100 + (80 - f) * 8.0)
    return pd.DataFrame({"frame": f, "x": x, "y": y, "conf": 0.9, "seg": 0})


def test_floor_bounce_has_the_down_then_up_signature_at_a_local_maximum_of_y():
    track = bounce_track()
    cands = pd.DataFrame({"frame": [40], "seg": [0], "x": [300.0], "y": [420.0], "speed_px": [8.0]})
    feats = candidate_features(cands, track)
    assert list(feats.columns) == FEATURES
    row = feats.iloc[0]
    assert row.flip_down_up == 1.0 and row.local_ymax == 1.0 and row.vy_pre > 0 > row.vy_post
