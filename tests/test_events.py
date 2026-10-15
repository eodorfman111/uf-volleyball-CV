import numpy as np
import pandas as pd

from vbcv.stages.events import find_contacts, label_touches, segment_rallies
from vbcv.stages.track import clean_ball


def test_clean_ball_drops_outlier_and_interpolates():
    f = np.arange(20)
    df = pd.DataFrame({"frame": f, "visible": 1, "x": f * 5.0, "y": 100.0, "conf": 0.9})
    df.loc[10, ["x", "y"]] = [900, 900]
    df = df.drop(index=[5])
    out = clean_ball(df, max_jump_px=50, max_gap=4, min_len=3)
    assert len(out) == 20
    assert out.x.max() < 200


def test_segment_rallies():
    meta = {"fps": 10.0, "start_frame": 0, "end_frame": 200}
    ball = pd.DataFrame({"frame": list(range(20, 80)) + list(range(120, 180))})
    c = {"min_ball_ratio": 0.5, "window_s": 1.0, "min_rally_s": 2.0, "merge_gap_s": 1.0, "pad_s": 0.0}
    r = segment_rallies(ball, meta, c)
    assert len(r) == 2


def test_contact_on_direction_change():
    t = np.arange(40)
    x = np.where(t < 20, t * 10.0, 200 - (t - 20) * 10.0)
    ball = pd.DataFrame({"frame": t, "x": x, "y": 300.0, "seg": 0})
    c = {"min_angle_deg": 35, "min_speed_px": 2.0, "min_sep_frames": 6}
    con = find_contacts(ball, c)
    assert len(con) >= 1 and abs(con.frame.iloc[0] - 20) <= 2


def test_rule_labels():
    con = pd.DataFrame(
        {
            "frame": [10, 40, 60, 80, 100],
            "side": ["near", "far", "far", "far", "near"],
            "court_x": [4, 4, 5, 4, 4],
            "court_y": [-1, 15, 10, 9.5, 8.5],
            "player_dist": [0.1] * 5,
        }
    )
    rallies = pd.DataFrame({"rally_id": [0], "start_frame": [0], "end_frame": [200]})
    out = label_touches(con, rallies, 1.5, 9.0, 30)
    assert out.label.tolist() == ["serve", "pass", "set", "attack", "block"]
