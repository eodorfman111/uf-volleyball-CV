import numpy as np
import pandas as pd

from vbcv.stages.track import drop_persistent

COURT = {"width_m": 9.0, "length_m": 18.0}


def test_static_person_at_court_edge_is_marked_off_court():
    f = np.arange(100)
    static = pd.DataFrame({"frame": f, "track_id": 1, "court_x": -0.4, "court_y": 9.0, "on_court": True})
    moving = pd.DataFrame({"frame": f, "track_id": 2, "court_x": 2 + f * 0.05, "court_y": 5 + f * 0.07, "on_court": True})
    out = drop_persistent(pd.concat([static, moving], ignore_index=True), {"persistent_frac": 0.7}, COURT)
    assert not out[out.track_id == 1].on_court.any()
    assert out[out.track_id == 2].on_court.all()


def test_persistent_filter_is_off_by_default_and_ignores_interior():
    f = np.arange(100)
    interior = pd.DataFrame({"frame": f, "track_id": 3, "court_x": 4.5, "court_y": 5.0, "on_court": True})
    assert drop_persistent(interior, {}, COURT).on_court.all()
    assert drop_persistent(interior, {"persistent_frac": 0.7}, COURT).on_court.all()
