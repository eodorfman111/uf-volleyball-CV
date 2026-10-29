import copy

from vbcv.config import DEFAULTS, set_dotted


def test_dotted_override_into_a_null_default_creates_the_dict():
    cfg = copy.deepcopy(DEFAULTS)
    assert cfg["track"]["ball"]["arc"] is None
    set_dotted(cfg, "track.ball.arc.radius", "4")
    assert cfg["track"]["ball"]["arc"] == {"radius": 4}
    set_dotted(cfg, "track.ball.arc.window", "10")
    assert cfg["track"]["ball"]["arc"] == {"radius": 4, "window": 10}


def test_dotted_override_replaces_whole_values_and_null():
    cfg = copy.deepcopy(DEFAULTS)
    set_dotted(cfg, "track.ball.min_len", "12")
    set_dotted(cfg, "track.ball.roi", "[[0, 0], [10, 0], [10, 10]]")
    set_dotted(cfg, "track.ball.roi", "null")
    assert cfg["track"]["ball"]["min_len"] == 12
    assert cfg["track"]["ball"]["roi"] is None
