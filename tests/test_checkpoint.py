from vbcv.checkpoint import stage_hash
from vbcv.config import load_config


def test_ball_roi_is_off_by_default_and_changes_track_hash(tmp_path):
    base = tmp_path / "a.yaml"
    base.write_text("name: t\n")
    with_roi = tmp_path / "b.yaml"
    with_roi.write_text("name: t\ntrack:\n  ball:\n    roi: [[0, 0], [10, 0], [10, 10]]\n")
    a, b = load_config(base), load_config(with_roi)
    assert a["track"]["ball"]["roi"] is None
    assert stage_hash(a, "track") != stage_hash(b, "track")
    assert stage_hash(a, "detect") == stage_hash(b, "detect")


def test_editing_court_points_invalidates_downstream_stages(tmp_path):
    pts = tmp_path / "pts.yaml"
    pts.write_text("points: {}\n")
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(f"name: t\ncourt:\n  points_file: {pts.as_posix()}\n")
    cfg = load_config(cfg_path)
    before = {s: stage_hash(cfg, s) for s in ["calibrate", "track", "stats"]}
    pts.write_text("points: {near_left: [1, 2]}\n")
    after = {s: stage_hash(cfg, s) for s in before}
    assert all(before[s] != after[s] for s in before)
    assert stage_hash(cfg, "ingest") == stage_hash(load_config(cfg_path), "ingest")
