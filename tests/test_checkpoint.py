from vbcv.checkpoint import stage_hash
from vbcv.config import load_config


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
