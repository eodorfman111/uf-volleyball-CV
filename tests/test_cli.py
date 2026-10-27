from vbcv import cli


def test_multiple_stages_run_in_pipeline_order(monkeypatch, tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("name: t\nruns_root: " + str(tmp_path / "runs") + "\n")
    ran = []
    monkeypatch.setattr(cli, "run_stage", lambda c, s, force=False: ran.append(s) or True)
    assert cli.main(["track", "ingest", "-c", str(cfg)]) == 0
    assert ran == ["ingest", "track"]
