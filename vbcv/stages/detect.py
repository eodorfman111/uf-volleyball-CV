import pandas as pd

from ..ball import VballNet
from ..checkpoint import stage_dir
from ..players import make_detector
from ..video import frames
from .ingest import load_meta


def detect_ball(cfg, meta, out):
    c = cfg["detect"]["ball"]
    if c["backend"] == "none":
        pd.DataFrame(columns=["frame", "visible", "x", "y", "conf"]).to_csv(out / "ball.csv", index=False)
        return 0
    if c["backend"] != "vballnet":
        raise ValueError(f"unknown ball backend {c['backend']}")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if c.get("device") == "cuda" else None
    model = VballNet(c["model"], c["threshold"], providers)
    rows = model.run_video(frames(meta["video"], meta["start_frame"], meta["end_frame"]), meta["width"], meta["height"])
    df = pd.DataFrame(rows, columns=["frame", "visible", "x", "y", "conf"])
    df.to_csv(out / "ball.csv", index=False, float_format="%.2f")
    return int(df.visible.sum())


def detect_players(cfg, meta, out, batch=4):
    c = cfg["detect"]["players"]
    if c["backend"] == "none":
        pd.DataFrame(columns=["frame", "x1", "y1", "x2", "y2", "score", "cls"]).to_csv(out / "players.csv", index=False)
        return 0
    det = make_detector(cfg)
    rows, buf, idxs = [], [], []

    def flush():
        for i, d in zip(idxs, det(buf)):
            rows.extend([(i, *r) for r in d.tolist()])

    for i, f in frames(meta["video"], meta["start_frame"], meta["end_frame"], c["stride"]):
        buf.append(f)
        idxs.append(i)
        if len(buf) == batch:
            flush()
            buf, idxs = [], []
    if buf:
        flush()
    df = pd.DataFrame(rows, columns=["frame", "x1", "y1", "x2", "y2", "score", "cls"])
    df.to_csv(out / "players.csv", index=False, float_format="%.2f")
    return len(df)


def run(cfg):
    meta = load_meta(cfg)
    out = stage_dir(cfg, "detect")
    n_ball = detect_ball(cfg, meta, out)
    n_players = detect_players(cfg, meta, out)
    return {"ball_frames": n_ball, "player_boxes": n_players}
