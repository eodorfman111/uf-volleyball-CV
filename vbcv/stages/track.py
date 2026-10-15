import numpy as np
import pandas as pd

from ..checkpoint import stage_dir
from ..court import image_to_court, load_homography
from .ingest import load_meta


def clean_ball(df, max_jump_px, max_gap, min_len):
    df = df[df.visible == 1].sort_values("frame")
    keep = []
    last = None
    for r in df.itertuples():
        if last is not None:
            gap = r.frame - last.frame
            if gap <= max_gap and np.hypot(r.x - last.x, r.y - last.y) > max_jump_px * gap:
                continue
        keep.append(r)
        last = r
    if not keep:
        return pd.DataFrame(columns=["frame", "x", "y", "conf", "interp", "seg"])
    d = pd.DataFrame(keep)[["frame", "x", "y", "conf"]]
    seg = (d.frame.diff().fillna(1) > max_gap + 1).cumsum()
    d["seg"] = seg.values
    parts = []
    for s, g in d.groupby("seg"):
        if len(g) < min_len:
            continue
        full = pd.DataFrame({"frame": np.arange(g.frame.min(), g.frame.max() + 1)})
        m = full.merge(g, on="frame", how="left")
        m["interp"] = m.x.isna().astype(int)
        m[["x", "y"]] = m[["x", "y"]].interpolate()
        m["conf"] = m.conf.fillna(0.0)
        m["seg"] = s
        parts.append(m)
    if not parts:
        return pd.DataFrame(columns=["frame", "x", "y", "conf", "interp", "seg"])
    out = pd.concat(parts, ignore_index=True)
    out["seg"] = pd.factorize(out.seg)[0]
    return out


def track_players(df, fps, stride, buffer):
    import supervision as sv

    tracker = sv.ByteTrack(lost_track_buffer=buffer, frame_rate=max(1, int(round(fps / stride))))
    rows = []
    for f in sorted(df.frame.unique()) if len(df) else []:
        g = df[df.frame == f]
        det = sv.Detections(
            xyxy=g[["x1", "y1", "x2", "y2"]].to_numpy(np.float32),
            confidence=g.score.to_numpy(np.float32),
            class_id=g.cls.to_numpy(int),
        )
        det = tracker.update_with_detections(det)
        for b, tid, s in zip(det.xyxy, det.tracker_id, det.confidence):
            rows.append((f, int(tid), *map(float, b), float(s)))
    return pd.DataFrame(rows, columns=["frame", "track_id", "x1", "y1", "x2", "y2", "score"])


def add_court(df, H, cfg, cam=None):
    c = cfg["court"]
    margin = cfg["track"]["players"]["court_margin_m"]
    feet = np.stack([(df.x1 + df.x2) / 2, df.y2], axis=1) if len(df) else np.zeros((0, 2))
    xy = image_to_court(H, feet, cam) if H is not None else np.full((len(df), 2), np.nan)
    df = df.copy()
    df["foot_x"], df["foot_y"] = feet[:, 0] if len(df) else [], feet[:, 1] if len(df) else []
    df["court_x"], df["court_y"] = xy[:, 0] if len(df) else [], xy[:, 1] if len(df) else []
    if H is None:
        df["on_court"] = True
        df["side"] = "unknown"
        return df
    df["on_court"] = (
        (df.court_x > -margin) & (df.court_x < c["width_m"] + margin) & (df.court_y > -margin) & (df.court_y < c["length_m"] + margin)
    )
    df["side"] = np.where(df.court_y < c["length_m"] / 2, "near", "far")
    return df


def run(cfg):
    meta = load_meta(cfg)
    det_dir = stage_dir(cfg, "detect")
    out = stage_dir(cfg, "track")
    hpath = stage_dir(cfg, "calibrate") / "homography.json"
    H, _, cam = load_homography(hpath) if hpath.exists() else (None, None, None)
    tb = cfg["track"]["ball"]
    ball = clean_ball(pd.read_csv(det_dir / "ball.csv"), tb["max_jump_px"], tb["max_gap"], tb["min_len"])
    ball.to_csv(out / "ball.csv", index=False, float_format="%.2f")
    tp = cfg["track"]["players"]
    raw = pd.read_csv(det_dir / "players.csv")
    players = add_court(track_players(raw, meta["fps"], cfg["detect"]["players"]["stride"], tp["track_buffer"]), H, cfg, cam)
    players.to_csv(out / "players.csv", index=False, float_format="%.3f")
    on = players[players.on_court] if len(players) else players
    return {
        "ball_frames": int(len(ball)),
        "ball_segments": int(ball.seg.nunique()) if len(ball) else 0,
        "player_tracks": int(players.track_id.nunique()) if len(players) else 0,
        "on_court_tracks": int(on.track_id.nunique()) if len(on) else 0,
    }
