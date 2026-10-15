import argparse

import cv2
import pandas as pd

from vbcv.config import load_config, run_dir
from vbcv.stages.ingest import load_meta
from vbcv.video import read_frame


def clip_rows(rd, fps, pad_s, last_frame):
    rows = []
    rallies = pd.read_csv(rd / "events" / "rallies.csv")
    for r in rallies.itertuples():
        for edge, f in (("rally_start", r.start_frame), ("rally_end", r.end_frame)):
            rows.append({"task": "rally_boundary", "priority": 1, "frame": int(f), "why": edge})
    contacts = pd.read_csv(rd / "events" / "contacts.csv")
    for r in contacts.itertuples():
        rows.append(
            {
                "task": "touch",
                "priority": 2 if r.label in ("serve", "attack", "block") else 3,
                "frame": int(r.frame),
                "why": f"pred={r.label} touch_n={r.touch_n} player_dist_m={r.player_dist:.2f}",
            }
        )
    jumps = pd.read_csv(rd / "events" / "jumps.csv")
    for r in jumps.itertuples():
        rows.append({"task": "jump", "priority": 3, "frame": int(r.takeoff_frame), "why": f"track={r.track_id} pred_height_cm={r.height_cm_flight:.0f}"})
    q = pd.DataFrame(rows)
    q["start_frame"] = (q.frame - pad_s * fps).clip(lower=0).astype(int)
    q["end_frame"] = (q.frame + pad_s * fps).clip(upper=last_frame).astype(int)
    return q


def frame_rows(rd, n):
    det = pd.read_csv(rd / "detect" / "ball.csv")
    trk = pd.read_csv(rd / "track" / "ball.csv")
    kept = set(trk[trk.interp == 0].frame)
    det["status"] = "missed"
    det.loc[det.frame.isin(kept), "status"] = "kept"
    det.loc[(det.visible == 1) & ~det.frame.isin(kept), "status"] = "rejected_by_tracker"
    det.loc[(det.visible == 1) & det.frame.isin(kept) & (det.conf < 0.7), "status"] = "low_conf"
    picks = []
    for status, k in (("rejected_by_tracker", n // 3), ("missed", n // 3), ("low_conf", n // 6), ("kept", n // 6)):
        g = det[det.status == status]
        if len(g):
            picks.append(g.iloc[:: max(len(g) // max(k, 1), 1)].head(k))
    q = pd.concat(picks)[["frame", "status", "conf"]]
    q["task"] = "ball_point"
    return q


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--pad-s", type=float, default=2.0)
    p.add_argument("--export-images", action="store_true")
    a = p.parse_args()
    cfg = load_config(a.config)
    rd = run_dir(cfg)
    meta = load_meta(cfg)
    out = rd / "labeling_queue"
    out.mkdir(parents=True, exist_ok=True)
    clips = clip_rows(rd, meta["fps"], a.pad_s, meta["end_frame"]).sort_values(["priority", "frame"])
    clips.insert(0, "video", cfg["video"])
    clips.to_csv(out / "clips.csv", index=False)
    frames_q = frame_rows(rd, a.frames)
    frames_q.insert(0, "video", cfg["video"])
    frames_q.to_csv(out / "ball_frames.csv", index=False)
    if a.export_images:
        img_dir = out / "images"
        img_dir.mkdir(exist_ok=True)
        for f in frames_q.frame:
            cv2.imwrite(str(img_dir / f"{cfg['name']}_{int(f):06d}.jpg"), read_frame(cfg["video"], int(f)))
    print(out, len(clips), "clips", len(frames_q), "frames")


if __name__ == "__main__":
    main()
