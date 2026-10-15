import json

import numpy as np
import pandas as pd

from ..checkpoint import stage_dir
from .ingest import load_meta


def _num(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else round(float(v), 3)
    return v


def rally_table(rallies, contacts, serves):
    rows = []
    for r in rallies.itertuples():
        g = contacts[contacts.rally_id == r.rally_id]
        s = serves[serves.rally_id == r.rally_id]
        rows.append(
            {
                "rally_id": r.rally_id,
                "start_s": r.start_s,
                "end_s": r.end_s,
                "duration_s": r.end_s - r.start_s,
                "contacts": len(g),
                "serve_side": g.side.iloc[0] if len(g) and isinstance(g.side.iloc[0], str) else None,
                "n_attacks": int((g.label == "attack").sum()),
                "n_blocks": int((g.label == "block").sum()),
                "serve_speed_kmh": float(s.avg_ground_speed_kmh.iloc[0]) if len(s) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def side_aggregates(contacts, jumps):
    out = {}
    for side in ["near", "far"]:
        g = contacts[contacts.side == side]
        att = g[g.label == "attack"]
        j = jumps[jumps.side == side]
        out[side] = {
            "touches": {k: int(v) for k, v in g.label.value_counts().items()},
            "attack_origin_court_xy": att[["court_x", "court_y"]].round(2).values.tolist(),
            "jumps": len(j),
            "jump_height_cm_median": _num(j.height_cm_flight.median()) if len(j) else None,
        }
    return out


def run(cfg):
    meta = load_meta(cfg)
    e = stage_dir(cfg, "events")
    out = stage_dir(cfg, "stats")
    rallies = pd.read_csv(e / "rallies.csv")
    contacts = pd.read_csv(e / "contacts.csv")
    serves = pd.read_csv(e / "serves.csv")
    jumps = pd.read_csv(e / "jumps.csv")
    rt = rally_table(rallies, contacts, serves)
    rt.to_csv(out / "rallies.csv", index=False, float_format="%.2f")
    per_player = (
        jumps.groupby("track_id").agg(jumps=("air_s", "size"), max_height_cm=("height_cm_flight", "max"), side=("side", "first")).reset_index()
        if len(jumps)
        else pd.DataFrame(columns=["track_id", "jumps", "max_height_cm", "side"])
    )
    per_player.to_csv(out / "player_jumps.csv", index=False, float_format="%.1f")
    summary = {
        "video": meta["video"],
        "duration_s": round((meta["end_frame"] - meta["start_frame"]) / meta["fps"], 2),
        "rallies": len(rallies),
        "rally_time_s": _num(rt.duration_s.sum()) if len(rt) else 0,
        "contacts": len(contacts),
        "touch_labels": {k: int(v) for k, v in contacts.label.value_counts().items()} if len(contacts) else {},
        "serve_speed_kmh": [_num(v) for v in serves.avg_ground_speed_kmh] if len(serves) else [],
        "jumps": len(jumps),
        "sides": side_aggregates(contacts, jumps),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=_num))
    return {"rallies": summary["rallies"], "contacts": summary["contacts"], "jumps": summary["jumps"]}
