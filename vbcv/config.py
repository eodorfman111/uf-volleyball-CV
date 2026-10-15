import copy
import hashlib
import json
from pathlib import Path

import yaml

DEFAULTS = {
    "name": "run",
    "video": None,
    "runs_root": "runs",
    "court": {
        "length_m": 18.0,
        "width_m": 9.0,
        "attack_line_m": 3.0,
        "net_height_m": 2.43,
        "points_file": None,
    },
    "ingest": {"start_s": 0.0, "end_s": None},
    "detect": {
        "ball": {
            "backend": "vballnet",
            "model": "models/VballNetV4c_seq9_grayscale_20260908_213829.onnx",
            "threshold": 0.5,
        },
        "players": {
            "backend": "dfine",
            "model": "grdesignbuild/volleyball-person-ball-detector",
            "conf": 0.4,
            "stride": 1,
            "device": "cpu",
        },
    },
    "track": {
        "ball": {"max_jump_px": 120, "max_gap": 8, "min_len": 5},
        "players": {"court_margin_m": 1.5, "max_per_side": 8, "track_buffer": 30},
    },
    "events": {
        "rally": {"min_ball_ratio": 0.5, "window_s": 1.0, "min_rally_s": 2.0, "merge_gap_s": 1.5, "pad_s": 1.0},
        "contact": {"min_angle_deg": 35, "min_speed_px": 2.0, "max_player_dist_m": 1.5, "min_sep_frames": 6},
        "jump": {"min_rise_frac": 0.12, "min_air_s": 0.25, "max_air_s": 1.2},
    },
    "stats": {},
}


def deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def set_dotted(cfg, key, value):
    node = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = yaml.safe_load(value)


def load_config(path, overrides=()):
    with open(path) as f:
        user = yaml.safe_load(f) or {}
    cfg = deep_merge(DEFAULTS, user)
    for item in overrides:
        k, v = item.split("=", 1)
        set_dotted(cfg, k, v)
    cfg["_config_path"] = str(Path(path).resolve())
    return cfg


def run_dir(cfg):
    return Path(cfg["runs_root"]) / cfg["name"]


def section_hash(cfg, keys):
    payload = {k: cfg.get(k) for k in keys}
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]
