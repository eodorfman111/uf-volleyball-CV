import hashlib
import json
import subprocess
import time
from pathlib import Path

from .config import run_dir, section_hash

STAGES = ["ingest", "calibrate", "detect", "track", "events", "stats"]
STAGE_FILES = {
    "calibrate": [("court", "points_file")],
    "detect": [("detect", "ball", "model")],
}
STAGE_KEYS = {
    "ingest": ["video", "ingest"],
    "calibrate": ["court"],
    "detect": ["video", "ingest", "detect"],
    "track": ["court", "track"],
    "events": ["court", "events"],
    "stats": ["stats"],
}


def git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, cwd=Path(__file__).parent).decode().strip()
    except Exception:
        return None


def stage_dir(cfg, stage):
    d = run_dir(cfg) / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def file_digest(path):
    if not path or not Path(path).is_file():
        return None
    h = hashlib.sha1()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def stage_hash(cfg, stage):
    upstream = STAGES[: STAGES.index(stage) + 1]
    keys = sorted({k for s in upstream for k in STAGE_KEYS[s]})
    files = {}
    for s in upstream:
        for path in STAGE_FILES.get(s, []):
            node = cfg
            for k in path:
                node = (node or {}).get(k)
            files["/".join(path)] = file_digest(node)
    return section_hash(cfg, keys, files)


def is_done(cfg, stage):
    marker = run_dir(cfg) / stage / "_done.json"
    if not marker.exists():
        return False
    return json.loads(marker.read_text()).get("hash") == stage_hash(cfg, stage)


def mark_done(cfg, stage, started, extra=None):
    marker = stage_dir(cfg, stage) / "_done.json"
    info = {
        "stage": stage,
        "hash": stage_hash(cfg, stage),
        "git": git_rev(),
        "config": cfg.get("_config_path"),
        "seconds": round(time.time() - started, 2),
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    info.update(extra or {})
    marker.write_text(json.dumps(info, indent=2))
    return info


def log_failure(cfg, stage, err):
    d = stage_dir(cfg, stage)
    fails = d / "_failures.jsonl"
    with fails.open("a") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "error": repr(err)}) + "\n")
    return sum(1 for _ in fails.open())
