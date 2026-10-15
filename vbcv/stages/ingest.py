import json
import subprocess
from fractions import Fraction
from pathlib import Path

import cv2

from ..checkpoint import stage_dir
from ..video import probe


def run(cfg):
    video = Path(cfg["video"])
    if not video.exists():
        raise FileNotFoundError(video)
    out = stage_dir(cfg, "ingest")
    info = probe(video)
    vs = next(s for s in info["streams"] if s["codec_type"] == "video")
    has_audio = any(s["codec_type"] == "audio" for s in info["streams"])
    fps = float(Fraction(vs.get("avg_frame_rate") or vs["r_frame_rate"]))
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot decode {video}")
    start_f = int(round(cfg["ingest"]["start_s"] * fps))
    end_s = cfg["ingest"]["end_s"]
    end_f = min(n, int(round(end_s * fps))) if end_s else n
    meta = {
        "video": str(video.resolve()),
        "width": int(vs["width"]),
        "height": int(vs["height"]),
        "fps": fps,
        "n_frames": n,
        "start_frame": start_f,
        "end_frame": end_f,
        "duration_s": float(info["format"].get("duration", n / fps)),
        "codec": vs.get("codec_name"),
        "has_audio": has_audio,
    }
    cv2.imwrite(str(out / "first_frame.jpg"), frame)
    if has_audio:
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", str(out / "audio.wav")],
            check=False,
        )
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return {"n_frames": end_f - start_f, "fps": fps}


def load_meta(cfg):
    return json.loads((stage_dir(cfg, "ingest") / "meta.json").read_text())
