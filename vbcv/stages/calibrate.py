import json

import cv2
import numpy as np
import yaml

from ..checkpoint import stage_dir
from ..court import camera_model, court_points, draw_court, estimate_k1, fit_homography, image_to_court, undistort


def load_spec(cfg):
    with open(cfg["court"]["points_file"]) as f:
        return yaml.safe_load(f)


def collect(spec, key, ref):
    items = [(k, v) for k, v in (spec.get(key) or {}).items() if v is not None]
    names = [k for k, _ in items]
    img = np.array([v for _, v in items], float).reshape(-1, 2)
    crt = np.array([spec.get("court", {}).get(k, ref.get(k)) for k in names], float).reshape(-1, 2)
    return names, img, crt


def line_targets(name, c):
    return {
        "near_end": (1, 0.0),
        "far_end": (1, c["length_m"]),
        "left_side": (0, 0.0),
        "right_side": (0, c["width_m"]),
        "center": (1, c["length_m"] / 2),
    }.get(name)


def rmse(v):
    v = np.asarray(v, float)
    return float(np.sqrt(np.mean(v**2))) if len(v) else None


def run(cfg):
    c = cfg["court"]
    out = stage_dir(cfg, "calibrate")
    spec = load_spec(cfg)
    ref = court_points(c["length_m"], c["width_m"], c["attack_line_m"])
    size = spec.get("image_size")
    lines = spec.get("lines") or {}
    dist = spec.get("distortion")
    k1_info = None
    if dist == "estimate" and lines:
        k1, before, after = estimate_k1([np.array(v, float) for v in lines.values()], size)
        dist = {"k1": k1}
        k1_info = {"k1": k1, "line_rms_px_before": before, "line_rms_px_after": after}
    cam = camera_model(dist, size) if isinstance(dist, dict) else None
    names, img, crt = collect(spec, "points", ref)
    h_names, h_img, h_crt = collect(spec, "holdout", ref)
    if len(img) < 4:
        raise ValueError("need at least 4 calibration points")
    uimg = undistort(img, cam)
    H = fit_homography(uimg, crt)
    fit_err = np.linalg.norm(image_to_court(H, img, cam) - crt, axis=1)
    loo = []
    if len(img) > 4:
        for i in range(len(img)):
            keep = np.arange(len(img)) != i
            Hi = fit_homography(uimg[keep], crt[keep])
            loo.append(float(np.linalg.norm(image_to_court(Hi, img[i : i + 1], cam) - crt[i : i + 1])))
    held = np.linalg.norm(image_to_court(H, h_img, cam) - h_crt, axis=1) if len(h_img) else np.array([])
    line_err = {}
    for name, pts in lines.items():
        t = line_targets(name, c)
        if t is None:
            continue
        proj = image_to_court(H, np.array(pts, float), cam)
        line_err[name] = rmse(proj[:, t[0]] - t[1])
    report = {
        "n_points": len(img),
        "distortion": dist if isinstance(dist, dict) else None,
        "k1_estimate": k1_info,
        "fit_err_m": dict(zip(names, map(float, fit_err))),
        "fit_rmse_m": rmse(fit_err),
        "loo_err_m": dict(zip(names, loo)),
        "loo_rmse_m": rmse(loo) if loo else None,
        "holdout_err_m": dict(zip(h_names, map(float, held))),
        "holdout_rmse_m": rmse(held) if len(held) else None,
        "line_rmse_m": line_err,
    }
    data = dict(report, image_size=size, source=c["points_file"], H_img_to_court=H.tolist())
    (out / "homography.json").write_text(json.dumps(data, indent=2))
    (out / "report.json").write_text(json.dumps(report, indent=2))
    frame = cv2.imread(str(stage_dir(cfg, "ingest") / "first_frame.jpg"))
    if frame is not None:
        vis = draw_court(frame, H, c["length_m"], c["width_m"], c["attack_line_m"], cam=cam)
        for p in img:
            cv2.circle(vis, tuple(int(v) for v in p), 6, (0, 0, 255), -1)
        for p in h_img:
            cv2.circle(vis, tuple(int(v) for v in p), 6, (255, 0, 255), 2)
        for pts in lines.values():
            for p in pts[::3]:
                cv2.circle(vis, tuple(int(v) for v in p), 2, (255, 128, 0), -1)
        cv2.imwrite(str(out / "overlay.jpg"), vis)
    return {k: report[k] for k in ["fit_rmse_m", "loo_rmse_m", "holdout_rmse_m"]}
