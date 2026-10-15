import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vbcv.config import DEFAULTS
from vbcv.court import load_homography
from vbcv.kinematics import camera_from_h, fit_ballistic
from vbcv.stages.events import find_jumps

G = 9.81


def jump_track(h_m, fps, stride, px_per_m=100.0, body_m=1.8, pre_s=2.0):
    v = np.sqrt(2 * G * h_m)
    t_air = 2 * v / G
    n = int((2 * pre_s + t_air) * fps)
    f = np.arange(0, n, stride)
    t = f / fps - pre_s
    z = np.where((t > 0) & (t < t_air), v * t - G * t * t / 2, 0.0)
    y2 = 800 - z * px_per_m
    y1 = y2 - body_m * px_per_m
    return pd.DataFrame({"track_id": 1, "frame": f, "x1": 0, "x2": 50, "y1": y1, "y2": y2, "on_court": True, "side": "near", "court_x": 4.5, "court_y": 4.0})


def eval_jumps(fps_list, strides, heights):
    out = {}
    c = DEFAULTS["events"]["jump"]
    for fps in fps_list:
        for s in strides:
            rows = []
            for h in heights:
                j = find_jumps(jump_track(h, fps, s), fps, c)
                rows.append({"true_cm": round(h * 100), "detected": len(j), "est_cm": round(float(j.height_cm_flight.iloc[0]), 1) if len(j) else None})
            out[f"fps{fps}_stride{s}"] = rows
    return out


def eval_serves(H, P, fps, speeds_kmh, contact_z=2.8, receive_z=0.8, noise_px=2.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for v_kmh in speeds_kmh:
        start = np.array([4.5, -1.0, contact_z])
        end = np.array([4.5, 15.0, receive_z])
        dist = np.linalg.norm(end[:2] - start[:2])
        T = dist / (v_kmh / 3.6)
        vz = (end[2] - start[2] + G * T * T / 2) / T
        t = np.arange(0, T, 1 / fps)
        pts = np.stack([np.full_like(t, 4.5), -1.0 + (end[1] - start[1]) * t / T, start[2] + vz * t - G * t * t / 2, np.ones_like(t)], 1)
        img = (P @ pts.T).T
        img = img[:, :2] / img[:, 2:]
        g = np.c_[img, np.ones(len(img))] @ H.T
        g = g[:, :2] / g[:, 2:]
        proxy = np.linalg.norm(g[-1] - g[0]) / (t[-1] - t[0]) * 3.6
        true_avg = np.linalg.norm(pts[-1, :3] - pts[0, :3]) / (t[-1] - t[0]) * 3.6
        fits = [fit_ballistic(P, t, img + rng.normal(0, noise_px, img.shape)) for _ in range(20)]
        avg = np.array([q["avg_kmh"] for q in fits])
        rows.append(
            {
                "true_ground_kmh": v_kmh,
                "true_3d_avg_kmh": round(float(true_avg), 1),
                "proxy_kmh": round(float(proxy), 1),
                "proxy_err_pct": round(float((proxy - v_kmh) / v_kmh * 100), 1),
                "ballistic_avg_kmh_median": round(float(np.median(avg)), 1),
                "ballistic_err_pct_median": round(float((np.median(avg) - true_avg) / true_avg * 100), 1),
                "ballistic_err_pct_p90_abs": round(float(np.percentile(np.abs(avg - true_avg) / true_avg * 100, 90)), 1),
                "noise_px": noise_px,
            }
        )
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--homography", default="runs/public_gtu/calibrate/homography.json")
    p.add_argument("--out", default="runs/eval_kinematics")
    a = p.parse_args()
    res = {"jump": eval_jumps([30, 60], [1, 2], [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]), "jump_rule": DEFAULTS["events"]["jump"]}
    H, meta, cam = load_homography(a.homography)
    size = meta.get("image_size", [1920, 1080]) if isinstance(meta, dict) else [1920, 1080]
    c = camera_from_h(H, size)
    if c is None:
        res["serve"] = "focal length not recoverable from this homography"
    else:
        res["serve_camera_f_px"] = round(c["f_px"], 1)
        res["serve"] = {f"fps{fps}": eval_serves(H, c["P"], fps, [40, 60, 80, 100]) for fps in (30, 60)}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
