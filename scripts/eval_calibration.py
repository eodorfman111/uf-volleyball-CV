import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from vbcv.court import court_points, fit_homography, image_to_court


def grid(length=18.0, width=9.0, step=0.5):
    xs, ys = np.meshgrid(np.arange(0, width + 1e-9, step), np.arange(0, length + 1e-9, step))
    return np.stack([xs.ravel(), ys.ravel()], 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--points", required=True)
    p.add_argument("--noise-px", nargs="+", type=float, default=[1, 2, 4, 8])
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/eval_calibration")
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    spec = yaml.safe_load(open(a.points))
    ref = court_points()
    names = [k for k, v in spec["points"].items() if v is not None]
    img = np.array([spec["points"][k] for k in names], float)
    crt = np.array([ref[k] for k in names], float)
    H0 = fit_homography(img, crt)
    g = grid()
    gimg = image_to_court(np.linalg.inv(H0), g)
    near = g[:, 1] <= 9
    subsets = {"4_corners_near_center": ["near_left", "near_right", "center_left", "center_right"], "all": names}
    res = {"points_file": a.points, "trials": a.trials, "seed": a.seed, "results": {}}
    for sname, sub in subsets.items():
        sub = [s for s in sub if s in names]
        idx = [names.index(s) for s in sub]
        src = image_to_court(np.linalg.inv(H0), crt[idx])
        for sigma in a.noise_px:
            errs_near, errs_far = [], []
            for _ in range(a.trials):
                noisy = src + rng.normal(0, sigma, src.shape)
                H = fit_homography(noisy, crt[idx])
                e = np.linalg.norm(image_to_court(H, gimg) - g, axis=1)
                errs_near.append(np.median(e[near]))
                errs_far.append(np.median(e[~near]))
            res["results"][f"{sname}@{sigma}px"] = {
                "n_points": len(sub),
                "median_err_m_near_half": round(float(np.median(errs_near)), 4),
                "p90_err_m_near_half": round(float(np.percentile(errs_near, 90)), 4),
                "median_err_m_far_half": round(float(np.median(errs_far)), 4),
                "p90_err_m_far_half": round(float(np.percentile(errs_far, 90)), 4),
            }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
