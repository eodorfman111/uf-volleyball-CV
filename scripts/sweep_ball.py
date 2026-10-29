import argparse
import copy
import itertools
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.score_ball import score
from vbcv.config import load_config, run_dir, set_dotted
from vbcv.stages.ingest import load_meta
from vbcv.stages.track import ball_track


def parse_grid(items):
    grid = {}
    for item in items:
        key, vals = item.split("=", 1)
        grid[key] = vals.split("|")
    return grid


def check_keys(cfg, grid):
    known = cfg["track"]["ball"]
    for key in grid:
        parts = key.split(".")
        if key != "roi_top" and not (parts[:2] == ["track", "ball"] and len(parts) > 2 and parts[2] in known):
            raise ValueError(f"unsupported sweep key {key!r}: only roi_top and existing track.ball.* keys ({', '.join(sorted(known))}) are evaluated")


def apply(cfg, key, val, size):
    c = copy.deepcopy(cfg)
    if key == "roi_top":
        top, (w, h) = int(val), size
        c["track"]["ball"]["roi"] = [[0, h - 1], [0, top], [w - 1, top], [w - 1, h - 1]]
    else:
        set_dotted(c, key, val)
    return c


def sweep(cfg, det, labels, grid, split, size, radius=12.0, drop_interp=False):
    tune = labels[labels.frame < split]
    test = labels[labels.frame >= split]
    check_keys(cfg, grid)
    keys = list(grid)
    rows = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        c = cfg
        for k, v in zip(keys, combo):
            c = apply(c, k, v, size)
        out = ball_track(det, c["track"]["ball"])
        s_tune, s_test, s_all = (score(part, out, radius, drop_interp) for part in (tune, test, labels))
        rows.append(
            {
                "params": " ".join(f"{k}={v}" for k, v in zip(keys, combo)),
                "tune_p": s_tune["precision"],
                "tune_r": s_tune["recall"],
                "tune_f1": s_tune["f1"],
                "test_p": s_test["precision"],
                "test_r": s_test["recall"],
                "test_f1": s_test["f1"],
                "all_f1": s_all["f1"],
            }
        )
    return pd.DataFrame(rows).sort_values(["tune_f1", "all_f1"], ascending=False, kind="stable").reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--grid", action="append", default=[], metavar="KEY=V1|V2|V3")
    p.add_argument("--split", type=int, default=None)
    p.add_argument("--radius", type=float, default=12.0)
    p.add_argument("--drop-interp", action="store_true")
    a = p.parse_args()
    cfg = load_config(a.config)
    meta = load_meta(cfg)
    det = pd.read_csv(run_dir(cfg) / "detect" / "ball.csv")
    labels = pd.read_csv(a.labels)
    split = a.split if a.split is not None else int(labels.frame.median())
    table = sweep(cfg, det, labels, parse_grid(a.grid), split, (meta["width"], meta["height"]), a.radius, a.drop_interp)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 90)
    print(f"tune on frames < {split}, test on frames >= {split}")
    print(table.to_string())
    best = table.iloc[0]
    print(f"\nbest on the tuning half: {best.params}\n  tuning F1 {best.tune_f1}, held-out F1 {best.test_f1}")


if __name__ == "__main__":
    main()
