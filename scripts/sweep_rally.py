import argparse
import copy
import itertools
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vbcv.config import load_config, run_dir, set_dotted
from vbcv.stages.events import segment_rallies
from vbcv.stages.ingest import load_meta


def parse_grid(items):
    grid = {}
    for item in items:
        key, vals = item.split("=", 1)
        grid[key] = vals.split("|")
    return grid


def check_keys(cfg, grid):
    known = cfg["events"]["rally"]
    for key in grid:
        parts = key.split(".")
        if not (parts[:2] == ["events", "rally"] and len(parts) == 3 and parts[2] in known):
            raise ValueError(f"unsupported sweep key {key!r}: only existing events.rally.* keys ({', '.join(sorted(known))}) are evaluated")


def evaluate(rallies, truth, tol):
    spans = list(zip(rallies.start_frame, rallies.end_frame))
    covered = [any(s - tol <= t <= e for s, e in spans) for t in truth]
    false = [not any(s - tol <= t <= e for t in truth) for s, e in spans] if truth else []
    return sum(covered), sum(false)


def sweep(cfg, ball, meta, truth, grid, tol, target=None):
    check_keys(cfg, grid)
    target = len(truth) if truth else target
    keys = list(grid)
    rows = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        c = copy.deepcopy(cfg)
        for k, v in zip(keys, combo):
            set_dotted(c, k, v)
        rallies = segment_rallies(ball, meta, c["events"]["rally"])
        covered, false = evaluate(rallies, truth, tol)
        rows.append(
            {
                "params": " ".join(f"{k.split('.')[-1]}={v}" for k, v in zip(keys, combo)),
                "rallies": len(rallies),
                "play_s": round(float((rallies.end_frame - rallies.start_frame).sum() / meta["fps"]), 1),
                "plays_found": f"{covered}/{len(truth)}",
                "missed": len(truth) - covered,
                "false_rallies": false,
            }
        )
    out = pd.DataFrame(rows)
    out["off_by"] = (out.rallies - (target if target is not None else len(truth))).abs()
    return out.sort_values(["missed", "false_rallies", "off_by", "play_s"], kind="stable").reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--plays", default="", help="comma-separated frame numbers, one per real play (serve contact)")
    p.add_argument("--target", type=int, default=None, help="known number of plays when the frames are not available")
    p.add_argument("--grid", action="append", default=[], metavar="KEY=V1|V2|V3")
    p.add_argument("--tol", type=int, default=45)
    a = p.parse_args()
    cfg = load_config(a.config)
    meta = load_meta(cfg)
    ball = pd.read_csv(run_dir(cfg) / "track" / "ball.csv")
    truth = [int(v) for v in a.plays.split(",") if v]
    pd.set_option("display.width", 250)
    print(sweep(cfg, ball, meta, truth, parse_grid(a.grid), a.tol, a.target).to_string())


if __name__ == "__main__":
    main()
