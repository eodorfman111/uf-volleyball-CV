import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vbcv.config import load_config, run_dir


def in_rally_mask(n, rallies):
    mask = np.zeros(n, bool)
    for r in rallies.itertuples():
        mask[max(int(r.start_frame), 0) : min(int(r.end_frame) + 1, n)] = True
    return mask


def pick(pool, count, rng, taken, gap):
    order = list(rng.permutation(pool))
    chosen = []
    for f in order:
        if len(chosen) >= count:
            break
        if all(abs(f - t) >= gap for t in taken) and all(abs(f - c) >= gap for c in chosen):
            chosen.append(int(f))
    return chosen


def sample(ball, rallies, n_rally, n_uncertain, n_negative, exclude, gap=15, seed=0, fractions=(0.6, 0.75)):
    n = int(ball.frame.max()) + 1
    vis = np.zeros(n, bool)
    conf = np.zeros(n)
    idx = ball.frame.to_numpy(int)
    vis[idx] = ball.visible.to_numpy() == 1
    conf[idx] = ball.conf.to_numpy()
    rally = in_rally_mask(n, rallies)
    allowed = np.ones(n, bool)
    for lo, hi in exclude:
        allowed[lo:hi] = False
    rng = np.random.default_rng(seed)
    taken, rows = [], []
    pools = [
        ("rally", np.flatnonzero(rally & allowed), n_rally),
        ("uncertain", np.flatnonzero(rally & allowed & (~vis | (conf < 0.8))), n_uncertain),
        ("false_alarm", np.flatnonzero(~rally & allowed & vis), n_negative // 2),
        ("dead_time", np.flatnonzero(~rally & allowed), n_negative - n_negative // 2),
    ]
    for reason, pool, count in pools:
        got = pick(pool, count, rng, taken, gap)
        taken += got
        rows += [(f, reason) for f in got]
    out = pd.DataFrame(rows, columns=["frame", "reason"]).sort_values("frame").reset_index(drop=True)
    out["split"] = np.where(out.frame < fractions[0] * n, "train", np.where(out.frame < fractions[1] * n, "val", "test"))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-rally", type=int, default=120)
    p.add_argument("--n-uncertain", type=int, default=80)
    p.add_argument("--n-negative", type=int, default=100)
    p.add_argument("--exclude", nargs="*", default=[], metavar="START:END")
    p.add_argument("--gap", type=int, default=15)
    a = p.parse_args()
    cfg = load_config(a.config)
    rd = run_dir(cfg)
    ball = pd.read_csv(rd / "detect" / "ball.csv")
    rallies = pd.read_csv(rd / "events" / "rallies.csv")
    exclude = [tuple(int(v) for v in e.split(":")) for e in a.exclude]
    out = sample(ball, rallies, a.n_rally, a.n_uncertain, a.n_negative, exclude, a.gap)
    out.to_csv(a.out, index=False)
    print(out.groupby(["split", "reason"]).size().to_string())
    print(f"wrote {len(out)} frames to {a.out}")


if __name__ == "__main__":
    main()
