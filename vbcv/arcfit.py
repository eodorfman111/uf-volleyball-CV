from itertools import combinations

import numpy as np


def _fit_exact(t, x, y):
    v = np.stack([np.ones_like(t), t, t * t], axis=1)
    try:
        return np.linalg.solve(v, np.stack([x, y], axis=1))
    except np.linalg.LinAlgError:
        return None


def _best_inliers(t, x, y, radius, rng, iters):
    n = len(t)
    if n <= 12:
        triples = list(combinations(range(n), 3))
    else:
        triples = [rng.choice(n, 3, replace=False) for _ in range(iters)]
    v = np.stack([np.ones_like(t), t, t * t], axis=1)
    best = np.zeros(n, bool)
    for tri in triples:
        idx = np.asarray(tri)
        coef = _fit_exact(t[idx], x[idx], y[idx])
        if coef is None:
            continue
        pred = v @ coef
        inl = np.hypot(pred[:, 0] - x, pred[:, 1] - y) <= radius
        if inl.sum() > best.sum():
            best = inl
    return best


def arc_filter_mask(frame, x, y, window=15, radius=6.0, min_inliers=5, iters=200, seed=0):
    frame = np.asarray(frame, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if min_inliers < 4:
        raise ValueError("min_inliers must be at least 4, since any 3 points fit a quadratic exactly")
    keep = np.zeros(len(frame), bool)
    if len(frame) < min_inliers:
        return keep
    order = np.argsort(frame, kind="stable")
    f, px, py = frame[order], x[order], y[order]
    rng = np.random.default_rng(seed)
    stride = max(1, window // 2)
    start = f[0]
    last = f[-1]
    while start <= last:
        lo = np.searchsorted(f, start, side="left")
        hi = np.searchsorted(f, start + window, side="left")
        if hi - lo >= min_inliers:
            sel = np.arange(lo, hi)
            inl = _best_inliers(f[sel] - start, px[sel], py[sel], radius, rng, iters)
            if inl.sum() >= min_inliers:
                keep[order[sel[inl]]] = True
        if hi == lo:
            start = f[lo] if lo < len(f) else last + 1
        else:
            start += stride
    return keep
