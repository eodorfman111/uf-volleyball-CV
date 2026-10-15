import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def list_frames(root):
    rows = []
    for label, sub in [(1, "rally"), (0, "non_rally")]:
        for p in sorted((Path(root) / sub).glob("*.jpg")):
            m = re.match(r"(match\d+)_(\d+)", p.stem)
            rows.append((str(p), label, m.group(1), int(m.group(2))))
    return rows


def embed(paths, device, batch=64):
    w = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = torchvision.models.mobilenet_v3_small(weights=w).to(device).eval()
    model.classifier = torch.nn.Identity()
    tf = w.transforms()
    feats = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i : i + batch]]).to(device)
            feats.append(model(x).cpu().numpy())
            if i % (batch * 20) == 0:
                print(f"embedded {i}/{len(paths)}", flush=True)
    return np.concatenate(feats)


def smooth_by_match(prob, matches, idx, k):
    out = prob.copy()
    if k <= 1:
        return out
    for m in np.unique(matches):
        sel = np.flatnonzero(matches == m)
        order = sel[np.argsort(idx[sel])]
        p = prob[order]
        pad = np.pad(p, (k // 2, k // 2), mode="edge")
        out[order] = np.convolve(pad, np.ones(k) / k, mode="valid")
    return out


def metrics(y, p):
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, p)),
        "precision": float(precision_score(y, p)),
        "recall": float(recall_score(y, p)),
        "f1": float(f1_score(y, p)),
    }


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--data", required=True)
    a.add_argument("--out", default="runs/eval_rally_frames")
    a.add_argument("--test-matches", nargs="+", default=["match13", "match15", "match4"])
    a.add_argument("--device", default="cpu")
    a.add_argument("--smooth", type=int, default=5)
    args = a.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = list_frames(args.data)
    paths = [r[0] for r in rows]
    y = np.array([r[1] for r in rows])
    matches = np.array([r[2] for r in rows])
    idx = np.array([r[3] for r in rows])
    cache = out / "features.npy"
    if cache.exists():
        X = np.load(cache)
    else:
        X = embed(paths, args.device)
        np.save(cache, X)
    test = np.isin(matches, args.test_matches)
    clf = LogisticRegression(max_iter=3000, C=1.0)
    clf.fit(X[~test], y[~test])
    prob = clf.predict_proba(X)[:, 1]
    sm = smooth_by_match(prob, matches, idx, args.smooth)
    res = {
        "data": args.data,
        "train_matches": sorted(set(matches[~test])),
        "test_matches": args.test_matches,
        "n_train": int((~test).sum()),
        "majority_baseline_acc": float(max(y[test].mean(), 1 - y[test].mean())),
        "test_raw": metrics(y[test], (prob[test] > 0.5).astype(int)),
        f"test_smoothed_k{args.smooth}": metrics(y[test], (sm[test] > 0.5).astype(int)),
        "per_match": {m: metrics(y[matches == m], (sm[matches == m] > 0.5).astype(int)) for m in args.test_matches},
        "seconds": round(time.time() - t0, 1),
    }
    errs = [(paths[i], int(y[i]), float(sm[i])) for i in np.flatnonzero(test & ((sm > 0.5).astype(int) != y))]
    errs.sort(key=lambda e: -abs(e[2] - e[1]))
    (out / "failures.json").write_text(json.dumps(errs[:50], indent=1))
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    import joblib

    joblib.dump(clf, out / "rally_logreg.joblib")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
