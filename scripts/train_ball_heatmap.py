import argparse
import json
import random
import re
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def match_of(name):
    m = re.match(r"(match\d+)_", name)
    return m.group(1) if m else None


def items(split, matches=None):
    out = []
    for im in sorted(Path(split, "images").glob("*.jpg")):
        if matches and match_of(im.name) not in matches:
            continue
        lbl = Path(split, "labels", im.stem + ".txt")
        xy = None
        if lbl.exists():
            v = lbl.read_text().split()
            if len(v) >= 5:
                xy = (float(v[1]), float(v[2]))
        out.append((im, xy))
    return out


def block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True)
    )


class HeatNet(nn.Module):
    def __init__(self, w=16):
        super().__init__()
        self.e1, self.e2, self.e3 = block(3, w), block(w, 2 * w), block(2 * w, 4 * w)
        self.d2, self.d1 = block(6 * w, 2 * w), block(3 * w, w)
        self.head = nn.Conv2d(w, 1, 1)

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(F.max_pool2d(a, 2))
        c = self.e3(F.max_pool2d(b, 2))
        b = self.d2(torch.cat([F.interpolate(c, scale_factor=2), b], 1))
        a = self.d1(torch.cat([F.interpolate(b, scale_factor=2), a], 1))
        return self.head(a)


def load(item, size, sigma):
    im = cv2.imread(str(item[0]))
    im = cv2.resize(im, (size, size)).astype(np.float32) / 255.0
    hm = np.zeros((size, size), np.float32)
    if item[1] is not None:
        yy, xx = np.mgrid[0:size, 0:size]
        hm = np.exp(-((xx - item[1][0] * size) ** 2 + (yy - item[1][1] * size) ** 2) / (2 * sigma**2)).astype(np.float32)
    return torch.from_numpy(im.transpose(2, 0, 1)), torch.from_numpy(hm)[None]


def peak(hm, thr):
    v, i = hm.flatten().max(0)
    if float(v) < thr:
        return None
    return float(i % hm.shape[-1]), float(i // hm.shape[-1]), float(v)


def evaluate(model, data, size, thr, max_dist_px, orig):
    model.eval()
    tp = fp = fn = 0
    errs = []
    with torch.no_grad():
        for it in data:
            x, _ = load(it, size, 1)
            p = peak(torch.sigmoid(model(x[None]))[0, 0], thr)
            s = orig / size
            if it[1] is None:
                fp += p is not None
                continue
            if p is None:
                fn += 1
                continue
            d = np.hypot(p[0] * s - it[1][0] * orig, p[1] * s - it[1][1] * orig)
            if d <= max_dist_px:
                tp += 1
                errs.append(d)
            else:
                fp += 1
                fn += 1
    pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(pr, 4),
        "recall": round(rc, 4),
        "f1": round(2 * pr * rc / max(pr + rc, 1e-9), 4),
        "median_center_err_px": round(float(np.median(errs)), 2) if errs else None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--test-matches", nargs="+", default=["match17", "match18"])
    p.add_argument("--size", type=int, default=320)
    p.add_argument("--sigma", type=float, default=2.5)
    p.add_argument("--width", type=int, default=16)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-train", type=int, default=0)
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--max-dist-px", type=float, default=10.0)
    p.add_argument("--orig", type=int, default=960)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/train_ball_heatmap")
    a = p.parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    train = [it for s in ("train", "valid") for it in items(Path(a.data, s)) if match_of(it[0].name) not in a.test_matches]
    if a.max_train:
        random.shuffle(train)
        train = train[: a.max_train]
    test = items(Path(a.data, "test"), a.test_matches)
    model = HeatNet(a.width).to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    ck = out / "last.pt"
    step = 0
    if ck.exists():
        s = torch.load(ck, map_location=a.device)
        model.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        step = s["step"]
    t0 = time.time()
    log = open(out / "train_log.jsonl", "a")
    model.train()
    while step < a.steps:
        batch = random.sample(train, a.batch)
        xs, ys = zip(*[load(it, a.size, a.sigma) for it in batch])
        x, y = torch.stack(xs).to(a.device), torch.stack(ys).to(a.device)
        logit = model(x)
        loss = F.binary_cross_entropy_with_logits(logit, y, pos_weight=torch.tensor(100.0, device=a.device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
        if step % 50 == 0 or step == a.steps:
            log.write(json.dumps({"step": step, "loss": round(float(loss), 5), "t": round(time.time() - t0, 1)}) + "\n")
            log.flush()
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, ck)
    res = {"args": vars(a), "n_train": len(train), "n_test": len(test), "train_seconds": round(time.time() - t0, 1)}
    model.cpu()
    res["test"] = {str(t): evaluate(model, test, a.size, t, a.max_dist_px, a.orig) for t in (0.3, 0.5, 0.7)}
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
