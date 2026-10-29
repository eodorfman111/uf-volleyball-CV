import json
import os
from pathlib import Path

import cv2
import numpy as np


class VballNet:
    def __init__(self, model_path, threshold=0.5, providers=None):
        import onnxruntime as ort

        self.path = Path(model_path)
        meta_path = self.path.with_suffix(".json")
        self.meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        opts = ort.SessionOptions()
        threads = int(os.environ.get("SLURM_CPUS_PER_TASK", 0))
        if threads:
            opts.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(self.path), sess_options=opts, providers=providers or ["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        out = self.sess.get_outputs()[0]
        self.input_name = inp.name
        self.output_name = out.name
        _, self.seq, self.h, self.w = [d if isinstance(d, int) else None for d in inp.shape]
        self.seq = self.seq or self.meta.get("seq", 9)
        self.h = self.h or 288
        self.w = self.w or 512
        out_c = out.shape[1] if isinstance(out.shape[1], int) else self.seq
        self.planes = 2 if out_c == 2 * self.seq else 1
        self.threshold = threshold

    def prep(self, frame):
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.resize(g, (self.w, self.h)).astype(np.float32) / 255.0

    def infer(self, stack):
        x = np.stack(stack, axis=0)[None]
        return self.sess.run([self.output_name], {self.input_name: x})[0][0]

    def decode(self, hm):
        _, binary = cv2.threshold(hm, self.threshold, 1.0, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours((binary * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        m = cv2.moments(c)
        if m["m00"] == 0:
            x, y, _, _ = cv2.boundingRect(c)
            cx, cy = float(x), float(y)
        else:
            cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        conf = float(hm[int(min(cy, self.h - 1)), int(min(cx, self.w - 1))])
        return cx, cy, conf

    def run_video(self, frame_iter, width, height):
        buf = []
        idxs = []
        rows = []
        sx, sy = width / self.w, height / self.h
        for idx, frame in frame_iter:
            buf.append(self.prep(frame))
            idxs.append(idx)
            if len(buf) == self.seq:
                rows += self._flush(buf, idxs, sx, sy)
                buf, idxs = [], []
        if buf:
            n = len(buf)
            pad = buf + [buf[-1]] * (self.seq - n)
            rows += self._flush(pad, idxs + [None] * (self.seq - n), sx, sy)
        return [r for r in rows if r[0] is not None]

    def _flush(self, buf, idxs, sx, sy):
        out = self.infer(buf)
        rows = []
        for k, idx in enumerate(idxs):
            d = self.decode(out[k])
            if d is None:
                rows.append((idx, 0, -1.0, -1.0, 0.0))
            else:
                rows.append((idx, 1, d[0] * sx, d[1] * sy, d[2]))
        return rows


class HeatmapBall:
    def __init__(self, checkpoint, threshold=0.5, device="cpu", input_size=(960, 540), width=16):
        import torch

        from .heatnet import HeatNet

        self.torch = torch
        self.device = device
        self.threshold = threshold
        self.w, self.h = int(input_size[0]), int(input_size[1])
        self.net = HeatNet(width).to(device).eval()
        state = torch.load(checkpoint, map_location=device)
        self.net.load_state_dict(state["model"] if "model" in state else state)

    def prep(self, frame):
        return np.ascontiguousarray(cv2.resize(frame, (self.w, self.h)).astype(np.float32).transpose(2, 0, 1) / 255.0)

    def decode(self, hm):
        v, i = hm.flatten().max(0)
        v, i = float(v), int(i)
        if v < self.threshold:
            return None
        return float(i % self.w), float(i // self.w), v

    def run_video(self, frame_iter, width, height, batch=8):
        sx, sy = width / self.w, height / self.h
        rows, idxs, buf = [], [], []

        def flush():
            x = self.torch.from_numpy(np.stack(buf)).to(self.device)
            with self.torch.no_grad():
                hms = self.torch.sigmoid(self.net(x))[:, 0].cpu()
            for idx, hm in zip(idxs, hms):
                d = self.decode(hm)
                rows.append((idx, 0, -1.0, -1.0, 0.0) if d is None else (idx, 1, d[0] * sx, d[1] * sy, d[2]))

        for idx, frame in frame_iter:
            buf.append(self.prep(frame))
            idxs.append(idx)
            if len(buf) == batch:
                flush()
                buf, idxs = [], []
        if buf:
            flush()
        return rows
