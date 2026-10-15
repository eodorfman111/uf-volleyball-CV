import numpy as np


class DFineDetector:
    def __init__(self, model_id, conf=0.4, device="cpu", classes=("person",)):
        import torch
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        self.torch = torch
        self.proc = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForObjectDetection.from_pretrained(model_id).to(device).eval()
        self.device = device
        self.conf = conf
        self.keep = {i for i, n in self.model.config.id2label.items() if n in classes}
        self.names = self.model.config.id2label

    def __call__(self, frames_bgr):
        from PIL import Image

        imgs = [Image.fromarray(f[:, :, ::-1]) for f in frames_bgr]
        with self.torch.no_grad():
            inputs = self.proc(images=imgs, return_tensors="pt").to(self.device)
            out = self.model(**inputs)
        sizes = self.torch.tensor([(f.shape[0], f.shape[1]) for f in frames_bgr])
        res = self.proc.post_process_object_detection(out, threshold=self.conf, target_sizes=sizes)
        dets = []
        for r in res:
            rows = [
                (*b, s, int(l))
                for b, s, l in zip(r["boxes"].tolist(), r["scores"].tolist(), r["labels"].tolist())
                if int(l) in self.keep
            ]
            dets.append(np.array(rows, dtype=np.float32).reshape(-1, 6))
        return dets


class UltralyticsDetector:
    def __init__(self, model_id, conf=0.4, device="cpu", classes=("person",)):
        from ultralytics import YOLO

        self.model = YOLO(model_id)
        self.conf = conf
        self.device = device
        self.keep = [i for i, n in self.model.names.items() if n in classes]

    def __call__(self, frames_bgr):
        res = self.model.predict(frames_bgr, conf=self.conf, classes=self.keep, device=self.device, verbose=False)
        dets = []
        for r in res:
            b = r.boxes
            rows = np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None], b.cls.cpu().numpy()[:, None]], axis=1)
            dets.append(rows.astype(np.float32))
        return dets


def make_detector(cfg):
    c = cfg["detect"]["players"]
    if c["backend"] == "dfine":
        return DFineDetector(c["model"], c["conf"], c["device"])
    if c["backend"] == "ultralytics":
        return UltralyticsDetector(c["model"], c["conf"], c["device"])
    raise ValueError(f"unknown player backend {c['backend']}")
