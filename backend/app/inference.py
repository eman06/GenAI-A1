"""ONNX Runtime model registry + pre/post-processing shared by all endpoints."""
import base64
import io
import os
import time

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

IMG = 128
CLASSES = ["clean", "salt_pepper", "blur", "occlusion"]
EXPERT_NAMES = {0: "identity bypass", 1: "salt & pepper specialist", 2: "blur specialist", 3: "occlusion specialist"}
BRANCHES = ["identity", "salt_pepper", "blur", "occlusion"]
STYLES = ["Style 1", "Style 2", "Style 3"]

MODEL_FILES = {
    "universal": "task1_universal_dae.onnx",
    "universal_skip": "task1_universal_dae_skip.onnx",
    "classifier": "task2_classifier.onnx",
    "spec_salt_pepper": "task2_specialist_salt_pepper.onnx",
    "spec_blur": "task2_specialist_blur.onnx",
    "spec_occlusion": "task2_specialist_occlusion.onnx",
    "soft_moe": "task3_soft_moe.onnx",
    "sketch": "task4_generator.onnx",
}


class Registry:
    def __init__(self, models_dir):
        self.models_dir = models_dir
        self.sessions, self.errors, self.meta = {}, {}, {}
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "2"))
        for key, fname in MODEL_FILES.items():
            path = os.path.join(models_dir, fname)
            if not os.path.exists(path):
                self.errors[key] = f"missing {fname}"
                continue
            try:
                s = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
                self.sessions[key] = s
                self.meta[key] = {
                    "file": fname, "size_mb": round(os.path.getsize(path) / 2 ** 20, 2),
                    "inputs": [{"name": i.name, "shape": i.shape, "type": i.type} for i in s.get_inputs()],
                    "outputs": [{"name": o.name, "shape": o.shape, "type": o.type} for o in s.get_outputs()],
                }
            except Exception as e:  # corrupted file etc.
                self.errors[key] = f"{fname}: {e}"

    def require(self, *keys):
        missing = [k for k in keys if k not in self.sessions]
        if missing:
            raise KeyError(", ".join(f"{k} ({self.errors.get(k, 'not loaded')})" for k in missing))

    def run(self, key, feeds):
        s = self.sessions[key]
        t0 = time.perf_counter()
        out = s.run(None, feeds)
        return out, (time.perf_counter() - t0) * 1000.0


# ------------------------------------------------------------- images
def load_rgb(data: bytes) -> np.ndarray:
    """Decode, fix EXIF rotation, RGB, resize to 128x128 (bicubic, as in training), float32 [0,1] HWC."""
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img).convert("RGB")
    orig = img.size
    img = img.resize((IMG, IMG), Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0, orig


def to_nchw(img):
    return np.ascontiguousarray(img.transpose(2, 0, 1)[None]).astype(np.float32)


def from_nchw(x):
    x = np.clip(x[0], 0, 1)
    return x.transpose(1, 2, 0) if x.shape[0] == 3 else x[0]


def png_b64(img, upscale=None) -> str:
    arr = (np.clip(img, 0, 1) * 255).round().astype(np.uint8)
    im = Image.fromarray(arr)
    if upscale:
        im = im.resize((upscale, upscale), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def error_map(a, b):
    """|a-b| averaged over channels, mapped to a simple inferno-like colour ramp."""
    e = np.abs(a - b).mean(-1) if a.ndim == 3 else np.abs(a - b)
    t = np.clip(e / 0.5, 0, 1)[..., None]
    c0, c1, c2 = np.array([0, 0, 0.02]), np.array([0.73, 0.21, 0.33]), np.array([0.99, 0.95, 0.6])
    return np.where(t < 0.5, c0 + (c1 - c0) * (t / 0.5), c1 + (c2 - c1) * ((t - 0.5) / 0.5)).astype(np.float32)


def quality(a, b):
    mse = float(np.mean((a - b) ** 2))
    return {"psnr": round(min(50.0, 10 * np.log10(1.0 / max(mse, 1e-10))), 2), "l1": round(float(np.mean(np.abs(a - b))), 4)}
