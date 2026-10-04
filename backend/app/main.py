"""FastAPI backend: validates uploads, preprocesses, runs the ONNX models, returns images + routing + timing."""
import io
import os
import platform
import sys
import time
from typing import Optional

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
if not os.path.exists(os.path.join(HERE, "corruptions.py")):      # local dev: use the training code directly
    sys.path.insert(0, os.path.join(HERE, "..", "..", "genai"))
from corruptions import CLASSES, SEVERITY_NAMES, TEST_SEVERITIES, apply_corruption, make_level_spec  # noqa: E402
from inference import (BRANCHES, EXPERT_NAMES, STYLES, Registry, error_map, from_nchw, load_rgb,  # noqa: E402
                       png_b64, quality, to_nchw)

MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "models_onnx"))
SAMPLES_DIR = os.environ.get("SAMPLES_DIR", os.path.join(os.path.dirname(__file__), "samples"))
MAX_BYTES = 10 * 2 ** 20
ALLOWED = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/bmp"}
STARTED = time.time()

app = FastAPI(title="GenAI Assignment 1 - Restoration & Face-to-Sketch API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
REG = Registry(MODELS_DIR)


# ------------------------------------------------------------- helpers
async def read_image(file: Optional[UploadFile], sample: Optional[str]):
    if sample:
        path = os.path.join(SAMPLES_DIR, os.path.basename(sample))
        if not os.path.isfile(path):
            raise HTTPException(404, f"unknown sample {sample!r}")
        data, name = open(path, "rb").read(), os.path.basename(path)
    elif file is not None:
        if file.content_type not in ALLOWED:
            raise HTTPException(415, f"unsupported content type {file.content_type}; use PNG/JPEG/WebP/BMP")
        data, name = await file.read(), file.filename
        if len(data) > MAX_BYTES:
            raise HTTPException(413, "file larger than 10 MB")
    else:
        raise HTTPException(422, "provide an image file or a sample name")
    try:
        img, orig = load_rgb(data)
    except Exception:
        raise HTTPException(400, "could not decode the image")
    return img, {"name": name, "original_size": list(orig), "model_size": [128, 128]}


def corrupt(img, corruption: str, level: int, seed: int):
    if corruption in ("", "none", None):
        return img, None
    if corruption not in CLASSES:
        raise HTTPException(422, f"corruption must be one of none/{'/'.join(CLASSES)}")
    if corruption == "clean":
        return img, {"type": "clean"}
    if level not in (0, 1, 2):
        raise HTTPException(422, "level must be 0 (low), 1 (medium) or 2 (high)")
    spec = make_level_spec(np.random.default_rng(seed), corruption, level)
    spec["seed_used"] = seed
    spec["severity"] = SEVERITY_NAMES[level]
    return apply_corruption(img, spec), spec


def need(*keys):
    try:
        REG.require(*keys)
    except KeyError as e:
        raise HTTPException(503, f"model not available: {e}")


def package(clean, noisy, out, spec, meta, timing, extra):
    res = {"input": meta, "corruption": spec, "timing_ms": timing,
           "images": {"input": png_b64(noisy), "output": png_b64(out)}, **extra}
    if spec is not None:   # clean reference known -> report quality and an error map
        res["images"]["clean"] = png_b64(clean)
        res["images"]["error_map"] = png_b64(error_map(out, clean))
        res["metrics"] = {"input_vs_clean": quality(noisy, clean), "output_vs_clean": quality(out, clean)}
    else:
        res["images"]["error_map"] = png_b64(error_map(out, noisy))
        res["metrics"] = {"output_vs_input": quality(out, noisy)}
    return res


# ------------------------------------------------------------- endpoints
@app.get("/api/health")
def health():
    return {"status": "ok" if not REG.errors else "degraded", "uptime_s": round(time.time() - STARTED, 1),
            "models_loaded": sorted(REG.sessions), "models_missing": REG.errors,
            "onnxruntime": ort.__version__, "providers": ort.get_available_providers(),
            "python": platform.python_version(), "models_dir": os.path.abspath(MODELS_DIR)}


@app.get("/api/info")
def info():
    return {"models": REG.meta, "corruptions": {"types": CLASSES[1:], "levels": SEVERITY_NAMES,
                                                "settings": TEST_SEVERITIES},
            "styles": STYLES, "branches": BRANCHES}


@app.get("/api/samples")
def samples():
    if not os.path.isdir(SAMPLES_DIR):
        return {"samples": []}
    return {"samples": sorted(f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg")))}


@app.get("/api/samples/{name}")
def sample_file(name: str):
    path = os.path.join(SAMPLES_DIR, os.path.basename(name))
    if not os.path.isfile(path):
        raise HTTPException(404, "not found")
    return FileResponse(path)


@app.post("/api/corrupt")
async def corrupt_only(file: Optional[UploadFile] = File(None), sample: Optional[str] = Form(None),
                       corruption: str = Form("salt_pepper"), level: int = Form(1), seed: int = Form(0)):
    img, meta = await read_image(file, sample)
    noisy, spec = corrupt(img, corruption, level, seed)
    return {"input": meta, "corruption": spec, "images": {"clean": png_b64(img), "corrupted": png_b64(noisy)}}


@app.post("/api/restore/universal")
async def restore_universal(file: Optional[UploadFile] = File(None), sample: Optional[str] = Form(None),
                            corruption: str = Form("none"), level: int = Form(1), seed: int = Form(0)):
    need("universal")
    img, meta = await read_image(file, sample)
    noisy, spec = corrupt(img, corruption, level, seed)
    (out,), ms = REG.run("universal", {"input": to_nchw(noisy)})
    return package(img, noisy, from_nchw(out), spec, meta, {"universal_dae": round(ms, 2), "total": round(ms, 2)},
                   {"model": "Task 1 universal denoising autoencoder"})


@app.post("/api/restore/hard")
async def restore_hard(file: Optional[UploadFile] = File(None), sample: Optional[str] = Form(None),
                       corruption: str = Form("none"), level: int = Form(1), seed: int = Form(0)):
    need("classifier", "spec_salt_pepper", "spec_blur", "spec_occlusion")
    img, meta = await read_image(file, sample)
    noisy, spec = corrupt(img, corruption, level, seed)
    x = to_nchw(noisy)
    (probs,), t_clf = REG.run("classifier", {"input": x})
    probs = probs[0]
    k = int(np.argmax(probs))
    if k == 0:
        out, t_exp = noisy.copy(), 0.0           # identity bypass: clean images are not processed
    else:
        (o,), t_exp = REG.run(f"spec_{CLASSES[k]}", {"input": x})
        out = from_nchw(o)
    return package(img, noisy, out, spec, meta,
                   {"classifier": round(t_clf, 2), "expert": round(t_exp, 2), "total": round(t_clf + t_exp, 2)},
                   {"model": "Task 2 classifier + hard-routed specialists",
                    "probabilities": {c: round(float(p), 5) for c, p in zip(CLASSES, probs)},
                    "predicted": CLASSES[k], "selected_expert": EXPERT_NAMES[k],
                    "true_label": spec["type"] if spec else None})


@app.post("/api/restore/soft")
async def restore_soft(file: Optional[UploadFile] = File(None), sample: Optional[str] = Form(None),
                       corruption: str = Form("none"), level: int = Form(1), seed: int = Form(0)):
    need("soft_moe")
    img, meta = await read_image(file, sample)
    noisy, spec = corrupt(img, corruption, level, seed)
    (out, w), ms = REG.run("soft_moe", {"input": to_nchw(noisy)})
    w = w[0]
    order = np.argsort(-w)
    return package(img, noisy, from_nchw(out), spec, meta, {"soft_moe": round(ms, 2), "total": round(ms, 2)},
                   {"model": "Task 3 jointly trained soft mixture-of-experts",
                    "weights": {b: round(float(v), 5) for b, v in zip(BRANCHES, w)},
                    "dominant_expert": BRANCHES[int(order[0])],
                    "contributors": [BRANCHES[i] for i in order if w[i] >= 0.10],
                    "true_label": spec["type"] if spec else None})


@app.post("/api/sketch")
async def sketch(file: Optional[UploadFile] = File(None), sample: Optional[str] = Form(None), style: int = Form(0)):
    need("sketch")
    if style not in (0, 1, 2):
        raise HTTPException(422, "style must be 0, 1 or 2")
    img, meta = await read_image(file, sample)
    (sk,), ms = REG.run("sketch", {"photo": to_nchw(img), "style": np.array([style], dtype=np.int64)})
    return {"input": meta, "style": STYLES[style], "timing_ms": {"generator": round(ms, 2), "total": round(ms, 2)},
            "model": "Task 4 style-conditioned U-Net cGAN generator",
            "images": {"photo": png_b64(img), "sketch": png_b64(from_nchw(sk))}}
