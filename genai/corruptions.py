"""Corruption definitions shared by training, evaluation and the FastAPI backend.

Pure NumPy on purpose: the backend container can import this file without
PyTorch, so the corruptions applied in the app are bit-identical to the ones
used for training and testing.

Images are float32 arrays in [0, 1] with shape (H, W, 3).
"""
import numpy as np

CLASSES = ["clean", "salt_pepper", "blur", "occlusion"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
SEVERITY_NAMES = ["low", "medium", "high"]

# Fixed test severities (assignment spec).
TEST_SEVERITIES = {
    "salt_pepper": [{"prob": 0.03}, {"prob": 0.08}, {"prob": 0.15}],
    "blur": [{"kernel": 3, "sigma": 0.7}, {"kernel": 5, "sigma": 1.5}, {"kernel": 7, "sigma": 2.5}],
    "occlusion": [{"n_rects": 1, "target_frac": 0.10},
                  {"n_rects": 2, "target_frac": 0.20},
                  {"n_rects": 3, "target_frac": 0.35}],
}

# Training ranges (assignment spec).
TRAIN_SP_RANGE = (0.02, 0.15)
TRAIN_BLUR_KERNELS = (3, 5, 7)
TRAIN_BLUR_SIGMA = (0.5, 2.5)
TRAIN_OCC_RECTS = (1, 3)
TRAIN_OCC_FRAC = (0.10, 0.35)


# ---------------------------------------------------------------- primitives
def salt_pepper(img, prob, seed):
    """Replace a `prob` fraction of pixels (all channels) by black or white, 50/50."""
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    out = img.copy()
    hit = rng.random((h, w)) < prob
    white = rng.random((h, w)) < 0.5
    out[hit & white] = 1.0
    out[hit & ~white] = 0.0
    return out


def gaussian_kernel1d(kernel, sigma):
    x = np.arange(kernel, dtype=np.float64) - (kernel - 1) / 2.0
    g = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return (g / g.sum()).astype(np.float32)


def gaussian_blur(img, kernel, sigma):
    """Separable Gaussian blur with reflect padding (same convention as torchvision)."""
    g = gaussian_kernel1d(kernel, sigma)
    r = kernel // 2
    h, w = img.shape[:2]
    p = np.pad(img, ((r, r), (r, r), (0, 0)), mode="reflect")
    tmp = sum(g[i] * p[:, i:i + w] for i in range(kernel))      # (H+2r, W, C)
    out = sum(g[i] * tmp[i:i + h] for i in range(kernel))       # (H, W, C)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def occlude(img, rects):
    out = img.copy()
    for y0, x0, y1, x1 in rects:
        out[y0:y1, x0:x1] = 0.0
    return out


def rect_coverage(rects, size):
    mask = np.zeros((size, size), dtype=bool)
    for y0, x0, y1, x1 in rects:
        mask[y0:y1, x0:x1] = True
    return float(mask.mean())


def sample_rects(rng, n_rects, target_frac, size=128, equal_split=False, tol=0.015, max_tries=500):
    """Sample `n_rects` non-overlapping rectangles jointly covering ~target_frac of the image.

    Area is split equally (test) or by a Dirichlet draw (training), aspect ratio is
    log-uniform in [1/2, 2], and placement is uniform with rejection of overlaps.
    """
    total = target_frac * size * size
    for _ in range(max_tries):
        shares = np.full(n_rects, 1.0 / n_rects) if equal_split else rng.dirichlet(np.full(n_rects, 3.0))
        rects, ok = [], True
        mask = np.zeros((size, size), dtype=bool)
        for share in shares:
            area = share * total
            aspect = float(np.exp(rng.uniform(np.log(0.5), np.log(2.0))))
            h = int(round(np.sqrt(area * aspect)))
            w = int(round(area / max(h, 1)))
            h, w = int(np.clip(h, 4, size)), int(np.clip(w, 4, size))
            placed = False
            for _ in range(100):
                y0 = int(rng.integers(0, size - h + 1))
                x0 = int(rng.integers(0, size - w + 1))
                if not mask[y0:y0 + h, x0:x0 + w].any():
                    mask[y0:y0 + h, x0:x0 + w] = True
                    rects.append([y0, x0, y0 + h, x0 + w])
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok and abs(mask.mean() - target_frac) <= tol:
            return rects
    raise RuntimeError(f"could not place {n_rects} rects covering {target_frac:.2f}")


# ---------------------------------------------------------------- specs
def apply_corruption(img, spec):
    t = spec["type"]
    if t == "clean":
        return img.copy()
    if t == "salt_pepper":
        return salt_pepper(img, spec["prob"], spec["seed"])
    if t == "blur":
        return gaussian_blur(img, spec["kernel"], spec["sigma"])
    if t == "occlusion":
        return occlude(img, spec["rects"])
    raise ValueError(f"unknown corruption type {t!r}")


def sample_train_spec(rng, ctype=None, size=128):
    """Draw a random training corruption. ctype=None -> uniform over the 4 conditions."""
    if ctype is None:
        ctype = CLASSES[int(rng.integers(len(CLASSES)))]
    spec = {"type": ctype}
    if ctype == "salt_pepper":
        spec.update(prob=float(rng.uniform(*TRAIN_SP_RANGE)), seed=int(rng.integers(2 ** 31 - 1)))
    elif ctype == "blur":
        spec.update(kernel=int(rng.choice(TRAIN_BLUR_KERNELS)), sigma=float(rng.uniform(*TRAIN_BLUR_SIGMA)))
    elif ctype == "occlusion":
        n = int(rng.integers(TRAIN_OCC_RECTS[0], TRAIN_OCC_RECTS[1] + 1))
        frac = float(rng.uniform(*TRAIN_OCC_FRAC))
        rects = sample_rects(rng, n, frac, size)
        spec.update(n_rects=n, target_frac=frac, rects=rects, coverage=rect_coverage(rects, size))
    return spec


def make_level_spec(rng, ctype, level, size=128):
    """Fixed-severity spec (level 0/1/2 = low/medium/high) used for test and in the app."""
    if ctype == "clean":
        return {"type": "clean", "level": -1}
    spec = {"type": ctype, "level": level, **TEST_SEVERITIES[ctype][level]}
    if ctype == "salt_pepper":
        spec["seed"] = int(rng.integers(2 ** 31 - 1))
    elif ctype == "occlusion":
        rects = sample_rects(rng, spec["n_rects"], spec["target_frac"], size, equal_split=True)
        spec.update(rects=rects, coverage=rect_coverage(rects, size))
    return spec
