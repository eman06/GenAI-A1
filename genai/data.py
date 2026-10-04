"""Oxford-IIIT Pet loading, the fixed 80/20 split, corruption manifests and DataLoaders."""
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .corruptions import CLASSES, CLASS_TO_ID, apply_corruption, make_level_spec, sample_train_spec

IMG_SIZE = 128
SPLIT_SEED = 42
VAL_MANIFEST_SEED = 1_000_003
TEST_MANIFEST_SEED = 2_000_003


# ---------------------------------------------------------------- raw images
def load_pet_arrays(data_root, cache_path):
    """Return {'trainval': uint8 (N,128,128,3), 'test': uint8 (M,128,128,3)}, cached as .npz."""
    if os.path.exists(cache_path):
        z = np.load(cache_path)
        return {"trainval": z["trainval"], "test": z["test"]}
    from torchvision.datasets import OxfordIIITPet
    out = {}
    for split in ("trainval", "test"):
        ds = OxfordIIITPet(data_root, split=split, target_types="category", download=True)
        imgs = [np.asarray(Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC))
                for p in ds._images]
        out[split] = np.stack(imgs).astype(np.uint8)
        print(f"{split}: {out[split].shape}")
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.savez(cache_path, **out)
    return out


def make_split(n, seed=SPLIT_SEED, val_frac=0.2):
    perm = np.random.default_rng(seed).permutation(n)
    n_val = int(round(n * val_frac))
    return sorted(perm[n_val:].tolist()), sorted(perm[:n_val].tolist())


# ---------------------------------------------------------------- manifests
def build_val_manifest(val_indices):
    """One random (training-distribution) corruption per validation image, fixed by seed."""
    entries = []
    for idx in val_indices:
        seed = VAL_MANIFEST_SEED + int(idx)
        spec = sample_train_spec(np.random.default_rng(seed))
        entries.append({"index": int(idx), "seed": seed, "label": CLASS_TO_ID[spec["type"]], **spec})
    return entries


def build_test_manifest(n_test):
    """Every test image x {clean, 3 corruptions x 3 fixed severities} = 10 entries per image."""
    entries = []
    for idx in range(n_test):
        entries.append({"index": idx, "seed": None, "label": 0, "type": "clean", "level": -1})
        for cid, ctype in enumerate(CLASSES[1:], start=1):
            for level in range(3):
                seed = TEST_MANIFEST_SEED + idx * 100 + cid * 10 + level
                spec = make_level_spec(np.random.default_rng(seed), ctype, level)
                entries.append({"index": idx, "seed": seed, "label": cid, **spec})
    return entries


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------- datasets
def to_chw(img):
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))


class CleanImages(Dataset):
    """Clean images only; corruption is applied at batch time by RuntimeCorruptCollate."""

    def __init__(self, images, indices):
        self.images, self.indices = images, list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.images[self.indices[i]].astype(np.float32) / 255.0


class RuntimeCorruptCollate:
    """Samples a fresh corruption type + severity for every image every time it is loaded.

    mode='uniform'  : each image independently draws a type uniformly from `types` (Task 1).
    mode='balanced' : every batch contains the same number of each type (Tasks 2-3 classifier/gate).
    Runs inside DataLoader workers; torch seeds each worker differently, so draws differ per worker.
    """

    def __init__(self, types=CLASSES, mode="uniform", size=IMG_SIZE):
        self.types, self.mode, self.size = list(types), mode, size

    def __call__(self, batch):
        rng = np.random.default_rng(int(torch.randint(0, 2 ** 31 - 1, (1,)).item()))
        n = len(batch)
        if self.mode == "balanced":
            chosen = [self.types[i % len(self.types)] for i in range(n)]
            rng.shuffle(chosen)
        else:
            chosen = [self.types[int(rng.integers(len(self.types)))] for _ in range(n)]
        noisy, clean, labels = [], [], []
        for img, ctype in zip(batch, chosen):
            spec = sample_train_spec(rng, ctype, self.size)
            noisy.append(to_chw(apply_corruption(img, spec)))
            clean.append(to_chw(img))
            labels.append(CLASS_TO_ID[ctype])
        return torch.stack(noisy), torch.stack(clean), torch.tensor(labels, dtype=torch.long)


class ManifestDataset(Dataset):
    """Deterministic corrupted samples defined by a validation/test manifest."""

    def __init__(self, images, manifest, types=None):
        self.images = images
        self.entries = [e for e in manifest if types is None or e["type"] in types]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        e = self.entries[i]
        clean = self.images[e["index"]].astype(np.float32) / 255.0
        noisy = apply_corruption(clean, e)
        return to_chw(noisy), to_chw(clean), e["label"], e.get("level", -1), i


# ---------------------------------------------------------------- convenience
class PetData:
    """Everything Tasks 1-3 need, loaded from the prepared outputs (see scripts/prepare_data.py)."""

    def __init__(self, data_root, cache_path, manifest_dir):
        arrays = load_pet_arrays(data_root, cache_path)
        self.trainval, self.test = arrays["trainval"], arrays["test"]
        split = load_json(os.path.join(manifest_dir, "split.json"))
        self.train_idx, self.val_idx = split["train"], split["val"]
        self.val_manifest = load_json(os.path.join(manifest_dir, "val_manifest.json"))
        self.test_manifest_path = os.path.join(manifest_dir, "test_manifest.json")
        self._test_manifest = None

    @property
    def test_manifest(self):
        if self._test_manifest is None:
            self._test_manifest = load_json(self.test_manifest_path)
        return self._test_manifest

    def train_loader(self, batch_size, types=CLASSES, mode="uniform", num_workers=2):
        return DataLoader(CleanImages(self.trainval, self.train_idx), batch_size=batch_size, shuffle=True,
                          drop_last=True, num_workers=num_workers, pin_memory=True,
                          persistent_workers=num_workers > 0,
                          collate_fn=RuntimeCorruptCollate(types, mode))

    def val_loader(self, batch_size=128, types=None, num_workers=2):
        return DataLoader(ManifestDataset(self.trainval, self.val_manifest, types), batch_size=batch_size,
                          shuffle=False, num_workers=num_workers, pin_memory=True)

    def test_loader(self, batch_size=128, types=None, num_workers=2):
        return DataLoader(ManifestDataset(self.test, self.test_manifest, types), batch_size=batch_size,
                          shuffle=False, num_workers=num_workers, pin_memory=True)
