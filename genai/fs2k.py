"""FS2K loading: official train/test annotations, 15% style-stratified validation split (seed 42),
paired 128x128 photo / sketch arrays, and pair-consistent augmentation."""
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMG = 128
STYLE_NAMES = ["Style 1", "Style 2", "Style 3"]


def _find(base):
    for ext in (".jpg", ".png", ".jpeg", ".JPG", ".PNG"):
        if os.path.exists(base + ext):
            return base + ext
    raise FileNotFoundError(base)


def _load_anno(path):
    with open(path) as f:
        a = json.load(f)
    if isinstance(a, dict):                       # tolerate {"...": [entries]} layouts
        a = next(v for v in a.values() if isinstance(v, list))
    return a


def find_root(data_root):
    for dirpath, dirnames, filenames in os.walk(data_root):
        if "anno_train.json" in filenames and "photo" in dirnames:
            return dirpath
    raise FileNotFoundError(f"FS2K (anno_train.json + photo/) not found under {data_root}")


def load_split(root, split):
    photos, sketches, styles, names = [], [], [], []
    for e in _load_anno(os.path.join(root, f"anno_{split}.json")):
        name = e["image_name"]                                   # e.g. photo1/image0110
        p = _find(os.path.join(root, "photo", name))
        s = _find(os.path.join(root, "sketch", name.replace("image", "sketch").replace("photo", "sketch")))
        photos.append(np.asarray(Image.open(p).convert("RGB").resize((IMG, IMG), Image.BICUBIC)))
        sketches.append(np.asarray(Image.open(s).convert("L").resize((IMG, IMG), Image.BICUBIC)))
        styles.append(int(e["style"]))
        names.append(name)
    return {"photo": np.stack(photos), "sketch": np.stack(sketches)[..., None],
            "style": np.array(styles, dtype=np.int64), "name": np.array(names)}


def prepare(data_root, cache_path, split_path, val_frac=0.15, seed=42):
    from sklearn.model_selection import train_test_split
    root = find_root(data_root)
    tr, te = load_split(root, "train"), load_split(root, "test")
    idx = np.arange(len(tr["style"]))
    tr_idx, va_idx = train_test_split(idx, test_size=val_frac, random_state=seed, stratify=tr["style"])
    np.savez(cache_path, **{f"train_{k}": v for k, v in tr.items()}, **{f"test_{k}": v for k, v in te.items()})
    split = {"seed": seed, "val_frac": val_frac, "train": sorted(tr_idx.tolist()), "val": sorted(va_idx.tolist()),
             "n_official_train": int(len(idx)), "n_test": int(len(te["style"])),
             "style_counts": {part: np.bincount(tr["style"][ids] if part != "test" else te["style"], minlength=3).tolist()
                              for part, ids in (("train", tr_idx), ("val", va_idx), ("test", None))}}
    with open(split_path, "w") as f:
        json.dump(split, f, indent=1)
    return split


class PairDataset(Dataset):
    """Returns (photo [-1,1] 3xHxW, sketch [-1,1] 1xHxW, style). With augment=True the SAME random
    flip and the SAME reflect-pad-and-crop offset are applied to the photo and its sketch."""

    def __init__(self, photo, sketch, style, augment=False, pad=8):
        self.photo, self.sketch, self.style, self.augment, self.pad = photo, sketch, style, augment, pad

    def __len__(self):
        return len(self.style)

    def __getitem__(self, i):
        p = self.photo[i].astype(np.float32) / 127.5 - 1
        s = self.sketch[i].astype(np.float32) / 127.5 - 1
        if self.augment:
            if np.random.rand() < 0.5:
                p, s = p[:, ::-1], s[:, ::-1]
            k = self.pad
            pp = np.pad(p, ((k, k), (k, k), (0, 0)), mode="reflect")
            sp = np.pad(s, ((k, k), (k, k), (0, 0)), mode="reflect")
            y, x = np.random.randint(0, 2 * k + 1, size=2)
            p, s = pp[y:y + IMG, x:x + IMG], sp[y:y + IMG, x:x + IMG]
        return (torch.from_numpy(np.ascontiguousarray(p.transpose(2, 0, 1))),
                torch.from_numpy(np.ascontiguousarray(s.transpose(2, 0, 1))), int(self.style[i]))


class FS2KData:
    def __init__(self, cache_path, split_path):
        z = np.load(cache_path)
        with open(split_path) as f:
            self.split = json.load(f)
        tr, va = self.split["train"], self.split["val"]
        self.train = {k: z[f"train_{k}"][tr] for k in ("photo", "sketch", "style", "name")}
        self.val = {k: z[f"train_{k}"][va] for k in ("photo", "sketch", "style", "name")}
        self.test = {k: z[f"test_{k}"] for k in ("photo", "sketch", "style", "name")}

    def loader(self, part, batch_size, shuffle=False, augment=False, workers=2):
        d = getattr(self, part)
        return DataLoader(PairDataset(d["photo"], d["sketch"], d["style"], augment), batch_size=batch_size,
                          shuffle=shuffle, drop_last=shuffle, num_workers=workers, pin_memory=True,
                          worker_init_fn=lambda w: np.random.seed(torch.initial_seed() % 2 ** 32))
