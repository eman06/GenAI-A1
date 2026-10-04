"""Download Oxford-IIIT Pet, resize to 128x128 RGB, make the 80/20 split (seed 42)
and generate the deterministic validation and test corruption manifests.

    python scripts/prepare_data.py
"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genai import paths  # noqa: E402
from genai.data import build_test_manifest, build_val_manifest, load_pet_arrays, make_split, save_json  # noqa: E402


def main():
    paths.ensure_dirs()
    arrays = load_pet_arrays(paths.DATA_ROOT, paths.PET_CACHE)
    n_tv, n_test = len(arrays["trainval"]), len(arrays["test"])
    os.makedirs(paths.MANIFEST_DIR, exist_ok=True)

    split_path = os.path.join(paths.MANIFEST_DIR, "split.json")
    train, val = make_split(n_tv)
    save_json({"seed": 42, "n_trainval": n_tv, "train": train, "val": val}, split_path)
    print(f"split: {len(train)} train / {len(val)} val / {n_test} test (official test untouched)")

    val_m = build_val_manifest(val)
    save_json(val_m, os.path.join(paths.MANIFEST_DIR, "val_manifest.json"))
    print("val manifest:", Counter(e["type"] for e in val_m))

    test_path = os.path.join(paths.MANIFEST_DIR, "test_manifest.json")
    test_m = build_test_manifest(n_test)
    save_json(test_m, test_path)
    print(f"test manifest: {len(test_m)} entries ->", Counter((e["type"], e.get("level")) for e in test_m))


if __name__ == "__main__":
    main()
