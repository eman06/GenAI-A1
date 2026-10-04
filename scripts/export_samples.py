"""Save a few official-test-split images as app samples (outputs/samples): 6 pets + 4 FS2K faces."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from genai import paths  # noqa: E402

out = os.path.join(paths.OUT_ROOT, "samples")
os.makedirs(out, exist_ok=True)
pets = np.load(paths.PET_CACHE)["test"]
for i in np.linspace(0, len(pets) - 1, 6).astype(int):
    Image.fromarray(pets[i]).save(os.path.join(out, f"pet_test_{i:04d}.png"))
fs = os.path.join(paths.DATA_ROOT, "fs2k_128.npz")
if os.path.exists(fs):
    z = np.load(fs)
    for i in np.linspace(0, len(z["test_photo"]) - 1, 4).astype(int):
        Image.fromarray(z["test_photo"][i]).save(os.path.join(out, f"face_test_{i:04d}.png"))
print(sorted(os.listdir(out)))
