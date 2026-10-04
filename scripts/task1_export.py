"""Task 1 - export the universal DAE to ONNX and check it against PyTorch on real test images.

    python scripts/task1_export.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from genai import paths  # noqa: E402
from genai.data import PetData  # noqa: E402
from genai.models import build_autoencoder  # noqa: E402
from genai.onnx_utils import export_onnx, verify_onnx  # noqa: E402


def main():
    ck = torch.load(os.path.join(paths.CKPT_DIR, "task1_udae_final.pt"), map_location="cpu", weights_only=False)
    model = build_autoencoder(ck["model_cfg"]).eval()
    model.load_state_dict(ck["state_dict"])
    path = os.path.join(paths.ONNX_DIR, "task1_universal_dae.onnx")
    export_onnx(model, (torch.rand(1, 3, 128, 128),), path, ["input"], ["output"])

    data = PetData(paths.DATA_ROOT, paths.PET_CACHE, paths.MANIFEST_DIR)
    ds = data.test_loader(num_workers=0).dataset
    x = torch.stack([ds[i][0] for i in range(40)])   # 4 images x all 10 conditions
    out = os.path.join(paths.RESULTS_DIR, "task1")
    os.makedirs(out, exist_ok=True)
    rep = verify_onnx(model, path, x, report_path=os.path.join(out, "onnx_check.json"))
    assert rep["passed"], "ONNX output differs from PyTorch"


if __name__ == "__main__":
    main()
