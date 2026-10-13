"""Re-export the trained soft MoE with a different gate temperature tau (for live demos).

    python scripts/export_moe_with_tau.py --tau 0.5      # sharper routing
    python scripts/export_moe_with_tau.py --tau 5        # more spread-out routing
    python scripts/export_moe_with_tau.py --restore      # back to the trained value
    docker compose restart backend                       # backend reloads the ONNX file

tau is a constant inside the exported graph: w = softmax(G(x) / tau). Changing it does not change any
weights, only how sharp the routing distribution is (tau -> 0: hard argmax routing, tau -> inf: uniform).
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from genai.models import ConvAutoencoder, CorruptionClassifier  # noqa: E402
from genai.onnx_utils import export_onnx  # noqa: E402
from genai.routing import SoftMoEExport, SoftMoERestorer  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX = os.path.join(ROOT, "models_onnx", "task3_soft_moe.onnx")
BACKUP = ONNX.replace(".onnx", "_trained_tau.onnx")
CFG = json.load(open(os.path.join(ROOT, "configs", "final_configs.json")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float)
    ap.add_argument("--restore", action="store_true")
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "drive_results", "outputs", "checkpoints", "task3_moe.pt"))
    a = ap.parse_args()
    if a.restore:
        shutil.copyfile(BACKUP, ONNX)
        print("restored the originally exported model (tau =", CFG["task3_soft_moe"]["tau"], ")")
        return
    if not os.path.exists(BACKUP):
        shutil.copyfile(ONNX, BACKUP)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    clf, spec = CFG["task2_classifier"], CFG["task2_specialists"]
    ch = {"small": (16, 32, 64, 128), "medium": (32, 64, 128, 256), "wide": (48, 96, 192, 256)}[clf["channels"]]
    moe = SoftMoERestorer(CorruptionClassifier(ch, clf["dropout"]),
                          [ConvAutoencoder(spec["base_ch"], spec["latent_ch"]) for _ in range(3)])
    moe.load_state_dict(ck["state_dict"])
    moe.tau = a.tau
    export_onnx(SoftMoEExport(moe).eval(), (torch.rand(1, 3, 128, 128),), ONNX, ["input"], ["output", "weights"])
    print(f"exported {ONNX} with tau = {a.tau} (trained value {ck['cfg']['tau']:.2f}); now run: docker compose restart backend")


if __name__ == "__main__":
    main()
