"""Task 2 - export the classifier and the three specialists to ONNX and verify them.

    python scripts/task2_export.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from genai import paths  # noqa: E402
from genai.data import PetData  # noqa: E402
from genai.onnx_utils import export_onnx, verify_onnx  # noqa: E402
from genai.routing import ClassifierExport  # noqa: E402
from task2_classifier import load_classifier  # noqa: E402
from task2_specialists import EXPERTS, load_specialists  # noqa: E402


def main():
    out = os.path.join(paths.RESULTS_DIR, "task2")
    os.makedirs(out, exist_ok=True)
    cpu = torch.device("cpu")
    data = PetData(paths.DATA_ROOT, paths.PET_CACHE, paths.MANIFEST_DIR)
    ds = data.test_loader(num_workers=0).dataset
    x = torch.stack([ds[i][0] for i in range(40)])
    dummy = (torch.rand(1, 3, 128, 128),)
    reports = {}

    clf = ClassifierExport(load_classifier(cpu)).eval()
    path = os.path.join(paths.ONNX_DIR, "task2_classifier.onnx")
    export_onnx(clf, dummy, path, ["input"], ["probs"])
    reports["classifier"] = verify_onnx(clf, path, x, atol=1e-4)

    for e, m in load_specialists(cpu).items():
        path = os.path.join(paths.ONNX_DIR, f"task2_specialist_{e}.onnx")
        export_onnx(m, dummy, path, ["input"], ["output"])
        reports[e] = verify_onnx(m, path, x, atol=1e-4)

    json.dump(reports, open(os.path.join(out, "onnx_check.json"), "w"), indent=2)
    assert all(r["passed"] for r in reports.values()), "ONNX mismatch"
    print("all Task 2 ONNX models verified:", list(reports))


if __name__ == "__main__":
    main()
