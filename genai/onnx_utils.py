"""ONNX export + numerical consistency check against PyTorch."""
import json
import os

import numpy as np
import torch


def export_onnx(model, example_inputs, path, input_names, output_names, opset=17):
    model.eval()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    dynamic_axes = {n: {0: "batch"} for n in input_names + output_names}
    kwargs = dict(input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes, opset_version=opset)
    try:  # TorchScript exporter (stable for these simple CNNs)
        torch.onnx.export(model, example_inputs, path, dynamo=False, **kwargs)
    except TypeError:  # older torch without the `dynamo` kwarg
        torch.onnx.export(model, example_inputs, path, **kwargs)
    import onnx
    onnx.checker.check_model(onnx.load(path))
    return path


def verify_onnx(model, path, inputs, atol=1e-4, report_path=None):
    """Run PyTorch and ONNX Runtime on the same real inputs and compare every output."""
    import onnxruntime as ort
    model.eval()
    if not isinstance(inputs, (tuple, list)):
        inputs = (inputs,)
    with torch.no_grad():
        ref = model(*inputs)
    ref = ref if isinstance(ref, (tuple, list)) else (ref,)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    feeds = {i.name: x.cpu().numpy() for i, x in zip(sess.get_inputs(), inputs)}
    outs = sess.run(None, feeds)
    report = {"onnx_path": os.path.basename(path), "n_samples": int(inputs[0].shape[0]), "atol": atol, "outputs": []}
    for o_meta, r, o in zip(sess.get_outputs(), ref, outs):
        diff = np.abs(r.cpu().numpy() - o)
        report["outputs"].append({"name": o_meta.name, "max_abs_diff": float(diff.max()),
                                  "mean_abs_diff": float(diff.mean()), "passed": bool(diff.max() <= atol)})
    report["passed"] = all(o["passed"] for o in report["outputs"])
    if report_path:
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return report
