"""Checks that the corruption pipeline matches the assignment specification.

    python -m pytest tests -q
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genai.corruptions import (TEST_SEVERITIES, apply_corruption, make_level_spec,  # noqa: E402
                               rect_coverage, sample_train_spec)

IMG = np.random.default_rng(0).random((128, 128, 3)).astype(np.float32) * 0.8 + 0.1


def test_train_ranges():
    rng = np.random.default_rng(1)
    counts = {}
    for _ in range(2000):
        s = sample_train_spec(rng)
        counts[s["type"]] = counts.get(s["type"], 0) + 1
        if s["type"] == "salt_pepper":
            assert 0.02 <= s["prob"] <= 0.15
        elif s["type"] == "blur":
            assert s["kernel"] in (3, 5, 7) and 0.5 <= s["sigma"] <= 2.5
        elif s["type"] == "occlusion":
            assert 1 <= len(s["rects"]) <= 3
            assert 0.10 - 0.016 <= rect_coverage(s["rects"], 128) <= 0.35 + 0.016
    assert all(400 < c < 600 for c in counts.values()), counts   # ~uniform over 4 conditions


def test_salt_pepper_fraction_and_values():
    out = apply_corruption(IMG, {"type": "salt_pepper", "prob": 0.15, "seed": 3})
    changed = np.any(out != IMG, axis=2)
    assert abs(changed.mean() - 0.15) < 0.02
    vals = out[changed]
    assert np.all((vals == 0) | (vals == 1))
    assert abs((vals[:, 0] == 1).mean() - 0.5) < 0.05


def test_deterministic_given_spec():
    rng = np.random.default_rng(7)
    for ctype in ("salt_pepper", "blur", "occlusion"):
        for lvl in range(3):
            spec = make_level_spec(rng, ctype, lvl)
            assert np.array_equal(apply_corruption(IMG, spec), apply_corruption(IMG, spec))


def test_test_occlusion_levels():
    rng = np.random.default_rng(9)
    for lvl, target in enumerate([0.10, 0.20, 0.35]):
        spec = make_level_spec(rng, "occlusion", lvl)
        assert len(spec["rects"]) == lvl + 1
        assert abs(spec["coverage"] - target) <= 0.015
        out = apply_corruption(IMG, spec)
        assert abs((out.sum(2) == 0).mean() - target) <= 0.015


def test_blur_matches_torchvision():
    import torch
    from torchvision.transforms.functional import gaussian_blur
    for cfg in TEST_SEVERITIES["blur"]:
        ours = apply_corruption(IMG, {"type": "blur", **cfg})
        tv = gaussian_blur(torch.from_numpy(IMG.transpose(2, 0, 1)), cfg["kernel"], cfg["sigma"]).numpy().transpose(1, 2, 0)
        assert np.abs(ours - tv).max() < 1e-5
