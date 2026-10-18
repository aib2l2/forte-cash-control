import json

import numpy as np
import pytest
import torch

from cash_ml.description_guidance import (
    GuidedHead,
    fit,
    load,
    save,
    subset_targets,
    target_masks,
)
from cash_ml.review_text_targets import collect_train_windows, window_texts
from cash_ml.temporal_model import infer, point_loss


def test_held_out_descriptions_rejected_before_any_file_access(tmp_path):
    for split in ("validation", "test"):
        with pytest.raises(ValueError, match="Only training"):
            collect_train_windows([{"split": split}], tmp_path / "does-not-exist")
        with pytest.raises(ValueError, match="original training"):
            subset_targets(tmp_path / "does-not-exist.npz", [{"split": split}])


def test_uses_only_structured_draft_and_rejects_bad_times():
    window = {
        "format_issues": [],
        "raw_local_output": "raw must not enter targets",
        "source_reference": {"type": "label must not enter targets"},
        "draft": {
            "observations": [{"action": "synthetic action", "time_reference_valid": True}],
            "ordinary_actions": [{"action": "ordinary", "why_ordinary": "context"}],
            "suspicious_actions": [],
            "uncertainties": ["not an observation"],
        },
    }
    assert window_texts(window) == ["synthetic action", "ordinary context"]
    window["draft"]["observations"][0]["time_reference_valid"] = False
    with pytest.raises(ValueError, match="time reference"):
        window_texts(window)


def test_review_windows_preserve_actual_sampling_coordinates(tmp_path):
    record = {"id": "a", "sha256": "digest", "split": "train"}
    root = tmp_path / "videos/digest"
    root.mkdir(parents=True)
    result = {
        "sha256": "digest",
        "review_status": "local_model_draft_complete",
        "expected_windows": 1,
        "duration_sec": 2.1,
        "windows": [
            {
                "index": 0,
                "sampled_range_sec": [0.0, 2.0],
                "format_issues": [],
                "draft": {"observations": [{"action": "synthetic", "time_reference_valid": True}]},
            }
        ],
    }
    (root / "result.json").write_text(json.dumps(result))
    rows, _ = collect_train_windows([record], tmp_path)
    assert rows[0]["start"] == 0.0 and rows[0]["end"] == 2.0
    masks = target_masks(
        {"centers": np.array([0.8, 1.0, 2.0])},
        {"starts": np.array([0.0, 2.01]), "ends": np.array([0.9, 2.1])},
    )
    assert [x.tolist() for x in masks] == [[0], [2]]


def test_unknown_positive_windows_never_become_negative_labels():
    logits = torch.zeros(4, requires_grad=True)
    typed = torch.zeros((4, 2), requires_grad=True)
    record = {"label": 1, "annotations": [{"type": "safe_left_open", "timecode": 1.0}]}
    b, t = point_loss(
        logits, typed, record, {"centers": np.arange(4)}, ["safe_left_open", "cash_left_unattended"]
    )
    (b + t).backward()
    assert logits.grad[1] < 0
    assert torch.equal(logits.grad[[0, 2, 3]], torch.zeros(3))


def test_semantic_loss_reaches_shared_visual_head_without_extra_inputs():
    torch.manual_seed(42)
    head = GuidedHead(8, 2, 4)
    _, _, h = head.training_outputs(torch.randn(5, 8))
    prediction = head.text(h.mean(0, keepdim=True))
    loss = 1 - torch.nn.functional.cosine_similarity(prediction, torch.randn(1, 4)).mean()
    loss.backward()
    assert head.input.weight.grad.abs().sum() > 0
    assert head.binary.weight.grad is None
    head.eval()
    binary, types = head(torch.randn(5, 8))
    assert binary.shape == (5,) and types.shape == (5, 2)


def test_training_rejects_fold_leakage_and_checkpoint_roundtrip(tmp_path):
    train = [{"id": "a", "split": "train", "scene_group": "g1"}]
    val = [{"id": "v", "scene_group": "g2"}]
    with pytest.raises(ValueError, match="exactly match"):
        fit(train, val, {}, {"a": {}, "v": {}}, ["a"], 0.1, 42)
    val[0]["scene_group"] = "g1"
    with pytest.raises(ValueError, match="Scene leakage"):
        fit(train, val, {}, {"a": {}}, ["a"], 0.1, 42)
    head = GuidedHead(8, 2, 4).eval()
    mean, scale = np.zeros(8, np.float32), np.ones(8, np.float32)
    data = {
        "sample": {"features": np.random.default_rng(42).normal(size=(7, 8)).astype(np.float32)}
    }
    before = infer(head, data, mean, scale)
    path = tmp_path / "model.pt"
    save(
        path,
        (head, mean, scale, np.zeros(4, np.float32), []),
        {"allowed_types": ["safe_left_open", "cash_left_unattended"]},
    )
    restored, rm, rs, metadata = load(path)
    after = infer(restored, data, rm, rs)
    for a, b in zip(before["sample"], after["sample"], strict=True):
        np.testing.assert_array_equal(a, b)
    assert metadata["allowed_types"] == ["safe_left_open", "cash_left_unattended"]
