import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from openpyxl import Workbook

from cash_ml.context_features import (
    ContextExtractor,
    FeatureSpec,
    letterbox,
    normalize_clip,
    windows,
)
from cash_ml.experiment import (
    deployment_gate,
    fit_head,
    head_logits,
    load_model,
    match_points,
    save_model,
    train_heads,
)
from cash_ml.train import read_labels


def test_letterbox_preserves_portrait_geometry_and_edges():
    portrait = np.zeros((848, 464, 3), np.uint8)
    portrait[:] = [255, 0, 0]
    portrait[:30] = [0, 255, 0]
    portrait[-30:] = [0, 0, 255]
    image = letterbox(portrait)
    assert image.shape == (112, 112, 3)
    content = image[56, :, 0] == 255
    assert content.sum() == round(464/848*112)
    assert image[0, 56, 1] == 255 and image[-1, 56, 2] == 255
    assert np.array_equal(image[:, 0], image[:, -1])
    clip = normalize_clip([image]*16)
    assert clip.shape == (3, 16, 112, 112)
    assert torch.isfinite(clip).all()


@pytest.mark.parametrize("seconds", [4, 8, 16])
def test_context_windows_cover_edges_and_overlap(seconds):
    spec = FeatureSpec(seconds, seconds/2)
    assert windows(1, spec) == [(0, 1)]
    spans = windows(37.3, spec)
    assert spans[0][0] == 0 and spans[-1][1] == 37.3
    assert all(a[1] > b[0] for a, b in zip(spans, spans[1:], strict=False))
    assert len({s for s, _ in spans}) == len(spans)
    with pytest.raises(ValueError):
        windows(float("nan"), spec)


def test_short_video_uses_actual_decoder_times_and_same_context_spec(tmp_path):
    path = tmp_path/"anonymous.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 96))
    assert writer.isOpened()
    for i in range(20):
        writer.write(np.full((96, 64, 3), i*10, np.uint8))
    writer.release()
    extractor = object.__new__(ContextExtractor)
    extractor.spec = FeatureSpec(16, 8)

    class FakeNet:
        def __call__(self, clips):
            assert clips.shape == (2, 3, 16, 112, 112)
            return torch.zeros((2, 512))

    extractor.net = FakeNet()
    data = extractor.extract(path)
    assert data["features"].shape == (1, 1024)
    assert data["frame_times"].shape == (1, 16)
    assert np.all(np.diff(data["frame_times"][0]) >= 0)
    assert data["frame_times"][0, 0] == 0
    assert data["frame_times"][0, -1] == pytest.approx(1.9)
    assert data["centers"][0] == data["frame_times"][0, 8]
    assert json.loads(str(data["spec"])) == json.loads(extractor.spec.to_json())


def test_multiple_rows_for_same_video_preserve_distinct_events(tmp_path):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "labels"
    sheet.append(["video", "violation", "timecode_sec", "type"])
    for timestamp, kind in [(1, "safe_left_open"), (3, "cash_left_unattended"), (1, "safe_left_open")]:
        sheet.append(["same", 1, timestamp, kind])
    path = tmp_path/"labels.xlsx"
    workbook.save(path)
    (tmp_path/"same.mov").write_bytes(b"example")
    records = read_labels(path, tmp_path)
    assert len(records) == 1
    assert len(records[0]["annotations"]) == 2
    assert len(records[0]["source_rows"]) == 3


def test_point_matching_maximizes_count_and_obeys_type():
    annotations = [{"timecode": 3., "type": "a"}, {"timecode": 7., "type": "a"}]
    events = [{"timecode_sec": 5., "type": "a"}, {"timecode_sec": 0., "type": "b"}]
    assert len(match_points(annotations, events)) == 2
    assert len(match_points(annotations, events, typed=True)) == 1
    assert match_points(annotations, []) == []
    # One alert cannot explain two annotation points.
    assert len(match_points(annotations, events[:1])) == 1


@pytest.mark.parametrize("family", ["logistic", "svm", "centroid"])
def test_numeric_export_preserves_head_predictions(tmp_path, family):
    rng = np.random.default_rng(3)
    x = rng.normal(size=(20, 6))
    binary = fit_head(x, np.array([0, 1]*10), np.ones(20), family)
    typed = fit_head(x, np.array(["a", "b", "c", "d"]*5), np.ones(20), family)
    model = {"binary": binary, "typed": typed}
    path = tmp_path/"candidate.npz"
    save_model(path, model, FeatureSpec().to_json(), .4, {"allowed_types": ["a", "b", "c", "d"]})
    loaded, spec, threshold, metadata = load_model(path)
    assert threshold == .4 and json.loads(spec)["seconds"] == 8
    assert list(loaded["typed"]["classes"]) == metadata["allowed_types"]
    for kind in ("binary", "typed"):
        np.testing.assert_allclose(head_logits(model[kind], x), head_logits(loaded[kind], x))


def test_unlabeled_positive_windows_are_never_used_as_normal(monkeypatch):
    import cash_ml.experiment as experiment

    captured = []
    real_fit = experiment.fit_head

    def record_fit(x, y, weights, family):
        captured.append((np.asarray(x), np.asarray(y)))
        return real_fit(x, y, weights, family)

    monkeypatch.setattr(experiment, "fit_head", record_fit)
    records = [{"id": "pos1", "label": 1, "annotations": [{"timecode": 5, "type": "a"}]},
               {"id": "pos2", "label": 1, "annotations": [{"timecode": 5, "type": "b"}]},
               {"id": "normal", "label": 0, "annotations": []}]
    data = {name: {"features": np.array([[100., 100], [value, value], [200., 200]]),
                   "centers": np.array([1, 5, 9])} for name, value in [("pos1", 1), ("pos2", 2)]}
    data["normal"] = {"features": np.array([[-1., 0], [-2., 0]]), "centers": np.array([1, 5])}
    train_heads(records, data, "centroid", reviewed_negatives=["normal", "not_in_train"])
    for x, _ in captured:
        assert not np.any(x == 100) and not np.any(x == 200)
    assert list(captured[0][1]) == [1, 1, 0, 0]


def test_deploy_gate_rejects_apparent_gain_on_one_scene():
    old_rows = [{"video": str(i), "scene_group": "one", "label": i % 2, "predicted": 1-i % 2} for i in range(5)]
    new_rows = [{**r, "predicted": r["label"]} for r in old_rows]
    old = {"videos": old_rows, "type_accuracy_at_points": 0, "timing": {"typed_point_recall": 0}}
    new = {"videos": new_rows, "type_accuracy_at_points": 1, "timing": {"typed_point_recall": 1}}
    gate = deployment_gate(old, new, True)
    assert not gate["passed"]
    assert not gate["checks"]["at_least_10_positive_and_negative_test_groups"]
    assert gate["group_bootstrap_accuracy_delta_95pct"] is None


def test_frozen_real_split_never_separates_scene_or_hash():
    latest = Path(__file__).resolve().parents[1]/"models/experiments/latest_run.txt"
    if not latest.exists():
        pytest.skip("No local experiment artifacts")
    records = json.loads((Path(latest.read_text().strip())/"grouped_records.json").read_text())
    for key in ("scene_group", "sha256"):
        for value in {r[key] for r in records}:
            assert len({r["split"] for r in records if r[key] == value}) == 1


def test_frontend_displays_comparison_without_claiming_candidate_is_active(monkeypatch):
    import httpx
    from streamlit.testing.v1 import AppTest

    metric = {"detection": {"videos": 5, "balanced_accuracy": .5,
                            "confusion_matrix_tn_fp_fn_tp": [[3, 0], [2, 0]]},
              "type_accuracy_at_points": .5, "timing": {"typed_point_recall": 0.}}
    status = {"ready": True, "threshold": .5, "test": {}, "type_test": {},
              "allowed_types": ["safe_left_open"],
              "latest_experiment": {"state": "complete", "candidate_active": False,
                                    "supports": {}, "deployment_gate": {"passed": False},
                                    "comparison": {n: {"test": metric} for n in ("baseline", "candidate")}}}
    monkeypatch.setenv("API_KEY", "test-only")
    monkeypatch.setattr(httpx, "get", lambda *a, **k: httpx.Response(
        200, json=status, request=httpx.Request("GET", "http://localhost/api/v1/model")))
    app = AppTest.from_file(str(Path(__file__).with_name("app.py"))).run(timeout=20)
    assert not app.exception
    assert len(app.dataframe) == 1 and len(app.dataframe[0].value) == 2
    assert any("исходная v2" in info.value for info in app.info)
