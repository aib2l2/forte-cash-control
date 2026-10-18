from pathlib import Path

import numpy as np
import pytest
from openpyxl import Workbook

from backend.schemas import EvidenceFrame
from cash_ml import service
from cash_ml.features import window_starts
from cash_ml.train import read_labels, split_with_type_coverage


def test_windows_cover_tail_and_short_video():
    assert window_starts(1) == [0]
    assert window_starts(10) == [0, 4, 6]
    with pytest.raises(ValueError):
        window_starts(float("nan"))


def test_merge_only_adjacent_positive_windows():
    scores = np.array([0.8, 0.9, 0.1, 0.7])
    assert service.event_peaks(np.array([0, 4, 8, 12]), np.array([4, 8, 12, 16]), scores, 0.5) == [1, 3]


def test_excel_exact_unicode_names_and_invalid_label(tmp_path):
    wb = Workbook()
    sheet = wb.active
    sheet.title = "labels"
    sheet.append(["video", "violation", "timecode_sec", "type"])
    sheet.append(["Х-17", 1, 7, "cash_taken_before_verification"])
    (tmp_path / "Х-17.mov").write_bytes(b"test")
    path = tmp_path / "labels.xlsx"
    wb.save(path)
    records = read_labels(path, tmp_path)
    assert records[0]["timecode"] == 7
    assert records[0]["path"].name == "Х-17.mov"
    sheet["B2"] = 22
    wb.save(path)
    with pytest.raises(ValueError, match="Некорректная метка"):
        read_labels(path, tmp_path)


def test_inference_uses_features_and_actual_evidence_timestamp(monkeypatch):
    analyzer = object.__new__(service.Analyzer)
    analyzer.mean = np.zeros(2)
    analyzer.scale = np.ones(2)
    analyzer.coef = np.array([1.0, 0.0])
    analyzer.intercept = 0.0
    analyzer.threshold = 0.7
    analyzer.type_mean = np.zeros(2)
    analyzer.type_scale = np.ones(2)
    analyzer.type_coef = np.array([[1.0, 0], [-1.0, 0]])
    analyzer.type_intercept = np.zeros(2)
    analyzer.type_classes = np.array(["cash_left_unattended", "safe_left_open"])
    analyzer.event_model = service.violation_contract(list(analyzer.type_classes))

    class FakeExtractor:
        def extract(self, path):
            return {"features": np.array([[-2.0, 0], [2.0, 0]]),
                    "starts": np.array([0, 4]), "ends": np.array([4, 8]),
                    "centers": np.array([2, 6])}

    analyzer.extractor = FakeExtractor()
    monkeypatch.setattr(service, "extract_evidence_frame", lambda p, t: EvidenceFrame(
        timestamp_sec=t + 0.01, image_base64="/9j/2Q=="))
    result = analyzer(Path("arbitrary-name.mov"))
    assert result.verdict == "violation"
    assert len(result.violations) == 1
    assert result.violations[0].timecode_sec == 6.01
    assert result.violations[0].type == "cash_left_unattended"
    analyzer.threshold = 0.99
    assert analyzer(Path("other.mov")).verdict == "no_violation"


def test_duplicate_video_keeps_both_events_in_one_group(tmp_path):
    wb = Workbook()
    sheet = wb.active
    sheet.title = "labels"
    sheet.append(["video", "violation", "timecode_sec", "type"])
    for name, timestamp in [("Х-4", 57), ("Х-47", 68)]:
        sheet.append([name, 1, timestamp, "event"])
        (tmp_path / f"{name}.mov").write_bytes(b"same video")
    path = tmp_path / "labels.xlsx"
    wb.save(path)
    records = read_labels(path, tmp_path)
    assert len(records) == 1
    assert records[0]["aliases"] == ["Х-4", "Х-47"]
    assert [a["timecode"] for a in records[0]["annotations"]] == [57, 68]
    sheet["B3"] = 0
    sheet["D3"] = None
    wb.save(path)
    with pytest.raises(ValueError, match="противоречивые"):
        read_labels(path, tmp_path)


def test_corrupt_video_never_returns_normal(tmp_path):
    from cash_ml.features import Extractor

    path = tmp_path / "invalid.mov"
    path.write_bytes(b"not a video")
    extractor = object.__new__(Extractor)
    with pytest.raises(ValueError, match="не декодируется"):
        extractor.extract(path)


def test_frontend_loads_with_model_status(monkeypatch):
    import httpx
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("API_KEY", "test-only")
    monkeypatch.setattr(httpx, "get", lambda *a, **k: httpx.Response(
        200, json={"ready": True, "threshold": 0.5, "test": {}, "type_test": {},
                   "allowed_types": ["safe_left_open"]},
        request=httpx.Request("GET", "http://localhost/api/v1/model")))
    app = AppTest.from_file(str(Path(__file__).with_name("app.py"))).run(timeout=20)
    assert not app.exception
    assert app.button[0].disabled


def test_adjacent_different_types_remain_separate():
    assert service.event_peaks(np.array([0, 4, 8]), np.array([4, 8, 12]),
                               np.array([0.8, 0.9, 0.7]), 0.5,
                               np.array(["safe_left_open", "cash_left_unattended", "cash_left_unattended"])) == [0, 1]


def test_type_contract_rejects_values_outside_excel_vocabulary():
    model = service.violation_contract(["safe_left_open", "unclassified_violation"])
    payload = {"type": "safe_left_open", "timecode_sec": 1, "description": "test",
               "evidence_frames": [{"timestamp_sec": 1, "image_base64": "/9j/2Q=="}]}
    assert model(**payload).type == "safe_left_open"
    assert set(model.model_json_schema()["properties"]["type"]["enum"]) == {"safe_left_open", "unclassified_violation"}
    for invalid in ["suspected_violation", "other", "no_violation", "SAFE_LEFT_OPEN"]:
        with pytest.raises(ValueError):
            model(**{**payload, "type": invalid})


def test_split_preserves_rare_types_without_duplicate_leakage():
    records = [{"label": 1, "sha256": f"pos-{i}",
                "annotations": [{"type": f"rare-{i}" if i < 3 else "common"}]}
               for i in range(12)]
    records += [{"label": 0, "sha256": f"neg-{i}", "annotations": []} for i in range(10)]
    train, val, test = split_with_type_coverage(records)
    assert {a["type"] for r in train for a in r["annotations"]} == {"rare-0", "rare-1", "rare-2", "common"}
    groups = [{r["sha256"] for r in split} for split in [train, val, test]]
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
    assert len(groups[0] | groups[1] | groups[2]) == len(records)
