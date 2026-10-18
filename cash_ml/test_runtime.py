import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.config import Settings
from backend.schemas import AnalysisResult, EvidenceFrame
from cash_ml.evidence import context_frames
from cash_ml.features import sha256
from cash_ml.review import append_review, review_tasks
from cash_ml.rules import describe_hypothesis, validate_registry
from cash_ml.runtime import create_runtime_app


def stub_analyzer(path):
    if path.read_bytes() == b"hang":
        time.sleep(30)
    if path.read_bytes() == b"broken":
        raise ValueError("bad media")
    return AnalysisResult(verdict="no_violation", violations=[])


def poll(client, job, headers, timeout=12):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        response = client.get("/api/v1/jobs/" + job, headers=headers)
        assert response.status_code == 200
        result = response.json()
        if result["status"] != "processing":
            return result
        time.sleep(0.03)
    raise AssertionError("job never completed")


def test_deadline_includes_queue_kills_hang_and_worker_recovers(tmp_path):
    settings = Settings(api_key="test-key", storage_dir=tmp_path, max_pending_jobs=2)
    headers = {"X-API-Key": settings.api_key}
    app = create_runtime_app(settings, stub_analyzer, deadline_seconds=4.0)
    with TestClient(app) as client:
        before = time.monotonic()
        first = client.post("/api/v1/analyze", headers=headers,
                            files={"file": ("one.mov", b"hang")}).json()["job_id"]
        second = client.post("/api/v1/analyze", headers=headers,
                             files={"file": ("two.mov", b"normal")}).json()["job_id"]
        assert client.post("/api/v1/analyze", headers=headers,
                           files={"file": ("full.mov", b"normal")}).status_code == 429
        assert poll(client, first, headers)["status"] == "failed"
        assert poll(client, second, headers)["status"] == "failed"
        assert time.monotonic() - before < 7.0
        end = time.monotonic() + 3
        while app.state.jobs.pending and time.monotonic() < end:
            time.sleep(0.03)
        assert not app.state.jobs.pending
        third = client.post("/api/v1/analyze", headers=headers,
                            files={"file": ("ok.mov", b"normal")}).json()["job_id"]
        assert poll(client, third, headers)["verdict"] == "no_violation"
        assert client.get("/api/v1/jobs/" + third).status_code == 401
        assert client.post("/api/v1/analyze", files={"file": ("x.mov", b"x")}).status_code == 401
        bad = client.post("/api/v1/analyze", headers=headers,
                          files={"file": ("bad.mov", b"broken")}).json()["job_id"]
        assert poll(client, bad, headers)["status"] == "failed"
    assert not list((tmp_path / "uploads").glob("*"))


def test_evidence_keeps_actual_primary_and_deduplicates_short_video():
    calls = []

    def reader(path, timestamp):
        calls.append(timestamp)
        return EvidenceFrame(timestamp_sec=round(timestamp, 1), image_base64="/9j/2Q==")

    actual, frames = context_frames(Path("unused"), 0.031, 0.05, reader=reader)
    assert actual == 0.0
    assert len(frames) == 1
    actual, frames = context_frames(Path("unused"), 5.031, 10, reader=reader)
    assert actual == 5.0
    assert [f.timestamp_sec for f in frames] == [3.0, 5.0, 7.0]
    assert 5.0 in [f.timestamp_sec for f in frames]


def test_registry_matches_model_and_sources_are_real_snapshots():
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / "models/cash_r3d/typed_report.json").read_text())
    data = validate_registry(report["allowed_types"])
    for card in data["types"].values():
        for ref in card["references"]:
            source = data["sources"][ref["source"]]
            paragraphs = json.loads((root / source["snapshot"]).read_text())["paragraphs"]
            assert any(p["id"] == ref["anchor"] for p in paragraphs)
    assert not data["types"]["personal_items_in_work_area"]["references"]
    assert data["types"]["unauthorized_person_in_cash_area"]["status"] == "authorization_and_scope_needed"
    with pytest.raises(ValueError):
        validate_registry(["new_invented_type"])
    text = describe_hypothesis("safe_key_left_in_lock", 0, 4, 0.3, 0.15)
    assert "Гипотеза модели" in text and "автоматически не подтверждены" in text
    assert "п. 80" in text and "не является вероятностью" in text


def test_review_keeps_all_events_and_only_training_data(tmp_path):
    source = tmp_path / "original.mov"
    source.write_bytes(b"immutable")
    record = {"id": "sample", "aliases": ["a", "b"], "sha256": sha256(source),
              "path": str(source), "split": "train", "annotations": [
                  {"type": "safe_left_open", "timecode": 1},
                  {"type": "safe_left_open", "timecode": 3}]}
    tasks = review_tasks([record, {**record, "split": "test"}])
    assert len(tasks) == 2 and tasks[0]["task_id"] != tasks[1]["task_id"]
    assert all(t["start_sec"] is None and t["end_sec"] is None for t in tasks)
    payload = {"reviewer": "human", "locally_viewed": True, "review_status": "confirmed_event",
               "type": "safe_left_open", "point_sec": 1, "start_sec": None, "end_sec": None,
               "observable_actions": "Human-reviewed synthetic test", "uncertainties": ""}
    destination = tmp_path / "annotations.jsonl"
    append_review(destination, tasks[0], payload, ["safe_left_open"], 5)
    append_review(destination, tasks[0], {**payload, "point_sec": 2}, ["safe_left_open"], 5)
    rows = [json.loads(line) for line in destination.read_text().splitlines()]
    assert len(rows) == 2 and all(not r["used_for_training"] for r in rows)
    assert rows[1]["original_point_sec"] == 1
    assert source.read_bytes() == b"immutable"
    invalid = [
        {"locally_viewed": False}, {"reviewer": ""}, {"type": "other"},
        {"start_sec": 0, "end_sec": 2}, {"point_sec": 6},
        {"review_status": "confirmed_normal_interval", "type": None, "point_sec": None},
    ]
    for extra in invalid:
        with pytest.raises(ValueError):
            append_review(destination, tasks[0], {**payload, **extra}, ["safe_left_open"], 5)
    assert len(destination.read_text().splitlines()) == 2
