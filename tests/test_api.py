import base64
import io
import time
from pathlib import Path
from threading import Event

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.config import Settings
from backend.jobs import JobManager
from backend.main import create_app
from backend.schemas import AnalysisResult, EvidenceFrame, JobResponse, Violation

KEY = "test-only-api-key"
HEADERS = {"X-API-Key": KEY}


@pytest.fixture
def settings(tmp_path):
    return Settings(api_key=KEY, storage_dir=tmp_path / "storage", max_upload_mb=1)


def upload(client, name="example.mov", content=b"API transport fixture, not real video"):
    return client.post(
        "/api/v1/analyze", headers=HEADERS,
        files={"file": (name, content, "video/quicktime")},
    )


def wait_for_job(client, job_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/jobs/{job_id}", headers=HEADERS)
        assert response.status_code == 200
        job = response.json()
        if job["status"] != "processing":
            return job
        time.sleep(0.01)
    pytest.fail("Worker did not finish within the test timeout")


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "incorrect"}])
def test_auth_required_for_both_endpoints(settings, headers):
    with TestClient(create_app(settings)) as client:
        assert client.post(
            "/api/v1/analyze", headers=headers,
            files={"file": ("test.mov", b"test")},
        ).status_code == 401
        assert client.get("/api/v1/jobs/unknown", headers=headers).status_code == 401


@pytest.mark.parametrize("name,content,status", [
    ("test.mp4", b"test", 415),
    ("test.mov", b"", 400),
    ("test.mov", b"x" * (1024 * 1024 + 1), 413),
])
def test_invalid_uploads_do_not_leave_files(settings, name, content, status):
    with TestClient(create_app(settings)) as client:
        assert upload(client, name, content).status_code == status
        assert list((settings.storage_dir / "uploads").iterdir()) == []


def test_unconfigured_model_is_failure_not_a_negative_verdict(settings):
    with TestClient(create_app(settings)) as client:
        response = upload(client, "EXAMPLE.MOV")
        assert response.status_code == 202
        job = wait_for_job(client, response.json()["job_id"])
        assert job["status"] == "failed"
        assert "ещё не подключена" in job["error"]
        assert "verdict" not in job
        assert "violations" not in job
    assert list((settings.storage_dir / "uploads").iterdir()) == []


def test_processing_then_done_and_persistence(settings):
    entered, release = Event(), Event()

    def analyzer(path):
        assert path.exists()
        assert path.parent == settings.storage_dir / "uploads"
        entered.set()
        assert release.wait(5)
        return AnalysisResult(verdict="no_violation", violations=[])

    with TestClient(create_app(settings, analyzer)) as client:
        response = upload(client, "../../outside.mov")
        job_id = response.json()["job_id"]
        try:
            assert entered.wait(2)
            assert client.get(
                f"/api/v1/jobs/{job_id}", headers=HEADERS,
            ).json() == {"job_id": job_id, "status": "processing"}
        finally:
            release.set()
        assert wait_for_job(client, job_id) == {
            "job_id": job_id, "status": "done", "verdict": "no_violation", "violations": [],
        }
    with TestClient(create_app(settings)) as client:
        assert client.get(f"/api/v1/jobs/{job_id}", headers=HEADERS).json()["status"] == "done"


def test_multiple_events_and_real_jpeg_evidence(settings):
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), "white").save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    events = [Violation(
        type="Тестовое событие",
        timecode_sec=timestamp,
        description="Синтетический ответ только для проверки контракта API.",
        evidence_frames=[EvidenceFrame(timestamp_sec=timestamp, image_base64=encoded)],
    ) for timestamp in (2.0, 8.0)]
    with TestClient(create_app(
        settings, lambda _: AnalysisResult(verdict="violation", violations=events),
    )) as client:
        job_id = upload(client).json()["job_id"]
        job = wait_for_job(client, job_id)
        assert job["verdict"] == "violation"
        assert len(job["violations"]) == 2
        assert job["violations"][1]["evidence_frames"][0]["image_base64"] == encoded


def test_invalid_model_output_fails_without_fabricated_verdict(settings):
    with TestClient(create_app(
        settings, lambda _: {"verdict": "violation", "violations": []},
    )) as client:
        job = wait_for_job(client, upload(client).json()["job_id"])
        assert job["status"] == "failed"
        assert "verdict" not in job


def test_queue_limit(settings):
    limited = Settings(api_key=KEY, storage_dir=settings.storage_dir, max_pending_jobs=1)
    release = Event()

    def analyzer(_):
        assert release.wait(5)
        return AnalysisResult(verdict="no_violation", violations=[])

    with TestClient(create_app(limited, analyzer)) as client:
        try:
            assert upload(client).status_code == 202
            assert upload(client).status_code == 429
        finally:
            release.set()


def test_restart_marks_unfinished_jobs_failed(settings):
    settings.storage_dir.mkdir()
    uploads = settings.storage_dir / "uploads"
    uploads.mkdir()
    manager = JobManager(settings, lambda _: None)
    manager.save(JobResponse(job_id="interrupted", status="processing"))
    manager.close()
    (uploads / "interrupted.mov").write_bytes(b"interrupted upload")
    with TestClient(create_app(settings)) as client:
        job = client.get("/api/v1/jobs/interrupted", headers=HEADERS).json()
        assert job["status"] == "failed"
        assert "перезапущен" in job["error"]
        assert list(uploads.iterdir()) == []


def test_health_and_unknown_job(settings):
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/api/v1/jobs/missing", headers=HEADERS).status_code == 404


def test_schema_rejects_missing_or_invalid_evidence():
    with pytest.raises(ValueError):
        EvidenceFrame(timestamp_sec=1, image_base64="not-base64")
    with pytest.raises(ValueError):
        Violation(type="event", timecode_sec=1, description="test", evidence_frames=[])


def test_frontend_renders_without_model(monkeypatch):
    import httpx
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("API_KEY", KEY)
    monkeypatch.setattr(
        httpx, "get",
        lambda *args, **kwargs: httpx.Response(
            200, json={"status": "ok"}, request=httpx.Request("GET", "http://localhost/health"),
        ),
    )
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / "frontend" / "app.py")).run(timeout=10)
    assert not app.exception
    assert "AI-модель ещё не подключена" in app.warning[0].value
    assert app.button[0].disabled
