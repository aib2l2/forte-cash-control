"""Local end-to-end checks. Print only technical counts and aggregate metrics."""

import argparse
import base64
import json
import os
import socket
import time
from pathlib import Path

import cv2
import httpx
import numpy as np

from backend.config import Settings
from backend.schemas import AnalysisResult
from cash_ml.evidence import context_frames
from cash_ml.experiment import load_model, load_npz, predict, write_json
from cash_ml.features import MODEL_DIR, PREPROCESS_VERSION, Extractor
from cash_ml.service import analyze_video, event_peaks


def deny_network():
    def denied(*args, **kwargs):
        raise RuntimeError("Network access blocked during local inference")

    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    socket.getaddrinfo = denied


def offline_analyze(path):
    deny_network()
    return analyze_video(path)


def offline_candidate(path):
    deny_network()
    from cash_ml.temporal_service import TemporalCandidateAnalyzer

    return TemporalCandidateAnalyzer(Path(os.environ["CANDIDATE_RUN"]))(path)


def verify_frames(path, events):
    requested = {}
    for event in events:
        assert 1 <= len(event["evidence_frames"]) <= 3
        assert any(abs(f["timestamp_sec"] - event["timecode_sec"]) < 1e-6
                   for f in event["evidence_frames"])
        for frame in event["evidence_frames"]:
            raw = base64.b64decode(frame["image_base64"], validate=True)
            assert raw.startswith(b"\xff\xd8") and raw.endswith(b"\xff\xd9")
            requested.setdefault(frame["timestamp_sec"], []).append(raw)
    cap = cv2.VideoCapture(str(path))
    try:
        while requested:
            ok, frame = cap.read()
            assert ok, "Evidence timestamp absent from source"
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            found = [k for k in requested if abs(k - t) <= 1e-6]
            if found:
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                assert ok
                for key in found:
                    assert all(content == encoded.tobytes() for content in requested.pop(key))
    finally:
        cap.release()
    return sum(len(e["evidence_frames"]) for e in events)


def wait_job(client, identifier, headers=None):
    states = set()
    deadline = time.monotonic() + 305
    while time.monotonic() < deadline:
        response = client.get("/api/v1/jobs/" + identifier, headers=headers)
        response.raise_for_status()
        job = response.json()
        states.add(job["status"])
        if job["status"] != "processing":
            return job, sorted(states)
        time.sleep(0.1)
    raise AssertionError("Job exceeded wall-clock limit")


def api_check(run, records, offline=False, candidate=False):
    from fastapi.testclient import TestClient

    from cash_ml.runtime import create_runtime_app

    allowed = json.loads((run / "protocol.json").read_text())["allowed_types"]
    chosen = [r for r in records if r["split"] == "test"][:2]
    durations = {r["id"]: float(load_npz(MODEL_DIR / "features" /
                 f"{r['sha256']}-{PREPROCESS_VERSION}.npz")["duration"]) for r in records}
    longest = max(records, key=lambda r: durations[r["id"]])
    if longest not in chosen:
        chosen.append(longest)
    mode = "offline_candidate" if candidate else ("offline_active" if offline else "live_active")
    if offline or candidate:
        settings = Settings(api_key="isolated-offline-check", storage_dir=run / mode, max_pending_jobs=4)
        os.environ["CANDIDATE_RUN"] = str(run.resolve())
        client_context = TestClient(create_runtime_app(settings, offline_candidate if candidate else offline_analyze))
    else:
        settings = Settings.from_env()
        client_context = httpx.Client(base_url=settings.api_base_url, timeout=60, trust_env=False)
    headers = {"X-API-Key": settings.api_key}
    rows = []
    with client_context as client:
        assert client.post("/api/v1/analyze", files={"file": ("bad.mov", b"bad")}).status_code == 401
        starts, pending = [], []
        for index, record in enumerate(chosen):
            starts.append(time.monotonic())
            with Path(record["path"]).open("rb") as file:
                response = client.post("/api/v1/analyze", headers=headers,
                                       files={"file": (f"anonymous-{index}.mov", file, "video/quicktime")})
            assert response.status_code == 202
            pending.append(response.json()["job_id"])
        for index, record in enumerate(chosen):
            job, states = wait_job(client, pending[index], headers)
            seconds = time.monotonic() - starts[index]
            assert seconds < 300 and job["status"] == "done"
            AnalysisResult.model_validate({k: job[k] for k in ("verdict", "violations")})
            assert all(e["type"] in allowed for e in job["violations"])
            frames = verify_frames(Path(record["path"]), job["violations"])
            rows.append({"case": index, "source_sha256": record["sha256"],
                         "duration_seconds": durations[record["id"]], "states": states,
                         "elapsed_seconds_including_upload_and_queue": seconds,
                         "event_count": len(job["violations"]), "verified_real_jpegs": frames})
        for filename, content, code in [("wrong.mp4", b"bad", 415), ("empty.mov", b"", 400)]:
            assert client.post("/api/v1/analyze", headers=headers,
                               files={"file": (filename, content)}).status_code == code
        bad = client.post("/api/v1/analyze", headers=headers,
                          files={"file": ("broken.mov", b"invalid media")})
        assert bad.status_code == 202
        bad_job, _ = wait_job(client, bad.json()["job_id"], headers)
        assert bad_job["status"] == "failed" and "verdict" not in bad_job
        assert client.get("/api/v1/jobs/" + pending[0]).status_code == 401
        if not offline and not candidate:
            vocab = client.get("/api/v1/model", headers=headers).json()["allowed_types"]
            assert vocab == allowed
            assert set(client.get("/api/v1/rules", headers=headers).json()["types"]) == set(allowed)
    report = {"mode": mode, "rows": rows, "failed_corrupt_input": True,
              "unauthorized_rejected": True, "all_passed": True,
              "timing_note": "Observed polling completion; later rows include prior verification overhead (conservative).",
              "network_disabled_in_analysis_process": offline or candidate}
    write_json(run / (mode + "_api.json"), report)
    print(json.dumps({"check": mode, "videos": len(rows), "real_jpegs": sum(r["verified_real_jpegs"] for r in rows),
                      "max_total_seconds": max(r["elapsed_seconds_including_upload_and_queue"] for r in rows)}), flush=True)


def benchmark(run, records):
    from cash_ml.temporal_finetune import prefix_data
    from cash_ml.temporal_service import TemporalCandidateAnalyzer

    candidate = TemporalCandidateAnalyzer(run)
    tests = [r for r in records if r["split"] == "test"]
    expected = prefix_data(run, tests)
    baseline, _, threshold, _ = load_model(run / "baseline_refit.npz")
    extractor = Extractor()
    rows = []
    for record in tests:
        source = Path(record["path"])
        began = time.monotonic()
        original = extractor.extract(source)
        reference = load_npz(MODEL_DIR / "features" / f"{record['sha256']}-{PREPROCESS_VERSION}.npz")
        np.testing.assert_allclose(original["features"], reference["features"], rtol=1e-5, atol=2e-5)
        scores, types = predict(baseline, original)
        baseline_events = []
        for i in event_peaks(original["starts"], original["ends"], scores, threshold, types):
            actual, frames = context_frames(source, float(original["centers"][i]), float(original["duration"]))
            baseline_events.append({"timecode_sec": actual, "evidence_frames": [f.model_dump() for f in frames]})
        baseline_seconds = time.monotonic() - began
        baseline_frames = verify_frames(source, baseline_events)
        began = time.monotonic()
        item = candidate.extract(source)
        binary, typed = candidate.logits(item)
        result = candidate.from_features(source, item)
        candidate_seconds = time.monotonic() - began
        reference = expected[record["id"]]
        np.testing.assert_array_equal(item["centers"], reference["centers"])
        np.testing.assert_allclose(item["features"], reference["features"], atol=2e-5, rtol=1e-5)
        expected_binary, expected_typed = candidate.logits(reference)
        np.testing.assert_allclose(binary, expected_binary, atol=1e-5, rtol=1e-5)
        np.testing.assert_array_equal(typed.argmax(1), expected_typed.argmax(1))
        nframes = verify_frames(source, [e.model_dump() for e in result.violations])
        rows.append({"source_sha256": record["sha256"], "duration_seconds": float(item["duration"]),
                     "baseline_decode_features_heads_evidence_seconds": baseline_seconds,
                     "candidate_decode_features_heads_evidence_seconds": candidate_seconds,
                     "baseline_verified_jpegs": baseline_frames,
                     "candidate_verified_jpegs": nframes, "training_inference_parity": True})
        write_json(run / "inference_benchmark.json", {"rows": rows, "includes_upload_queue_model_load": False})
        print(json.dumps({"benchmark_case": len(rows), "baseline_seconds": baseline_seconds,
                          "candidate_seconds": candidate_seconds, "parity": True}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--mode", choices=["live", "offline", "candidate", "benchmark"], required=True)
    args = parser.parse_args()
    records = json.loads((args.run / "previous_grouped_records.json").read_text())
    if args.mode == "benchmark":
        benchmark(args.run, records)
    else:
        api_check(args.run, records, offline=args.mode == "offline", candidate=args.mode == "candidate")


if __name__ == "__main__":
    main()
