"""Tests use synthetic video and text, never case footage or cloud descriptions."""

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import cv2
import numpy as np

from cash_ml.local_video_review import (
    extract_samples,
    group_inventory,
    parse_output,
    repair_structural_issues,
    update_index,
    windows,
)


def test_windows_cover_short_long_and_boundary_durations():
    for last in (0.0, 0.1, 11.6, 24.0, 24.01, 40.0, 142.5):
        spans = windows(last)
        assert spans[0][0] == 0 and spans[-1][1] == last
        assert all(0 <= start <= end <= last for start, end in spans)
        assert all(spans[i + 1][0] <= spans[i][1] for i in range(len(spans) - 1))


def test_actual_frames_and_final_timestamp_are_kept(tmp_path):
    source = tmp_path / "synthetic.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (80, 160))
    assert writer.isOpened()
    for i in range(26):
        writer.write(np.full((160, 80, 3), i * 8, dtype=np.uint8))
    writer.release()
    result = extract_samples(source, tmp_path / "frames", step=2, long_side=80)
    assert result["decoded_frames"] == 26
    assert [f["source_frame_index"] for f in result["samples"]] == [0, 20, 25]
    assert abs(result["samples"][-1]["time_sec"] - 2.5) < 0.01
    assert all((f["width"], f["height"]) == (40, 80) for f in result["samples"])
    assert all(Path(f["image"]).is_file() for f in result["samples"])


def test_frame_references_are_mapped_not_invented():
    frames = [{"time_sec": 20.1}, {"time_sec": 22.1}]
    payload = {"observations": [{"first_frame": 1, "last_frame": 2, "action": "Synthetic observation"}],
               "ordinary_actions": [], "suspicious_actions": [], "uncertainties": []}
    parsed, issues = parse_output(json.dumps(payload), frames)
    assert not issues
    assert parsed["observations"][0]["first_observed_sec"] == 20.1
    payload["observations"][0]["last_frame"] = 200
    parsed, issues = parse_output(json.dumps(payload), frames)
    assert issues == ["invalid_frame_reference"]
    assert parsed["observations"][0]["time_reference_valid"] is False
    assert "last_observed_sec" not in parsed["observations"][0]
    assert parse_output("incomplete {", frames) == (None, ["invalid_json"])


def test_duplicate_sources_reuse_review_but_keep_every_file(tmp_path):
    rows = [{"sha256": "a", "file": "one.mov", "path": str(tmp_path / "one.mov")},
            {"sha256": "a", "file": "two.mov", "path": str(tmp_path / "two.mov")}]
    assert len(group_inventory(rows)) == 1
    stats = update_index(tmp_path, rows, {})
    assert stats["files"] == 2 and stats["unique_videos"] == 1
    assert stats["completed_files"] == 0 and not stats["used_for_training"]
    report = (tmp_path / "report.html").read_text()
    assert report.count("<article class='video'>") == 2
    assert "one.mov" in report and "two.mov" in report


def test_report_escapes_generated_html_and_marks_unverified(tmp_path):
    rows = [{"sha256": "a", "file": "one.mov", "path": str(tmp_path / "one.mov")}]
    work = tmp_path / "videos/a"
    work.mkdir(parents=True)
    result = {"review_status": "local_model_draft_complete", "expected_windows": 1,
              "windows": [{"sampled_range_sec": [0, 1], "frames": [], "draft": None,
                           "raw_local_output": "<script>evil()</script>", "metrics": {},
                           "format_issues": ["invalid_json"]}]}
    (work / "result.json").write_text(json.dumps(result))
    stats = update_index(tmp_path, rows, {})
    report = (tmp_path / "report.html").read_text()
    assert "<script>evil()" not in report and "&lt;script&gt;evil()" in report
    assert "человеком не проверен" in report
    assert stats["human_verified"] == 0 and stats["used_for_training"] is False
    assert stats["files_needing_technical_review"] == 1 and stats["structurally_valid_files"] == 0


def test_reexamination_keeps_original_and_maps_valid_reference(tmp_path, monkeypatch):
    from cash_ml.local_video_review import digest

    image = tmp_path / "synthetic.jpg"
    image.write_bytes(b"synthetic fixture; no video data")
    frame = {"image": str(image), "image_sha256": digest(image), "time_sec": 4.5,
             "source_frame_index": 45}
    work = tmp_path / "videos/a"
    work.mkdir(parents=True)
    row = {"sha256": "a", "file": "synthetic.mov", "path": str(tmp_path / "synthetic.mov")}
    original = {"raw_local_output": "invalid initial reference", "draft": None,
                "format_issues": ["invalid_frame_reference"], "metrics": {"finish_reason": "stop"}}
    entry = {"index": 0, "sampled_range_sec": [0, 5], "frames": [frame], **original}
    result = {"review_status": "local_model_draft_complete", "expected_windows": 1, "windows": [entry]}
    (work / "result.json").write_text(json.dumps(result))
    payload = {"observations": [{"first_frame": 1, "last_frame": 1, "action": "Synthetic observation"}],
               "ordinary_actions": [], "suspicious_actions": [], "uncertainties": []}
    generated = SimpleNamespace(text=json.dumps(payload), prompt_tokens=10, generation_tokens=10,
                                peak_memory=0, finish_reason="stop")
    mx = ModuleType("mlx.core")
    mx.clear_cache = lambda: None
    mlx = ModuleType("mlx")
    mlx.core = mx
    vlm = ModuleType("mlx_vlm")
    vlm.generate = lambda *a, **kw: generated
    prompt_utils = ModuleType("mlx_vlm.prompt_utils")
    prompt_utils.apply_chat_template = lambda *a, **kw: "synthetic prompt"
    for name, module in [("mlx", mlx), ("mlx.core", mx), ("mlx_vlm", vlm),
                         ("mlx_vlm.prompt_utils", prompt_utils)]:
        monkeypatch.setitem(sys.modules, name, module)
    processor = SimpleNamespace(apply_chat_template=lambda *a, **kw: "synthetic prompt")
    repair_structural_issues(SimpleNamespace(config={}), processor, tmp_path, [row], {})
    saved = json.loads((work / "result.json").read_text())["windows"][0]
    assert saved["original_response"] == original
    assert not saved["format_issues"]
    assert saved["draft"]["observations"][0]["first_observed_sec"] == 4.5
    assert len(saved["structural_reexaminations"]) == 1
