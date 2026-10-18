"""Compatibility adapter to the active local typed_v2 analyzer."""

from pathlib import Path

from backend.schemas import AnalysisResult


class AnalyzerNotConfigured(RuntimeError):
    """The installed model artifacts are missing or unavailable."""


def analyze_video(video_path: Path) -> AnalysisResult:
    """Use the same local model as the main API, without a second implementation."""
    from cash_ml.service import analyze_video as active_analyze_video

    return active_analyze_video(video_path)
