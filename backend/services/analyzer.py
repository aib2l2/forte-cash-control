"""Replace this integration point once a trained local model is available."""

from pathlib import Path

from backend.schemas import AnalysisResult


class AnalyzerNotConfigured(RuntimeError):
    pass


def analyze_video(video_path: Path) -> AnalysisResult:
    """Analyze the entire video without using labels, filenames, or external AI APIs.

    Implementation checklist:
    1. Decode the video and retain source timestamps.
    2. Extract clip features using locally stored pretrained weights.
    3. Apply the trained classifier and threshold selected on validation data.
    4. Merge duplicate detections but retain distinct events.
    5. Classify events, select 1–3 real evidence frames, attach verified rules.
    6. Return AnalysisResult; no_violation is valid only after actual analysis.

    Load models once and reuse them, rather than loading weights for each clip.
    See ml/README.md and rules/rules.json. Nothing is downloaded here.
    """
    raise AnalyzerNotConfigured(
        "Видео принято, но AI-модель ещё не подключена. "
        "Реализуйте analyze_video в backend/services/analyzer.py. "
        "Вердикт по этому видео не сформирован."
    )
