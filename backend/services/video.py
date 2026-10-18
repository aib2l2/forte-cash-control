"""Optional local decoding helpers. Install the [ml] dependencies first."""

import base64
import math
from pathlib import Path

from backend.schemas import EvidenceFrame


def extract_evidence_frame(video_path: Path, timestamp_sec: float) -> EvidenceFrame:
    """Read a frame near the requested time and return its decoder-reported timestamp.

    Seeking precision depends on the codec. The final model should retain frame
    timestamps during sequential decoding, especially for variable-frame-rate video.
    """
    if not math.isfinite(timestamp_sec) or timestamp_sec < 0:
        raise ValueError("Таймкод должен быть конечным неотрицательным числом.")
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Установите ML-зависимости: pip install -e '.[ml]'") from exc

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError("Не удалось открыть видео.")
        if not capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000):
            raise ValueError("Декодер не поддерживает переход к заданному таймкоду.")
        ok, frame = capture.read()
        if not ok:
            raise ValueError("Не удалось прочитать кадр по этому таймкоду.")
        actual_time = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if not math.isfinite(actual_time) or actual_time < 0:
            raise ValueError("Декодер не вернул корректный таймкод кадра.")
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise ValueError("Не удалось сохранить JPEG.")
        return EvidenceFrame(
            timestamp_sec=actual_time,
            image_base64=base64.b64encode(encoded.tobytes()).decode("ascii"),
        )
    finally:
        capture.release()
