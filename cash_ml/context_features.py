"""Aspect-preserving R3D features, shared by experiments and candidate inference.

The second view was chosen after inspecting the source contact sheets: the desk,
metal cabinet and hands occupy y=.32:.70. It is an image crop, not a detector.
No event label, filename, or annotation is used to construct inference windows.
"""

import json
import math
import time
from dataclasses import asdict, dataclass

import cv2
import numpy as np
import torch

from cash_ml.features import FRAMES, WEIGHTS, Extractor, sha256

VERSION = "r3d18-context-letterbox-v1"
MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)
WORK_CROP = (0.0, 0.32, 1.0, 0.70)


@dataclass(frozen=True)
class FeatureSpec:
    seconds: float = 8.0
    stride: float = 4.0
    views: str = "full_work"
    version: str = VERSION

    def __post_init__(self):
        if (not math.isfinite(self.seconds) or not math.isfinite(self.stride)
                or self.seconds <= 0 or not 0 < self.stride <= self.seconds):
            raise ValueError("Invalid temporal specification")
        if self.views not in ("full", "full_work") or self.version != VERSION:
            raise ValueError("Unknown feature specification")

    def to_json(self):
        return json.dumps(asdict(self), sort_keys=True)


def windows(duration, spec):
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid video duration")
    last = max(0., duration - spec.seconds)
    starts = list(np.arange(0., last + 1e-8, spec.stride))
    if last - starts[-1] > 1e-6:
        starts.append(last)
    return [(s, min(duration, s + spec.seconds)) for s in starts]


def letterbox(rgb, size=112):
    """Preserve aspect ratio, filling unused pixels with the pretrained mean."""
    h, w = rgb.shape[:2]
    if min(h, w) < 1:
        raise ValueError("Empty image")
    factor = min(size / h, size / w)
    nh, nw = max(1, round(h * factor)), max(1, round(w * factor))
    scaled = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    result = np.empty((size, size, 3), dtype=np.uint8)
    result[:] = np.round(MEAN * 255).astype(np.uint8)
    y, x = (size - nh) // 2, (size - nw) // 2
    result[y:y+nh, x:x+nw] = scaled
    return result


def frame_views(frame, views):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = [letterbox(rgb)]
    if views == "full_work":
        h, w = rgb.shape[:2]
        x1, y1, x2, y2 = WORK_CROP
        result.append(letterbox(rgb[round(h*y1):round(h*y2), round(w*x1):round(w*x2)]))
    return result


def normalize_clip(frames):
    array = (np.asarray(frames, dtype=np.float32) / 255. - MEAN) / STD
    return torch.from_numpy(array).permute(3, 0, 1, 2)


class ContextExtractor(Extractor):
    def __init__(self, spec=None):
        super().__init__(download=False)
        self.spec = spec or FeatureSpec()

    @torch.inference_mode()
    def extract(self, path):
        began = time.monotonic()
        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                raise ValueError("Видео не декодируется")
            fps, count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if not math.isfinite(fps) or fps <= 0 or not math.isfinite(count) or count < 1:
                raise ValueError("Некорректная длительность видео")
            duration = count / fps
            if duration > 600:
                raise ValueError("Прототип поддерживает видео не длиннее 10 минут")
            spans = windows(duration, self.spec)
            targets_by_window = [np.linspace(s, max(s, e - 1/fps), FRAMES) for s, e in spans]
            targets = sorted({float(t) for ts in targets_by_window for t in ts})
            sampled, cursor, previous, last = {}, 0, -1., None
            while cursor < len(targets):
                ok, frame = cap.read()
                if not ok:
                    break
                actual = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                if not math.isfinite(actual) or actual < 0 or actual < previous:
                    raise ValueError("Некорректный таймкод декодера")
                previous, last = actual, (frame, actual)
                if actual + 1e-6 >= targets[cursor]:
                    prepared = frame_views(frame, self.spec.views)
                    while cursor < len(targets) and actual + 1e-6 >= targets[cursor]:
                        sampled[targets[cursor]] = (prepared, actual)
                        cursor += 1
            if cursor < len(targets):
                if last is None or targets[-1] - last[1] > max(.25, 2/fps):
                    raise ValueError("Видео обрывается до заявленного конца")
                prepared = frame_views(last[0], self.spec.views)
                for t in targets[cursor:]:
                    sampled[t] = (prepared, last[1])
            vectors, actual_times = [], []
            nviews = 2 if self.spec.views == "full_work" else 1
            for ts in targets_by_window:
                clips = torch.stack([normalize_clip([sampled[float(t)][0][v] for t in ts])
                                     for v in range(nviews)])
                vectors.append(self.net(clips).numpy().reshape(-1))
                actual_times.append([sampled[float(t)][1] for t in ts])
            actual_times = np.asarray(actual_times)
            return {"features": np.asarray(vectors), "frame_times": actual_times,
                    "centers": actual_times[:, FRAMES//2],
                    "starts": np.asarray([s for s, _ in spans]),
                    "ends": np.asarray([e for _, e in spans]),
                    "duration": np.asarray(duration), "spec": np.asarray(self.spec.to_json()),
                    "elapsed_seconds": np.asarray(time.monotonic()-began)}
        finally:
            cap.release()


def cache_key(digest, spec):
    import hashlib
    identity = f"{digest}:{spec.to_json()}:{sha256(WEIGHTS)}:{WORK_CROP}:{FRAMES}"
    return hashlib.sha256(identity.encode()).hexdigest()
