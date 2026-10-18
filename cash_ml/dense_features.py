"""Dense short R3D clips and ordered context, shared by training and inference.

ROIs were chosen from TRAIN frames only. They are camera-specific image regions,
not object detections. Context boundaries are not annotated event boundaries.
"""

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass

import cv2
import numpy as np
import torch

from cash_ml.context_features import letterbox, normalize_clip
from cash_ml.features import WEIGHTS, Extractor, sha256

VERSION = "r3d18-dense-v1"
REGIONS = ((.02, .38, .78, .66), (.28, .28, .98, .63))


@dataclass(frozen=True)
class DenseSpec:
    fps: int = 8
    views: str = "full"
    context: int = 8
    pooling: str = "ordered"
    version: str = VERSION

    def __post_init__(self):
        if (self.fps not in (8, 15) or self.views not in ("full", "details")
                or self.context not in (0, 8, 16) or self.pooling not in ("center", "mean", "ordered")
                or (self.context == 0) != (self.pooling == "center") or self.version != VERSION):
            raise ValueError("Invalid dense feature specification")

    def to_json(self):
        return json.dumps(asdict(self), sort_keys=True)


def clip_schedule(duration, source_fps, fps):
    if not math.isfinite(duration) or duration <= 0 or not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("Invalid video timing")
    if fps not in (8, 15):
        raise ValueError("Invalid sampling rate")
    last_frame = max(0., duration - 1/source_fps)
    span = 15/fps
    last_start = max(0., last_frame-span)
    # Both rates overlap; fixed sampling, independent of labels and filenames.
    stride = 1. if fps == 8 else .75
    starts = list(np.arange(0., last_start+1e-9, stride))
    if last_start-starts[-1] > 1e-7:
        starts.append(last_start)
    return np.minimum(np.asarray(starts)[:, None] + np.arange(16)[None, :]/fps, last_frame)


def dense_views(frame, views):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = [letterbox(rgb)]
    if views == "details":
        h, w = rgb.shape[:2]
        result += [letterbox(rgb[round(y1*h):round(y2*h), round(x1*w):round(x2*w)])
                   for x1, y1, x2, y2 in REGIONS]
    elif views != "full":
        raise ValueError("Unknown views")
    return result


def raw_identity(digest, fps, views):
    return json.dumps({"video": digest, "fps": fps, "views": views, "version": VERSION,
                       "weights": sha256(WEIGHTS), "regions": REGIONS}, sort_keys=True)


def raw_cache_key(digest, fps, views):
    return hashlib.sha256(raw_identity(digest, fps, views).encode()).hexdigest()


class DenseExtractor(Extractor):
    def __init__(self, spec=None):
        super().__init__(download=False)
        self.spec = spec or DenseSpec()

    @torch.inference_mode()
    def extract_raw(self, path, fps=None, views=None):
        began = time.monotonic()
        fps, views = fps or self.spec.fps, views or self.spec.views
        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                raise ValueError("Video cannot be decoded")
            source_fps, count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if not math.isfinite(source_fps) or source_fps <= 0 or not math.isfinite(count) or count < 1:
                raise ValueError("Invalid video metadata")
            duration = count/source_fps
            if duration > 600:
                raise ValueError("Video exceeds 10 minutes")
            times = clip_schedule(duration, source_fps, fps)
            targets = np.unique(times)
            sampled, cursor, previous, last = {}, 0, -1., None
            while cursor < len(targets):
                ok, frame = cap.read()
                if not ok:
                    break
                actual = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                if not math.isfinite(actual) or actual < 0 or actual < previous:
                    raise ValueError("Invalid decoded timestamp")
                last, previous = (frame, actual), actual
                if actual + 1e-6 >= targets[cursor]:
                    prepared = dense_views(frame, views)
                    while cursor < len(targets) and actual + 1e-6 >= targets[cursor]:
                        sampled[float(targets[cursor])] = (prepared, actual)
                        cursor += 1
            if cursor < len(targets):
                if last is None or targets[-1]-last[1] > max(.25, 2/source_fps):
                    raise ValueError("Truncated video")
                prepared = dense_views(last[0], views)
                for t in targets[cursor:]:
                    sampled[float(t)] = (prepared, last[1])
            nviews = 3 if views == "details" else 1
            vectors, actual_times = [], []
            for ts in times:
                clips = torch.stack([normalize_clip([sampled[float(t)][0][v] for t in ts])
                                     for v in range(nviews)])
                vectors.append(self.net(clips).numpy().reshape(-1))
                actual_times.append([sampled[float(t)][1] for t in ts])
            actual_times = np.asarray(actual_times)
            return {"features": np.asarray(vectors), "frame_times": actual_times,
                    "target_times": times, "centers": actual_times[:, 8],
                    "starts": actual_times[:, 0], "ends": actual_times[:, -1],
                    "duration": np.asarray(duration), "source_fps": np.asarray(source_fps),
                    "fps": np.asarray(fps), "views": np.asarray(views),
                    "elapsed_seconds": np.asarray(time.monotonic()-began)}
        finally:
            cap.release()

    def extract(self, path):
        return aggregate(self.extract_raw(path), self.spec)


def aggregate(raw, spec):
    """2-second decision grid; the same anchors for all context alternatives.

    Context is clipped at edges without shifting the decision timestamp. Empty
    temporal bins repeat their nearest available clip and carry an edge flag.
    """
    if int(raw["fps"]) != spec.fps:
        raise ValueError("Sampling rate mismatch")
    if spec.views == "details" and str(raw["views"]) != "details":
        raise ValueError("Missing detail views")
    x = raw["features"][:, :512] if spec.views == "full" else raw["features"]
    centers, duration = raw["centers"], float(raw["duration"])
    targets = np.r_[np.arange(0., duration, 2.), centers[-1]]
    anchors = sorted({int(np.argmin(abs(centers-t))) for t in targets})
    values, cstarts, cends, edge_flags = [], [], [], []
    for i in anchors:
        center = centers[i]
        if spec.pooling == "center":
            values.append(x[i])
            cstarts.append(raw["starts"][i])
            cends.append(raw["ends"][i])
            edge_flags.append(False)
            continue
        start, end = max(0., center-spec.context/2), min(duration, center+spec.context/2)
        cstarts.append(start)
        cends.append(end)
        edge_flags.append(center-spec.context/2 < 0 or center+spec.context/2 > duration)
        if spec.pooling == "mean":
            indices = np.flatnonzero((centers >= start) & (centers <= end))
            values.append(x[indices].mean(axis=0))
        else:
            boundaries = center + np.linspace(-spec.context/2, spec.context/2, 4)
            bins = []
            for j in range(3):
                mask = (centers >= boundaries[j]) & ((centers <= boundaries[j+1]) if j == 2
                                                      else (centers < boundaries[j+1]))
                indices = np.flatnonzero(mask)
                bins.append(x[indices].mean(axis=0) if len(indices)
                            else x[int(np.argmin(abs(centers-(boundaries[j]+boundaries[j+1])/2)))])
            values.append(np.concatenate(bins))
    selected = centers[anchors]
    # Merge on local decision support, not the entire 8/16-second context.
    return {"features": np.asarray(values), "centers": selected,
            "starts": np.maximum(0., selected-1), "ends": np.minimum(duration, selected+1),
            "context_starts": np.asarray(cstarts), "context_ends": np.asarray(cends),
            "edge_context": np.asarray(edge_flags), "duration": np.asarray(duration),
            "frame_times": raw["frame_times"][anchors], "spec": np.asarray(spec.to_json())}
