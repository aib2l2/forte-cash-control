"""Fixed R3D-18 features. Weights are downloaded only by the training command."""

import hashlib
import math
import os
from pathlib import Path

import certifi
import cv2
import numpy as np
import torch
from torchvision.models.video import R3D_18_Weights, r3d_18

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "cash_r3d"
WEIGHTS = MODEL_DIR / "r3d18_kinetics400.pth"
WINDOW = 4.0
STRIDE = 4.0
FRAMES = 16
PREPROCESS_VERSION = "r3d18-kinetics400-v1-16frames-4sec-stride4-v2"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def window_starts(duration):
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Некорректная длительность видео")
    last = max(0.0, duration - WINDOW)
    starts = list(np.arange(0.0, last + 1e-6, STRIDE))
    if last - starts[-1] > 0.05:
        starts.append(last)
    return starts


class Extractor:
    def __init__(self, download=False):
        torch.set_num_threads(4)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.net = r3d_18(weights=None)
        if not WEIGHTS.exists():
            if not download:
                raise FileNotFoundError("Локальные веса R3D-18 отсутствуют. Сначала обучите модель.")
            os.environ.setdefault("SSL_CERT_FILE", certifi.where())
            state = R3D_18_Weights.KINETICS400_V1.get_state_dict(progress=True, check_hash=True)
            temporary = WEIGHTS.with_suffix(".tmp")
            torch.save(state, temporary)
            temporary.replace(WEIGHTS)
        self.net.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
        self.net.fc = torch.nn.Identity()
        self.net.eval()
        self.transform = R3D_18_Weights.KINETICS400_V1.transforms()

    @torch.inference_mode()
    def extract(self, path):
        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                raise ValueError("Видео не декодируется")
            fps = cap.get(cv2.CAP_PROP_FPS)
            count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if not math.isfinite(fps) or fps <= 0 or not math.isfinite(count) or count < 1:
                raise ValueError("Не удалось определить длительность видео")
            duration = count / fps
            if duration > 600:
                raise ValueError("Прототип поддерживает видео не длиннее 10 минут")
            windows = []
            for start in window_starts(duration):
                end = min(duration, start + WINDOW)
                target_times = np.linspace(start, max(start, end - 1 / fps), FRAMES)
                windows.append((start, end, target_times))
            # Decode sequentially: seeking near the tail of some MOV files fails.
            targets = sorted({float(t) for _, _, times in windows for t in times})
            sampled, cursor, previous, last = {}, 0, -1.0, None
            while cursor < len(targets):
                ok, frame = cap.read()
                if not ok:
                    break
                actual = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                if not math.isfinite(actual) or actual < 0 or actual < previous:
                    raise ValueError("Некорректный таймкод декодера")
                previous = actual
                last = (frame, actual)
                if actual + 1e-6 >= targets[cursor]:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    resized = cv2.resize(rgb, (171, 128), interpolation=cv2.INTER_AREA)
                    while cursor < len(targets) and actual + 1e-6 >= targets[cursor]:
                        sampled[targets[cursor]] = (resized, actual)
                        cursor += 1
            if cursor < len(targets):
                if last is None or targets[-1] - last[1] > max(0.25, 2 / fps):
                    raise ValueError("Видео обрывается до заявленного конца")
                rgb = cv2.cvtColor(last[0], cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb, (171, 128), interpolation=cv2.INTER_AREA)
                for target in targets[cursor:]:
                    sampled[target] = (resized, last[1])
            features, centers, starts_out, ends = [], [], [], []
            for start, end, target_times in windows:
                frames = [sampled[float(t)][0] for t in target_times]
                actual_times = [sampled[float(t)][1] for t in target_times]
                clip = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
                features.append(self.net(self.transform(clip).unsqueeze(0))[0].numpy())
                centers.append(actual_times[FRAMES // 2])
                starts_out.append(start)
                ends.append(end)
            return {"features": np.asarray(features), "centers": np.asarray(centers),
                    "starts": np.asarray(starts_out), "ends": np.asarray(ends),
                    "duration": np.asarray(duration)}
        finally:
            cap.release()
