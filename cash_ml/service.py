"""Local inference; never reads labels or downloads weights."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import create_model

from backend.schemas import AnalysisResult, Violation
from backend.services.analyzer import AnalyzerNotConfigured
from backend.services.video import extract_evidence_frame
from cash_ml.evidence import context_frames
from cash_ml.features import MODEL_DIR, PREPROCESS_VERSION, WEIGHTS, Extractor, sha256
from cash_ml.rules import describe_hypothesis, validate_registry


def violation_contract(allowed_types):
    if not allowed_types or any(not isinstance(t, str) or not t for t in allowed_types):
        raise ValueError("Пустой или некорректный словарь типов")
    return create_model("LabelViolation", __base__=Violation,
                        type=(Literal[tuple(allowed_types)], ...))


def model_status():
    ready = all((MODEL_DIR / p).is_file() for p in ["typed_classifier.npz", "typed_report.json"]) and WEIGHTS.is_file()
    report = json.loads((MODEL_DIR / "typed_report.json").read_text()) if ready else {}
    ready = ready and report.get("format_version") == 2
    return {"ready": ready, "experimental": True, "model": report.get("model"),
            "active_version": "typed_v2", "latest_experiment": experiment_status(),
            "format_version": 2, "allowed_types": report.get("allowed_types", []),
            "type_test": report.get("type_test"), "type_train_counts": report.get("type_train_counts", {}),
            "threshold": report.get("threshold"), "validation": report.get("validation"),
            "test": report.get("test"), "limitations": report.get("limitations", [])}


def experiment_status():
    """Expose the comparison without changing the active classifier or job store."""
    latest = MODEL_DIR.parent/"experiments/latest_run.txt"
    if not latest.exists():
        return None
    try:
        run = Path(latest.read_text().strip())
        path = run/"comparison.json"
        if not path.exists():
            return {"state": "running", "candidate_active": False}
        report = json.loads(path.read_text())
        return {"state": "complete", "candidate_active": False, "selected": report["selected"],
                "test_status": report["test_status"], "deployment_gate": report["deployment_gate"],
                "supports": report["supports"],
                "comparison": {name: {part: {k: v for k, v in report[name][part].items()
                                              if k not in ("videos", "point_predictions")}
                                      for part in ("validation", "test")}
                               for name in ("baseline", "candidate")}}
    except (OSError, ValueError, KeyError):
        return {"state": "report_unavailable", "candidate_active": False}


def event_peaks(starts, ends, scores, threshold, types=None):
    """Merge adjacent flagged windows only when the predicted types agree."""
    groups = []
    for i in np.flatnonzero(scores >= threshold):
        if (groups and starts[i] <= ends[groups[-1][-1]] + 0.05
                and (types is None or types[i] == types[groups[-1][-1]])):
            groups[-1].append(int(i))
        else:
            groups.append([int(i)])
    return [max(group, key=lambda i: scores[i]) for group in groups]


class Analyzer:
    def __init__(self):
        if not model_status()["ready"]:
            raise AnalyzerNotConfigured("Модель ещё не обучена. Запустите cash_ml.train.")
        report = json.loads((MODEL_DIR / "typed_report.json").read_text())
        path = MODEL_DIR / "typed_classifier.npz"
        if sha256(path) != report["classifier_sha256"]:
            raise ValueError("Классификатор не соответствует отчёту обучения")
        if sha256(WEIGHTS) != report["weights_sha256"]:
            raise ValueError("Веса R3D-18 не соответствуют отчёту обучения")
        with np.load(path, allow_pickle=False) as saved:
            if int(saved["format_version"]) != 2:
                raise ValueError("Нужна переобученная модель типов версии 2")
            if str(saved["preprocess_version"]) != PREPROCESS_VERSION:
                raise ValueError("Версия признаков отличается от версии обучения")
            self.mean, self.scale = saved["mean"], saved["scale"]
            self.coef = saved["coef"]
            self.intercept, self.threshold = float(saved["intercept"]), float(saved["threshold"])
            self.type_mean, self.type_scale = saved["type_mean"], saved["type_scale"]
            self.type_coef, self.type_intercept = saved["type_coef"], saved["type_intercept"]
            self.type_classes = saved["type_classes"].astype(str)
        if list(self.type_classes) != report["allowed_types"]:
            raise ValueError("Словарь классификатора отличается от типов в исходной разметке")
        self.event_model = violation_contract(report["allowed_types"])
        validate_registry(report["allowed_types"])
        self.extractor = Extractor()

    def __call__(self, path: Path):
        data = self.extractor.extract(path)
        logits = ((data["features"] - self.mean) / self.scale) @ self.coef + self.intercept
        scores = 1 / (1 + np.exp(-np.clip(logits, -700, 700)))
        type_logits = ((data["features"] - self.type_mean) / self.type_scale) @ self.type_coef.T + self.type_intercept
        types = self.type_classes[np.argmax(type_logits, axis=1)]
        events = []
        for index in event_peaks(data["starts"], data["ends"], scores, self.threshold, types):
            timestamp = float(data["centers"][index])
            actual, frames = context_frames(path, timestamp, float(data["ends"][-1]),
                                            reader=extract_evidence_frame)
            events.append(self.event_model(
                type=str(types[index]),
                timecode_sec=actual,
                description=describe_hypothesis(str(types[index]), float(data["starts"][index]),
                                               float(data["ends"][index]), float(scores[index]),
                                               self.threshold),
                evidence_frames=frames,
            ))
        return AnalysisResult(verdict="violation" if events else "no_violation", violations=events)


@lru_cache(maxsize=1)
def get_analyzer():
    return Analyzer()


def analyze_video(path):
    return get_analyzer()(path)
