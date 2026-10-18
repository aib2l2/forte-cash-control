"""Offline dense candidate. Production keeps v2 until the deployment gate passes."""

import json
from pathlib import Path

import numpy as np

from backend.schemas import AnalysisResult
from backend.services.video import extract_evidence_frame
from cash_ml.dense_features import DenseExtractor, DenseSpec
from cash_ml.experiment import load_model, predict
from cash_ml.features import WEIGHTS, sha256
from cash_ml.service import event_peaks, violation_contract


class DenseCandidateAnalyzer:
    def __init__(self, run: Path):
        selection = json.loads((run/"selection_frozen.json").read_text())
        protocol = json.loads((run/"protocol.json").read_text())
        path = run/(selection["selected"]+".npz")
        if sha256(path) != selection["model_sha256"]:
            raise ValueError("Candidate artifact hash mismatch")
        if sha256(run/"protocol.json") != selection["protocol_sha256"]:
            raise ValueError("Protocol hash mismatch")
        self.model, spec, self.threshold, metadata = load_model(path)
        if metadata["weights_sha256"] != sha256(WEIGHTS):
            raise ValueError("R3D weights mismatch")
        if metadata["protocol_sha256"] != selection["protocol_sha256"]:
            raise ValueError("Model protocol mismatch")
        if metadata["preprocessing_sha256"] != sha256(Path(__file__).with_name("dense_features.py")):
            raise ValueError("Preprocessing code mismatch")
        if (list(self.model["typed"]["classes"]) != protocol["allowed_types"]
                or metadata["allowed_types"] != protocol["allowed_types"]):
            raise ValueError("Candidate vocabulary mismatch")
        if not np.isfinite(self.threshold):
            raise ValueError("Invalid threshold")
        self.event_model = violation_contract(protocol["allowed_types"])
        self.extractor = DenseExtractor(DenseSpec(**json.loads(spec)))

    def from_features(self, path, item):
        scores, types = predict(self.model, item)
        events = []
        for i in event_peaks(item["starts"], item["ends"], scores, self.threshold, types):
            evidence = extract_evidence_frame(path, float(item["centers"][i]))
            events.append(self.event_model(
                type=str(types[i]), timecode_sec=evidence.timestamp_sec,
                description=(f"Экспериментальная R3D-18: {types[i]}. Контекст "
                             f"{item['context_starts'][i]:.2f}–{item['context_ends'][i]:.2f} с. "
                             f"Оценка {scores[i]:.3f} не является вероятностью. "
                             "Таймкод указывает на реальный кадр, а не границу события. "
                             "Тип и нормативное основание требуют проверки."), evidence_frames=[evidence]))
        return AnalysisResult(verdict="violation" if events else "no_violation", violations=events)

    def __call__(self, path: Path):
        return self.from_features(path, self.extractor.extract(path))
