"""Explicit offline candidate inference. The production API still uses v2.

Loading a candidate verifies the selected artifact and weights. Preprocessing,
numeric heads and event merging are identical to the experiment evaluator.
"""

import json
from pathlib import Path

from backend.schemas import AnalysisResult
from backend.services.video import extract_evidence_frame
from cash_ml.context_features import ContextExtractor, FeatureSpec
from cash_ml.experiment import load_model, predict
from cash_ml.features import WEIGHTS, sha256
from cash_ml.service import event_peaks, violation_contract


class CandidateAnalyzer:
    def __init__(self, run: Path):
        report = json.loads((run/"comparison.json").read_text())
        selection = json.loads((run/"selection_frozen.json").read_text())
        if selection["selected"] != report["selected"]:
            raise ValueError("Candidate selection does not match comparison")
        grid = json.loads((run/"validation_grid.json").read_text())
        entry = next(r for r in grid if r["name"] == report["selected"])
        path = run/(entry["name"]+".npz")
        if sha256(path) != entry["model_sha256"]:
            raise ValueError("Candidate artifact hash mismatch")
        self.model, spec, self.threshold, metadata = load_model(path)
        if metadata["weights_sha256"] != sha256(WEIGHTS):
            raise ValueError("R3D weights differ from training")
        if metadata["protocol_sha256"] != sha256(run/"protocol.json"):
            raise ValueError("Evaluation protocol differs from training")
        if list(self.model["typed"]["classes"]) != report["allowed_types"]:
            raise ValueError("Candidate vocabulary mismatch")
        self.event_model = violation_contract(report["allowed_types"])
        self.extractor = ContextExtractor(FeatureSpec(**json.loads(spec)))

    def __call__(self, path: Path):
        item = self.extractor.extract(path)
        scores, types = predict(self.model, item)
        events = []
        for i in event_peaks(item["starts"], item["ends"], scores, self.threshold, types):
            evidence = extract_evidence_frame(path, float(item["centers"][i]))
            events.append(self.event_model(
                type=str(types[i]), timecode_sec=evidence.timestamp_sec,
                description=(f"Экспериментальная модель: {types[i]}. Контекст "
                             f"{item['starts'][i]:.1f}–{item['ends'][i]:.1f} с; оценка {scores[i]:.3f}. "
                             "Оценка не является вероятностью. Таймкод — кадр внутри контекста, "
                             "а не установленная граница события. Нужна проверка оператором; "
                             "нормативное основание не установлено."),
                evidence_frames=[evidence]))
        return AnalysisResult(verdict="violation" if events else "no_violation", violations=events)
