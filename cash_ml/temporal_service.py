"""Isolated candidate inference. Never installed as the active production model."""

import json
from pathlib import Path

import numpy as np
from torch import nn

from backend.schemas import AnalysisResult
from cash_ml.dense_features import DenseExtractor, DenseSpec
from cash_ml.evidence import context_frames
from cash_ml.features import sha256
from cash_ml.rules import describe_hypothesis, validate_registry
from cash_ml.service import event_peaks, violation_contract
from cash_ml.temporal_finetune import infer_block, load_block, prefix_tensor
from cash_ml.temporal_model import infer, load_head


class TemporalCandidateAnalyzer:
    def __init__(self, run):
        run = Path(run)
        self.selection = json.loads((run / "selection_frozen.json").read_text())
        path = run / (self.selection["selected"] + ".pt")
        if sha256(path) != self.selection["model_sha256"]:
            raise ValueError("Selected candidate weights changed")
        if sha256(run / "protocol.json") != self.selection["protocol_sha256"]:
            raise ValueError("Selected candidate protocol changed")
        self.extractor = DenseExtractor(DenseSpec(8, "full", 0, "center"))
        if self.selection["kind"] == "last_block":
            self.model, self.threshold, meta = load_block(path)
            base = self.extractor.net
            self.extractor.net = nn.Sequential(base.stem, base.layer1, base.layer2, base.layer3).eval()
        else:
            self.model, self.mean, self.scale, self.threshold, meta = load_head(path)
        self.allowed = meta["allowed_types"]
        validate_registry(self.allowed)
        self.event_model = violation_contract(self.allowed)

    def extract(self, path):
        raw = self.extractor.extract_raw(path, 8, "full")
        centers = raw["centers"]
        raw["starts"] = np.maximum(0.0, centers - 0.5)
        raw["ends"] = np.minimum(float(raw["duration"]), centers + 0.5)
        if self.selection["kind"] == "last_block":
            raw["features"] = prefix_tensor(raw["features"], len(centers))
        return raw

    def logits(self, data):
        if self.selection["kind"] == "last_block":
            return infer_block(self.model, {"input": data})["input"]
        return infer(self.model, {"input": data}, self.mean, self.scale)["input"]

    def from_features(self, path, data):
        binary, typed = self.logits(data)
        scores = 1 / (1 + np.exp(-np.clip(binary, -700, 700)))
        types = np.asarray(self.allowed)[typed.argmax(axis=1)]
        events = []
        for index in event_peaks(data["starts"], data["ends"], scores, self.threshold, types):
            actual, frames = context_frames(path, float(data["centers"][index]), float(data["duration"]))
            events.append(self.event_model(
                type=str(types[index]), timecode_sec=actual,
                description=describe_hypothesis(str(types[index]), float(data["starts"][index]),
                                               float(data["ends"][index]), float(scores[index]), self.threshold),
                evidence_frames=frames,
            ))
        return AnalysisResult(verdict="violation" if events else "no_violation", violations=events)

    def __call__(self, path):
        return self.from_features(path, self.extract(path))
