"""Offline candidate inference; never switches the active API model."""

import json
from pathlib import Path

import numpy as np
import torch

from cash_ml.dense_features import DenseExtractor, DenseSpec
from cash_ml.description_guidance import load
from cash_ml.features import WEIGHTS, sha256
from cash_ml.rules import validate_registry
from cash_ml.service import violation_contract
from cash_ml.temporal_model import infer
from cash_ml.temporal_service import TemporalCandidateAnalyzer


class DescriptionCandidateAnalyzer(TemporalCandidateAnalyzer):
    def __init__(self, run):
        run = Path(run)
        selection = json.loads((run / "selection_frozen.json").read_text())
        if sha256(run / "protocol.json") != selection["protocol_sha256"]:
            raise ValueError("Protocol changed")
        if sha256(run / "main_training.json") != selection["training_sha256"]:
            raise ValueError("Selected ensemble changed")
        protocol = json.loads((run / "protocol.json").read_text())
        if sha256(WEIGHTS) != protocol["weights_sha256"]:
            raise ValueError("Video backbone changed")
        config = json.loads((run / "main_training.json").read_text())[selection["selected"]]
        self.models = []
        for name in config["checkpoints"]:
            if sha256(run / name) != config["sha256"][name]:
                raise ValueError("Candidate weights changed")
            model, mean, scale, meta = load(run / name)
            if meta["allowed_types"] != protocol["allowed_types"]:
                raise ValueError("Invalid type vocabulary")
            self.models.append((model, mean, scale))
        self.allowed = protocol["allowed_types"]
        self.threshold = config["threshold"]
        self.selection = {"kind": "description_guidance"}
        validate_registry(self.allowed)
        self.event_model = violation_contract(self.allowed)
        torch.set_num_threads(4)
        self.extractor = DenseExtractor(DenseSpec(8, "full", 0, "center"))

    def logits(self, data):
        predictions = [
            infer(model, {"input": data}, mean, scale)["input"]
            for model, mean, scale in self.models
        ]
        return tuple(np.mean([p[part] for p in predictions], axis=0) for part in (0, 1))
