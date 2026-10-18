"""Controlled train-only description distillation into a video-only temporal head."""

import copy
import json
from collections import Counter

import numpy as np
import torch
from torch import nn

from cash_ml.temporal_model import (
    TemporalHead,
    infer,
    normalized,
    point_loss,
    training_scaler,
)

VERSION = "r3d18-description-guidance-v1"


class GuidedHead(TemporalHead):
    def __init__(self, dimensions=512, classes=11, text_dimensions=384):
        super().__init__("tcn", dimensions, classes, hidden=16)
        self.text = nn.Linear(16, text_dimensions)

    def shared(self, x):
        h = self.input(x)
        return h + self.temporal(h.T.unsqueeze(0))[0].T

    def training_outputs(self, x):
        h = self.shared(x)
        dropped = self.dropout(h)
        return self.binary(dropped).squeeze(-1), self.typed(dropped), h

    def forward(self, x):
        binary, typed, _ = self.training_outputs(x)
        return binary, typed


def subset_targets(path, records):
    if any(r["split"] != "train" for r in records):
        raise ValueError("Descriptions may only come from the original training split")
    with np.load(path, allow_pickle=False) as saved:
        ids = saved["ids"].astype(str)
        result = {}
        for r in records:
            mask = ids == r["id"]
            if not mask.any():
                raise ValueError("Missing training description")
            result[r["id"]] = {k: saved[k][mask] for k in ("starts", "ends", "targets")}
    return result


def target_masks(item, targets):
    masks = []
    for start, end in zip(targets["starts"], targets["ends"], strict=True):
        # This is a review sampling window, NEVER an annotated event interval.
        ids = np.flatnonzero((item["centers"] >= start) & (item["centers"] <= end))
        if not len(ids):
            ids = np.asarray([np.argmin(abs(item["centers"] - (start + end) / 2))])
        masks.append(torch.from_numpy(ids))
    return masks


def fit(train, validation, data, targets, allowed, weight, seed, max_epochs=60, patience=8):
    if set(targets) != {r["id"] for r in train}:
        raise ValueError("Auxiliary targets must exactly match this fold's training records")
    if {r["scene_group"] for r in train} & {r["scene_group"] for r in validation}:
        raise ValueError("Scene leakage")
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mean, scale = training_scaler(train, data)
    text_mean = np.mean([targets[r["id"]]["targets"].mean(0) for r in train], axis=0)
    model = GuidedHead(len(mean), len(allowed), len(text_mean))
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
    cc = Counter(a["type"] for r in train for a in r["annotations"])
    tw = torch.tensor([sum(cc.values()) / (len(allowed) * max(1, cc[t])) for t in allowed])
    lc = Counter(r["label"] for r in train)
    prepared = {}
    for r in train:
        target = torch.from_numpy(targets[r["id"]]["targets"] - text_mean)
        prepared[r["id"]] = (target_masks(data[r["id"]], targets[r["id"]]), target)
    history, best, best_loss, stale = [], None, float("inf"), 0
    for epoch in range(max_epochs):
        model.train()
        opt.zero_grad()
        supervised_total, auxiliary_total = 0.0, 0.0
        for idx in rng.permutation(len(train)):
            r = train[int(idx)]
            binary, typed, h = model.training_outputs(normalized(data[r["id"]], mean, scale))
            b, t = point_loss(binary, typed, r, data[r["id"]], allowed, tw)
            supervised = b / (2 * lc[r["label"]]) + 0.5 * t / max(1, lc[1])
            if weight:
                masks, target = prepared[r["id"]]
                predicted = model.text(torch.stack([h[ids].mean(0) for ids in masks]))
                auxiliary = (1 - nn.functional.cosine_similarity(predicted, target, dim=1)).mean()
                auxiliary = auxiliary * weight / len(train)
            else:
                auxiliary = h.sum() * 0
            (supervised + auxiliary).backward()
            supervised_total += float(supervised.detach())
            auxiliary_total += float(auxiliary.detach())
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        pred = infer(model, {r["id"]: data[r["id"]] for r in validation}, mean, scale)
        losses = []
        for r in validation:
            b, t = point_loss(
                *(torch.from_numpy(p) for p in pred[r["id"]]), r, data[r["id"]], allowed
            )
            losses.append(float(b + 0.5 * t))
        vl = float(np.mean(losses))
        history.append(
            {
                "epoch": epoch + 1,
                "supervised_loss": supervised_total,
                "auxiliary_loss": auxiliary_total,
                "validation_loss": vl,
            }
        )
        if vl < best_loss - 1e-5:
            best_loss, best, stale = vl, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if epoch >= 9 and stale >= patience:
            break
    model.load_state_dict(best)
    return model, mean, scale, text_mean, history


def save(path, fitted, metadata):
    model, mean, scale, text_mean, history = fitted
    torch.save(
        {
            "version": VERSION,
            "state_dict": model.state_dict(),
            "mean": torch.from_numpy(mean),
            "scale": torch.from_numpy(scale),
            "text_mean": torch.from_numpy(text_mean),
            "metadata_json": json.dumps(metadata),
        },
        path,
    )
    return history


def load(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["version"] != VERSION:
        raise ValueError("Wrong checkpoint version")
    meta = json.loads(saved["metadata_json"])
    model = GuidedHead(len(saved["mean"]), len(meta["allowed_types"]), len(saved["text_mean"]))
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, saved["mean"].numpy(), saved["scale"].numpy(), meta


def ensemble(paths, data):
    predictions = []
    for path in paths:
        model, mean, scale, _ = load(path)
        predictions.append(infer(model, data, mean, scale))
    return {
        key: tuple(np.mean([p[key][part] for p in predictions], axis=0) for part in (0, 1))
        for key in data
    }
