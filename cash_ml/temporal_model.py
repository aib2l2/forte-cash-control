"""Local temporal heads with point supervision and a shared inference contract."""

import copy
import json
from collections import Counter

import numpy as np
import torch
from torch import nn

from cash_ml.experiment import evaluate, sigmoid
from cash_ml.train import metrics

VERSION = "r3d18-temporal-v1"


class TemporalHead(nn.Module):
    def __init__(self, family, dimensions=512, classes=11, hidden=16):
        super().__init__()
        self.family = family
        self.input = nn.Linear(dimensions, hidden) if family != "linear" else nn.Identity()
        if family == "tcn":
            self.temporal = nn.Sequential(
                nn.Conv1d(hidden, hidden, 3, padding=1),
                nn.GELU(),
                nn.Dropout(0.25),
                nn.Conv1d(hidden, hidden, 3, padding=2, dilation=2),
                nn.GELU(),
                nn.Conv1d(hidden, hidden, 3, padding=4, dilation=4),
                nn.GELU(),
            )
        elif family == "gru":
            self.temporal = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        elif family != "linear":
            raise ValueError("Unknown temporal family")
        width = dimensions if family == "linear" else hidden * (2 if family == "gru" else 1)
        self.dropout = nn.Dropout(0.25)
        self.binary = nn.Linear(width, 1)
        self.typed = nn.Linear(width, classes)

    def forward(self, x):
        # One unpadded video at a time; no leakage through batch padding or ordering.
        h = self.input(x)
        if self.family == "tcn":
            h = h + self.temporal(h.T.unsqueeze(0))[0].T
        elif self.family == "gru":
            h, _ = self.temporal(h.unsqueeze(0))
            h = h[0]
        h = self.dropout(h)
        return self.binary(h).squeeze(-1), self.typed(h)


def point_indices(record, item):
    return np.asarray(
        [int(np.argmin(abs(item["centers"] - a["timecode"]))) for a in record["annotations"]],
        dtype=np.int64,
    )


def point_loss(binary, typed, record, item, allowed, type_weights=None):
    """Unlabelled windows in positive videos never become negative targets."""
    if record["label"]:
        ids = point_indices(record, item)
        if not len(ids):
            raise ValueError("Positive video without point supervision")
        binary_loss = nn.functional.binary_cross_entropy_with_logits(
            binary[np.unique(ids)], torch.ones(len(np.unique(ids)))
        )
        targets = torch.tensor([allowed.index(a["type"]) for a in record["annotations"]])
        losses = nn.functional.cross_entropy(typed[ids], targets, reduction="none")
        type_loss = (
            (losses * type_weights[targets]).mean() if type_weights is not None else losses.mean()
        )
        return binary_loss, type_loss
    return nn.functional.binary_cross_entropy_with_logits(
        binary, torch.zeros_like(binary)
    ), binary.sum() * 0


def training_scaler(records, data):
    # Equal video weight; includes only training features, without labels as inputs.
    mean = np.mean([data[r["id"]]["features"].mean(0) for r in records], axis=0)
    var = np.mean([((data[r["id"]]["features"] - mean) ** 2).mean(0) for r in records], axis=0)
    return mean.astype(np.float32), np.maximum(np.sqrt(var), 1e-5).astype(np.float32)


def normalized(item, mean, scale):
    return torch.from_numpy(((item["features"] - mean) / scale).astype(np.float32))


@torch.inference_mode()
def infer(head, data, mean, scale):
    head.eval()
    return {
        name: tuple(t.numpy() for t in head(normalized(item, mean, scale)))
        for name, item in data.items()
    }


def threshold_for(predictions, records):
    scores = [float(sigmoid(predictions[r["id"]][0]).max()) for r in records]
    y = [r["label"] for r in records]
    candidates = np.r_[0.0, np.unique(scores), np.nextafter(max(scores), np.inf)]

    def rank(t):
        m = metrics(y, scores, t)
        return m["balanced_accuracy"], -m["confusion_matrix_tn_fp_fn_tp"][0][1], float(t)

    return float(max(candidates, key=rank))


def evaluate_logits(predictions, records, data, threshold, allowed):
    # Reuse the SAME event merging and one-to-one timing matcher as the baseline.
    n = len(allowed)
    adapter = {
        "binary": {
            "mean": np.zeros(n + 1),
            "scale": np.ones(n + 1),
            "coef": np.eye(n + 1)[:1],
            "intercept": np.zeros(1),
        },
        "typed": {
            "mean": np.zeros(n + 1),
            "scale": np.ones(n + 1),
            "coef": np.eye(n + 1)[1:],
            "intercept": np.zeros(n),
            "classes": np.asarray(allowed),
        },
    }
    merged = {
        r["id"]: {**data[r["id"]], "features": np.column_stack(predictions[r["id"]])}
        for r in records
    }
    result = evaluate(adapter, records, merged, threshold, allowed)
    tn, fp = result["detection"]["confusion_matrix_tn_fp_fn_tp"][0]
    fn, tp = result["detection"]["confusion_matrix_tn_fp_fn_tp"][1]
    result["detection"]["accuracy"] = (tn + tp) / len(records)
    result["detection"]["f1"] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    for support in result["per_type"].values():
        support["accuracy"] = (
            support["correct"] / support["support"] if support["support"] else None
        )
    result.pop("false_alerts_per_negative_minute", None)
    result["negative_exposure_seconds"] = sum(
        float(data[r["id"]]["duration"]) for r in records if not r["label"]
    )
    result["false_alerts_per_hour"] = None
    return result


def validation_rank(result):
    return (
        result["detection"]["balanced_accuracy"],
        result["timing"]["typed_point_recall"] or 0.0,
        result["type_accuracy_at_points"] or 0.0,
        -result["false_alerts_on_negative_videos"],
    )


def train_temporal(train, validation, data, allowed, family, seed=42, max_epochs=60, patience=8):
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mean, scale = training_scaler(train, data)
    model = TemporalHead(family, len(mean), len(allowed))
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
    class_counts = Counter(a["type"] for r in train for a in r["annotations"])
    tw = torch.tensor(
        [sum(class_counts.values()) / (len(allowed) * max(1, class_counts[t])) for t in allowed],
        dtype=torch.float32,
    )
    label_counts = Counter(r["label"] for r in train)
    history, best, best_loss, stale = [], None, float("inf"), 0
    for epoch in range(max_epochs):
        model.train()
        opt.zero_grad()
        total = 0.0
        for i in rng.permutation(len(train)):
            r = train[int(i)]
            logits = model(normalized(data[r["id"]], mean, scale))
            b, t = point_loss(*logits, r, data[r["id"]], allowed, tw)
            loss = b / (2 * label_counts[r["label"]]) + 0.5 * t / max(1, label_counts[1])
            loss.backward()
            total += float(loss.detach())
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        pred = infer(model, {r["id"]: data[r["id"]] for r in validation}, mean, scale)
        # Early stopping uses a fixed proper loss, not a test-tuned alarm threshold.
        losses = []
        for r in validation:
            b, t = point_loss(
                *(torch.from_numpy(p) for p in pred[r["id"]]), r, data[r["id"]], allowed
            )
            losses.append(float(b + 0.5 * t))
        vl = float(np.mean(losses))
        history.append({"epoch": epoch + 1, "train_loss": total, "validation_loss": vl})
        if vl < best_loss - 1e-5:
            best_loss, best, stale = vl, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if epoch >= 9 and stale >= patience:
            break
    model.load_state_dict(best)
    pred = infer(model, {r["id"]: data[r["id"]] for r in validation}, mean, scale)
    threshold = threshold_for(pred, validation)
    score = evaluate_logits(pred, validation, data, threshold, allowed)
    return model, mean, scale, threshold, score, history


def save_head(path, model, mean, scale, threshold, metadata):
    torch.save(
        {
            "state_dict": model.state_dict(),
            "mean": torch.from_numpy(mean),
            "scale": torch.from_numpy(scale),
            "threshold": threshold,
            "metadata_json": json.dumps(metadata),
            "version": VERSION,
        },
        path,
    )


def load_head(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["version"] != VERSION:
        raise ValueError("Unsupported temporal model")
    meta = json.loads(saved["metadata_json"])
    head = TemporalHead(meta["family"], len(saved["mean"]), len(meta["allowed_types"]))
    head.load_state_dict(saved["state_dict"])
    head.eval()
    return head, saved["mean"].numpy(), saved["scale"].numpy(), saved["threshold"], meta
