"""Train only R3D-18 layer4 and heads; frozen prefix is cached locally."""

import argparse
import copy
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

from cash_ml.dense_features import DenseExtractor, DenseSpec
from cash_ml.experiment import load_npz, write_json
from cash_ml.features import WEIGHTS, Extractor, sha256
from cash_ml.temporal_model import (
    evaluate_logits,
    load_head,
    point_indices,
    threshold_for,
    validation_rank,
)

PREFIX_VERSION = "r3d18-layer3-full-8fps-v1"


def prefix_tensor(features, clip_count):
    # For 16 x 112 x 112 input, layer3 is 256 x 4 x 14 x 14.
    # Recover the same flat values from a legacy cache with an incorrect view.
    expected = clip_count * 256 * 4 * 14 * 14
    if features.size != expected:
        raise ValueError("Unexpected R3D layer3 shape")
    return features.reshape(clip_count, 256, 4, 14, 14)


def prefix_path(run, record):
    return run / "layer3_cache" / (record["sha256"] + ".npz")


def extract_prefix(run, records):
    (run / "layer3_cache").mkdir(exist_ok=True)
    extractor = DenseExtractor(DenseSpec(8, "full", 0, "center"))
    base = extractor.net
    extractor.net = nn.Sequential(base.stem, base.layer1, base.layer2, base.layer3).eval()
    identity_base = {
        "version": PREFIX_VERSION,
        "weights_sha256": sha256(WEIGHTS),
        "preprocessing_sha256": sha256(Path(__file__).with_name("dense_features.py")),
    }
    for i, r in enumerate(records):
        p = prefix_path(run, r)
        identity = json.dumps({**identity_base, "source_sha256": r["sha256"]}, sort_keys=True)
        if not p.exists():
            if sha256(Path(r["path"])) != r["sha256"]:
                raise ValueError("Source hash changed")
            raw = extractor.extract_raw(Path(r["path"]), 8, "full")
            raw["features"] = prefix_tensor(raw["features"], len(raw["centers"]))
            raw["identity"] = np.asarray(identity)
            temp = p.with_suffix(".partial.npz")
            np.savez_compressed(temp, **raw)
            temp.replace(p)
        raw = load_npz(p)
        if str(raw["identity"]) != identity:
            raise ValueError("Prefix cache identity mismatch")
        print(
            json.dumps(
                {
                    "prefix_progress": i + 1,
                    "total": len(records),
                    "clips": len(raw["centers"]),
                    "seconds": round(float(raw["elapsed_seconds"]), 2),
                }
            ),
            flush=True,
        )


def prefix_data(run, records):
    data = {}
    for r in records:
        item = load_npz(prefix_path(run, r))
        ident = json.loads(str(item["identity"]))
        if (
            ident["source_sha256"] != r["sha256"]
            or ident["weights_sha256"] != sha256(WEIGHTS)
            or ident["version"] != PREFIX_VERSION
        ):
            raise ValueError("Invalid prefix cache")
        centers = item["centers"]
        item["features"] = prefix_tensor(item["features"], len(centers))
        data[r["id"]] = {
            **item,
            "starts": np.maximum(0.0, centers - 0.5),
            "ends": np.minimum(float(item["duration"]), centers + 0.5),
        }
    return data


class LastBlock(nn.Module):
    def __init__(self, head, mean, scale):
        super().__init__()
        base = Extractor().net
        self.layer4 = base.layer4
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.head = head
        self.register_buffer("mean", torch.from_numpy(mean))
        self.register_buffer("scale", torch.from_numpy(scale))

    def train(self, mode=True):
        super().train(mode)
        # BatchNorm statistics stay frozen despite tiny batches.
        for module in self.layer4.modules():
            if isinstance(module, nn.BatchNorm3d):
                module.eval()
                for p in module.parameters():
                    p.requires_grad_(False)
        return self

    def forward(self, prefix):
        features = self.pool(self.layer4(prefix)).flatten(1)
        return self.head((features - self.mean) / self.scale)


@torch.inference_mode()
def infer_block(model, data):
    model.eval()
    predictions = {}
    for name, item in data.items():
        chunks = [
            model(torch.from_numpy(item["features"][i : i + 8].astype(np.float32)))
            for i in range(0, len(item["features"]), 8)
        ]
        predictions[name] = tuple(torch.cat([c[k] for c in chunks]).numpy() for k in [0, 1])
    return predictions


def train_blocks(run, records, protocol):
    from cash_ml.temporal_experiment import sequence_data

    if (run / "selection_frozen.json").exists():
        raise ValueError("Selection is frozen; create a new experiment instead of retraining")

    train = [r for r in records if r["split"] == "train"]
    val = [r for r in records if r["split"] == "validation"]
    data = prefix_data(run, train + val)
    allowed = protocol["allowed_types"]
    counts = Counter(r["label"] for r in train)
    class_counts = Counter(a["type"] for r in train for a in r["annotations"])
    weights = torch.tensor(
        [sum(class_counts.values()) / (len(allowed) * max(1, class_counts[t])) for t in allowed]
    )
    rows = []
    for config in protocol["fine_tuning_grid"]:
        began = time.monotonic()
        torch.manual_seed(config["seed"])
        rng = np.random.default_rng(config["seed"])
        head, mean, scale, _, _ = load_head(run / "temporal_linear.pt")
        model = LastBlock(head, mean, scale)
        # Check that splitting the backbone did not alter frozen features/logits.
        if not (run / "prefix_parity.json").exists():
            one = train[:1]
            dense = sequence_data(one)
            frozen = infer_block(model, {one[0]["id"]: data[one[0]["id"]]})[one[0]["id"]]
            with torch.inference_mode():
                original = model.head(
                    torch.from_numpy((dense[one[0]["id"]]["features"] - mean) / scale)
                )
            delta = max(float(np.max(abs(frozen[k] - original[k].numpy()))) for k in [0, 1])
            write_json(
                run / "prefix_parity.json", {"max_logit_difference": delta, "tolerance": 1e-4}
            )
            if delta > 1e-4:
                raise ValueError("Prefix preparation differs from frozen features")
        model.train()
        initial = {
            n: p.detach().clone() for n, p in model.layer4.named_parameters() if p.requires_grad
        }
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": [p for p in model.layer4.parameters() if p.requires_grad],
                    "lr": config["layer4_lr"],
                },
                {"params": model.head.parameters(), "lr": config["head_lr"]},
            ],
            weight_decay=0.01,
        )
        history, best, best_loss, stale = [], None, float("inf"), 0
        for epoch in range(config["max_epochs"]):
            model.train()
            optimizer.zero_grad()
            total = 0.0
            for index in rng.permutation(len(train)):
                r = train[int(index)]
                item = data[r["id"]]
                if r["label"]:
                    ids = point_indices(r, item)
                else:
                    ids = np.sort(
                        rng.choice(
                            len(item["features"]), min(4, len(item["features"])), replace=False
                        )
                    )
                binary, typed = model(torch.from_numpy(item["features"][ids].astype(np.float32)))
                loss = nn.functional.binary_cross_entropy_with_logits(
                    binary, torch.full_like(binary, float(r["label"]))
                ) / (2 * counts[r["label"]])
                if r["label"]:
                    targets = torch.tensor([allowed.index(a["type"]) for a in r["annotations"]])
                    loss = (
                        loss
                        + 0.5
                        * (
                            nn.functional.cross_entropy(typed, targets, reduction="none")
                            * weights[targets]
                        ).mean()
                        / counts[1]
                    )
                loss.backward()
                total += float(loss.detach())
            reg = sum(
                (p - initial[n]).square().mean()
                for n, p in model.layer4.named_parameters()
                if p.requires_grad
            )
            (0.01 * reg).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            vp = infer_block(model, {r["id"]: data[r["id"]] for r in val})
            from cash_ml.temporal_model import point_loss

            vl = np.mean(
                [
                    sum(
                        float(x) * w
                        for x, w in zip(
                            point_loss(
                                *(torch.from_numpy(p) for p in vp[r["id"]]),
                                r,
                                data[r["id"]],
                                allowed,
                            ),
                            [1.0, 0.5],
                            strict=True,
                        )
                    )
                    for r in val
                ]
            )
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": total,
                    "validation_loss": float(vl),
                    "elapsed_seconds": time.monotonic() - began,
                }
            )
            if vl < best_loss - 1e-5:
                best_loss, best, stale = vl, copy.deepcopy(model.state_dict()), 0
            else:
                stale += 1
            print(
                json.dumps(
                    {
                        "finetune_lr": config["layer4_lr"],
                        "epoch": epoch + 1,
                        "validation_loss": round(float(vl), 4),
                        "seconds": round(time.monotonic() - began, 1),
                    }
                ),
                flush=True,
            )
            if epoch >= 3 and stale >= config["patience"]:
                break
        model.load_state_dict(best)
        pred = infer_block(model, {r["id"]: data[r["id"]] for r in val})
        threshold = threshold_for(pred, val)
        result = evaluate_logits(pred, val, data, threshold, allowed)
        delta = float(
            sum(
                (p.detach() - initial[n]).square().sum()
                for n, p in model.layer4.named_parameters()
                if n in initial
            ).sqrt()
        )
        name = "layer4_" + str(config["layer4_lr"]).replace(".", "p").replace("-", "m")
        meta = {
            "kind": "last_block",
            "name": name,
            "allowed_types": allowed,
            "protocol_sha256": sha256(run / "protocol.json"),
            "weights_sha256": sha256(WEIGHTS),
            "config": config,
            "prefix_version": PREFIX_VERSION,
        }
        path = run / (name + ".pt")
        torch.save(
            {
                "state_dict": model.state_dict(),
                "mean": torch.from_numpy(mean),
                "scale": torch.from_numpy(scale),
                "metadata_json": json.dumps(meta),
                "threshold": threshold,
            },
            path,
        )
        row = {
            "name": name,
            "kind": "last_block",
            "config": config,
            "threshold": threshold,
            "validation": result,
            "rank": validation_rank(result),
            "history": history,
            "model_sha256": sha256(path),
            "training_seconds": time.monotonic() - began,
            "layer4_weight_l2_change": delta,
        }
        rows.append(row)
        write_json(run / "finetuning_grid.json", rows)
        print(
            json.dumps({"finetune_complete": name, "rank": row["rank"], "weight_change": delta}),
            flush=True,
        )


def load_block(path):
    from cash_ml.temporal_model import TemporalHead

    saved = torch.load(path, map_location="cpu", weights_only=True)
    meta = json.loads(saved["metadata_json"])
    if meta["weights_sha256"] != sha256(WEIGHTS) or meta["prefix_version"] != PREFIX_VERSION:
        raise ValueError("Fine-tuned backbone identity mismatch")
    model = LastBlock(
        TemporalHead("linear", 512, len(meta["allowed_types"])),
        saved["mean"].numpy(),
        saved["scale"].numpy(),
    )
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, saved["threshold"], meta


def predict_saved(run, path, records):
    extract_prefix(run, records)
    data = prefix_data(run, records)
    model, threshold, _ = load_block(path)
    return infer_block(model, data), data, threshold


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--phase", choices=["extract", "train"], required=True)
    a = p.parse_args()
    records = json.loads((a.run / "previous_grouped_records.json").read_text())
    protocol = json.loads((a.run / "protocol.json").read_text())
    if a.phase == "extract":
        extract_prefix(a.run, [r for r in records if r["split"] != "test"])
    else:
        train_blocks(a.run, records, protocol)


if __name__ == "__main__":
    main()
