"""Small predeclared temporal comparison; no video content is printed."""

import argparse
import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from cash_ml.dense_experiment import cache_path, score
from cash_ml.dense_features import raw_identity
from cash_ml.experiment import (
    choose_threshold,
    deployment_gate,
    load_model,
    load_npz,
    save_model,
    train_heads,
    write_json,
)
from cash_ml.features import MODEL_DIR, PREPROCESS_VERSION, WEIGHTS, sha256
from cash_ml.temporal_model import (
    evaluate_logits,
    infer,
    load_head,
    save_head,
    train_temporal,
    validation_rank,
)

PREVIOUS = MODEL_DIR.parent / "experiments/20261017T223442Z_dense"


def sequence_data(records, cache=PREVIOUS):
    data = {}
    for r in records:
        raw = load_npz(cache_path(cache, r, 8, "details"))
        if str(raw["identity"]) != raw_identity(r["sha256"], 8, "details"):
            raise ValueError("Dense feature identity mismatch")
        c = raw["centers"]
        data[r["id"]] = {
            "features": raw["features"][:, :512].astype(np.float32),
            "centers": c,
            "starts": np.maximum(0.0, c - 0.5),
            "ends": np.minimum(float(raw["duration"]), c + 0.5),
            "duration": raw["duration"],
        }
    return data


def freeze_protocol(run):
    if (run / "protocol.json").exists():
        raise ValueError("Protocol already frozen")
    records = json.loads((run / "previous_grouped_records.json").read_text())
    (run / "grouped_records.json").write_bytes((run / "previous_grouped_records.json").read_bytes())
    allowed = json.loads((run / "baseline_snapshot/typed_report.json").read_text())["allowed_types"]
    for r in records:
        if sha256(Path(r["path"])) != r["sha256"]:
            raise ValueError("Original video changed")
    for group in {r["scene_group"] for r in records}:
        if len({r["split"] for r in records if r["scene_group"] == group}) != 1:
            raise ValueError("Scene leakage")
    protocol = {
        "version": 1,
        "utc": datetime.now(UTC).isoformat(),
        "allowed_types": allowed,
        "test_status": "reused exploratory test, NOT independent final evaluation",
        "independent_final_test": False,
        "head_grid": [
            {"family": f, "seed": 42, "max_epochs": 60, "patience": 8}
            for f in ["linear", "tcn", "gru"]
        ],
        "fine_tuning_grid": [
            {"layer4_lr": lr, "head_lr": 0.0003, "max_epochs": 12, "patience": 4, "seed": 42}
            for lr in [1e-5, 3e-5]
        ],
        "early_stop": "mean per-video point/negative loss on validation; unknown positive windows ignored",
        "selection": "validation balanced accuracy, typed point recall, point type accuracy, fewer negative alerts; fixed order on tie",
        "cv": {
            "seed": 73,
            "outer_test_groups": [
                ["scene_01", "scene_05", "scene_09"],
                ["scene_02", "scene_04"],
                ["scene_03", "scene_07"],
            ],
            "inner_validation": "last sorted remaining training group; all early stopping/thresholds on it",
            "purpose": "stability audit of temporal family selected on outer validation; not an independent final test",
        },
        "preprocessing": "full frame, letterbox112,16 frames at8fps,stride1sec,last real tail; no fixed ROI",
        "point_tolerance_seconds": 4.0,
        "tolerance_status": "internal research metric, not jury tolerance",
        "promotion": "existing deployment_gate AND independent final test AND end-to-end <=300sec AND service checks",
        "weights_sha256": sha256(WEIGHTS),
        "grouped_records_sha256": sha256(run / "previous_grouped_records.json"),
        "labels_sha256": sha256(run / "baseline_snapshot/labels_final.xlsx"),
        "source_code_sha256": {
            p.name: sha256(p) for p in Path(__file__).parent.glob("temporal*.py")
        },
        "state_features": "No object-state training labels or validated detectors; no invented state observations. Local review UI supplies future labels.",
        "augmentation": "feature dropout only; no claim of robustness to actual new camera angle",
        "budget": "3 temporal heads + 2 last-block rates; 12 epochs maximum for layer4; no automatic expanded search",
    }
    write_json(run / "protocol.json", protocol)
    write_json(
        run / "dataset_audit.json",
        {
            "unique_videos": len(records),
            "events": sum(len(r["annotations"]) for r in records),
            "split_counts": dict(Counter(r["split"] for r in records)),
            "groups_per_split": {
                s: len({r["scene_group"] for r in records if r["split"] == s})
                for s in ["train", "validation", "test"]
            },
            "all_source_hashes_unchanged": True,
            "point_only": True,
            "cloud_annotations_used": False,
        },
    )


def select_heads(run, records, protocol):
    if (run / "selection_frozen.json").exists():
        raise ValueError("Selection is frozen; create a new experiment instead of retraining")
    train = [r for r in records if r["split"] == "train"]
    val = [r for r in records if r["split"] == "validation"]
    data = sequence_data(train + val)
    allowed = protocol["allowed_types"]
    base_data = {
        r["id"]: load_npz(MODEL_DIR / "features" / f"{r['sha256']}-{PREPROCESS_VERSION}.npz")
        for r in train + val
    }
    baseline = train_heads(train, base_data, "logistic")
    threshold = choose_threshold(baseline, val, base_data, baseline=True)
    save_model(
        run / "baseline_refit.npz",
        baseline,
        PREPROCESS_VERSION,
        threshold,
        {"protocol_sha256": sha256(run / "protocol.json"), "allowed_types": allowed},
    )
    write_json(
        run / "baseline_validation.json", score(baseline, val, base_data, threshold, allowed)
    )
    rows = []
    for config in protocol["head_grid"]:
        began = time.monotonic()
        model, mean, scale, threshold, result, history = train_temporal(
            train, val, data, allowed, **config
        )
        name = f"temporal_{config['family']}"
        meta = {
            **config,
            "allowed_types": allowed,
            "weights_sha256": sha256(WEIGHTS),
            "protocol_sha256": sha256(run / "protocol.json"),
            "kind": "temporal",
            "name": name,
        }
        save_head(run / (name + ".pt"), model, mean, scale, threshold, meta)
        row = {
            "name": name,
            "kind": "temporal",
            "config": config,
            "threshold": threshold,
            "validation": result,
            "rank": validation_rank(result),
            "history": history,
            "model_sha256": sha256(run / (name + ".pt")),
            "training_seconds": time.monotonic() - began,
        }
        rows.append(row)
        write_json(run / "head_grid.json", rows)
        print(
            json.dumps(
                {
                    "head": config["family"],
                    "epochs": len(history),
                    "rank": row["rank"],
                    "seconds": round(row["training_seconds"], 2),
                }
            ),
            flush=True,
        )


def cross_validate(run, records, protocol):
    train = [r for r in records if r["split"] == "train"]
    data = sequence_data(train)
    rows = json.loads((run / "head_grid.json").read_text())
    best = max([r for r in rows if r["config"]["family"] != "linear"], key=lambda r: r["rank"])
    reports = []
    for family in ["linear", best["config"]["family"]]:
        for fold, groups in enumerate(protocol["cv"]["outer_test_groups"]):
            outer = [r for r in train if r["scene_group"] in groups]
            remaining = [r for r in train if r["scene_group"] not in groups]
            inner_group = sorted({r["scene_group"] for r in remaining})[-1]
            it = [r for r in remaining if r["scene_group"] != inner_group]
            iv = [r for r in remaining if r["scene_group"] == inner_group]
            model, mean, scale, threshold, _, history = train_temporal(
                it, iv, data, protocol["allowed_types"], family, seed=73
            )
            pred = infer(model, {r["id"]: data[r["id"]] for r in outer}, mean, scale)
            result = evaluate_logits(pred, outer, data, threshold, protocol["allowed_types"])
            absent = sorted(
                set(protocol["allowed_types"]) - {a["type"] for r in it for a in r["annotations"]}
            )
            reports.append(
                {
                    "family": family,
                    "fold": fold,
                    "train_groups": sorted({r["scene_group"] for r in it}),
                    "validation_group": inner_group,
                    "held_out_groups": groups,
                    "unseen_types_in_training": absent,
                    "epochs": len(history),
                    "result": result,
                }
            )
            write_json(run / "group_cross_validation.json", reports)
            print(
                json.dumps(
                    {
                        "cv_family": family,
                        "fold": fold,
                        "balanced_accuracy": result["detection"]["balanced_accuracy"],
                    }
                ),
                flush=True,
            )


def freeze_selection(run):
    if (run / "selection_frozen.json").exists():
        raise ValueError("Selection already frozen")
    rows = json.loads((run / "head_grid.json").read_text())
    if (run / "finetuning_grid.json").exists():
        rows += json.loads((run / "finetuning_grid.json").read_text())
    best = max(rows, key=lambda r: r["rank"])
    write_json(
        run / "selection_frozen.json",
        {
            "selected": best["name"],
            "kind": best["kind"],
            "model_sha256": best["model_sha256"],
            "utc": datetime.now(UTC).isoformat(),
            "protocol_sha256": sha256(run / "protocol.json"),
            "rank": best["rank"],
        },
    )
    print(json.dumps({"selected": best["name"], "rank": best["rank"]}), flush=True)


def test_selected(run, records, protocol):
    from cash_ml.temporal_finetune import predict_saved

    if (run / "comparison.json").exists():
        raise ValueError("Test comparison already recorded; it must not be silently overwritten")

    selection = json.loads((run / "selection_frozen.json").read_text())
    name = selection["selected"]
    if selection["model_sha256"] != sha256(run / (name + ".pt")) or selection[
        "protocol_sha256"
    ] != sha256(run / "protocol.json"):
        raise ValueError("Frozen selection modified")
    tests = [r for r in records if r["split"] == "test"]
    data = sequence_data(tests)
    if selection["kind"] == "temporal":
        model, mean, scale, threshold, _ = load_head(run / (name + ".pt"))
        pred = infer(model, data, mean, scale)
    else:
        pred, data, threshold = predict_saved(run, run / (name + ".pt"), tests)
    candidate = evaluate_logits(pred, tests, data, threshold, protocol["allowed_types"])
    model, _, threshold, _ = load_model(run / "baseline_refit.npz")
    bd = {
        r["id"]: load_npz(MODEL_DIR / "features" / f"{r['sha256']}-{PREPROCESS_VERSION}.npz")
        for r in tests
    }
    baseline = score(model, tests, bd, threshold, protocol["allowed_types"])
    rows = json.loads((run / "head_grid.json").read_text()) + json.loads(
        (run / "finetuning_grid.json").read_text()
    )
    selected = next(r for r in rows if r["name"] == name)
    bv = json.loads((run / "baseline_validation.json").read_text())
    gate = deployment_gate(
        baseline,
        candidate,
        selected["validation"]["detection"]["balanced_accuracy"]
        > bv["detection"]["balanced_accuracy"],
    )
    gate["checks"].update(
        independent_final_test=False, end_to_end_latency_verified=False, serving_verified=False
    )
    gate["passed"] = all(gate["checks"].values())
    supports = {
        t: {
            s: {
                "annotations": sum(
                    a["type"] == t for r in records if r["split"] == s for a in r["annotations"]
                ),
                "scene_groups": len(
                    {
                        r["scene_group"]
                        for r in records
                        if r["split"] == s and any(a["type"] == t for a in r["annotations"])
                    }
                ),
            }
            for s in ["train", "validation", "test"]
        }
        for t in protocol["allowed_types"]
    }
    report = {
        "format_version": 4,
        "experiment": "temporal_heads_and_last_block",
        "selected": name,
        "allowed_types": protocol["allowed_types"],
        "test_status": protocol["test_status"],
        "supports": supports,
        "baseline": {"validation": bv, "test": baseline},
        "candidate": {"validation": selected["validation"], "test": candidate},
        "deployment_gate": gate,
        "active_model": "typed_v2_original_weights_unchanged",
    }
    write_json(run / "comparison.json", report)
    print(
        json.dumps({"selected": name, "test": candidate["detection"], "gate": gate["passed"]}),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--phase", choices=["protocol", "heads", "cv", "select", "test"], required=True)
    a = p.parse_args()
    if a.phase == "protocol":
        freeze_protocol(a.run)
        return
    protocol = json.loads((a.run / "protocol.json").read_text())
    records = json.loads((a.run / "previous_grouped_records.json").read_text())
    if a.phase == "heads":
        select_heads(a.run, records, protocol)
    elif a.phase == "cv":
        cross_validate(a.run, records, protocol)
    elif a.phase == "select":
        freeze_selection(a.run)
    else:
        test_selected(a.run, records, protocol)


if __name__ == "__main__":
    main()
