"""Predeclared paired ablation; private descriptions are processed only locally."""

import argparse
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from cash_ml.dense_experiment import score
from cash_ml.description_guidance import ensemble, fit, save, subset_targets
from cash_ml.experiment import deployment_gate, load_model, load_npz, write_json
from cash_ml.features import MODEL_DIR, PREPROCESS_VERSION, WEIGHTS, sha256
from cash_ml.review_text_targets import require_offline
from cash_ml.temporal_experiment import sequence_data
from cash_ml.temporal_model import evaluate_logits, threshold_for

PREVIOUS = MODEL_DIR.parent / "experiments/20261017T235906Z_temporal"
LABELS = Path("/Users/almat/Downloads/labels_final (2).xlsx")


def prepare(run):
    if run.exists():
        raise ValueError("Create a new run; never overwrite an experiment")
    run.mkdir(parents=True)
    snapshot = run / "baseline_snapshot"
    snapshot.mkdir()
    for name in ("typed_classifier.npz", "typed_report.json"):
        shutil.copy2(MODEL_DIR / name, snapshot / name)
    shutil.copy2(LABELS, snapshot / "labels_final.xlsx")
    shutil.copy2(PREVIOUS / "grouped_records.json", run / "grouped_records.json")
    shutil.copy2(PREVIOUS / "baseline_refit.npz", run / "baseline_refit.npz")
    shutil.copy2(PREVIOUS / "comparison.json", run / "previous_comparison.json")
    records = json.loads((run / "grouped_records.json").read_text())
    old = json.loads((PREVIOUS / "protocol.json").read_text())
    if sha256(LABELS) != old["labels_sha256"]:
        raise ValueError("Source labels changed; explicit re-audit required")
    for r in records:
        if sha256(Path(r["path"])) != r["sha256"]:
            raise ValueError("Original video changed")
    for g in {r["scene_group"] for r in records}:
        if len({r["split"] for r in records if r["scene_group"] == g}) != 1:
            raise ValueError("Split leaks scenes")
    source = snapshot / "cash_ml"
    source.mkdir()
    for p in Path(__file__).parent.glob("*.py"):
        shutil.copy2(p, source / p.name)
    protocol = {
        "version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "experiment": "train_only_local_descriptions_auxiliary_distillation",
        "allowed_types": old["allowed_types"],
        "seeds": [42, 43, 44],
        "variants": {"control": 0.0, "descriptions": 0.1},
        "architecture": "frozen R3D-18; identical TCN hidden16 and three-seed ensemble",
        "preparation": "16 frames at8fps, full frame letterbox112,stride1s",
        "optimizer": {"lr": 0.001, "weight_decay": 0.05, "max_epochs": 60, "patience": 8},
        "target": "local multilingual E5-small; mean per-action embeddings within each review window; centered using only current fold training videos",
        "text_encoder_revision": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
        "auxiliary_loss": "0.1 times cosine loss, equal video weighting; descriptions are unverified and do not create normal/type/event labels",
        "inference": "video only; no text encoder, descriptions, filenames or Excel as input",
        "selection": "one fixed description candidate versus paired control; epochs by validation loss; ensemble threshold by validation balanced accuracy and fewer false positives",
        "cross_validation": old["cv"],
        "test_status": "reused exploratory test: 5 videos, one scene group; NOT independent",
        "promotion": "validation gain over control and refit, CV mean gain, existing gate AND independent final evaluation AND service/latency checks",
        "false_alerts_per_hour": "not estimated: insufficient long verified normal recordings",
        "labels_sha256": sha256(LABELS),
        "weights_sha256": sha256(WEIGHTS),
        "records_sha256": sha256(run / "grouped_records.json"),
        "source_sha256": {p.name: sha256(p) for p in source.glob("*.py")},
        "active_classifier_sha256": sha256(MODEL_DIR / "typed_classifier.npz"),
        "active_report_sha256": sha256(MODEL_DIR / "typed_report.json"),
    }
    write_json(run / "protocol.json", protocol)
    print(json.dumps({"prepared": str(run), "train": 29, "validation": 6, "test": 5}), flush=True)


def verify_inputs(run, protocol):
    if sha256(run / "grouped_records.json") != protocol["records_sha256"]:
        raise ValueError("Frozen records changed")
    for name in ("description_guidance.py", "description_experiment.py", "review_text_targets.py"):
        if sha256(Path(__file__).parent / name) != protocol["source_sha256"][name]:
            raise ValueError(
                "Protocol implementation changed; register an amendment before continuing"
            )
    audit = json.loads((run / "description_targets_audit.json").read_text())
    if sha256(run / "description_targets.npz") != audit["targets_sha256"]:
        raise ValueError("Description targets changed")
    if audit["revision"] != protocol["text_encoder_revision"]:
        raise ValueError("Wrong encoder")


def train_pair(run, protocol, train, val, data, tag):
    targets = subset_targets(run / "description_targets.npz", train)
    results = {}
    for variant, weight in protocol["variants"].items():
        paths, histories = [], []
        start = time.monotonic()
        for seed in protocol["seeds"]:
            path = run / f"{tag}_{variant}_{seed}.pt"
            if path.exists():
                raise ValueError("Training checkpoint already exists")
            fitted = fit(
                train,
                val,
                data,
                targets,
                protocol["allowed_types"],
                weight,
                seed,
                max_epochs=protocol["optimizer"]["max_epochs"],
                patience=protocol["optimizer"]["patience"],
            )
            history = save(
                path,
                fitted,
                {
                    "allowed_types": protocol["allowed_types"],
                    "seed": seed,
                    "auxiliary_weight": weight,
                    "protocol_sha256": sha256(run / "protocol.json"),
                    "training_ids": [r["id"] for r in train],
                },
            )
            paths.append(path)
            histories.append({"seed": seed, "epochs": len(history), "history": history})
            print(
                json.dumps(
                    {"phase": tag, "variant": variant, "seed": seed, "epochs": len(history)}
                ),
                flush=True,
            )
        vd = {r["id"]: data[r["id"]] for r in val}
        pred = ensemble(paths, vd)
        threshold = threshold_for(pred, val)
        result = evaluate_logits(pred, val, vd, threshold, protocol["allowed_types"])
        results[variant] = {
            "checkpoints": [p.name for p in paths],
            "sha256": {p.name: sha256(p) for p in paths},
            "threshold": threshold,
            "validation": result,
            "history": histories,
            "seconds": time.monotonic() - start,
        }
        write_json(run / (tag + "_training.json"), results)
    return results


def train_main(run, records, protocol):
    if (run / "main_training.json").exists():
        raise ValueError("Main training already recorded")
    train = [r for r in records if r["split"] == "train"]
    val = [r for r in records if r["split"] == "validation"]
    train_pair(run, protocol, train, val, sequence_data(train + val), "main")


def cross_validate(run, records, protocol):
    if (run / "cross_validation.json").exists():
        raise ValueError("Cross validation already recorded")
    train = [r for r in records if r["split"] == "train"]
    data = sequence_data(train)
    rows = []
    for fold, groups in enumerate(protocol["cross_validation"]["outer_test_groups"]):
        outer = [r for r in train if r["scene_group"] in groups]
        remaining = [r for r in train if r["scene_group"] not in groups]
        vg = sorted({r["scene_group"] for r in remaining})[-1]
        it = [r for r in remaining if r["scene_group"] != vg]
        iv = [r for r in remaining if r["scene_group"] == vg]
        result = train_pair(run, protocol, it, iv, data, f"cv{fold}")
        od = {r["id"]: data[r["id"]] for r in outer}
        for variant, trained in result.items():
            pred = ensemble([run / p for p in trained["checkpoints"]], od)
            scored = evaluate_logits(
                pred, outer, od, trained["threshold"], protocol["allowed_types"]
            )
            rows.append(
                {
                    "fold": fold,
                    "variant": variant,
                    "training_ids": [r["id"] for r in it],
                    "inner_validation_group": vg,
                    "outer_groups": groups,
                    "result": scored,
                }
            )
            print(
                json.dumps(
                    {
                        "cv": fold,
                        "variant": variant,
                        "balanced_accuracy": scored["detection"]["balanced_accuracy"],
                    }
                ),
                flush=True,
            )
    write_json(run / "cross_validation.json", rows)


def freeze(run, protocol):
    if (run / "selection_frozen.json").exists():
        raise ValueError("Comparison already frozen")
    main = json.loads((run / "main_training.json").read_text())
    cv = json.loads((run / "cross_validation.json").read_text())
    write_json(
        run / "selection_frozen.json",
        {
            "utc": datetime.now(UTC).isoformat(),
            "selected": "descriptions",
            "protocol_sha256": sha256(run / "protocol.json"),
            "training_sha256": sha256(run / "main_training.json"),
            "cv_sha256": sha256(run / "cross_validation.json"),
            "targets_sha256": sha256(run / "description_targets.npz"),
            "cv_balanced_accuracy": {
                v: float(
                    np.mean(
                        [
                            r["result"]["detection"]["balanced_accuracy"]
                            for r in cv
                            if r["variant"] == v
                        ]
                    )
                )
                for v in main
            },
            "reason": "fixed candidate; test not used for selection, threshold, epochs or description targets",
        },
    )


def test(run, records, protocol):
    if (run / "comparison.json").exists():
        raise ValueError("Test was already evaluated")
    selection = json.loads((run / "selection_frozen.json").read_text())
    for key, file in (
        ("protocol_sha256", "protocol.json"),
        ("training_sha256", "main_training.json"),
        ("cv_sha256", "cross_validation.json"),
        ("targets_sha256", "description_targets.npz"),
    ):
        if selection[key] != sha256(run / file):
            raise ValueError("Frozen selection changed")
    main = json.loads((run / "main_training.json").read_text())
    tests = [r for r in records if r["split"] == "test"]
    val = [r for r in records if r["split"] == "validation"]
    data = sequence_data(tests)
    result = {}
    for variant, trained in main.items():
        for name, digest in trained["sha256"].items():
            if sha256(run / name) != digest:
                raise ValueError("Checkpoint changed")
        pred = ensemble([run / p for p in trained["checkpoints"]], data)
        result[variant] = {
            "validation": trained["validation"],
            "test": evaluate_logits(
                pred, tests, data, trained["threshold"], protocol["allowed_types"]
            ),
        }
    model, _, threshold, _ = load_model(run / "baseline_refit.npz")
    bd = {
        r["id"]: load_npz(MODEL_DIR / "features" / f"{r['sha256']}-{PREPROCESS_VERSION}.npz")
        for r in val + tests
    }
    baseline = {
        part: score(model, rr, bd, threshold, protocol["allowed_types"])
        for part, rr in (("validation", val), ("test", tests))
    }

    def ba(x):
        return x["validation"]["detection"]["balanced_accuracy"]

    gate = deployment_gate(
        baseline["test"], result["descriptions"]["test"], ba(result["descriptions"]) > ba(baseline)
    )
    gate["checks"].update(
        {
            "validation_better_than_paired_control": ba(result["descriptions"])
            > ba(result["control"]),
            "cv_mean_better_than_paired_control": selection["cv_balanced_accuracy"]["descriptions"]
            > selection["cv_balanced_accuracy"]["control"],
            "independent_final_test": False,
            "end_to_end_latency_verified": False,
            "serving_verified": False,
        }
    )
    gate["passed"] = all(gate["checks"].values())
    old = json.loads((run / "previous_comparison.json").read_text())
    report = {
        "format_version": 5,
        "experiment": protocol["experiment"],
        "selected": "description_guidance_tcn_3seed",
        "allowed_types": protocol["allowed_types"],
        "test_status": protocol["test_status"],
        "supports": old["supports"],
        "baseline": baseline,
        "control": result["control"],
        "candidate": result["descriptions"],
        "cv_balanced_accuracy": selection["cv_balanced_accuracy"],
        "description_usage": json.loads((run / "description_targets_audit.json").read_text()),
        "deployment_gate": gate,
        "active_model": "typed_v2_original_weights_unchanged",
    }
    write_json(run / "comparison.json", report)
    for name in ("baseline", "control", "candidate"):
        print(
            json.dumps(
                {
                    "model": name,
                    "test": report[name]["test"]["detection"],
                    "point_type_accuracy": report[name]["test"]["type_accuracy_at_points"],
                }
            ),
            flush=True,
        )
    print(
        json.dumps({"promotion": gate["passed"], "cv": selection["cv_balanced_accuracy"]}),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--phase", choices=["prepare", "train", "cv", "freeze", "test"], required=True)
    a = p.parse_args()
    require_offline()
    if a.phase == "prepare":
        prepare(a.run)
        return
    protocol = json.loads((a.run / "protocol.json").read_text())
    verify_inputs(a.run, protocol)
    records = json.loads((a.run / "grouped_records.json").read_text())
    if a.phase == "train":
        train_main(a.run, records, protocol)
    elif a.phase == "cv":
        cross_validate(a.run, records, protocol)
    elif a.phase == "freeze":
        freeze(a.run, protocol)
    else:
        test(a.run, records, protocol)


if __name__ == "__main__":
    main()
