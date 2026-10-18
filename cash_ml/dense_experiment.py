"""Frozen protocol, dense feature extraction, validation selection, reused test.

python -m cash_ml.dense_experiment --run <run> --phase extract
python -m cash_ml.dense_experiment --run <run> --phase select
python -m cash_ml.dense_experiment --run <run> --phase test
"""

import argparse
import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from cash_ml.dense_features import DenseExtractor, DenseSpec, aggregate, raw_cache_key, raw_identity
from cash_ml.experiment import (
    choose_threshold,
    deployment_gate,
    evaluate,
    fit_head,
    load_model,
    load_npz,
    save_model,
    train_heads,
    write_json,
)
from cash_ml.features import MODEL_DIR, PREPROCESS_VERSION, WEIGHTS, sha256


def grid():
    specs = [DenseSpec(fps, "full", 0, "center") for fps in (8, 15)]
    specs += [DenseSpec(8, view, context, pool) for view in ("full", "details")
              for context, pool in ((8, "mean"), (8, "ordered"), (16, "ordered"))]
    specs += [DenseSpec(8, "details", 0, "center")]
    return [{"name": f"dense_{s.fps}fps_{s.views}_{s.context}s_{s.pooling}_{family}",
             "spec": json.loads(s.to_json()), "family": family, "hard_negatives": True}
            for s in specs for family in ("logistic", "svm")] + [
                {"name": "dense_8fps_full_8s_ordered_logistic_no_mining",
                 "spec": json.loads(DenseSpec().to_json()), "family": "logistic", "hard_negatives": False}]


def cache_path(run, record, fps, views):
    return run/"dense_features"/(raw_cache_key(record["sha256"], fps, views)+".npz")


def extract(run, records, combinations):
    (run/"dense_features").mkdir(exist_ok=True)
    extractor = DenseExtractor()
    for fps, views in combinations:
        for index, r in enumerate(records):
            source = Path(r["path"])
            if sha256(source) != r["sha256"]:
                raise ValueError("Source changed since audit")
            path = cache_path(run, r, fps, views)
            if not path.exists():
                item = extractor.extract_raw(source, fps, views)
                item["identity"] = np.asarray(raw_identity(r["sha256"], fps, views))
                temporary = path.with_suffix(".partial.npz")
                np.savez_compressed(temporary, **item)
                temporary.replace(path)
            item = load_npz(path)
            if str(item["identity"]) != raw_identity(r["sha256"], fps, views):
                raise ValueError("Feature cache identity mismatch")
            print(f"{fps}fps/{views} {index+1}/{len(records)} {r['id']} "
                  f"{len(item['centers'])} clips {float(item['elapsed_seconds']):.1f}s", flush=True)


def data_for(run, records, spec):
    raw_views = "details" if spec.fps == 8 else "full"
    data = {}
    for r in records:
        raw = load_npz(cache_path(run, r, spec.fps, raw_views))
        if str(raw["identity"]) != raw_identity(r["sha256"], spec.fps, raw_views):
            raise ValueError("Feature cache identity mismatch")
        data[r["id"]] = aggregate(raw, spec)
    return data


def train_dense(records, data, family, hard_points):
    model = train_heads(records, data, family)
    if not hard_points:
        return model
    if any(name not in {r["id"] for r in records if not r["label"]} for name in hard_points):
        raise ValueError("Hard negatives must belong to negative TRAIN videos")
    x, y, weights, counts = [], [], [], Counter(r["label"] for r in records)
    selected_counts = {}
    for r in records:
        item = data[r["id"]]
        indices = (sorted({int(np.argmin(abs(item["centers"]-a["timecode"]))) for a in r["annotations"]})
                   if r["label"] else list(range(len(item["features"]))))
        hard = {int(np.argmin(abs(item["centers"]-t))) for t in hard_points.get(r["id"], [])}
        w = np.asarray([2. if i in hard else 1. for i in indices])
        w /= w.sum()*counts[r["label"]]
        x.extend(item["features"][indices])
        y.extend([r["label"]]*len(indices))
        weights.extend(w)
        if hard:
            selected_counts[r["id"]] = len(hard)
    weights = np.asarray(weights)*len(weights)/np.sum(weights)
    model["binary"] = fit_head(x, y, weights, family)
    model["hard_negative_windows"] = selected_counts
    return model


def score(model, records, data, threshold, allowed):
    result = evaluate(model, records, data, threshold, allowed)
    # Honest exposure information, no extrapolation of short clips to hourly rate.
    result["negative_exposure_seconds"] = sum(float(data[r["id"]]["duration"]) for r in records if not r["label"])
    result.pop("false_alerts_per_negative_minute", None)
    result["false_alerts_per_hour"] = None
    result["hourly_rate_reason"] = "Only short staged clips; no long independent normal recording"
    for row in result["videos"]:
        item = data[row["video"]]
        for event in row["events"]:
            i = int(np.argmin(abs(item["centers"]-event["timecode_sec"])))
            if "context_starts" in item:
                event["decision_start"] = event.pop("context_start")
                event["decision_end"] = event.pop("context_end")
                event["context_start"] = float(item["context_starts"][i])
                event["context_end"] = float(item["context_ends"][i])
    return result


def select(run, records, protocol):
    if (run/"selection_frozen.json").exists():
        raise ValueError("Selection already frozen; use a new run for new experiments")
    train = [r for r in records if r["split"] == "train"]
    val = [r for r in records if r["split"] == "validation"]
    allowed = protocol["allowed_types"]
    if sorted({a["type"] for r in train for a in r["annotations"]}) != allowed:
        raise ValueError("Training data does not cover the frozen vocabulary")
    baseline_data = {r["id"]: load_npz(MODEL_DIR/"features"/f"{r['sha256']}-{PREPROCESS_VERSION}.npz")
                     for r in train+val}
    base = train_heads(train, baseline_data, "logistic")
    threshold = choose_threshold(base, val, baseline_data, baseline=True)
    baseline_val = score(base, val, baseline_data, threshold, allowed)
    metadata = {"allowed_types": allowed, "weights_sha256": sha256(WEIGHTS),
                "protocol_sha256": sha256(run/"protocol.json"),
                "preprocessing_sha256": protocol["preprocessing_sha256"]}
    save_model(run/"baseline_refit.npz", base, PREPROCESS_VERSION, threshold, metadata)
    rows = []
    for config in protocol["grid"]:
        began = time.monotonic()
        spec = DenseSpec(**config["spec"])
        data = data_for(run, train+val, spec)
        model = train_dense(train, data, config["family"], protocol["hard_negative_points"] if config["hard_negatives"] else {})
        threshold = choose_threshold(model, val, data)
        result = score(model, val, data, threshold, allowed)
        cost = data[val[0]["id"]]["features"].shape[1]
        rank = [result["detection"]["balanced_accuracy"], result["timing"]["typed_point_recall"],
                result["type_accuracy_at_points"], -result["false_alerts_on_negative_videos"], -cost]
        path = run/(config["name"]+".npz")
        save_model(path, model, spec.to_json(), threshold, {**metadata, "name": config["name"],
                                                          "hard_negative_windows": model["hard_negative_windows"]})
        rows.append({**config, "threshold": threshold, "validation": result, "rank": rank,
                     "model_sha256": sha256(path), "training_seconds": time.monotonic()-began,
                     "hard_negative_windows": model["hard_negative_windows"]})
        write_json(run/"validation_grid.json", rows)
        print(config["name"], rank, flush=True)
    best = max(rows, key=lambda row: row["rank"])
    write_json(run/"baseline_validation.json", baseline_val)
    write_json(run/"selection_frozen.json", {"selected": best["name"], "utc": datetime.now(UTC).isoformat(),
                                            "protocol_sha256": sha256(run/"protocol.json"),
                                            "model_sha256": best["model_sha256"], "rank": best["rank"]})


def test(run, records, protocol):
    selection = json.loads((run/"selection_frozen.json").read_text())
    if selection["protocol_sha256"] != sha256(run/"protocol.json"):
        raise ValueError("Frozen protocol changed")
    model_path = run/(selection["selected"]+".npz")
    if selection["model_sha256"] != sha256(model_path):
        raise ValueError("Selected model changed")
    model, spec_json, threshold, metadata = load_model(model_path)
    spec = DenseSpec(**json.loads(spec_json))
    test_records = [r for r in records if r["split"] == "test"]
    extract(run, test_records, [(spec.fps, "details" if spec.fps == 8 else "full")])
    data = data_for(run, test_records, spec)
    candidate_test = score(model, test_records, data, threshold, protocol["allowed_types"])
    baseline, _, bt, _ = load_model(run/"baseline_refit.npz")
    baseline_data = {r["id"]: load_npz(MODEL_DIR/"features"/f"{r['sha256']}-{PREPROCESS_VERSION}.npz") for r in test_records}
    baseline_test = score(baseline, test_records, baseline_data, bt, protocol["allowed_types"])
    rows = json.loads((run/"validation_grid.json").read_text())
    chosen = next(row for row in rows if row["name"] == selection["selected"])
    baseline_val = json.loads((run/"baseline_validation.json").read_text())
    gate = deployment_gate(baseline_test, candidate_test,
                           chosen["validation"]["detection"]["balanced_accuracy"] > baseline_val["detection"]["balanced_accuracy"])
    gate["checks"]["independent_final_test"] = bool(protocol.get("independent_final_test", False))
    gate["passed"] = all(gate["checks"].values())
    supports = {t: {s: {"annotations": sum(a["type"] == t for r in records if r["split"] == s for a in r["annotations"]),
                       "scene_groups": len({r["scene_group"] for r in records if r["split"] == s
                                             and any(a["type"] == t for a in r["annotations"])})}
                   for s in ("train", "validation", "test")} for t in protocol["allowed_types"]}
    report = {"format_version": 3, "experiment": "dense_short_clips", "allowed_types": protocol["allowed_types"],
              "selected": chosen["name"], "test_status": protocol["test_status"], "supports": supports,
              "protocol_sha256": sha256(run/"protocol.json"),
              "baseline": {"validation": baseline_val, "test": baseline_test},
              "candidate": {"validation": chosen["validation"], "test": candidate_test},
              "deployment_gate": gate, "active_model": "typed_v2_original_unchanged",
              "fine_tuning": protocol["fine_tuning"], "limitations": protocol["label_limitations"]}
    write_json(run/"comparison.json", report)
    print(json.dumps({"selected": chosen["name"], "test": candidate_test["detection"], "gate": gate}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--phase", choices=("extract", "select", "test"), required=True)
    args = parser.parse_args()
    records = json.loads((args.run/"grouped_records.json").read_text())
    protocol = json.loads((args.run/"protocol.json").read_text())
    if sha256(Path(__file__).with_name("dense_features.py")) != protocol["preprocessing_sha256"]:
        raise ValueError("Preprocessing code differs from frozen protocol")
    for group in {r["scene_group"] for r in records}:
        if len({r["split"] for r in records if r["scene_group"] == group}) != 1:
            raise ValueError("Scene leakage across splits")
    if args.phase == "extract":
        extract(args.run, [r for r in records if r["split"] != "test"], [(8, "details"), (15, "full")])
    elif args.phase == "select":
        select(args.run, records, protocol)
    else:
        test(args.run, records, protocol)


if __name__ == "__main__":
    main()
