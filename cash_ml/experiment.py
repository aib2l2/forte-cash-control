"""Reproducible, local, grouped comparison. Never replaces the serving model.

python -m cash_ml.experiment --run models/experiments/<run>
The reviewed protocol and frozen feature caches must exist first.
"""

import argparse
import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from cash_ml.context_features import ContextExtractor, FeatureSpec, cache_key
from cash_ml.features import MODEL_DIR, PREPROCESS_VERSION, WEIGHTS, sha256
from cash_ml.service import event_peaks
from cash_ml.train import annotation_examples, metrics


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def load_npz(path):
    with np.load(path, allow_pickle=False) as saved:
        return {k: saved[k] for k in saved.files}


def sigmoid(logits):
    return 1 / (1 + np.exp(-np.clip(logits, -700, 700)))


def fit_head(x, y, weights, family):
    x, y = np.asarray(x), np.asarray(y)
    scaler = StandardScaler().fit(x, sample_weight=weights)
    z = scaler.transform(x)
    if family == "logistic":
        model = LogisticRegression(C=.1, max_iter=3000, random_state=42).fit(z, y, sample_weight=weights)
        coef, intercept, classes = model.coef_, model.intercept_, model.classes_
    elif family == "svm":
        model = LinearSVC(C=.01, max_iter=10000, random_state=42, dual="auto").fit(z, y, sample_weight=weights)
        coef, intercept, classes = model.coef_, model.intercept_, model.classes_
    elif family == "centroid":
        classes = np.unique(y)
        centers = np.stack([np.average(z[y == c], axis=0, weights=weights[y == c]) for c in classes])
        coef, intercept = 2*centers/z.shape[1], -np.sum(centers**2, axis=1)/z.shape[1]
        if len(classes) == 2:
            coef, intercept = coef[1:] - coef[:1], intercept[1:] - intercept[:1]
    else:
        raise ValueError("Unknown head family")
    return {"mean": scaler.mean_, "scale": scaler.scale_, "coef": coef,
            "intercept": intercept, "classes": classes}


def head_logits(head, x):
    return ((x-head["mean"])/head["scale"]) @ head["coef"].T + head["intercept"]


def predict(model, item):
    scores = sigmoid(head_logits(model["binary"], item["features"]).reshape(-1))
    types = model["typed"]["classes"][head_logits(model["typed"], item["features"]).argmax(axis=1)]
    return scores, types


def train_heads(records, data, family, reviewed_negatives=()):
    x, y, weights, owners = [], [], [], []
    counts = Counter(r["label"] for r in records)
    for r in records:
        item = data[r["id"]]
        indices = (sorted({int(np.argmin(abs(item["centers"]-a["timecode"]))) for a in r["annotations"]})
                   if r["label"] else range(len(item["features"])))
        for index in indices:
            x.append(item["features"][index])
            y.append(r["label"])
            weights.append(1/(len(indices)*counts[r["label"]]))
            owners.append(r["id"])
    x, y, weights, owners = np.asarray(x), np.asarray(y), np.asarray(weights), np.asarray(owners)
    weights *= len(weights)/weights.sum()
    first = fit_head(x, y, weights, family)
    # Hard-negative mining only among reviewed, Excel-negative TRAIN videos.
    hard_counts = {}
    if reviewed_negatives:
        scores = head_logits(first, x).reshape(-1)
        for name in reviewed_negatives:
            indices = np.flatnonzero((owners == name) & (y == 0))
            if not len(indices):
                continue
            chosen = indices[np.argsort(scores[indices])[-max(1, int(np.ceil(len(indices)*.2))):]]
            total = weights[indices].sum()
            weights[chosen] *= 2
            weights[indices] *= total/weights[indices].sum()
            hard_counts[name] = len(chosen)
        binary = fit_head(x, y, weights, family)
    else:
        binary = first
    tx, ty, _ = annotation_examples(records, data)
    tc = Counter(ty)
    tw = np.asarray([len(ty)/(len(tc)*tc[t]) for t in ty])
    # Original v2 type scaler was unweighted; retain that for the baseline.
    if not reviewed_negatives and family == "logistic":
        scaler = StandardScaler().fit(tx)
        head = LogisticRegression(C=.1, class_weight="balanced", max_iter=3000, random_state=42)
        head.fit(scaler.transform(tx), ty)
        typed = {"mean": scaler.mean_, "scale": scaler.scale_, "coef": head.coef_,
                 "intercept": head.intercept_, "classes": head.classes_}
    else:
        typed = fit_head(tx, ty, tw, family)
    return {"binary": binary, "typed": typed, "hard_negative_windows": hard_counts}


def choose_threshold(model, records, data, baseline=False):
    scores = [float(predict(model, data[r["id"]])[0].max()) for r in records]
    labels = [r["label"] for r in records]
    candidates = np.r_[0., np.unique(scores), np.nextafter(max(scores), np.inf)]
    def rank(t):
        m = metrics(labels, scores, t)
        if baseline:
            return m["balanced_accuracy"], -abs(t-.5)
        return m["balanced_accuracy"], -m["confusion_matrix_tn_fp_fn_tp"][0][1], float(t)
    return float(max(candidates, key=rank))


def match_points(annotations, events, tolerance=4., typed=False):
    """One-to-one matching to point labels, not a fabricated interval metric."""
    if not annotations or not events:
        return []
    costs = np.full((len(annotations), len(events)), 1e9)
    for ai, a in enumerate(annotations):
        for ei, e in enumerate(events):
            error = abs(a["timecode"]-e["timecode_sec"])
            if error <= tolerance and (not typed or a["type"] == e["type"]):
                costs[ai, ei] = error
    rows, cols = linear_sum_assignment(costs)
    return [{"annotation": int(ai), "event": int(ei), "absolute_error_seconds": float(costs[ai, ei])}
            for ai, ei in zip(rows, cols, strict=True) if costs[ai, ei] <= tolerance]


def evaluate(model, records, data, threshold, allowed_types):
    labels, video_scores, video_rows, point_rows, errors = [], [], [], [], []
    point_hits = typed_hits = negative_alerts = unknown_alerts = 0
    negative_duration = 0.
    for r in records:
        item = data[r["id"]]
        scores, types = predict(model, item)
        indices = event_peaks(item["starts"], item["ends"], scores, threshold, types)
        events = [{"timecode_sec": float(item["centers"][i]), "type": str(types[i]),
                   "score": float(scores[i]), "context_start": float(item["starts"][i]),
                   "context_end": float(item["ends"][i])} for i in indices]
        matches = match_points(r["annotations"], events)
        typed_matches = match_points(r["annotations"], events, typed=True)
        point_hits += len(matches)
        typed_hits += len(typed_matches)
        errors.extend(m["absolute_error_seconds"] for m in matches)
        if not r["label"]:
            negative_alerts += len(events)
            negative_duration += float(item["duration"])
        else:
            unknown_alerts += len(events)-len(matches)
        for a in r["annotations"]:
            i = int(np.argmin(abs(item["centers"]-a["timecode"])))
            point_rows.append({"video": r["id"], "expected": a["type"], "point": a["timecode"],
                               "predicted": str(types[i]), "detected_at_nearest_window": bool(scores[i] >= threshold),
                               "nearest_window_center": float(item["centers"][i])})
        labels.append(r["label"])
        video_scores.append(float(scores.max()))
        video_rows.append({"video": r["id"], "scene_group": r["scene_group"], "label": r["label"],
                           "predicted": int(bool(events)), "score": float(scores.max()),
                           "events": events, "point_matches": matches, "typed_point_matches": typed_matches})
    total = len(point_rows)
    return {"detection": metrics(labels, video_scores, threshold),
            "type_accuracy_at_points": sum(p["expected"] == p["predicted"] for p in point_rows)/total if total else None,
            "per_type": {t: {"support": sum(p["expected"] == t for p in point_rows),
                              "correct": sum(p["expected"] == t == p["predicted"] for p in point_rows)} for t in allowed_types},
            "timing": {"point_count": total, "tolerance_seconds": 4., "detected_points": point_hits,
                       "typed_detected_points": typed_hits, "point_recall": point_hits/total if total else None,
                       "typed_point_recall": typed_hits/total if total else None,
                       "matched_mean_absolute_error_seconds": float(np.mean(errors)) if errors else None,
                       "missed_points": total-point_hits,
                       "interval_iou": None, "reason": "Only point annotations; event boundaries unavailable"},
            "false_alerts_on_negative_videos": negative_alerts,
            "false_alerts_per_negative_minute": negative_alerts/(negative_duration/60) if negative_duration else None,
            "unmatched_alerts_on_positive_videos_unknown": unknown_alerts,
            "videos": video_rows, "point_predictions": point_rows}


def save_model(path, model, spec, threshold, metadata):
    arrays = {f"{kind}_{k}": v for kind in ("binary", "typed") for k, v in model[kind].items()}
    np.savez_compressed(path, **arrays, threshold=threshold, spec=spec,
                        metadata=json.dumps(metadata, ensure_ascii=False), format_version=3)


def load_model(path):
    saved = load_npz(path)
    if int(saved["format_version"]) != 3:
        raise ValueError("Unknown model format")
    model = {kind: {k: saved[f"{kind}_{k}"] for k in ("mean", "scale", "coef", "intercept", "classes")}
             for kind in ("binary", "typed")}
    return model, str(saved["spec"]), float(saved["threshold"]), json.loads(str(saved["metadata"]))


def deployment_gate(baseline, candidate, validation_better):
    old, new = baseline["videos"], candidate["videos"]
    if [r["video"] for r in old] != [r["video"] for r in new]:
        raise ValueError("Cannot compare different test videos")
    # Resample scene groups, not correlated videos.
    group_names = sorted({r["scene_group"] for r in old})
    delta = np.array([np.mean([int(n["predicted"] == n["label"])-int(o["predicted"] == o["label"])
                              for o, n in zip(old, new, strict=True) if o["scene_group"] == g]) for g in group_names])
    lower = upper = None
    if len(delta) >= 2:
        rng = np.random.default_rng(42)
        boot = rng.choice(delta, (10000, len(delta)), replace=True).mean(axis=1)
        lower, upper = np.quantile(boot, [.025, .975])
    pos_groups = {r["scene_group"] for r in old if r["label"]}
    neg_groups = {r["scene_group"] for r in old if not r["label"]}
    checks = {"validation_improved": bool(validation_better),
              "paired_group_bootstrap_lower_above_zero": lower is not None and bool(lower > 0),
              "point_type_accuracy_not_worse": candidate["type_accuracy_at_points"] >= baseline["type_accuracy_at_points"],
              "typed_point_recall_not_worse": candidate["timing"]["typed_point_recall"] >= baseline["timing"]["typed_point_recall"],
              "at_least_10_positive_and_negative_test_groups": len(pos_groups) >= 10 and len(neg_groups) >= 10}
    return {"passed": all(checks.values()), "checks": checks,
            "group_bootstrap_accuracy_delta_95pct": [float(lower), float(upper)] if lower is not None else None,
            "test_scene_groups": len(group_names),
            "warning": "Exploratory reused test; bootstrap with very few groups cannot establish generalization."}


def extract_features(run, records):
    cache = run/"context_features"
    cache.mkdir(exist_ok=True)
    extractor = ContextExtractor()
    manifest = {}
    for seconds in (4, 8, 16):
        spec = FeatureSpec(seconds, seconds/2, "full_work")
        extractor.spec = spec
        manifest[str(seconds)] = {}
        for i, record in enumerate(records):
            source = Path(record["path"])
            if sha256(source) != record["sha256"]:
                raise ValueError("Source video differs from frozen audit")
            path = cache/(cache_key(record["sha256"], spec)+".npz")
            if not path.exists():
                data = extractor.extract(source)
                np.savez_compressed(path, **data)
            item = load_npz(path)
            if str(item["spec"]) != spec.to_json():
                raise ValueError("Cached preprocessing does not match specification")
            if any(a["timecode"] >= float(item["duration"]) for a in record["annotations"]):
                raise ValueError("Point annotation outside video")
            manifest[str(seconds)][record["id"]] = {
                "path": str(path), "elapsed_seconds": float(item["elapsed_seconds"]),
                "windows": len(item["centers"])}
            write_json(run/"feature_manifest.json", manifest)
            print(f"{seconds}s {i+1}/{len(records)} {record['id']}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--extract", action="store_true", help="Create/resume verified feature caches first")
    args = parser.parse_args()
    run = args.run
    records = json.loads((run/"grouped_records.json").read_text())
    protocol = json.loads((run/"protocol.json").read_text())
    if args.extract:
        extract_features(run, records)
    manifest = json.loads((run/"feature_manifest.json").read_text())
    splits = {s: [r for r in records if r["split"] == s] for s in ("train", "validation", "test")}
    for sec in ("4", "8", "16"):
        if set(manifest.get(sec, {})) != {r["id"] for r in records}:
            raise ValueError(f"Feature extraction incomplete: {sec}s")
    allowed = sorted({a["type"] for r in records for a in r["annotations"]})
    if sorted({a["type"] for r in splits["train"] for a in r["annotations"]}) != allowed:
        raise ValueError("Training split lacks a type")
    baseline_data = {r["id"]: load_npz(MODEL_DIR/"features"/f"{r['sha256']}-{PREPROCESS_VERSION}.npz") for r in records}
    baseline = train_heads(splits["train"], baseline_data, "logistic")
    base_threshold = choose_threshold(baseline, splits["validation"], baseline_data, baseline=True)
    baseline_val = evaluate(baseline, splits["validation"], baseline_data, base_threshold, allowed)
    save_model(run/"baseline_refit.npz", baseline, PREPROCESS_VERSION, base_threshold, {"allowed_types": allowed})
    rows, candidates = [], []
    for seconds in (4, 8, 16):
        for record in records:
            expected_cache = cache_key(record["sha256"], FeatureSpec(seconds, seconds/2, "full_work"))+".npz"
            if Path(manifest[str(seconds)][record["id"]]["path"]).name != expected_cache:
                raise ValueError("Feature cache belongs to a different source or model")
        all_data = {r["id"]: load_npz(manifest[str(seconds)][r["id"]]["path"]) for r in records}
        expected = FeatureSpec(seconds, seconds/2, "full_work").to_json()
        if any(str(d["spec"]) != expected for d in all_data.values()):
            raise ValueError("Feature cache specification mismatch")
        for views in ("full", "full_work"):
            spec = FeatureSpec(seconds, seconds/2, views)
            data = {name: {**d, "features": d["features"][:, :512] if views == "full" else d["features"]}
                    for name, d in all_data.items()}
            for family in ("logistic", "svm", "centroid"):
                began = time.monotonic()
                model = train_heads(splits["train"], data, family, protocol["normal_review"]["training_videos"])
                threshold = choose_threshold(model, splits["validation"], data)
                val = evaluate(model, splits["validation"], data, threshold, allowed)
                name = f"context_{seconds}s_{views}_{family}"
                # Cost tie-break is a deterministic window/view count, not a noisy runtime.
                cost = sum(len(data[r["id"]]["centers"]) for r in splits["validation"]) * (2 if views == "full_work" else 1)
                rank = (val["detection"]["balanced_accuracy"], val["timing"]["typed_point_recall"],
                        val["type_accuracy_at_points"], -val["false_alerts_on_negative_videos"], -cost)
                metadata = {"name": name, "allowed_types": allowed, "weights_sha256": sha256(WEIGHTS),
                            "protocol_sha256": sha256(run/"protocol.json"), "hard_negative_windows": model["hard_negative_windows"]}
                path = run/(name+".npz")
                save_model(path, model, spec.to_json(), threshold, metadata)
                row = {"name": name, "spec": json.loads(spec.to_json()), "family": family, "threshold": threshold,
                       "validation": val, "training_seconds": time.monotonic()-began, "rank": list(rank),
                       "model_sha256": sha256(path), "hard_negative_windows": model["hard_negative_windows"]}
                rows.append(row)
                candidates.append((rank, row, model, data))
                print(name, rank, flush=True)
    write_json(run/"validation_grid.json", rows)
    _, chosen, candidate, candidate_data = max(candidates, key=lambda x: x[0])
    # Freeze selection BEFORE computing test metrics. No test-driven fallback selection.
    write_json(run/"selection_frozen.json", {"selected": chosen["name"], "utc": datetime.now(UTC).isoformat(),
                                            "protocol_sha256": sha256(run/"protocol.json"), "rank": chosen["rank"]})
    baseline_test = evaluate(baseline, splits["test"], baseline_data, base_threshold, allowed)
    candidate_test = evaluate(candidate, splits["test"], candidate_data, chosen["threshold"], allowed)
    gate = deployment_gate(baseline_test, candidate_test,
                           chosen["validation"]["detection"]["balanced_accuracy"] > baseline_val["detection"]["balanced_accuracy"])
    supports = {t: {s: {"annotations": sum(a["type"] == t for r in rs for a in r["annotations"]),
                       "scene_groups": len({r["scene_group"] for r in rs if any(a["type"] == t for a in r["annotations"])})}
                   for s, rs in splits.items()} for t in allowed}
    report = {"format_version": 3, "allowed_types": allowed, "selected": chosen["name"],
              "test_status": protocol["test_status"], "protocol_sha256": sha256(run/"protocol.json"),
              "supports": supports, "baseline": {"validation": baseline_val, "test": baseline_test},
              "candidate": {"validation": chosen["validation"], "test": candidate_test},
              "deployment_gate": gate, "active_model": "typed_v2_original_unchanged",
              "feature_extraction_seconds": {sec: sum(v["elapsed_seconds"] for v in items.values()) for sec, items in manifest.items()},
              "limitations": protocol["label_limitations"]}
    write_json(run/"comparison.json", report)
    print(json.dumps({"selected": chosen["name"], "baseline_test": baseline_test["detection"],
                      "candidate_test": candidate_test["detection"], "gate": gate}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
