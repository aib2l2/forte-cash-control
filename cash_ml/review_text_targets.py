"""Create train-only semantic targets locally. Never print descriptions or images."""

import argparse
import hashlib
import json
import socket
from pathlib import Path

import numpy as np


def digest(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def require_offline():
    try:
        socket.create_connection(("1.1.1.1", 443), timeout=1).close()
    except PermissionError:
        return
    except OSError as exc:
        raise RuntimeError("Run under an OS network-denial sandbox") from exc
    raise RuntimeError("Network is available; refusing private text processing")


def window_texts(window):
    """Only the locally generated draft. Exclude file names, Excel and old cloud drafts."""
    if window.get("format_issues"):
        raise ValueError("Unresolved draft format issues")
    draft = window["draft"]
    texts = []
    for item in draft.get("observations", []):
        if not item.get("time_reference_valid"):
            raise ValueError("Invalid observation time reference")
        texts.append(str(item["action"]).strip())
    for item in draft.get("ordinary_actions", []):
        texts.append(" ".join(str(item.get(k, "")) for k in ("action", "why_ordinary")).strip())
    for item in draft.get("suspicious_actions", []):
        if not item.get("time_reference_valid"):
            raise ValueError("Invalid suspicious-action time reference")
        texts.append(
            " ".join(
                str(item.get(k, "")) for k in ("action", "why_check", "needed_context")
            ).strip()
        )
    return list(dict.fromkeys(t for t in texts if t))


def collect_train_windows(records, review_root):
    if any(r["split"] != "train" for r in records):
        raise ValueError("Only training records may be read for description targets")
    rows, sources = [], {}
    for record in records:
        path = review_root / "videos" / record["sha256"] / "result.json"
        result = json.loads(path.read_text())
        if (
            result["sha256"] != record["sha256"]
            or result["review_status"] != "local_model_draft_complete"
        ):
            raise ValueError("Incomplete or mismatched local draft")
        if len(result["windows"]) != result["expected_windows"]:
            raise ValueError("Missing review windows")
        sources[record["id"]] = {"sha256": record["sha256"], "draft_sha256": digest(path)}
        for window in result["windows"]:
            start, end = map(float, window["sampled_range_sec"])
            if not (0 <= start <= end <= result["duration_sec"] + 0.1):
                raise ValueError("Invalid review sampling range")
            texts = window_texts(window)
            if not texts:
                raise ValueError("Empty description window")
            rows.append(
                {
                    "id": record["id"],
                    "window": window["index"],
                    "start": start,
                    "end": end,
                    "texts": texts,
                }
            )
    return rows, sources


def encode(texts, model_dir):
    # Import only in the isolated ONNX environment; service dependencies are unchanged.
    import onnxruntime as ort
    from tokenizers import Tokenizer

    manifest = json.loads((model_dir / "download_manifest.json").read_text())
    for name, metadata in manifest["files"].items():
        if digest(model_dir / name) != metadata["sha256"]:
            raise ValueError("Text encoder checksum mismatch")
    tokenizer = Tokenizer.from_file(str(model_dir / "onnx/tokenizer.json"))
    pad = tokenizer.token_to_id("<pad>")
    tokenizer.enable_truncation(max_length=512)
    tokenizer.enable_padding(pad_id=pad, pad_token="<pad>")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(model_dir / "onnx/model.onnx"), sess_options=options, providers=["CPUExecutionProvider"]
    )
    vectors, truncated = [], 0
    input_names = {i.name for i in session.get_inputs()}
    for start in range(0, len(texts), 8):
        batch = tokenizer.encode_batch(["query: " + text for text in texts[start : start + 8]])
        truncated += sum(bool(t.overflowing) for t in batch)
        inputs = {
            "input_ids": np.asarray([t.ids for t in batch], dtype=np.int64),
            "attention_mask": np.asarray([t.attention_mask for t in batch], dtype=np.int64),
            "token_type_ids": np.asarray([t.type_ids for t in batch], dtype=np.int64),
        }
        hidden = session.run(None, {k: v for k, v in inputs.items() if k in input_names})[0]
        mask = inputs["attention_mask"][..., None]
        pooled = (hidden * mask).sum(1) / np.maximum(mask.sum(1), 1)
        pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-8)
        vectors.append(pooled.astype(np.float32))
    return np.concatenate(vectors), truncated, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--review-root", type=Path, default=Path("outputs/video_review/local_full_review")
    )
    parser.add_argument("--encoder", type=Path, default=Path("models/local_text_encoder/e5-small"))
    args = parser.parse_args()
    require_offline()
    out = args.run / "description_targets.npz"
    if out.exists():
        raise ValueError("Existing targets will not be overwritten")
    records = json.loads((args.run / "grouped_records.json").read_text())
    rows, sources = collect_train_windows(
        [r for r in records if r["split"] == "train"], args.review_root
    )
    texts = [text for row in rows for text in row["texts"]]
    vectors, truncated, manifest = encode(texts, args.encoder)
    targets, cursor = [], 0
    for row in rows:
        n = len(row["texts"])
        vector = vectors[cursor : cursor + n].mean(0)
        targets.append(vector / max(np.linalg.norm(vector), 1e-8))
        cursor += n
    np.savez_compressed(
        out,
        ids=np.asarray([r["id"] for r in rows]),
        window_indices=np.asarray([r["window"] for r in rows]),
        starts=np.asarray([r["start"] for r in rows]),
        ends=np.asarray([r["end"] for r in rows]),
        targets=np.asarray(targets, dtype=np.float32),
    )
    metadata = {
        "encoder": manifest["model"],
        "revision": manifest["revision"],
        "encoder_manifest_sha256": digest(args.encoder / "download_manifest.json"),
        "source_drafts": sources,
        "videos": len(sources),
        "windows": len(rows),
        "descriptions_encoded": len(texts),
        "truncated_descriptions": truncated,
        "dimensions": targets[0].shape[0],
        "human_verified": False,
        "supervision": "auxiliary semantic target, not a violation or normal label",
        "no_event_boundaries_created": True,
        "held_out_descriptions_read": False,
        "network_denied": True,
        "targets_sha256": digest(out),
    }
    (args.run / "description_targets_audit.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({k: v for k, v in metadata.items() if k != "source_drafts"}), flush=True)


if __name__ == "__main__":
    main()
