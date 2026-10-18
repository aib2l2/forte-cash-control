"""Verify the published model without loading pickle or connecting to a network."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify():
    folder = ROOT / "models" / "cash_r3d"
    report = json.loads((folder / "typed_report.json").read_text())
    for name, key in [("r3d18_kinetics400.pth", "weights_sha256"),
                      ("typed_classifier.npz", "classifier_sha256")]:
        with (folder / name).open("rb") as source:
            actual = hashlib.file_digest(source, "sha256").hexdigest()
        if actual != report[key]:
            raise ValueError(f"Checksum mismatch: {name}")
    import numpy as np
    with np.load(folder / "typed_classifier.npz", allow_pickle=False) as model:
        if int(model["format_version"]) != 2:
            raise ValueError("Unsupported model format")
        if list(model["type_classes"].astype(str)) != report["allowed_types"]:
            raise ValueError("Model vocabulary mismatch")
    print("typed_v2: weights, classifier and 11-type vocabulary verified")


if __name__ == "__main__":
    verify()
