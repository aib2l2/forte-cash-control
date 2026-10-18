"""Create a local inference archive using an explicit allowlist, never raw datasets."""

import hashlib
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def build():
    paths = set()
    for folder in ("backend", "cash_ml"):
        paths.update(p for p in (ROOT / folder).rglob("*.py") if "__pycache__" not in p.parts)
    paths.update((ROOT / "cash_ml/assets").rglob("*"))
    paths.update((ROOT / "deploy").rglob("*"))
    paths.update((ROOT / "outputs/case_review/legal_sources").glob("*_paragraphs.json"))
    for name in ["pyproject.toml", ".env.example", "scripts/setup_env.py", "cash_ml/requirements.txt",
                 "cash_ml/rule_registry.json", "cash_ml/README.md", "README.md",
                 "models/cash_r3d/r3d18_kinetics400.pth", "models/cash_r3d/typed_classifier.npz",
                 "models/cash_r3d/typed_report.json", "outputs/case_review/execution_report.txt",
                 "outputs/case_review/data_collection_plan.txt", "outputs/case_review/implementation_prompt.txt"]:
        paths.add(ROOT / name)
    files = sorted(p for p in paths if p.is_file() and "__pycache__" not in p.parts
                   and p.name not in {"site.env", ".env"} and not p.is_symlink())
    for p in files:
        relative = p.relative_to(ROOT)
        assert relative.parts[0] not in {"storage", "sources", ".venv", ".git"}
        assert p.suffix.lower() not in {".mov", ".mp4", ".xlsx", ".sqlite3", ".db"}
    manifest = {"built_utc": datetime.now(UTC).isoformat(), "purpose": "inference only; training data excluded",
                "active_model": "typed_v2", "public_deployment_verified": False,
                "files": {str(p.relative_to(ROOT)): digest(p) for p in files}}
    target = ROOT / "outputs/releases/forte-cash-control-inference-20261018.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "x", zipfile.ZIP_DEFLATED, compresslevel=2) as archive:
        for p in files:
            archive.write(p, "forte-cash-control/" + str(p.relative_to(ROOT)))
        archive.writestr("forte-cash-control/RELEASE_MANIFEST.json", json.dumps(manifest, indent=2))
        archive.writestr("forte-cash-control/START_HERE.txt", (ROOT / "deploy/README.txt").read_text())
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        assert not any(n.endswith(("/.env", "/site.env", ".mov", ".sqlite3", ".xlsx")) for n in archive.namelist())
    checksum = digest(target)
    target.with_suffix(".zip.sha256").write_text(checksum + "  " + target.name + "\n")
    print(json.dumps({"archive": str(target), "files": len(files), "bytes": target.stat().st_size, "sha256": checksum}))


if __name__ == "__main__":
    build()
