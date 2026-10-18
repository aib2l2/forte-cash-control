"""Shared configuration for the API and Streamlit. Paths are project-relative."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    api_key: str
    storage_dir: Path
    api_base_url: str = "http://127.0.0.1:8000"
    max_upload_mb: int = 100
    max_pending_jobs: int = 4

    def __post_init__(self) -> None:
        if not self.api_key or self.api_key == "replace-with-a-random-key":
            raise ValueError("API_KEY не настроен. Выполните: python scripts/setup_env.py")
        if self.max_upload_mb < 1 or self.max_pending_jobs < 1:
            raise ValueError("MAX_UPLOAD_MB и MAX_PENDING_JOBS должны быть положительными.")

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        storage = Path(os.getenv("STORAGE_DIR", "storage"))
        if not storage.is_absolute():
            storage = PROJECT_ROOT / storage
        return cls(
            api_key=os.getenv("API_KEY", ""),
            storage_dir=storage,
            api_base_url=os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/"),
            max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "100")),
            max_pending_jobs=int(os.getenv("MAX_PENDING_JOBS", "4")),
        )
