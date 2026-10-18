"""One local worker and persistent job results. Run exactly one API process."""

import logging
import sqlite3
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import BoundedSemaphore

from backend.config import Settings
from backend.schemas import AnalysisResult, JobResponse
from backend.services.analyzer import AnalyzerNotConfigured

logger = logging.getLogger(__name__)
Analyzer = Callable[[Path], AnalysisResult]


class QueueFull(RuntimeError):
    pass


class JobManager:
    def __init__(self, settings: Settings, analyzer: Analyzer) -> None:
        self.analyzer = analyzer
        self.db_path = settings.storage_dir / "jobs.sqlite3"
        self.slots = BoundedSemaphore(settings.max_pending_jobs)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-analysis")
        with sqlite3.connect(self.db_path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
            previous = db.execute("SELECT payload FROM jobs").fetchall()
        for (payload,) in previous:
            job = JobResponse.model_validate_json(payload)
            if job.status == "processing":
                self.save(JobResponse(
                    job_id=job.job_id,
                    status="failed",
                    error="Сервис перезапущен во время обработки. Загрузите видео повторно.",
                ))
        # Uploaded videos are temporary; results contain only selected evidence frames.
        for path in (settings.storage_dir / "uploads").glob("*.mov"):
            path.unlink(missing_ok=True)

    def save(self, job: JobResponse) -> None:
        with sqlite3.connect(self.db_path) as db:
            db.execute(
                "INSERT OR REPLACE INTO jobs (id, payload) VALUES (?, ?)",
                (job.job_id, job.model_dump_json(exclude_none=True)),
            )

    def get(self, job_id: str) -> JobResponse | None:
        with sqlite3.connect(self.db_path) as db:
            row = db.execute("SELECT payload FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return JobResponse.model_validate_json(row[0]) if row else None

    def submit(self, video_path: Path) -> str:
        if not self.slots.acquire(blocking=False):
            raise QueueFull("Очередь заполнена. Повторите загрузку после завершения текущих задач.")
        job_id = video_path.stem
        try:
            self.save(JobResponse(job_id=job_id, status="processing"))
            self.executor.submit(self._run, job_id, video_path)
        except Exception:
            self.slots.release()
            raise
        return job_id

    def _run(self, job_id: str, video_path: Path) -> None:
        try:
            result = AnalysisResult.model_validate(self.analyzer(video_path))
            job = JobResponse(job_id=job_id, status="done", **result.model_dump())
        except AnalyzerNotConfigured as exc:
            job = JobResponse(job_id=job_id, status="failed", error=str(exc))
        except Exception:
            logger.exception("Analysis failed for job %s", job_id)
            job = JobResponse(
                job_id=job_id,
                status="failed",
                error="Ошибка анализа. Проверьте файл и журнал сервера.",
            )
        try:
            self.save(job)
        finally:
            try:
                video_path.unlink(missing_ok=True)
            finally:
                self.slots.release()

    def close(self) -> None:
        self.executor.shutdown(wait=True)
