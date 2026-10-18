"""A bounded local worker with a deadline covering queue time and inference.

Each video runs in a spawned process so a stalled decoder can actually be stopped.
No network calls, audio processing or video contents are logged by this module.
"""

import multiprocessing as mp
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from multiprocessing.process import BaseProcess
from threading import RLock, Timer

from starlette.concurrency import run_in_threadpool

from backend.app_factory import create_app
from backend.jobs import JobManager, QueueFull
from backend.schemas import AnalysisResult, JobResponse


def _child(analyzer, path, sender):
    try:
        result = AnalysisResult.model_validate(analyzer(path))
        sender.send((True, result.model_dump(mode="json")))
    except Exception:
        sender.send((False, "Не удалось декодировать или проанализировать видео."))
    finally:
        sender.close()


@dataclass
class Pending:
    deadline: float
    timer: Timer | None = None
    expired: bool = False
    process: BaseProcess | None = None


class DeadlineJobManager(JobManager):
    def __init__(self, settings, analyzer, deadline_seconds=290.0):
        if not 0 < deadline_seconds <= 290:
            raise ValueError("Deadline must leave at least 10 seconds within the five-minute limit")
        super().__init__(settings, analyzer)
        self.deadline_seconds = deadline_seconds
        self.pending = {}
        self.lock = RLock()
        self.context = mp.get_context("spawn")

    def _expire(self, job_id):
        with self.lock:
            pending = self.pending.get(job_id)
            if pending is not None:
                pending.expired = True
                self.save(JobResponse(job_id=job_id, status="failed", error=(
                    "Истёк лимит обработки с учётом очереди. Повторите загрузку позднее."
                )))
                # Also interrupt a blocked recv/decode, not just a poll timeout.
                try:
                    if pending.process is not None and pending.process.is_alive():
                        pending.process.terminate()
                except (OSError, ValueError):
                    pass  # The worker may already have closed the process.

    def submit(self, video_path):
        if not self.slots.acquire(blocking=False):
            raise QueueFull("Очередь заполнена. Дождитесь завершения текущих задач.")
        job_id = video_path.stem
        timer = Timer(self.deadline_seconds, self._expire, args=(job_id,))
        timer.daemon = True
        try:
            with self.lock:
                self.pending[job_id] = Pending(time.monotonic() + self.deadline_seconds, timer)
                self.save(JobResponse(job_id=job_id, status="processing"))
                timer.start()
                self.executor.submit(self._run, job_id, video_path)
        except Exception:
            timer.cancel()
            with self.lock:
                self.pending.pop(job_id, None)
            self.slots.release()
            raise
        return job_id

    def _run(self, job_id, video_path):
        process, receiver, sender = None, None, None
        job = None
        try:
            with self.lock:
                state = self.pending[job_id]
                remaining = state.deadline - time.monotonic()
                if state.expired or remaining <= 0:
                    self._expire(job_id)
                    return
            receiver, sender = self.context.Pipe(duplex=False)
            process = self.context.Process(target=_child, args=(self.analyzer, video_path, sender))
            with self.lock:
                state.process = process
            process.start()
            sender.close()
            remaining = max(0.0, state.deadline - time.monotonic())
            if receiver.poll(remaining):
                ok, payload = receiver.recv()
                if ok:
                    result = AnalysisResult.model_validate(payload)
                    job = JobResponse(job_id=job_id, status="done", **result.model_dump())
                else:
                    job = JobResponse(job_id=job_id, status="failed", error=payload)
            else:
                self._expire(job_id)
        except Exception:
            job = JobResponse(job_id=job_id, status="failed", error="Ошибка локального обработчика видео.")
        finally:
            if process is not None and process.pid is not None:
                process.join(timeout=0.2)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=0.5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=0.5)
                process.close()
            for connection in (receiver, sender):
                if connection is not None:
                    connection.close()
            with self.lock:
                state = self.pending.pop(job_id)
                state.timer.cancel()
                if not state.expired:
                    if time.monotonic() > state.deadline:
                        job = JobResponse(job_id=job_id, status="failed", error="Истёк лимит обработки.")
                    if job is not None:
                        self.save(job)
            try:
                video_path.unlink(missing_ok=True)
            finally:
                self.slots.release()


def create_runtime_app(settings, analyzer, deadline_seconds=None):
    application = create_app(settings, analyzer)
    limit = float(os.getenv("JOB_DEADLINE_SECONDS", "290")) if deadline_seconds is None else deadline_seconds

    @asynccontextmanager
    async def lifespan(app):
        settings.storage_dir.mkdir(parents=True, exist_ok=True)
        (settings.storage_dir / "uploads").mkdir(exist_ok=True)
        app.state.settings = settings
        app.state.jobs = DeadlineJobManager(settings, analyzer, limit)
        try:
            yield
        finally:
            await run_in_threadpool(app.state.jobs.close)

    application.router.lifespan_context = lifespan
    return application
