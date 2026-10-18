from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from backend.config import Settings
from backend.jobs import Analyzer, JobManager, QueueFull
from backend.schemas import JobCreated, JobResponse
from backend.security import require_api_key
from backend.services.analyzer import analyze_video


def create_app(settings: Settings | None = None, analyzer: Analyzer | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        config = settings or Settings.from_env()
        config.storage_dir.mkdir(parents=True, exist_ok=True)
        (config.storage_dir / "uploads").mkdir(exist_ok=True)
        application.state.settings = config
        application.state.jobs = JobManager(config, analyzer or analyze_video)
        try:
            yield
        finally:
            await run_in_threadpool(application.state.jobs.close)

    application = FastAPI(
        title="Forte Cash Control",
        version="0.1.0",
        description="Каркас Datathon. AI-модель ещё не подключена; фиктивных вердиктов нет.",
        lifespan=lifespan,
    )

    @application.get("/health", tags=["service"])
    def health():
        # Liveness only: this endpoint intentionally makes no model-readiness claim.
        return {"status": "ok"}

    @application.post(
        "/api/v1/analyze",
        response_model=JobCreated,
        status_code=202,
        dependencies=[Depends(require_api_key)],
        tags=["analysis"],
    )
    async def analyze(request: Request, file: Annotated[UploadFile, File()]):
        target: Path | None = None
        try:
            if Path(file.filename or "").suffix.lower() != ".mov":
                raise HTTPException(status_code=415, detail="Загрузите видео в формате .mov.")
            config = request.app.state.settings
            limit = config.max_upload_mb * 1024 * 1024
            target = config.storage_dir / "uploads" / f"{uuid4().hex}.mov"
            size = 0
            with target.open("xb") as output:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > limit:
                        raise HTTPException(
                            status_code=413,
                            detail=f"Файл превышает лимит {config.max_upload_mb} МБ.",
                        )
                    await run_in_threadpool(output.write, chunk)
            if size == 0:
                raise HTTPException(status_code=400, detail="Файл пустой.")
            try:
                job_id = await run_in_threadpool(request.app.state.jobs.submit, target)
            except QueueFull as exc:
                raise HTTPException(status_code=429, detail=str(exc)) from exc
            return JobCreated(job_id=job_id)
        except BaseException:
            if target is not None:
                target.unlink(missing_ok=True)
            raise
        finally:
            await file.close()

    @application.get(
        "/api/v1/jobs/{job_id}",
        response_model=JobResponse,
        response_model_exclude_none=True,
        dependencies=[Depends(require_api_key)],
        tags=["analysis"],
    )
    def get_job(request: Request, job_id: str):
        job = request.app.state.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Задача не найдена.")
        return job

    return application


app = create_app()
