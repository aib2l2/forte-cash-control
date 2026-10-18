"""Run: python -m uvicorn cash_ml.api:app --host 127.0.0.1 --port 8000 --workers 1."""

from dataclasses import replace

from fastapi import Depends, HTTPException, Request
from pydantic import create_model

from backend.config import Settings
from backend.schemas import JobResponse
from backend.security import require_api_key
from cash_ml.rules import validate_registry
from cash_ml.runtime import create_runtime_app
from cash_ml.service import analyze_video, model_status, violation_contract

settings = Settings.from_env()
app = create_runtime_app(settings=replace(settings, storage_dir=settings.storage_dir / "typed_v2"),
                 analyzer=analyze_video)
app.title = "ForteAI Cash Control"
app.description = "Локальная демонстрационная модель typed_v2 на R3D-18. Типы строго из разметки Excel; срабатывания проверяет оператор."

# Encode the same whitelist in OpenAPI and validate stored responses as well.
status = model_status()
if status["ready"]:
    event_model = violation_contract(status["allowed_types"])
    response_model = create_model("TypedJobResponse", __base__=JobResponse,
                                  violations=(list[event_model] | None, None))
    app.router.routes = [r for r in app.router.routes if getattr(r, "path", None) != "/api/v1/jobs/{job_id}"]

    @app.get("/api/v1/jobs/{job_id}", response_model=response_model,
             response_model_exclude_none=True, dependencies=[Depends(require_api_key)])
    def get_job(request: Request, job_id: str):
        result = request.app.state.jobs.get(job_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Задача не найдена.")
        return result


@app.get("/api/v1/model", dependencies=[Depends(require_api_key)])
def model():
    return model_status()


@app.get("/api/v1/rules", dependencies=[Depends(require_api_key)])
def rules():
    return validate_registry(model_status()["allowed_types"])
