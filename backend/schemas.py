"""The jury API contract; successful results must satisfy these invariants."""

import base64
import binascii
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class EvidenceFrame(ContractModel):
    timestamp_sec: float = Field(ge=0)
    image_base64: str = Field(min_length=1)

    @field_validator("image_base64")
    @classmethod
    def check_jpeg(cls, value: str) -> str:
        try:
            content = base64.b64decode(value, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("Evidence must contain base64-encoded JPEG bytes") from exc
        if not content.startswith(b"\xff\xd8") or not content.endswith(b"\xff\xd9"):
            raise ValueError("Evidence must be a JPEG image, without a data: URL prefix")
        return value


class Violation(ContractModel):
    type: str = Field(min_length=1)
    timecode_sec: float = Field(ge=0)
    description: str = Field(min_length=1)
    evidence_frames: list[EvidenceFrame] = Field(min_length=1, max_length=3)


class AnalysisResult(ContractModel):
    verdict: Literal["no_violation", "violation"]
    violations: list[Violation]

    @model_validator(mode="after")
    def check_verdict(self) -> "AnalysisResult":
        if (self.verdict == "violation") != bool(self.violations):
            raise ValueError("violation requires events; no_violation requires an empty list")
        return self


class JobCreated(ContractModel):
    job_id: str


class JobResponse(ContractModel):
    job_id: str
    status: Literal["processing", "done", "failed"]
    verdict: Literal["no_violation", "violation"] | None = None
    violations: list[Violation] | None = None
    error: str | None = None
