import secrets
from typing import Annotated

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(
    request: Request,
    supplied: Annotated[str | None, Security(api_key_header)],
) -> None:
    expected = request.app.state.settings.api_key
    if supplied is None or not secrets.compare_digest(supplied.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Неверный или отсутствующий X-API-Key.")
