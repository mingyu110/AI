import json
from typing import Any, Dict, Iterable

from fastapi.responses import JSONResponse, StreamingResponse


def json_response(payload: Dict[str, Any], status: int = 200) -> JSONResponse:
    return JSONResponse(content=payload, status_code=status)


def error_response(status: int, message: str) -> JSONResponse:
    return JSONResponse(content={"error": message}, status_code=status)


def streaming_ndjson_response(event_iter: Iterable[Dict[str, Any]]) -> StreamingResponse:
    def gen():
        for obj in event_iter:
            yield (json.dumps(obj) + "\n").encode("utf-8")

    return StreamingResponse(gen(), media_type="application/x-ndjson")
