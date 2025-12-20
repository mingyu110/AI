from typing import Dict, Any

from botocore.exceptions import ClientError
from fastapi.responses import JSONResponse, StreamingResponse

from core.settings import AppSettings
from core.bedrock_client import get_bedrock_client
from services.chat_service import ChatService
from services.embedding_service import EmbeddingService
from .response import json_response, error_response, streaming_ndjson_response

settings = AppSettings.instance()
bedrock_client = get_bedrock_client()

chat_service = ChatService(bedrock_client, settings)
embedding_service = EmbeddingService(bedrock_client, settings)


async def handle_chat_request(body: Dict[str, Any]) -> JSONResponse | StreamingResponse:
    """
    /chat 端点的主入口函数。
    在嵌入/聊天模式、流式/非流式响应之间做路由决策。
    """
    mode = body.get("mode", "chat")
    print("Incoming body:", body)

    # ---------- 嵌入模式 ----------
    if mode == "embed":
        text = body.get("text") or body.get("input") or ""
        if not text:
            return error_response(400, "Missing 'text' or 'input' for embed mode")

        try:
            result = embedding_service.embed(text, body.get("model"))
        except ClientError as ce:
            return error_response(
                500,
                f"Bedrock embedding error: {ce.response.get('Error', {}).get('Message', str(ce))}",
            )
        except ValueError as ve:
            return error_response(400, str(ve))

        return json_response(result)

    # ---------- 聊天模式 ----------
    try:
        prepared = chat_service.prepare_request(body)
    except ValueError as ve:
        return error_response(400, str(ve))

    payload, model_id, has_tools, use_stream, invoke_kwargs = prepared

    # 流式响应分支
    if use_stream:
        try:
            events_iter_factory, full_text_parts, meta = chat_service.chat_stream(prepared)
        except ClientError as ce:
            return error_response(
                500,
                f"Bedrock error: {ce.response.get('Error', {}).get('Message', str(ce))}",
            )

        # events_iter_factory 是一个创建生成器的函数
        return streaming_ndjson_response(events_iter_factory())

    # 非流式响应分支
    try:
        result = chat_service.chat_sync(body, prepared=prepared)
    except ClientError as ce:
        return error_response(
            500,
            f"Bedrock error: {ce.response.get('Error', {}).get('Message', str(ce))}",
        )

    return json_response(result)
