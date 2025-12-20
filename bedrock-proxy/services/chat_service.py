import json
from typing import Dict, Any, Tuple, List, Optional

from botocore.exceptions import ClientError

from core.settings import AppSettings


class ChatService:
    def __init__(self, bedrock_client, settings: AppSettings):
        self.client = bedrock_client
        self.settings = settings


    def prepare_request(
        self, body: Dict[str, Any], force_stream: Optional[bool] = None
    ) -> Tuple[Dict[str, Any], str, bool, bool, Dict[str, Any]]:
        """
        规范化输入，添加防护栏配置，判断是否流式响应，
        并构建 Bedrock 调用的参数。
        """
        payload = body.get("payload", {})
        model_id = body.get("model", self.settings.bedrock_model_id)

        if not isinstance(payload, dict):
            raise ValueError("payload must be a JSON object")

        has_tools = bool(payload.get("tools"))
        if has_tools:
            print("Payload includes tools definition")

        # 将防护栏配置加入请求负载和请求头中
        if self.settings.guardrail_id and self.settings.guardrail_version:
            payload.setdefault("amazon-bedrock-guardrailConfig", {})

        use_stream = (
            force_stream
            if force_stream is not None
            else bool(body.get("stream") or body.get("transport") == "streamable-http")
        )

        invoke_kwargs = self._build_invoke_kwargs(model_id, payload)
        print(f"Calling Bedrock model: {model_id}")
        print(f"Streaming: {use_stream}")

        return payload, model_id, has_tools, use_stream, invoke_kwargs

    def chat_sync(
        self,
        body: Dict[str, Any],
        prepared: Optional[Tuple[Dict[str, Any], str, bool, bool, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        非流式聊天调用。返回包含答案、用量、工具调用等信息的字典。
        """
        if prepared is None:
            prepared = self.prepare_request(body, force_stream=False)

        payload, model_id, has_tools, use_stream, invoke_kwargs = prepared
        if use_stream:
            raise ValueError("流式聊天请求必须通过 chat_stream() 方法处理")

        try:
            response = self.client.invoke_model(**invoke_kwargs)
            raw_body = response["body"].read()
            print("Raw Bedrock body:", raw_body)
            response_body = json.loads(raw_body)
        except ClientError as ce:
            print("Bedrock ClientError:", ce)
            raise

        answer, tool_calls = self._extract_text_and_tools(response_body)
        if not answer:
            answer = json.dumps(response_body)

        usage_info = response_body.get("usage", {}) if response_body else {}
        clean_payload = dict(payload)

        result = {
            "mode": "chat",
            "model": model_id,
            "payload": clean_payload,
            "answer": answer,
            "Usage": usage_info,
            "tool_calls": tool_calls,
            "has_tools": has_tools,
            "streaming": False,
        }

        if response_body:
            result["amazon-bedrock-guardrailAction"] = response_body.get("amazon-bedrock-guardrailAction")
            result["stopReason"] = response_body.get("stop_reason")
            result["stop_sequence"] = response_body.get("stop_sequence")

        return result

    def chat_stream(
        self,
        prepared: Tuple[Dict[str, Any], str, bool, bool, Dict[str, Any]],
    ):
        """
        流式聊天调用：生成 NDJSON 字典（已解析），供 FastAPI StreamingResponse 使用。
        同时返回最终拼接的文本和事件列表（如果需要）。
        """
        payload, model_id, has_tools, use_stream, invoke_kwargs = prepared
        if not use_stream:
            raise ValueError("chat_stream called with use_stream=False")

        try:
            response = self.client.invoke_model_with_response_stream(**invoke_kwargs)
        except ClientError as ce:
            print("Bedrock ClientError (stream):", ce)
            raise

        stream_body = response.get("body")
        full_text_parts: List[str] = []

        # 已解析事件的生成器（用于 UI/调试/工具）
        def events():
            for event in stream_body:
                if "chunk" not in event:
                    yield {"event": event}
                    continue

                raw_bytes = event["chunk"]["bytes"]
                decoded = raw_bytes.decode("utf-8", errors="ignore")

                for line in decoded.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        obj = {"raw": line}

                    delta_text = self._extract_text_from_stream_event(obj)
                    if delta_text:
                        full_text_parts.append(delta_text)

                    yield obj

        return events, full_text_parts, {
            "mode": "chat",
            "model": model_id,
            "payload": payload,
            "has_tools": has_tools,
            "streaming": True,
        }


    def _build_invoke_kwargs(self, model_id: str, payload: dict) -> dict:
        kwargs = {
            "modelId": model_id,
            "body": json.dumps(payload),
            "contentType": "application/json",
            "accept": "application/json",
        }

        if self.settings.guardrail_id and self.settings.guardrail_version:
            kwargs["guardrailIdentifier"] = self.settings.guardrail_id
            kwargs["guardrailVersion"] = self.settings.guardrail_version
            # kwargs["trace"] = "ENABLED"

        return kwargs

    @staticmethod
    def _extract_text_and_tools(response_body: dict):
        """从 Bedrock 响应体中提取文本和工具调用信息。"""
        texts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []

        try:
            content = response_body.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    t = item.get("type")
                    if t == "text":
                        txt = item.get("text", "")
                        if txt:
                            texts.append(txt)
                    elif t == "tool_use":
                        tool_calls.append(
                            {
                                "id": item.get("id"),
                                "name": item.get("name"),
                                "input": item.get("input"),
                            }
                        )
        except Exception as e:
            print("解析 Bedrock 内容时出错:", e)

        answer = "\n".join(texts) if texts else None
        return answer, tool_calls

    @staticmethod
    def _extract_text_from_stream_event(obj: dict) -> str:
        delta = obj.get("delta")
        if isinstance(delta, dict) and "text" in delta:
            return delta["text"]

        output_text = obj.get("outputText")
        if isinstance(output_text, dict) and "text" in output_text:
            return output_text["text"]

        return ""
