import json
from typing import Dict, Any, Tuple, Optional

from botocore.exceptions import ClientError

from core.settings import AppSettings


class EmbeddingService:
    def __init__(self, bedrock_client, settings: AppSettings):
        self.client = bedrock_client
        self.settings = settings

    def embed(self, text: str, model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        调用 Bedrock 上的嵌入模型，返回包含嵌入向量、输入词元数、维度等信息的字典。
        """
        if not text:
            raise ValueError("嵌入功能需要提供文本内容")

        model = model_id or self.settings.bedrock_embed_model_id
        payload = self._build_payload(text, model)

        try:
            resp = self.client.invoke_model(
                modelId=model,
                body=json.dumps(payload),
                contentType="application/json",
                accept="application/json",
            )
        except ClientError as ce:
            print("Bedrock 客户端错误 (嵌入):", ce)
            raise

        body_bytes = resp["body"].read()
        body = json.loads(body_bytes)

        embedding, input_tokens = self._parse_response(model, body)

        return {
            "mode": "embed",
            "model": model,
            "text": text,
            "embedding": embedding,
            "inputTokens": input_tokens,
            "dimensions": len(embedding) if embedding else None,
        }

    # ---------- 内部辅助方法 ----------

    @staticmethod
    def _build_payload(text: str, model_id: str) -> Dict[str, Any]:
        if model_id.startswith("amazon.titan-embed-text"):
            return {"inputText": text}
        elif model_id.startswith("cohere.embed"):
            return {"input_type": "search_document", "texts": [text]}
        else:
            raise ValueError(f"不支持的嵌入模型: {model_id}")

    @staticmethod
    def _parse_response(model_id: str, body: Dict[str, Any]) -> Tuple[Optional[list], Optional[int]]:
        # Titan v2 模型
        if model_id.startswith("amazon.titan-embed-text"):
            embedding = body.get("embedding")
            input_tokens = body.get("inputTextTokenCount")
        # Cohere 多语言 v3 模型
        elif model_id.startswith("cohere.embed"):
            embeddings = body.get("embeddings") or []
            embedding = embeddings[0] if embeddings else None
            input_tokens = None
        else:
            raise ValueError(f"不支持的嵌入模型: {model_id}")

        return embedding, input_tokens
