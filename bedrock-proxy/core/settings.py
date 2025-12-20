import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class AppSettings:
    # AWS / Bedrock 相关配置
    aws_region: str = os.getenv("AWS_REGION", "eu-central-1")
    bedrock_model_id: str = os.getenv(
        "BEDROCK_MODEL_ID",
        "inference-profile/eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
    )
    bedrock_embed_model_id: str = os.getenv(
        "BEDROCK_EMBED_MODEL_ID",
        "amazon.titan-embed-text-v2:0",
    )

    # Guardrails 防护栏配置
    guardrail_id: Optional[str] = os.getenv("BEDROCK_GUARDRAIL_ID")
    guardrail_version: Optional[str] = os.getenv("BEDROCK_GUARDRAIL_VERSION", "1")

    # HTTP / CORS 配置
    allowed_origins: str = os.getenv("ALLOWED_ORIGINS", "*")

    # Bedrock / HTTP 调优配置
    max_pool_connections: int = int(os.getenv("MAX_POOL_CONNECTIONS", "50"))
    max_retry_attempts: int = int(os.getenv("MAX_RETRY_ATTEMPTS", "5"))

    _instance: "AppSettings" = None  # type: ignore

    @classmethod
    def instance(cls) -> "AppSettings":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
