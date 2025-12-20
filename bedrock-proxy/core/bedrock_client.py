import boto3
from botocore.config import Config
from .settings import AppSettings

_bedrock_client = None


def get_bedrock_client():
    """
    类似单例的 Bedrock 客户端，支持连接池和重试机制。
    对于典型的 FastAPI 使用场景来说是线程安全的。
    """
    global _bedrock_client
    if _bedrock_client is None:
        settings = AppSettings.instance()
        cfg = Config(
            region_name=settings.aws_region,
            max_pool_connections=settings.max_pool_connections,
            retries={"max_attempts": settings.max_retry_attempts, "mode": "adaptive"},
        )
        _bedrock_client = boto3.client("bedrock-runtime", config=cfg)
    return _bedrock_client
