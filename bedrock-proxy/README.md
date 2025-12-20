# bedrock-proxy

一个轻量级的 FastAPI 网关，部署在 Amazon Bedrock 前端，提供统一的 `/chat` 端点用于聊天补全和文本嵌入，支持可选的流式响应和 Bedrock Guardrails。

## 特性

- 基于 Amazon Bedrock 的简单 HTTP API（无需在应用中使用 AWS SDK）。
- `POST /chat` 支持：
  - 聊天补全（同步或流式响应）。
  - 文本嵌入（Titan / Cohere）。
- 共享 Bedrock 客户端，支持连接池和重试机制。
- 可通过环境变量配置可选的 Bedrock Guardrails。
- 支持 CORS，适用于浏览器前端。

## 要求

- Python 3.12+
- 拥有 Amazon Bedrock 访问权限的 AWS 账户。
- 已配置 AWS 凭据（例如 `AWS_ACCESS_KEY_ID`、`AWS_SECRET_ACCESS_KEY`、`AWS_REGION`）。

## 配置

所有配置均从环境变量读取（详见 `core/settings.py`）：

- `AWS_REGION`（默认值：`eu-central-1`）
- `BEDROCK_MODEL_ID`（默认值：Claude 模型 ID）
- `BEDROCK_EMBED_MODEL_ID`（默认值：`amazon.titan-embed-text-v2:0`）
- `BEDROCK_GUARDRAIL_ID`（可选）
- `BEDROCK_GUARDRAIL_VERSION`（默认值：`1`）
- `ALLOWED_ORIGINS`（CORS，默认值：`*`）
- `MAX_POOL_CONNECTIONS`（默认值：`50`）
- `MAX_RETRY_ATTEMPTS`（默认值：`5`）

示例 `.env` 文件：

```env
AWS_REGION=eu-central-1
BEDROCK_MODEL_ID=inference-profile/your-model-id
BEDROCK_EMBED_MODEL_ID=amazon.titan-embed-text-v2:0
BEDROCK_GUARDRAIL_ID=gr-xxxxxxxx
BEDROCK_GUARDRAIL_VERSION=1
ALLOWED_ORIGINS=http://localhost:3000,https://your-frontend.app
```

## 本地运行

```bash
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

健康检查：

```bash
curl http://localhost:8000/health
# -> {"status": "ok"}
```

## 使用 Docker 运行

```bash
docker build -t bedrock-proxy .
docker run --rm -p 8000:8000 --env-file .env bedrock-proxy
```

## 使用方法

### 聊天补全（非流式）

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "chat",
    "payload": {
      "messages": [
        {
          "role": "user",
          "content": [
            { "type": "text", "text": "Hello, Bedrock!" }
          ]
        }
      ]
    }
  }'
```

### 文本嵌入

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "embed",
    "text": "文档或句子内容"
  }'
```

### 流式聊天（高级）

发送 `mode: "chat"` 并设置 `stream: true`（使用支持流式响应的模型）。API 返回 NDJSON 事件，您可以在生成答案时实时渲染。

## 项目结构

- `api/` — FastAPI 应用、路由和响应辅助函数。
- `core/` — 设置和 Bedrock 客户端（boto3）。
- `services/` — 聊天和嵌入服务。

