from fastapi import FastAPI, Request
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware

from core.settings import AppSettings
from .chat_handler import handle_chat_request

settings = AppSettings.instance()

app = FastAPI()

# CORS 配置
origins = ["*"] if settings.allowed_origins == "*" else settings.allowed_origins.split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat")
async def chat(request: Request):
    try:
        body = await request.json()
    # except Exception:
    #     body = {}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {e}")
    return await handle_chat_request(body)


@app.get("/health")
async def health():
    return {"status": "ok"}
