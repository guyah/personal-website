from fastapi import FastAPI, WebSocket

from .voice.ws import talk_websocket_handler

from . import models  # noqa: F401
from .database import Base, engine

app = FastAPI(title="personal-website-backend")


@app.on_event("startup")
def on_startup() -> None:
	Base.metadata.create_all(bind=engine)


@app.get("/health")
def health() -> dict[str, str]:
	return {"status": "ok"}


@app.websocket("/ws/talk")
async def ws_talk(websocket: WebSocket):
	await talk_websocket_handler(websocket)
