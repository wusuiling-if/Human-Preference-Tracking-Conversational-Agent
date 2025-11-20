"""FastAPI server that exposes the latent aligner conversation as a web API."""
import os
from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from session_core import ConversationSession

session = ConversationSession()
subscribers: List[WebSocket] = []

app = FastAPI(title="Latent Aligner Web API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def broadcast_snapshot():
    """Push latest snapshot to all connected WebSocket clients."""
    payload = session.snapshot()
    stale: List[WebSocket] = []
    for ws in subscribers:
        try:
            await ws.send_json(payload)
        except Exception:
            stale.append(ws)
    for ws in stale:
        try:
            subscribers.remove(ws)
        except ValueError:
            pass


class ChatRequest(BaseModel):
    message: str


@app.post("/api/chat")
async def api_chat(payload: ChatRequest):
    resp = session.handle_message(payload.message.strip())
    await broadcast_snapshot()
    return resp


@app.get("/api/state")
def api_state():
    return session.snapshot()


@app.websocket("/ws/state")
async def ws_state(websocket: WebSocket):
    await websocket.accept()
    subscribers.append(websocket)
    try:
        await websocket.send_json(session.snapshot())
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in subscribers:
            subscribers.remove(websocket)


FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "web_frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def index():
    if os.path.isdir(FRONTEND_DIR):
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
    return {"message": "Frontend directory missing. Build assets in web_frontend/ first."}
