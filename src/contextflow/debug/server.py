"""Debug server – FastAPI app serving the debug frontend and REST/WebSocket APIs."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from contextflow.debug.session import DebugSession, LLMCallRecord, StateChangeRecord

logger = logging.getLogger("contextflow.debug")

STATIC_DIR = Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(min_length=1)


class TransitionRequest(BaseModel):
    target_state: str = Field(min_length=1)


# ---------------------------------------------------------------------------
# WebSocket manager
# ---------------------------------------------------------------------------

class _WSManager:
    """Simple fan-out to all connected WebSocket clients."""

    def __init__(self) -> None:
        self._clients: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._clients.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self._clients:
            self._clients.remove(ws)

    async def broadcast(self, event: dict[str, Any]) -> None:
        data = json.dumps(event, default=str)
        dead: list[WebSocket] = []
        for ws in self._clients:
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_debug_app(session: DebugSession) -> FastAPI:
    """Build the FastAPI app wired to *session*."""

    app = FastAPI(title="ContextFlow Debug", version="0.1.0")
    ws_manager = _WSManager()

    # -- Real-time hooks ---------------------------------------------------

    def _on_llm_call(record: LLMCallRecord) -> None:
        asyncio.ensure_future(ws_manager.broadcast({
            "type": "llm_call",
            "data": record.to_dict(),
        }))

    def _on_state_change(record: StateChangeRecord) -> None:
        asyncio.ensure_future(ws_manager.broadcast({
            "type": "state_change",
            "data": record.to_dict(),
        }))

    def _on_conversation(msg: dict[str, Any]) -> None:
        asyncio.ensure_future(ws_manager.broadcast({
            "type": "conversation",
            "data": msg,
        }))

    session._on_llm_call.append(_on_llm_call)
    session._on_state_change.append(_on_state_change)
    session._on_conversation_update.append(_on_conversation)

    # -- Frontend ----------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def index():
        html_path = STATIC_DIR / "index.html"
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))

    # -- REST API ----------------------------------------------------------

    @app.get("/api/status")
    async def api_status():
        return JSONResponse(session.snapshot_status())

    @app.post("/api/chat")
    async def api_chat(req: ChatRequest):
        try:
            response_text = await session.chat(req.message)
            return JSONResponse({
                "ok": True,
                "response": response_text,
                "status": session.snapshot_status(),
            })
        except Exception as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

    @app.post("/api/transition")
    async def api_transition(req: TransitionRequest):
        try:
            new_state = await session.transition_to(req.target_state)
            return JSONResponse({
                "ok": True,
                "new_state": new_state,
                "status": session.snapshot_status(),
            })
        except Exception as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

    @app.get("/api/conversation")
    async def api_conversation():
        return JSONResponse(session.snapshot_conversation())

    @app.get("/api/calls")
    async def api_calls():
        return JSONResponse(session.snapshot_llm_calls())

    @app.get("/api/calls/{call_id}")
    async def api_call_detail(call_id: str):
        record = session.snapshot_llm_call(call_id)
        if record is None:
            return JSONResponse({"error": "Not found"}, status_code=404)
        return JSONResponse(record)

    @app.get("/api/state")
    async def api_state():
        sm = session.snapshot_state_machine()
        if sm is None:
            return JSONResponse({"error": "No state machine attached"}, status_code=404)
        return JSONResponse(sm)

    @app.get("/api/state-changes")
    async def api_state_changes():
        return JSONResponse(session.snapshot_state_changes())

    # -- WebSocket ---------------------------------------------------------

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        await ws_manager.connect(ws)
        try:
            while True:
                # Keep the connection alive; client may send pings
                data = await ws.receive_text()
                if data == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))
        except WebSocketDisconnect:
            ws_manager.disconnect(ws)

    return app


# ---------------------------------------------------------------------------
# Convenience launcher
# ---------------------------------------------------------------------------

def launch_debug(
    session: DebugSession,
    *,
    host: str = "127.0.0.1",
    port: int = 8790,
    open_browser: bool = True,
) -> None:
    """Start the debug server (blocking).

    Usage::

        from contextflow.debug import DebugSession, launch_debug

        session = DebugSession(agent)
        launch_debug(session, port=8790)
    """
    import uvicorn
    import webbrowser
    import threading

    app = create_debug_app(session)

    if open_browser:
        def _open():
            import time
            time.sleep(1.5)
            webbrowser.open(f"http://{host}:{port}")
        threading.Thread(target=_open, daemon=True).start()

    print(f"\n{'='*60}")
    print(f"  ContextFlow Debug Frontend")
    print(f"  http://{host}:{port}")
    print(f"{'='*60}\n")

    uvicorn.run(app, host=host, port=port, log_level="info")
