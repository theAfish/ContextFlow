from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from contextflow.core.models import ContextNode, MessageRole
from contextflow.state import SessionManager


class CreateSessionResponse(BaseModel):
    session_id: str


class TurnRequest(BaseModel):
    content: str = Field(min_length=1)


class TurnResponse(BaseModel):
    session_id: str
    messages: list[dict]


def create_app() -> FastAPI:
    app = FastAPI(title="ContextFlow API", version="0.1.0")
    sessions = SessionManager()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/sessions", response_model=CreateSessionResponse)
    async def create_session() -> CreateSessionResponse:
        session = sessions.create()
        return CreateSessionResponse(session_id=session.session_id)

    @app.post("/sessions/{session_id}/turn", response_model=TurnResponse)
    async def append_turn(session_id: str, payload: TurnRequest) -> TurnResponse:
        session = sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")

        sessions.add_node(
            session_id,
            ContextNode(role=MessageRole.USER, content=payload.content),
        )
        return TurnResponse(session_id=session_id, messages=sessions.render_messages(session_id))

    return app


app = create_app()
