"""FastAPI entry point: ``uvicorn app.main:app``."""
from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool

from .llm import AnswerLLM, GroqLLM
from .schemas import AskRequest, AskResponse, HealthResponse
from .service import AskService
from .settings import AppSettings, load_app_settings


def create_app(service: AskService | None = None, settings: AppSettings | None = None,
               llm: AnswerLLM | None = None, warm: bool = True) -> FastAPI:
    """App factory. Tests inject a service with a mocked provider."""
    settings = settings or load_app_settings()
    svc = service or AskService(settings, llm or GroqLLM(settings))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Load the index and the models once, off the event loop. A missing index
        # does not stop the server; /api/health reports it.
        await run_in_threadpool(svc.load, warm)
        yield

    app = FastAPI(
        title="Passport Seva Application Guide API", version="0.2.0", lifespan=lifespan,
        description="Answers questions about adult, in-India, ordinary-passport, Normal-process "
                    "applications from indexed official Passport Seva sources, with citations. "
                    "Do not send passport numbers, ARNs or documents.")
    app.state.service = svc

    # The browser build uses the Vite proxy (same origin), so it needs no CORS. The
    # Capacitor Android WebView is served from http(s)://localhost and calls this API
    # cross-origin. Override with CORS_ALLOW_ORIGINS (comma separated) if needed.
    origins = [o.strip() for o in os.environ.get(
        "CORS_ALLOW_ORIGINS",
        "http://localhost,https://localhost,capacitor://localhost,http://localhost:5173",
    ).split(",") if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type", "Accept"])

    @app.get("/", tags=["system"], summary="API overview")
    async def root() -> dict[str, str]:
        return {
            "name": "Passport Seva Application Guide API",
            "version": "0.2.0",
            "docs": "/docs",
            "health": "/api/health",
            "ask": "/api/ask",
        }

    @app.get("/api/health", response_model=HealthResponse, tags=["system"],
             summary="Backend, index and Groq configuration status (never calls Groq)")
    async def health() -> HealthResponse:
        return svc.health()

    @app.post(
        "/api/ask", response_model=AskResponse, tags=["ask"],
        summary="Ask a question; returns an answer with verified citations or a routed response",
        responses={
            422: {"description": "Blank, over-long or otherwise invalid request."},
            429: {"model": AskResponse, "description": "Groq rate limit (status=error, error.code=groq_rate_limited)."},
            502: {"model": AskResponse, "description": "Groq provider failure."},
            503: {"model": AskResponse, "description": "Groq not configured or key rejected, or index unavailable."},
            504: {"model": AskResponse, "description": "Groq timeout."},
        })
    async def ask(req: AskRequest, response: Response) -> AskResponse:
        if len(req.question) > svc.s.max_question_chars:
            from fastapi import HTTPException
            raise HTTPException(422, f"question exceeds {svc.s.max_question_chars} characters")
        # Retrieval, reranking and the Groq call are blocking: run them in a worker thread.
        result, http_status = await asyncio.to_thread(svc.ask, req.question, req.mode)
        response.status_code = http_status
        return result

    return app


app = create_app()

