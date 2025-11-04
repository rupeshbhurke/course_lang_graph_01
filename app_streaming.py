"""
app_streaming.py
----------------
FastAPI app that serves a local web UI (index.html)
and exposes a streaming chat endpoint using your
Ollama + LangGraph backend, with session memory.

Features:
- Real-time token streaming via SSE
- Session-based context memory
- Static HTML frontend served under /
"""

import os
import json
import uuid
import httpx
from typing import AsyncGenerator, Optional, List, Dict
from fastapi import FastAPI, Request, Query
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------------
load_dotenv()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3")

# ---------------------------------------------------------------------------
# Initialize app and session storage
# ---------------------------------------------------------------------------
app = FastAPI(title="LangGraph + Ollama Streaming Chat API")
session_contexts: Dict[str, List[int]] = {}  # in-memory session store

# ---------------------------------------------------------------------------
# Stream NDJSON from Ollama
# ---------------------------------------------------------------------------
async def stream_from_ollama(prompt: str, context: Optional[List[int]] = None) -> AsyncGenerator[str, None]:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True,
    }
    if context:
        payload["context"] = context

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", f"{OLLAMA_BASE_URL}/api/generate", json=payload) as r:
            async for line in r.aiter_lines():
                if not line.strip():
                    continue
                yield line  # raw NDJSON line

# ---------------------------------------------------------------------------
# /chat/stream endpoint with session context
# ---------------------------------------------------------------------------
@app.post("/chat/stream")
async def chat_stream(request: Request, session_id: str = Query(...)):
    """
    Expects:
    - Query param: ?session_id=<uuid>
    - JSON body: { "message": "Hello!" }

    Streams model output with per-session context memory.
    """
    body = await request.json()
    prompt = body.get("message", "")
    existing_context = session_contexts.get(session_id)

    async def event_generator():
        async for chunk in stream_from_ollama(prompt, existing_context):
            try:
                data = json.loads(chunk)
            except json.JSONDecodeError:
                continue

            # Send the raw event to the client
            yield f"data: {json.dumps(data)}\n\n"

            # Update session context if finished
            if data.get("done"):
                new_context = data.get("context")
                if new_context:
                    session_contexts[session_id] = new_context

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "model": OLLAMA_MODEL, "base_url": OLLAMA_BASE_URL, "sessions": len(session_contexts)}

# ---------------------------------------------------------------------------
# Static frontend (index.html)
# ---------------------------------------------------------------------------
app.mount("/", StaticFiles(directory="static", html=True), name="static")
