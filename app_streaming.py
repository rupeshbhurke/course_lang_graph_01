"""
app_streaming_redis.py
----------------------
FastAPI app with:
- Real-time Ollama streaming (SSE)
- Persistent chat memory via Redis
- Multiple conversation sessions per user
"""

import os
import json
import uuid
import httpx
import redis
from typing import AsyncGenerator, Optional, List, Dict
from fastapi import FastAPI, Request, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------------
load_dotenv()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# ---------------------------------------------------------------------------
# Initialize app + Redis connection
# ---------------------------------------------------------------------------
app = FastAPI(title="Ollama Streaming Chat (Redis Memory)")
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ---------------------------------------------------------------------------
# Stream NDJSON from Ollama
# ---------------------------------------------------------------------------
async def stream_from_ollama(prompt: str, context: Optional[List[int]] = None) -> AsyncGenerator[str, None]:
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": True}
    if context:
        payload["context"] = context

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", f"{OLLAMA_BASE_URL}/api/generate", json=payload) as rstream:
            async for line in rstream.aiter_lines():
                if not line.strip():
                    continue
                yield line

# ---------------------------------------------------------------------------
# Helper functions for Redis chat persistence
# ---------------------------------------------------------------------------
def get_context(session_id: str) -> Optional[List[int]]:
    val = r.get(f"context:{session_id}")
    return json.loads(val) if val else None

def set_context(session_id: str, context: List[int]):
    r.set(f"context:{session_id}", json.dumps(context))

def append_message(session_id: str, role: str, content: str):
    msg = json.dumps({"role": role, "content": content})
    r.rpush(f"session:{session_id}", msg)

def get_conversation(session_id: str) -> List[Dict]:
    msgs = r.lrange(f"session:{session_id}", 0, -1)
    return [json.loads(m) for m in msgs]

def list_conversations() -> List[str]:
    keys = r.keys("session:*")
    return [k.split(":")[1] for k in keys]

# ---------------------------------------------------------------------------
# Stream endpoint
# ---------------------------------------------------------------------------
@app.post("/chat/stream")
async def chat_stream(request: Request, session_id: str = Query(...)):
    """
    Streams a conversation for a given session_id (SSE).
    """
    body = await request.json()
    prompt = body.get("message", "").strip()
    if not prompt:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    # Save user message
    append_message(session_id, "user", prompt)
    existing_context = get_context(session_id)

    async def event_generator():
        async for chunk in stream_from_ollama(prompt, existing_context):
            try:
                data = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            yield f"data: {json.dumps(data)}\n\n"

            if data.get("done"):
                new_context = data.get("context")
                if new_context:
                    set_context(session_id, new_context)
                # Store assistant message at completion
                append_message(session_id, "assistant", data.get("response", ""))

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# ---------------------------------------------------------------------------
# New conversation
# ---------------------------------------------------------------------------
@app.post("/new_conversation")
async def new_conversation():
    session_id = str(uuid.uuid4())
    r.delete(f"session:{session_id}", f"context:{session_id}")
    append_message(session_id, "system", "New conversation started.")
    return {"session_id": session_id}

# ---------------------------------------------------------------------------
# List conversations
# ---------------------------------------------------------------------------
@app.get("/conversations")
async def conversations():
    sessions = list_conversations()
    return {"sessions": sessions}

# ---------------------------------------------------------------------------
# Fetch entire conversation (history)
# ---------------------------------------------------------------------------
@app.get("/conversation/{session_id}")
async def conversation_history(session_id: str):
    return {"session_id": session_id, "messages": get_conversation(session_id)}

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "model": OLLAMA_MODEL, "conversations": len(list_conversations())}

# ---------------------------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------------------------
app.mount("/", StaticFiles(directory="static", html=True), name="static")
