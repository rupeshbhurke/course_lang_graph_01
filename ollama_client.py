import httpx
import os
from dotenv import load_dotenv
from typing import List, Dict, Optional

# Load environment variables from .env
load_dotenv()

class OllamaClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "phi3")
        self.stream_default = os.getenv("OLLAMA_STREAM", "false").lower() == "true"

    def generate(
        self,
        prompt: str,
        context: Optional[List[int]] = None,
        stream: Optional[bool] = None
    ) -> Dict:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": self.stream_default if stream is None else stream,
        }
        if context:
            payload["context"] = context

        response = httpx.post(f"{self.base_url}/api/generate", json=payload, timeout=None)
        response.raise_for_status()
        return response.json()
