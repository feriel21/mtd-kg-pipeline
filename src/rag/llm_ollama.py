"""
rag/llm_ollama.py — FIXED
═════════════════════════

Fix: Use Ollama HTTP API with proper chat format instead of subprocess with raw tags.
"""

import json
import requests
from typing import Optional


def ollama_chat(
    model: str,
    system: str,
    user: str,
    temperature: float = 0.0,
    base_url: str = "http://localhost:11434"
) -> str:
    """
    Call Ollama LLM via HTTP API with proper chat format.

    Args:
        model: Ollama model name (e.g., "qwen2.5:7b-instruct")
        system: System prompt
        user: User prompt
        temperature: Sampling temperature (0.0 = greedy)
        base_url: Ollama server URL

    Returns:
        Generated text
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 450,
        }
    }

    try:
        resp = requests.post(
            f"{base_url}/api/chat",
            json=payload,
            timeout=120
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "").strip()
    except requests.ConnectionError:
        raise RuntimeError(
            f"Cannot connect to Ollama at {base_url}. "
            "Is Ollama running? Start with: ollama serve"
        )
    except requests.HTTPError as e:
        raise RuntimeError(f"Ollama HTTP error: {e}")
    except Exception as e:
        raise RuntimeError(f"Ollama error: {e}")