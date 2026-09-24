"""Claude Opus 5.5 backend: ask the model for the option letter through the local fcc proxy.

The Anthropic API exposes no logits, so this is a categorical choice, not a
logit readout: probabilities are one-hot and option_logits is null.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request

from .core import LETTERS, digest, direct_messages

PROMPT_VERSION = "direct-options-v1-anthropic"
DEFAULT_MODEL = "claude-opus-5-5"
BASE_URL = os.environ.get("SEMIF_ANTHROPIC_BASE_URL", "http://127.0.0.1:8082")
API_KEY = os.environ.get("SEMIF_ANTHROPIC_API_KEY") or os.environ.get("JCODE_PROVIDER_FCC_API_KEY", "freecc")


def load_model(source: str | None):
    model = source or DEFAULT_MODEL
    return model, None, {"source": model, "backend": "anthropic", "endpoint": BASE_URL}


def _ask(model: str, messages: list[dict]) -> str:
    # Opus 5.5 may think adaptively before answering; thinking tokens count
    # against max_tokens, so a tiny cap returns only an empty thinking block.
    body = {"model": model, "max_tokens": 2048, "system": messages[0]["content"],
            "messages": [{"role": "user", "content": messages[1]["content"]}]}
    request = urllib.request.Request(
        BASE_URL.rstrip("/") + "/v1/messages", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "x-api-key": API_KEY, "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(request, timeout=120) as response:
        reply = json.load(response)
    return "".join(block.get("text", "") for block in reply.get("content", []) if block.get("type") == "text")


def score(model, tokenizer, row: dict, metadata: dict, max_tokens: int = 4096) -> dict:
    started = time.perf_counter()
    messages = direct_messages(row)
    prompt_hash = digest(json.dumps(messages, ensure_ascii=False))
    text = _ask(model, messages).strip()
    letters = LETTERS[:len(row["options"])]
    if not text or text[0] not in letters:
        raise ValueError(f"Row {row['id']}: model answered {text!r}, expected one of {list(letters)}")
    choice = letters.index(text[0])
    return {
        "id": row["id"],
        "option_ids": [option["id"] for option in row["options"]],
        "probabilities": [1.0 if index == choice else 0.0 for index in range(len(letters))],
        "option_logits": None,
        "total_seconds": time.perf_counter() - started,
        "prompt_sha256": prompt_hash,
        "prompt_version": PROMPT_VERSION,
        "model": metadata,
        "readout": "generated option letter (API exposes no logits)",
        "probability_status": "one-hot categorical choice; not a probability",
    }
