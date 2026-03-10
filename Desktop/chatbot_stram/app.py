from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from queue import Queue
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from functools import lru_cache
from typing import Any, Iterator, Optional

import edge_tts
import google.generativeai as genai
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from redis import Redis
from redis.exceptions import RedisError

load_dotenv()

DEFAULT_MODEL = os.getenv("MODEL_NAME", "gemini-2.5-flash")
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
RESPONSE_WORD_LIMIT = int(os.getenv("RESPONSE_WORD_LIMIT", "100"))
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
INPUT_COST_PER_1M_USD = float(os.getenv("INPUT_COST_PER_1M_USD", "0.30"))
OUTPUT_COST_PER_1M_USD = float(os.getenv("OUTPUT_COST_PER_1M_USD", "0.60"))
MODEL_TIMEOUT_SECONDS = int(os.getenv("MODEL_TIMEOUT_SECONDS", "30"))
MODEL_MAX_RETRIES = int(os.getenv("MODEL_MAX_RETRIES", "1"))
TTS_DEFAULT_VOICE = os.getenv("TTS_DEFAULT_VOICE", "en-US-AriaNeural")
TTS_DEFAULT_RATE = os.getenv("TTS_DEFAULT_RATE", "+2%")
TTS_DEFAULT_PITCH = os.getenv("TTS_DEFAULT_PITCH", "+0Hz")

app = Flask(__name__)
MODEL_EXECUTOR = ThreadPoolExecutor(max_workers=2)
TTS_VOICES = [
    {"id": "en-US-AriaNeural", "name": "Aria (US, Female)"},
    {"id": "en-US-JennyNeural", "name": "Jenny (US, Female)"},
    {"id": "en-US-GuyNeural", "name": "Guy (US, Male)"},
    {"id": "en-GB-SoniaNeural", "name": "Sonia (UK, Female)"},
    {"id": "en-GB-RyanNeural", "name": "Ryan (UK, Male)"},
    {"id": "en-IN-NeerjaNeural", "name": "Neerja (India, Female)"},
    {"id": "en-IN-PrabhatNeural", "name": "Prabhat (India, Male)"},
]
TTS_VOICE_IDS = {voice["id"] for voice in TTS_VOICES}


@lru_cache(maxsize=1)
def get_model() -> genai.GenerativeModel:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is missing. Set it in your .env file.")

    genai.configure(api_key=api_key)
    return genai.GenerativeModel(DEFAULT_MODEL)


@lru_cache(maxsize=1)
def get_redis_client() -> Optional[Redis]:
    try:
        client = Redis.from_url(REDIS_URL, decode_responses=True, socket_timeout=2)
        client.ping()
        return client
    except RedisError:
        return None


def build_cache_key(query: str, word_limit: int) -> str:
    digest = hashlib.sha256(query.strip().encode("utf-8")).hexdigest()
    return f"query_bot:response:{word_limit}:{digest}"


def get_cached_response(query: str, word_limit: int) -> Optional[dict[str, Any]]:
    client = get_redis_client()
    if client is None:
        return None

    try:
        raw_value = client.get(build_cache_key(query, word_limit))
        if raw_value is None:
            return None

        try:
            payload = json.loads(raw_value)
            if not isinstance(payload, dict):
                return None
            if "response" not in payload:
                return None
            return payload
        except json.JSONDecodeError:
            # Backward compatibility for old cache entries that stored plain text.
            return {
                "response": raw_value,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0,
            }
    except RedisError:
        return None


def save_cached_response(
    query: str,
    word_limit: int,
    response_text: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    estimated_cost_usd: float,
) -> None:
    client = get_redis_client()
    if client is None:
        return

    try:
        payload = {
            "response": response_text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": estimated_cost_usd,
        }
        client.setex(
            build_cache_key(query, word_limit),
            CACHE_TTL_SECONDS,
            json.dumps(payload),
        )
    except RedisError:
        return


def enforce_word_limit(text: str, word_limit: int) -> str:
    words = text.split()
    if len(words) <= word_limit:
        return text
    return " ".join(words[:word_limit])


def extract_usage_and_cost(response: Any) -> tuple[int, int, int, float]:
    usage_meta = getattr(response, "usage_metadata", None)
    prompt_tokens = int(getattr(usage_meta, "prompt_token_count", 0) or 0)
    completion_tokens = int(getattr(usage_meta, "candidates_token_count", 0) or 0)
    total_tokens = int(
        getattr(usage_meta, "total_token_count", prompt_tokens + completion_tokens)
        or (prompt_tokens + completion_tokens)
    )
    estimated_cost_usd = (
        (prompt_tokens * INPUT_COST_PER_1M_USD)
        + (completion_tokens * OUTPUT_COST_PER_1M_USD)
    ) / 1_000_000
    return prompt_tokens, completion_tokens, total_tokens, round(estimated_cost_usd, 8)


def normalize_tts_text(text: str) -> str:
    # Clean markdown and add punctuation pauses for more natural speech cadence.
    cleaned = re.sub(r"`|[*_>#]", "", text)
    cleaned = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", cleaned)
    cleaned = re.sub(r"https?://\S+", "", cleaned)
    cleaned = cleaned.replace("\n", ". ")
    cleaned = re.sub(r"\s*[:;]\s*", ". ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def stream_tts_audio(
    text: str,
    voice: str,
    rate: str,
    pitch: str,
) -> Iterator[bytes]:
    queue: Queue[bytes | None] = Queue()

    async def run_tts() -> None:
        communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                queue.put(chunk["data"])
        queue.put(None)

    def worker() -> None:
        try:
            asyncio.run(run_tts())
        except Exception:
            queue.put(None)

    Thread(target=worker, daemon=True).start()

    while True:
        item = queue.get()
        if item is None:
            break
        yield item


def generate_response(query: str) -> dict[str, Any]:
    clean_query = query.strip()
    if not clean_query:
        return {
            "response": "Please enter a query.",
            "cached": False,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
        }

    cached = get_cached_response(clean_query, RESPONSE_WORD_LIMIT)
    if cached is not None:
        return {
            "response": enforce_word_limit(str(cached.get("response", "")), RESPONSE_WORD_LIMIT),
            "cached": True,
            "prompt_tokens": int(cached.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(cached.get("completion_tokens", 0) or 0),
            "total_tokens": int(cached.get("total_tokens", 0) or 0),
            "estimated_cost_usd": float(cached.get("estimated_cost_usd", 0.0) or 0.0),
        }

    limited_prompt = (
        f"{clean_query}\n\n"
        f"Instruction: Respond in at most {RESPONSE_WORD_LIMIT} words."
    )
    response: Any | None = None
    last_timeout_exc: Exception | None = None
    for _ in range(MODEL_MAX_RETRIES + 1):
        future = MODEL_EXECUTOR.submit(get_model().generate_content, limited_prompt)
        try:
            response = future.result(timeout=MODEL_TIMEOUT_SECONDS)
            break
        except FutureTimeoutError as exc:
            last_timeout_exc = exc
            future.cancel()
            continue

    if response is None:
        raise TimeoutError(
            f"Model request timed out after {MODEL_TIMEOUT_SECONDS} seconds "
            f"(retries: {MODEL_MAX_RETRIES})."
        ) from last_timeout_exc

    response_text = enforce_word_limit(response.text or "", RESPONSE_WORD_LIMIT)
    prompt_tokens, completion_tokens, total_tokens, estimated_cost_usd = (
        extract_usage_and_cost(response)
    )

    if response_text:
        save_cached_response(
            clean_query,
            RESPONSE_WORD_LIMIT,
            response_text,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            estimated_cost_usd,
        )

    return {
        "response": response_text,
        "cached": False,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": estimated_cost_usd,
    }


@app.get("/")
def index() -> str:
    redis_status = "Connected" if get_redis_client() is not None else "Unavailable"
    return render_template(
        "chat.html",
        word_limit=RESPONSE_WORD_LIMIT,
        model_name=DEFAULT_MODEL,
        cache_ttl=CACHE_TTL_SECONDS,
        redis_status=redis_status,
    )


@app.post("/api/chat")
def chat() -> tuple[dict, int]:
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", ""))

    if not message.strip():
        return jsonify({"error": "Message is required."}), 400

    try:
        result = generate_response(message)
        return jsonify(
            {
                "response": result["response"],
                "cached": result["cached"],
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["total_tokens"],
                "estimated_cost_usd": result["estimated_cost_usd"],
                "word_limit": RESPONSE_WORD_LIMIT,
            }
        ), 200
    except TimeoutError as exc:
        return jsonify({"error": str(exc)}), 504
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Request failed: {exc}"}), 500


@app.get("/api/voices")
def voices() -> tuple[dict, int]:
    return jsonify({"voices": TTS_VOICES, "default_voice": TTS_DEFAULT_VOICE}), 200


@app.post("/api/tts")
def tts() -> Response | tuple[dict, int]:
    payload = request.get_json(silent=True) or {}
    text = normalize_tts_text(str(payload.get("text", "")))
    voice = str(payload.get("voice", TTS_DEFAULT_VOICE)).strip() or TTS_DEFAULT_VOICE
    rate = str(payload.get("rate", TTS_DEFAULT_RATE)).strip() or TTS_DEFAULT_RATE
    pitch = str(payload.get("pitch", TTS_DEFAULT_PITCH)).strip() or TTS_DEFAULT_PITCH

    if not text:
        return jsonify({"error": "Text is required for TTS."}), 400
    if voice not in TTS_VOICE_IDS:
        return jsonify({"error": "Unsupported voice selected."}), 400

    generator = stream_with_context(stream_tts_audio(text=text, voice=voice, rate=rate, pitch=pitch))
    return Response(
        generator,
        mimetype="audio/mpeg",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
def health() -> tuple[dict, int]:
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=False)
