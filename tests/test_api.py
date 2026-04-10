import io
import os
import asyncio
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from starlette.requests import Request

os.environ["AOEDE_DISABLE_TORCH"] = "1"

from aoede.api.app import create_app, get_service
from aoede.schemas import SynthesisRequest, VoiceDesignRequest


def make_wav_bytes(duration_s: float = 1.0, sample_rate: int = 24000):
    t = np.linspace(0.0, duration_s, num=int(sample_rate * duration_s), endpoint=False)
    signal = 0.2 * np.sin(2.0 * np.pi * 220.0 * t)
    pcm = (signal * 32767.0).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return buffer.getvalue()


@pytest.fixture()
def app(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    get_service.cache_clear()
    return create_app()


def route_for(app: Any, path: str, method: str = "GET"):
    for route in app.routes:
        if getattr(route, "path", None) == path and method in getattr(route, "methods", set()):
            return route.endpoint
    raise AssertionError(f"missing route {method} {path}")


async def read_streaming_response(response: Any):
    chunks = []
    iterator = response.body_iterator
    if hasattr(iterator, "__aiter__"):
        async for chunk in iterator:
            chunks.append(chunk)
    else:
        chunks.extend(iterator)
    return b"".join(chunks)


def make_request(path: str, body: bytes, content_type: str = "audio/wav"):
    state = {"sent": False}

    async def receive():
        if state["sent"]:
            return {"type": "http.disconnect"}
        state["sent"] = True
        return {"type": "http.request", "body": body, "more_body": False}

    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": [(b"content-type", content_type.encode("utf-8"))],
        "query_string": b"",
    }
    return Request(scope, receive)


def test_health(app):
    endpoint = route_for(app, "/api/v1/health")
    response = asyncio.run(endpoint())
    assert response.runtime == "mock"
    assert response.status == "ready"


def test_enroll_design_and_synthesize(app):
    health = asyncio.run(route_for(app, "/api/v1/health")())
    assert health.runtime == "mock"

    wav_bytes = make_wav_bytes()
    enroll = asyncio.run(
        route_for(app, "/api/v1/voices/enroll", "POST")(
            request=make_request("/api/v1/voices/enroll", wav_bytes),
            voice_id="api-test",
        )
    )
    assert enroll.voice_id == "api-test"

    design = asyncio.run(
        route_for(app, "/api/v1/voices/design", "POST")(
            VoiceDesignRequest(
                voice_id="designed-test",
                preset="bright",
                source_voice_id="api-test",
                style_controls={"pitch": 1.1, "pace": 0.95, "energy": 1.05, "brightness": 1.2},
            )
        )
    )
    assert design.voice_id == "designed-test"

    synth_response = asyncio.run(
        route_for(app, "/api/v1/synthesis", "POST")(
            SynthesisRequest(
                text="hello multilingual world",
                language_code="en",
                voice_id="designed-test",
                style_controls={"pitch": 1.0, "pace": 1.0, "energy": 1.0, "brightness": 1.0},
            )
        )
    )
    body = asyncio.run(read_streaming_response(synth_response))
    assert synth_response.media_type == "audio/wav"
    assert float(synth_response.headers["X-Duration-Seconds"]) > 0
    assert len(body) > 44


def test_streaming_generator(app):
    wav_bytes = make_wav_bytes()
    asyncio.run(
        route_for(app, "/api/v1/voices/enroll", "POST")(
            request=make_request("/api/v1/voices/enroll", wav_bytes),
            voice_id="stream-test",
        )
    )
    request = SynthesisRequest(
        text="stream this",
        language_code="en",
        voice_id="stream-test",
        style_controls={"pitch": 1.0, "pace": 1.0, "energy": 1.0, "brightness": 1.0},
        stream=True,
    )
    events = list(get_service().stream_synthesis(request))
    assert any(event.type == "audio_chunk" for event in events)
    assert events[-1].type == "done"
