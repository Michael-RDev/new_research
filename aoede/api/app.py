from __future__ import annotations

import asyncio
from functools import lru_cache

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

from aoede.config import default_config
from aoede.schemas import SynthesisRequest, VoiceDesignRequest
from aoede.service import build_service


@lru_cache(maxsize=1)
def get_service():
    return build_service(default_config())


def create_app():
    config = default_config()
    app = FastAPI(title="Aoede", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.service.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/v1/health")
    async def health():
        return get_service().health()

    @app.get("/api/v1/languages")
    async def languages():
        return get_service().list_languages()

    @app.get("/api/v1/voices")
    async def voices():
        return get_service().list_voices()

    @app.post("/api/v1/voices/enroll")
    async def enroll(request: Request, voice_id: str = None):
        filename = "upload.wav"
        content_type = request.headers.get("content-type", "")
        if "multipart/form-data" in content_type:
            try:
                form = await request.form()
            except Exception as exc:
                return JSONResponse({"error": f"multipart parsing unavailable: {exc}"}, status_code=500)
            upload = form.get("audio")
            if upload is None:
                return JSONResponse({"error": "missing audio field"}, status_code=400)
            if voice_id is None:
                voice_id = form.get("voice_id")
            payload = await upload.read()
            filename = getattr(upload, "filename", filename)
        else:
            payload = await request.body()
        if not payload:
            return JSONResponse({"error": "empty audio payload"}, status_code=400)
        response = get_service().enroll(payload, voice_id=voice_id, metadata={"filename": filename})
        return response

    @app.post("/api/v1/voices/design")
    async def design_voice(request: VoiceDesignRequest):
        return get_service().design_voice(request)

    @app.post("/api/v1/synthesis")
    async def synthesize(request: SynthesisRequest):
        try:
            audio_bytes, duration = get_service().synthesize(request)
        except FileNotFoundError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        return StreamingResponse(
            iter([audio_bytes]),
            media_type="audio/wav",
            headers={"X-Duration-Seconds": str(round(duration, 3))},
        )

    @app.websocket("/api/v1/synthesis/stream")
    async def synthesize_stream(ws: WebSocket):
        await ws.accept()
        try:
            payload = await ws.receive_json()
            request = SynthesisRequest.model_validate(payload)
            for event in get_service().stream_synthesis(request):
                await ws.send_json(event.model_dump())
                await asyncio.sleep(0)
        except WebSocketDisconnect:
            return
        except Exception as exc:
            await ws.send_json({"type": "error", "stage": "failed", "progress": 1.0, "payload": {"message": str(exc)}})
        finally:
            await ws.close()

    return app


app = create_app()


def run():
    config = default_config()
    uvicorn.run("aoede.api.app:app", host=config.service.host, port=config.service.port, reload=False)


if __name__ == "__main__":
    run()
