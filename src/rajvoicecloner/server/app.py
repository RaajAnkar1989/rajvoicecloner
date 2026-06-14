"""RajVoiceCloner: ElevenLabs-compatible TTS API + web studio.

Built on the RajVoiceCloner speech engine for high-quality synthesis, voice cloning,
and voice design.

Endpoint compatibility (subset of the ElevenLabs v1 API):
- POST /v1/text-to-speech/{voice_id}            full audio
- POST /v1/text-to-speech/{voice_id}/stream     chunked audio (wav/pcm)
- GET  /v1/voices                                voice library
- GET  /v1/voices/{voice_id}
- POST /v1/voices/add                            instant voice cloning
- DELETE /v1/voices/{voice_id}
- POST /v1/text-to-voice/design                  voice design preview
- POST /v1/text-to-voice/create                  save a designed voice
- GET  /v1/history, /v1/history/{id}/audio, DELETE /v1/history/{id}
- GET  /v1/models
Authentication via the ``xi-api-key`` header when an API key is configured.
"""

import base64
import json
import logging
import tempfile
import threading
import os
from pathlib import Path
from typing import Generator, Iterator

import numpy as np
import soundfile as sf
import librosa
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .agents import AgentStore, SessionManager
from .branding import APP_NAME, APP_VERSION, MODEL_DISPLAY_NAME, MODEL_ID
from .audio import OutputFormat, encode, float_to_pcm16, parse_output_format, resample, wav_stream_header
from .config import ServerConfig
from .engine import TTSEngine, log_to_stderr, split_for_pipelining
from .history import HistoryStore
from .llm import LLMClient
from .realtime import CALL_VOICES, RealtimeTTS, pick_realtime_voice
from .settings import SettingsStore, StudioSettings
from .schemas import (
    AddVoiceResponse,
    AgentCreateRequest,
    AgentResponse,
    HistoryListResponse,
    ModelResponse,
    SettingsResponse,
    SettingsUpdateRequest,
    TTSRequest,
    VoiceDesignRequest,
    VoicesListResponse,
)
from .voices import Voice, VoiceLibrary, seed_sample_transcript

logger = logging.getLogger("rajvoicecloner.server")

STATIC_DIR = Path(__file__).parent / "static"


class CacheStaticFiles(StaticFiles):
    """StaticFiles with conservative browser caching for immutable app assets."""

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if response.status_code == 200 and path != "index.html":
            response.headers.setdefault("Cache-Control", "public, max-age=3600")
        else:
            response.headers.setdefault("Cache-Control", "no-cache")
        return response


MODELS = [
    ModelResponse(
        model_id=MODEL_ID,
        name=MODEL_DISPLAY_NAME,
        description=(
            "RajVoiceCloner speech engine: voice design, instant cloning, "
            "live agents, 48 kHz output, 30+ languages."
        ),
        languages=[
            {"language_id": code, "name": name}
            for code, name in [
                ("en", "English"),
                ("zh", "Chinese"),
                ("ja", "Japanese"),
                ("ko", "Korean"),
                ("fr", "French"),
                ("de", "German"),
                ("es", "Spanish"),
                ("pt", "Portuguese"),
                ("ru", "Russian"),
                ("ar", "Arabic"),
                ("hi", "Hindi"),
                ("it", "Italian"),
                ("te", "Telugu"),
            ]
        ],
    )
]


def create_app(config: ServerConfig | None = None) -> FastAPI:
    config = config or ServerConfig()
    config.ensure_dirs()
    log_to_stderr()

    engine = TTSEngine(config)
    realtime = RealtimeTTS(config.data_dir / "realtime")
    voices = VoiceLibrary(config.voices_dir)
    history = HistoryStore(config.history_dir)
    agents = AgentStore(config.agents_dir)

    settings_store = SettingsStore(config.data_dir)
    settings = settings_store.load(
        StudioSettings(
            llm_base_url=config.llm_base_url,
            llm_model=config.llm_model,
            llm_api_key=config.llm_api_key,
        )
    )
    llm = LLMClient(settings.llm_base_url, settings.llm_model, settings.llm_api_key)
    sessions = SessionManager(agents, llm=llm)

    app = FastAPI(title=APP_NAME, version=APP_VERSION)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Turn-Meta"],
    )

    # ------------------------------------------------------------------ #
    # Auth
    # ------------------------------------------------------------------ #
    def require_api_key(
        xi_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> None:
        if config.api_key is None:
            return
        bearer = None
        if authorization and authorization.lower().startswith("bearer "):
            bearer = authorization[7:].strip()
        if xi_api_key != config.api_key and bearer != config.api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key (xi-api-key header)")

    auth = Depends(require_api_key)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def get_voice_or_404(voice_id: str) -> Voice:
        voice = voices.get(voice_id)
        if voice is None:
            raise HTTPException(status_code=404, detail=f"Voice not found: {voice_id}")
        return voice

    def ensure_voice_sample(voice: Voice) -> Voice:
        """Pin description-only voices (premade/designed) to a generated seed sample.

        One-time cost per voice; afterwards every generation reuses the cached
        encoded prompt and the voice keeps a single consistent identity.
        """
        if voice.sample_path or not voice.description:
            return voice
        transcript = seed_sample_transcript(voice.voice_id)
        wav = engine.synthesize(transcript, control=voice.description, denoise=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            seed_path = tmp.name
        try:
            sf.write(seed_path, wav, engine.sample_rate)
            return voices.attach_seed_sample(voice, Path(seed_path), transcript=transcript)
        finally:
            os.unlink(seed_path)

    def voice_generation_kwargs(voice: Voice, req: TTSRequest) -> dict:
        """Map a library voice onto RajVoiceCloner generation arguments."""
        control_parts = []
        if voice.category in ("premade", "designed") and voice.description:
            control_parts.append(voice.description)
        if req.control:
            control_parts.append(req.control)

        kwargs: dict = {
            "control": ", ".join(control_parts) or None,
            "cfg_value": req.voice_settings.resolved_cfg(),
            "inference_timesteps": req.voice_settings.resolved_timesteps(),
            "normalize": req.normalize,
            "denoise": req.denoise,
        }
        if voice.sample_path:
            kwargs["reference_wav_path"] = voice.sample_path
            if voice.transcript:
                kwargs["prompt_wav_path"] = voice.sample_path
                kwargs["prompt_text"] = voice.transcript
        return kwargs

    def save_history(text: str, audio: bytes, fmt: OutputFormat, voice: Voice | None) -> None:
        history.add(
            text=text,
            audio_bytes=audio,
            content_type=fmt.media_type,
            file_extension=fmt.file_extension,
            voice_id=voice.voice_id if voice else None,
            voice_name=voice.name if voice else None,
        )

    def stream_audio(chunks: Generator[np.ndarray, None, None], fmt: OutputFormat) -> Iterator[bytes]:
        src_rate = engine.sample_rate
        if fmt.codec == "wav":
            yield wav_stream_header(fmt.sample_rate)
        for chunk in chunks:
            yield float_to_pcm16(resample(chunk, src_rate, fmt.sample_rate))

    def transcode_upload(upload: UploadFile, dst_path: str) -> None:
        """Persist an uploaded audio file as mono wav."""
        suffix = Path(upload.filename or "sample.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(upload.file.read())
            tmp_path = tmp.name
        try:
            wav, sr = librosa.load(tmp_path, sr=None, mono=True)
            if len(wav) < int(0.5 * sr):
                raise HTTPException(status_code=400, detail="Audio sample is too short (need at least 0.5s)")
            sf.write(dst_path, wav, int(sr))
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not decode audio file: {exc}") from exc
        finally:
            os.unlink(tmp_path)

    # ------------------------------------------------------------------ #
    # Meta
    # ------------------------------------------------------------------ #
    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "app": APP_NAME,
            "model_loaded": engine.model_loaded,
            "llm_available": llm.available(),
            "llm_model": llm.model,
            "realtime_tts": realtime.available(),
        }

    @app.get("/v1/models", dependencies=[auth])
    def list_models() -> list[ModelResponse]:
        return MODELS

    @app.get("/v1/user", dependencies=[auth])
    def get_user() -> dict:
        return {
            "subscription": {"tier": "rajvoicecloner", "character_count": 0, "character_limit": -1},
            "is_new_user": False,
            "xi_api_key": "rajvoicecloner-self-hosted",
        }

    # ------------------------------------------------------------------ #
    # Settings
    # ------------------------------------------------------------------ #
    def settings_response() -> SettingsResponse:
        return SettingsResponse(
            llm_base_url=settings.llm_base_url,
            llm_model=settings.llm_model,
            llm_has_api_key=bool(settings.llm_api_key),
            llm_available=llm.available(ttl_seconds=0),
            default_voice_id=settings.default_voice_id,
        )

    @app.get("/v1/settings", dependencies=[auth])
    def get_settings() -> SettingsResponse:
        return settings_response()

    @app.put("/v1/settings", dependencies=[auth])
    def update_settings(req: SettingsUpdateRequest) -> SettingsResponse:
        if req.llm_base_url is not None:
            settings.llm_base_url = req.llm_base_url.strip() or settings.llm_base_url
        if req.llm_model is not None:
            settings.llm_model = req.llm_model.strip() or settings.llm_model
        if req.llm_api_key is not None:
            settings.llm_api_key = req.llm_api_key.strip() or None
        if req.default_voice_id is not None:
            if req.default_voice_id == "":
                settings.default_voice_id = None
            else:
                get_voice_or_404(req.default_voice_id)
                settings.default_voice_id = req.default_voice_id
        settings_store.save(settings)
        llm.configure(settings.llm_base_url, settings.llm_model, settings.llm_api_key)
        return settings_response()

    @app.get("/v1/settings/llm/models", dependencies=[auth])
    def list_llm_models() -> dict:
        try:
            models = llm.list_models()
            return {"available": True, "models": models, "error": None}
        except Exception as exc:
            return {"available": False, "models": [], "error": str(exc)}

    # ------------------------------------------------------------------ #
    # Voices
    # ------------------------------------------------------------------ #
    @app.get("/v1/voices", dependencies=[auth])
    def list_voices() -> VoicesListResponse:
        return VoicesListResponse(voices=[v.to_public_dict() for v in voices.list_voices()])

    @app.get("/v1/voices/{voice_id}", dependencies=[auth])
    def get_voice(voice_id: str) -> dict:
        return get_voice_or_404(voice_id).to_public_dict()

    @app.get("/v1/voices/{voice_id}/sample", dependencies=[auth])
    def get_voice_sample(voice_id: str) -> FileResponse:
        voice = get_voice_or_404(voice_id)
        if not voice.sample_path or not os.path.exists(voice.sample_path):
            raise HTTPException(status_code=404, detail="Voice has no audio sample")
        return FileResponse(voice.sample_path, media_type="audio/wav")

    @app.post("/v1/voices/add", dependencies=[auth])
    def add_voice(
        name: str = Form(...),
        description: str = Form(default=""),
        labels: str = Form(default="{}"),
        transcript: str = Form(default=""),
        auto_transcribe: bool = Form(default=True),
        files: list[UploadFile] = File(default=[]),
    ) -> AddVoiceResponse:
        try:
            labels_dict = json.loads(labels) if labels else {}
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="labels must be a JSON object")

        sample_path = None
        resolved_transcript = transcript.strip() or None
        if files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                sample_path = tmp.name
            transcode_upload(files[0], sample_path)
            if resolved_transcript is None and auto_transcribe:
                try:
                    resolved_transcript = engine.transcribe(sample_path) or None
                except Exception:
                    resolved_transcript = None
        elif not description.strip():
            raise HTTPException(
                status_code=400,
                detail="Provide an audio sample (cloned voice) or a description (designed voice)",
            )

        try:
            voice = voices.add(
                name=name,
                description=description.strip() or None,
                transcript=resolved_transcript,
                labels=labels_dict,
                sample_src_path=Path(sample_path) if sample_path else None,
            )
        finally:
            if sample_path and os.path.exists(sample_path):
                os.unlink(sample_path)
        return AddVoiceResponse(voice_id=voice.voice_id)

    @app.delete("/v1/voices/{voice_id}", dependencies=[auth])
    def delete_voice(voice_id: str) -> dict:
        voice = get_voice_or_404(voice_id)
        if voice.category == "premade":
            raise HTTPException(status_code=400, detail="Premade voices cannot be deleted")
        voices.delete(voice_id)
        return {"status": "ok"}

    # ------------------------------------------------------------------ #
    # Text to speech
    # ------------------------------------------------------------------ #
    @app.post("/v1/text-to-speech/{voice_id}", dependencies=[auth])
    def text_to_speech(
        voice_id: str,
        req: TTSRequest,
        output_format: str = Query(default="wav_48000"),
    ) -> Response:
        try:
            fmt = parse_output_format(output_format)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        voice = ensure_voice_sample(get_voice_or_404(voice_id))
        try:
            if voice.sample_path:
                # Library voices reuse the cached encoded sample (much faster).
                wav = engine.synthesize_with_voice(
                    req.text,
                    sample_path=voice.sample_path,
                    transcript=voice.transcript,
                    control=req.control,
                    cfg_value=req.voice_settings.resolved_cfg(),
                    inference_timesteps=req.voice_settings.resolved_timesteps(),
                )
            else:
                wav = engine.synthesize(req.text, **voice_generation_kwargs(voice, req))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        audio, actual_fmt = encode(wav, engine.sample_rate, fmt)
        save_history(req.text, audio, actual_fmt, voice)
        return Response(content=audio, media_type=actual_fmt.media_type)

    @app.post("/v1/text-to-speech/{voice_id}/stream", dependencies=[auth])
    def text_to_speech_stream(
        voice_id: str,
        req: TTSRequest,
        output_format: str = Query(default="wav_48000"),
    ) -> StreamingResponse:
        try:
            fmt = parse_output_format(output_format)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if fmt.codec == "mp3":
            # mp3 cannot be chunk-encoded with libsndfile; stream wav instead.
            fmt = OutputFormat(codec="wav", sample_rate=fmt.sample_rate)

        voice = ensure_voice_sample(get_voice_or_404(voice_id))
        if voice.sample_path:
            # Sentence-pipelined batch synthesis: ~3x faster end-to-end than
            # step-level streaming, with the first sentence arriving early.
            chunks = engine.synthesize_pipelined(
                req.text,
                sample_path=voice.sample_path,
                transcript=voice.transcript,
                control=req.control,
                cfg_value=req.voice_settings.resolved_cfg(),
                inference_timesteps=req.voice_settings.resolved_timesteps(),
            )
        else:
            chunks = engine.synthesize_streaming(req.text, **voice_generation_kwargs(voice, req))
        return StreamingResponse(stream_audio(chunks, fmt), media_type=fmt.media_type)

    # ------------------------------------------------------------------ #
    # Voice design
    # ------------------------------------------------------------------ #
    @app.post("/v1/text-to-voice/design", dependencies=[auth])
    def design_voice_preview(
        req: VoiceDesignRequest,
        output_format: str = Query(default="wav_48000"),
    ) -> Response:
        try:
            fmt = parse_output_format(output_format)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        try:
            wav = engine.synthesize(
                req.text,
                control=req.voice_description,
                cfg_value=req.voice_settings.resolved_cfg(),
                inference_timesteps=req.voice_settings.resolved_timesteps(),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        audio, actual_fmt = encode(wav, engine.sample_rate, fmt)
        save_history(req.text, audio, actual_fmt, None)
        return Response(content=audio, media_type=actual_fmt.media_type)

    @app.post("/v1/text-to-voice/create", dependencies=[auth])
    def create_designed_voice(payload: dict) -> AddVoiceResponse:
        name = (payload.get("voice_name") or payload.get("name") or "").strip()
        description = (payload.get("voice_description") or payload.get("description") or "").strip()
        if not name or not description:
            raise HTTPException(status_code=400, detail="voice_name and voice_description are required")
        voice = voices.add(name=name, description=description, labels=payload.get("labels") or {})
        return AddVoiceResponse(voice_id=voice.voice_id)

    # ------------------------------------------------------------------ #
    # History
    # ------------------------------------------------------------------ #
    @app.get("/v1/history", dependencies=[auth])
    def list_history() -> HistoryListResponse:
        return HistoryListResponse(history=[item.to_public_dict() for item in history.list_items()])

    @app.get("/v1/history/{history_item_id}/audio", dependencies=[auth])
    def get_history_audio(history_item_id: str) -> FileResponse:
        item = history.get(history_item_id)
        if item is None:
            raise HTTPException(status_code=404, detail="History item not found")
        path = history.audio_path(item)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Audio file missing")
        return FileResponse(path, media_type=item.content_type)

    @app.delete("/v1/history/{history_item_id}", dependencies=[auth])
    def delete_history(history_item_id: str) -> dict:
        if not history.delete(history_item_id):
            raise HTTPException(status_code=404, detail="History item not found")
        return {"status": "ok"}

    # ------------------------------------------------------------------ #
    # Voice agents (live questionnaire calls)
    # ------------------------------------------------------------------ #
    AGENT_STREAM_RATE = 24000

    def agent_turn_response(session, agent, utterance: str, user_transcript: str | None = None) -> StreamingResponse:
        """Stream the agent's utterance as raw PCM while it is generated.

        Turn metadata travels in the X-Turn-Meta header (base64 JSON) so the
        client can render text/progress before any audio arrives. Streaming
        means the agent starts talking after the first chunk instead of after
        the full synthesis.
        """
        voice = voices.get(agent.voice_id)
        if voice is None:
            raise HTTPException(status_code=409, detail=f"The agent's voice no longer exists: {agent.voice_id}")

        meta = {
            "session_id": session.session_id,
            "text": utterance,
            "finished": session.finished,
            "smart": session.smart,
            "questions_total": len(agent.questions),
            "questions_answered": len(session.answers),
            "user_transcript": user_transcript,
            "sample_rate": AGENT_STREAM_RATE,
        }
        def pcm_chunks() -> Iterator[bytes]:
            # Live calls use the realtime engine (Kokoro, faster than realtime
            # on CPU) unless the agent explicitly asks for its library voice
            # ("library" = RajVoiceCloner: most realistic + cloned voices, but slower).
            yielded = False
            if realtime.available() and agent.call_voice != "library":
                try:
                    kokoro_voice = agent.call_voice or pick_realtime_voice(voice.labels, seed=voice.voice_id)
                    # Kokoro outruns playback, so split only per sentence group:
                    # fewer joins sounds more natural.
                    for piece in split_for_pipelining(utterance, max_chars=400, cut_first=False):
                        wav = realtime.synthesize(piece, voice=kokoro_voice)
                        yield float_to_pcm16(resample(wav, realtime.sample_rate, AGENT_STREAM_RATE))
                        yielded = True
                    return
                except Exception as exc:
                    logger.warning("Realtime TTS failed (%s); falling back to RajVoiceCloner", exc)
                    if yielded:
                        return
            # Seed-sample creation (first ever use of a premade/designed voice)
            # runs inside the stream so the call is picked up immediately.
            v = ensure_voice_sample(voice)
            for chunk in engine.synthesize_pipelined(
                utterance,
                sample_path=v.sample_path,
                transcript=v.transcript,
                control=None if v.sample_path else v.description,
                inference_timesteps=6,
            ):
                yield float_to_pcm16(resample(chunk, engine.sample_rate, AGENT_STREAM_RATE))

        headers = {
            "X-Turn-Meta": base64.b64encode(json.dumps(meta, ensure_ascii=False).encode("utf-8")).decode("ascii"),
            "Cache-Control": "no-store",
        }
        return StreamingResponse(pcm_chunks(), media_type="application/octet-stream", headers=headers)

    def get_agent_or_404(agent_id: str):
        agent = agents.get(agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
        return agent

    def get_session_or_404(session_id: str):
        session = sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Call session not found: {session_id}")
        return session

    @app.get("/v1/agents", dependencies=[auth])
    def list_agents() -> list[AgentResponse]:
        return [AgentResponse(**vars(a)) for a in agents.list_agents()]

    # NOTE: declared before /v1/agents/{agent_id} so "call-voices" isn't
    # swallowed by the path parameter.
    @app.get("/v1/agents/call-voices", dependencies=[auth])
    def list_call_voices() -> dict:
        return {"available": realtime.available(), "voices": CALL_VOICES}

    def validate_call_voice(call_voice: str) -> str:
        # "library" = speak with the agent's library voice via RajVoiceCloner
        # (most realistic, supports cloned voices, but slower than realtime).
        call_voice = (call_voice or "").strip()
        if call_voice and call_voice != "library" and call_voice not in CALL_VOICES:
            raise HTTPException(status_code=400, detail=f"Unknown call voice: {call_voice}")
        return call_voice

    @app.post("/v1/agents", dependencies=[auth])
    def create_agent(req: AgentCreateRequest) -> AgentResponse:
        get_voice_or_404(req.voice_id)
        questions = [q.strip() for q in req.questions if q.strip()]
        if not questions:
            raise HTTPException(status_code=400, detail="The questionnaire needs at least one question")
        agent = agents.add(
            name=req.name.strip(),
            prompt=req.prompt.strip(),
            questions=questions,
            voice_id=req.voice_id,
            closing=(req.closing or "").strip() or None,
            call_voice=validate_call_voice(req.call_voice),
        )
        return AgentResponse(**vars(agent))

    @app.get("/v1/agents/{agent_id}", dependencies=[auth])
    def get_agent(agent_id: str) -> AgentResponse:
        return AgentResponse(**vars(get_agent_or_404(agent_id)))

    @app.put("/v1/agents/{agent_id}", dependencies=[auth])
    def update_agent(agent_id: str, req: AgentCreateRequest) -> AgentResponse:
        agent = get_agent_or_404(agent_id)
        get_voice_or_404(req.voice_id)
        questions = [q.strip() for q in req.questions if q.strip()]
        if not questions:
            raise HTTPException(status_code=400, detail="The questionnaire needs at least one question")
        agent = agents.update(
            agent,
            name=req.name.strip(),
            prompt=req.prompt.strip(),
            questions=questions,
            voice_id=req.voice_id,
            closing=(req.closing or "").strip() or None,
            call_voice=validate_call_voice(req.call_voice),
        )
        return AgentResponse(**vars(agent))

    @app.delete("/v1/agents/{agent_id}", dependencies=[auth])
    def delete_agent(agent_id: str) -> dict:
        if not agents.delete(agent_id):
            raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
        return {"status": "ok"}

    @app.get("/v1/agents/{agent_id}/sessions", dependencies=[auth])
    def list_agent_sessions(agent_id: str) -> list[dict]:
        get_agent_or_404(agent_id)
        return agents.list_saved_sessions(agent_id)

    @app.post("/v1/agents/{agent_id}/call", dependencies=[auth])
    def start_call(agent_id: str) -> StreamingResponse:
        agent = get_agent_or_404(agent_id)
        session, opening = sessions.start(agent)
        return agent_turn_response(session, agent, opening)

    @app.post("/v1/agents/calls/{session_id}/reply", dependencies=[auth])
    def call_reply(session_id: str, audio: UploadFile = File(...)) -> StreamingResponse:
        session = get_session_or_404(session_id)
        if session.finished:
            raise HTTPException(status_code=400, detail="This call has already ended")
        agent = get_agent_or_404(session.agent_id)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            answer_path = tmp.name
        try:
            transcode_upload(audio, answer_path)
            transcript = engine.transcribe(answer_path).strip()
        finally:
            if os.path.exists(answer_path):
                os.unlink(answer_path)

        if not transcript:
            return agent_turn_response(
                session, agent, "Sorry, I didn't catch that. Could you say it again?", user_transcript=""
            )

        utterance = sessions.record_answer(session, agent, transcript)
        return agent_turn_response(session, agent, utterance, user_transcript=transcript)

    @app.get("/v1/agents/calls/{session_id}", dependencies=[auth])
    def get_call(session_id: str) -> dict:
        session = get_session_or_404(session_id)
        return session.to_public_dict(get_agent_or_404(session.agent_id))

    @app.post("/v1/agents/calls/{session_id}/end", dependencies=[auth])
    def end_call(session_id: str) -> dict:
        session = get_session_or_404(session_id)
        agent = get_agent_or_404(session.agent_id)
        sessions.end(session, agent)
        return session.to_public_dict(agent)

    # ------------------------------------------------------------------ #
    # Frontend
    # ------------------------------------------------------------------ #
    if STATIC_DIR.is_dir():
        app.mount("/", CacheStaticFiles(directory=str(STATIC_DIR), html=True), name="studio")

    @app.exception_handler(Exception)
    def unhandled_exception_handler(request, exc):
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    if config.preload:
        engine.get_model()
    # Load + warm-up in the background so the first request pays neither the
    # model load nor the one-off first-generation kernel compilation.
    threading.Thread(target=engine.warm_up, daemon=True, name="rajvoicecloner-warmup").start()

    return app
