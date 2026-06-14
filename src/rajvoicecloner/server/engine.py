"""Model lifecycle + synthesis for the server.

A single RajVoiceCloner model instance is shared across requests; generation is
serialized with a lock since the underlying model is not thread-safe.
"""

import logging
import os
import re
import sys
import threading
import time
from collections import OrderedDict
from typing import Generator, Optional

import numpy as np
import torch

from ..core import RajVoiceCloner
from ..model.rajvoicecloner2 import RajVoiceCloner2Model
from .config import ServerConfig

logger = logging.getLogger("rajvoicecloner.server")

_PARENS_RE = re.compile(r"[()（）]")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?。！？])\s+")


def build_final_text(text: str, control: str | None) -> str:
    control = _PARENS_RE.sub("", (control or "")).strip()
    return f"({control}){text}" if control else text


def split_for_pipelining(text: str, max_chars: int = 200, min_chars: int = 12, cut_first: bool = True) -> list[str]:
    """Split text at sentence boundaries into pieces suitable for pipelined synthesis.

    The first piece is kept short so the first audio arrives quickly; tiny
    trailing fragments are merged so we don't synthesize one-word pieces.
    ``cut_first=False`` skips the mid-sentence comma cut for engines fast
    enough that prosody continuity matters more than first-chunk latency.
    """
    text = re.sub(r"\s+", " ", text).strip()
    sentences = [s for s in _SENTENCE_END_RE.split(text) if s]
    pieces: list[str] = []
    buf = ""
    for sent in sentences:
        if not buf:
            buf = sent
        elif len(buf) < min_chars or (pieces and len(buf) + len(sent) < max_chars):
            buf = f"{buf} {sent}"
        else:
            pieces.append(buf)
            buf = sent
    if buf:
        if len(buf) < min_chars and pieces:
            pieces[-1] = f"{pieces[-1]} {buf}"
        else:
            pieces.append(buf)
    if not pieces:
        return [text]
    # Cut a long opening piece at a comma so the first audio arrives sooner.
    if cut_first and len(pieces[0]) > 60:
        cut = pieces[0].find(", ", 10, 60)
        if cut != -1:
            pieces[0:1] = [pieces[0][: cut + 1], pieces[0][cut + 2 :]]
    return pieces


class TTSEngine:
    MAX_PROMPT_CACHES = 8

    def __init__(self, config: ServerConfig):
        self.config = config
        self._model: Optional[RajVoiceCloner] = None
        self._asr_model = None
        self._model_lock = threading.Lock()
        self._generate_lock = threading.Lock()
        self._asr_lock = threading.Lock()
        # Encoded voice-sample caches keyed by (path, mtime, transcript):
        # encoding a reference sample is expensive, the result is reusable.
        self._prompt_caches: OrderedDict[tuple, dict] = OrderedDict()

    # ------------------------------------------------------------------ #
    # Model loading
    # ------------------------------------------------------------------ #
    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    def get_model(self) -> RajVoiceCloner:
        with self._model_lock:
            if self._model is None:
                source = self.config.model_path or self.config.hf_model_id
                # torch.compile + warmup only pays off on CUDA; on MPS/CPU it
                # makes loading far slower and heavier (same policy as app.py).
                device = self.config.device
                on_cuda = (device or "").startswith("cuda") or (device in (None, "auto") and torch.cuda.is_available())
                optimize = self.config.optimize and on_cuda
                logger.info("Loading RajVoiceCloner model from %s (optimize=%s) ...", source, optimize)
                # Denoiser is heavy (modelscope pipeline); loaded lazily on first use.
                self._model = RajVoiceCloner.from_pretrained(
                    hf_model_id=source,
                    load_denoiser=False,
                    device=device,
                    optimize=optimize,
                )
                logger.info("RajVoiceCloner model loaded.")
            return self._model

    def _ensure_denoiser(self, model: RajVoiceCloner) -> bool:
        """Lazy-load the ZipEnhancer denoiser; returns False if unavailable."""
        if not self.config.load_denoiser:
            return False
        if model.denoiser is None:
            # Lazy import: modelscope's pipeline stack is heavy and only needed
            # when denoising is actually requested.
            from ..zipenhancer import ZipEnhancer

            with self._model_lock:
                if model.denoiser is None:
                    logger.info("Loading ZipEnhancer denoiser ...")
                    model.denoiser = ZipEnhancer("iic/speech_zipenhancer_ans_multiloss_16k_base")
        return True

    def warm_up(self) -> None:
        """Load the model and run a tiny throwaway generation.

        The first forward pass after load pays one-off kernel compilation
        (~20s on MPS); doing it here keeps it off the first user request.
        """
        try:
            self.synthesize("Hello.", denoise=False, inference_timesteps=4)
            logger.info("Engine warm-up complete.")
        except Exception as exc:
            logger.warning("Engine warm-up failed: %s", exc)

    @property
    def sample_rate(self) -> int:
        return self.get_model().tts_model.sample_rate

    # ------------------------------------------------------------------ #
    # ASR (for instant voice cloning transcripts)
    # ------------------------------------------------------------------ #
    def transcribe(self, wav_path: str) -> str:
        from funasr import AutoModel

        with self._asr_lock:
            if self._asr_model is None:
                device = self.config.device or ""
                asr_device = "cuda:0" if device.startswith("cuda") else "cpu"
                logger.info("Loading ASR model (SenseVoiceSmall) on %s ...", asr_device)
                self._asr_model = AutoModel(
                    model="iic/SenseVoiceSmall",
                    disable_update=True,
                    log_level="WARNING",
                    device=asr_device,
                )
            res = self._asr_model.generate(input=wav_path, language="auto", use_itn=True)
        if not res:
            return ""
        return res[0]["text"].split("|>")[-1].strip()

    # ------------------------------------------------------------------ #
    # Synthesis
    # ------------------------------------------------------------------ #
    def synthesize(
        self,
        text: str,
        *,
        control: str | None = None,
        prompt_wav_path: str | None = None,
        prompt_text: str | None = None,
        reference_wav_path: str | None = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        normalize: bool = False,
        denoise: bool = True,
    ) -> np.ndarray:
        model = self.get_model()
        final_text = build_final_text(text, control)
        has_audio = prompt_wav_path is not None or reference_wav_path is not None
        do_denoise = denoise and has_audio and self._ensure_denoiser(model)
        with self._generate_lock:
            return model.generate(
                text=final_text,
                prompt_wav_path=prompt_wav_path,
                prompt_text=prompt_text,
                reference_wav_path=reference_wav_path,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
                normalize=normalize,
                denoise=do_denoise,
            )

    def synthesize_with_voice(
        self,
        text: str,
        *,
        sample_path: str,
        transcript: str | None = None,
        control: str | None = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        retry_badcase_max_times: int = 2,
    ) -> np.ndarray:
        """Synthesize with a library voice, reusing the encoded sample.

        Re-encoding a reference recording on every generation is a large part
        of cloned-voice latency; here the encoded prompt cache is built once
        per voice and reused (LRU, invalidated when the file changes).
        """
        model = self.get_model()
        if not isinstance(model.tts_model, RajVoiceCloner2Model):
            return self.synthesize(
                text,
                control=control,
                prompt_wav_path=sample_path if transcript else None,
                prompt_text=transcript or None,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
                denoise=False,
            )

        final_text = build_final_text(text, control)
        with self._generate_lock:
            cache = self._get_or_build_prompt_cache(model, sample_path, transcript)
            wav, _, _ = model.tts_model.generate_with_prompt_cache(
                target_text=final_text,
                prompt_cache=cache,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
                retry_badcase=True,
                retry_badcase_max_times=retry_badcase_max_times,
            )
            return wav.squeeze(0).cpu().numpy()

    def _get_or_build_prompt_cache(self, model: RajVoiceCloner, sample_path: str, transcript: str | None) -> dict:
        """LRU-cached encoded voice sample. Caller must hold the generate lock."""
        key = (sample_path, os.path.getmtime(sample_path), transcript or "")
        cache = self._prompt_caches.get(key)
        if cache is None:
            logger.info("Encoding voice sample for prompt cache: %s", sample_path)
            cache = model.tts_model.build_prompt_cache(
                prompt_text=transcript or None,
                prompt_wav_path=sample_path if transcript else None,
                reference_wav_path=sample_path,
            )
            self._prompt_caches[key] = cache
            while len(self._prompt_caches) > self.MAX_PROMPT_CACHES:
                self._prompt_caches.popitem(last=False)
        else:
            self._prompt_caches.move_to_end(key)
        return cache

    def synthesize_pipelined(
        self,
        text: str,
        *,
        sample_path: str | None = None,
        transcript: str | None = None,
        control: str | None = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
    ) -> Generator[np.ndarray, None, None]:
        """Yield one waveform per sentence, each generated in batch mode.

        The model's step-level streaming decode is ~3x slower end-to-end than
        batch generation (per-step VAE decode + GPU->CPU sync), so we pipeline
        at sentence granularity instead: the first short sentence arrives
        quickly and total time matches plain batch synthesis.
        """
        for piece in split_for_pipelining(text):
            t0 = time.monotonic()
            if sample_path:
                wav = self.synthesize_with_voice(
                    piece,
                    sample_path=sample_path,
                    transcript=transcript,
                    control=control,
                    cfg_value=cfg_value,
                    inference_timesteps=inference_timesteps,
                    retry_badcase_max_times=1,
                )
            else:
                wav = self.synthesize(
                    piece,
                    control=control,
                    cfg_value=cfg_value,
                    inference_timesteps=inference_timesteps,
                    denoise=False,
                )
            logger.info(
                "piece %.40r: %.1fs for %.1fs audio",
                piece,
                time.monotonic() - t0,
                len(wav) / self.sample_rate,
            )
            yield wav

    def synthesize_streaming(
        self,
        text: str,
        *,
        control: str | None = None,
        prompt_wav_path: str | None = None,
        prompt_text: str | None = None,
        reference_wav_path: str | None = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        normalize: bool = False,
        denoise: bool = True,
    ) -> Generator[np.ndarray, None, None]:
        """Yields float32 waveform chunks. Holds the generate lock for the
        duration of the stream."""
        model = self.get_model()
        final_text = build_final_text(text, control)
        has_audio = prompt_wav_path is not None or reference_wav_path is not None
        do_denoise = denoise and has_audio and self._ensure_denoiser(model)
        with self._generate_lock:
            stream = model.generate_streaming(
                text=final_text,
                prompt_wav_path=prompt_wav_path,
                prompt_text=prompt_text,
                reference_wav_path=reference_wav_path,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
                normalize=normalize,
                denoise=do_denoise,
            )
            try:
                yield from stream
            finally:
                stream.close()


def log_to_stderr() -> None:
    if any(getattr(handler, "_rajvoicecloner", False) for handler in logger.handlers):
        return
    handler = logging.StreamHandler(sys.stderr)
    handler._rajvoicecloner = True
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
