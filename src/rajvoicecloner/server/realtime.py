"""Realtime lightweight TTS for live agent calls.

RajVoiceCloner2 (2B params) runs ~2.5-3x slower than realtime on Apple Silicon, which
makes live conversations sluggish. Kokoro (82M params, ONNX) runs ~4x faster
than realtime on the same hardware, so agent calls use it for instant replies
while RajVoiceCloner keeps serving the playground, cloning and offline generations.

The dependency is optional: when ``kokoro-onnx`` is not installed, callers
fall back to RajVoiceCloner synthesis.
"""

import logging
import threading
import urllib.request
from pathlib import Path

import numpy as np

try:
    from kokoro_onnx import Kokoro
except ImportError:  # optional dependency
    Kokoro = None

logger = logging.getLogger("rajvoicecloner.server")

_RELEASE_BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
MODEL_FILENAME = "kokoro-v1.0.onnx"
VOICES_FILENAME = "voices-v1.0.bin"

# Curated call voices: id -> human-friendly label (shown in the agent editor).
CALL_VOICES: dict[str, str] = {
    "af_heart": "Heart — US female, warm",
    "af_bella": "Bella — US female, bright",
    "af_nicole": "Nicole — US female, soft",
    "af_sky": "Sky — US female, crisp",
    "am_michael": "Michael — US male, calm",
    "am_adam": "Adam — US male, deep",
    "am_puck": "Puck — US male, upbeat",
    "bf_emma": "Emma — UK female",
    "bf_isabella": "Isabella — UK female, warm",
    "bm_george": "George — UK male",
    "bm_lewis": "Lewis — UK male, deep",
}

_FEMALE_POOL = ["af_heart", "af_bella", "af_nicole", "af_sky", "bf_emma", "bf_isabella"]
_MALE_POOL = ["am_michael", "am_adam", "am_puck", "bm_george", "bm_lewis"]

_VOICE_BY_GENDER_ACCENT = {
    ("male", "british"): "bm_george",
    ("male", "american"): "am_michael",
    ("female", "british"): "bf_emma",
    ("female", "american"): "af_heart",
}


def pick_realtime_voice(labels: dict | None, seed: str = "") -> str:
    """Map a library voice onto the closest Kokoro voice.

    With gender/accent labels (premade voices) the match is direct. Without
    them (cloned voices), ``seed`` (the library voice id) picks a stable voice
    so different agents don't all sound identical.
    """
    gender = (labels or {}).get("gender", "").strip().lower()
    accent = (labels or {}).get("accent", "").strip().lower()
    if gender in ("male", "female") and accent in ("british", "american"):
        return _VOICE_BY_GENDER_ACCENT[(gender, accent)]
    pool = _MALE_POOL if gender == "male" else _FEMALE_POOL if gender == "female" else _FEMALE_POOL + _MALE_POOL
    return pool[sum(seed.encode("utf-8")) % len(pool)] if seed else pool[0]


class RealtimeTTS:
    """Lazy-loaded Kokoro ONNX engine; model files are fetched on first use."""

    sample_rate = 24000

    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self._kokoro = None
        self._lock = threading.Lock()
        self._failed = False

    def available(self) -> bool:
        return Kokoro is not None and not self._failed

    def _get(self) -> "Kokoro":
        with self._lock:
            if self._kokoro is None:
                self.models_dir.mkdir(parents=True, exist_ok=True)
                model_path = self.models_dir / MODEL_FILENAME
                voices_path = self.models_dir / VOICES_FILENAME
                for path in (model_path, voices_path):
                    if not path.is_file():
                        url = f"{_RELEASE_BASE}/{path.name}"
                        logger.info("Downloading realtime TTS file %s ...", url)
                        tmp = path.with_suffix(path.suffix + ".part")
                        urllib.request.urlretrieve(url, tmp)
                        tmp.rename(path)
                logger.info("Loading realtime TTS (Kokoro) ...")
                self._kokoro = Kokoro(str(model_path), str(voices_path))
            return self._kokoro

    def synthesize(self, text: str, *, voice: str = "af_heart", speed: float = 1.0) -> np.ndarray:
        """Return a float32 waveform at ``sample_rate``. Raises on failure."""
        try:
            kokoro = self._get()
            samples, _sr = kokoro.create(text, voice=voice, speed=speed)
        except Exception:
            # Don't retry a broken engine on every turn; fall back to RajVoiceCloner.
            self._failed = True
            raise
        return np.asarray(samples, dtype=np.float32)
