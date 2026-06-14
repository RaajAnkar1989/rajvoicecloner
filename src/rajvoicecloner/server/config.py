import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .branding import (
    API_KEY_ENV,
    APP_SLUG,
    DATA_DIR_ENV,
    LEGACY_API_KEY_ENV,
    LEGACY_DATA_DIR_ENV,
    LEGACY_LLM_API_KEY_ENV,
    LEGACY_LLM_MODEL_ENV,
    LEGACY_LLM_URL_ENV,
    LLM_API_KEY_ENV,
    LLM_MODEL_ENV,
    LLM_URL_ENV,
)

DEFAULT_HF_MODEL_ID = "openbmb/" + "Vox" + "CPM2"


def _env(primary: str, legacy: str | None = None) -> str | None:
    value = os.environ.get(primary)
    if value:
        return value
    if legacy:
        return os.environ.get(legacy)
    return None


def default_data_dir() -> Path:
    override = _env(DATA_DIR_ENV, LEGACY_DATA_DIR_ENV)
    if override:
        path = Path(override)
        path.mkdir(parents=True, exist_ok=True)
        return path

    new_root = Path.home() / f".{APP_SLUG}"
    new = new_root / "studio"
    old = Path.home() / ".rajvoicecloner" / "studio"
    if not new.exists() and old.is_dir():
        new_root.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old), str(new))
    new.mkdir(parents=True, exist_ok=True)
    return new


@dataclass
class ServerConfig:
    """Runtime configuration for the RajVoiceCloner server."""

    model_path: str | None = None
    hf_model_id: str = DEFAULT_HF_MODEL_ID
    device: str | None = None
    load_denoiser: bool = True
    optimize: bool = True
    preload: bool = False
    api_key: str | None = field(default_factory=lambda: _env(API_KEY_ENV, LEGACY_API_KEY_ENV) or None)
    data_dir: Path = field(default_factory=default_data_dir)

    # Local LLM (OpenAI-compatible; defaults target Ollama) for smart agents
    llm_base_url: str = field(
        default_factory=lambda: _env(LLM_URL_ENV, LEGACY_LLM_URL_ENV) or "http://localhost:11434/v1"
    )
    llm_model: str = field(
        default_factory=lambda: _env(LLM_MODEL_ENV, LEGACY_LLM_MODEL_ENV) or "llama3.2"
    )
    llm_api_key: str | None = field(default_factory=lambda: _env(LLM_API_KEY_ENV, LEGACY_LLM_API_KEY_ENV) or None)

    @property
    def voices_dir(self) -> Path:
        return self.data_dir / "voices"

    @property
    def history_dir(self) -> Path:
        return self.data_dir / "history"

    @property
    def agents_dir(self) -> Path:
        return self.data_dir / "agents"

    def ensure_dirs(self) -> None:
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.agents_dir.mkdir(parents=True, exist_ok=True)
