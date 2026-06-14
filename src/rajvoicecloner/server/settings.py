"""Runtime-editable studio settings, persisted to ``<data_dir>/settings.json``.

These override the CLI/env defaults and can be changed from the web UI
without restarting the server.
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path

SETTINGS_FILENAME = "settings.json"


@dataclass
class StudioSettings:
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "llama3.2"
    llm_api_key: str | None = None
    default_voice_id: str | None = None


class SettingsStore:
    def __init__(self, data_dir: Path):
        self.path = data_dir / SETTINGS_FILENAME

    def load(self, defaults: StudioSettings) -> StudioSettings:
        if not self.path.is_file():
            return defaults
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return defaults
        known = {k: v for k, v in data.items() if k in StudioSettings.__dataclass_fields__}
        return StudioSettings(**{**asdict(defaults), **known})

    def save(self, settings: StudioSettings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(asdict(settings), ensure_ascii=False, indent=2), encoding="utf-8")
