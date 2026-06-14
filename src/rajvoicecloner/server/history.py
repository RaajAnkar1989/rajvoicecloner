"""Generation history: every TTS request's audio + metadata, ElevenLabs-style."""

import json
import shutil
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path

HISTORY_META_FILENAME = "item.json"


@dataclass
class HistoryItem:
    history_item_id: str
    text: str
    voice_id: str | None = None
    voice_name: str | None = None
    content_type: str = "audio/mpeg"
    file_extension: str = "mp3"
    date_unix: int = field(default_factory=lambda: int(time.time()))

    @property
    def character_count(self) -> int:
        return len(self.text)

    def to_public_dict(self) -> dict:
        data = asdict(self)
        data.pop("file_extension", None)
        data["character_count"] = self.character_count
        data["state"] = "created"
        return data


class HistoryStore:
    def __init__(self, history_dir: Path, max_items: int = 500):
        self.history_dir = history_dir
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.max_items = max_items

    def add(
        self,
        *,
        text: str,
        audio_bytes: bytes,
        content_type: str,
        file_extension: str,
        voice_id: str | None = None,
        voice_name: str | None = None,
    ) -> HistoryItem:
        item = HistoryItem(
            history_item_id=uuid.uuid4().hex[:20],
            text=text,
            voice_id=voice_id,
            voice_name=voice_name,
            content_type=content_type,
            file_extension=file_extension,
        )
        item_dir = self.history_dir / item.history_item_id
        item_dir.mkdir(parents=True, exist_ok=True)
        (item_dir / f"audio.{file_extension}").write_bytes(audio_bytes)
        (item_dir / HISTORY_META_FILENAME).write_text(
            json.dumps(asdict(item), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._prune()
        return item

    def list_items(self) -> list[HistoryItem]:
        items = []
        for item_dir in self.history_dir.iterdir():
            if not item_dir.is_dir():
                continue
            meta_path = item_dir / HISTORY_META_FILENAME
            if not meta_path.is_file():
                continue
            items.append(HistoryItem(**json.loads(meta_path.read_text(encoding="utf-8"))))
        items.sort(key=lambda i: i.date_unix, reverse=True)
        return items

    def get(self, history_item_id: str) -> HistoryItem | None:
        meta_path = self.history_dir / history_item_id / HISTORY_META_FILENAME
        if not meta_path.is_file():
            return None
        return HistoryItem(**json.loads(meta_path.read_text(encoding="utf-8")))

    def audio_path(self, item: HistoryItem) -> Path:
        return self.history_dir / item.history_item_id / f"audio.{item.file_extension}"

    def delete(self, history_item_id: str) -> bool:
        item_dir = self.history_dir / history_item_id
        if not (item_dir / HISTORY_META_FILENAME).is_file():
            return False
        shutil.rmtree(item_dir)
        return True

    def _prune(self) -> None:
        items = self.list_items()
        for stale in items[self.max_items :]:
            self.delete(stale.history_item_id)
