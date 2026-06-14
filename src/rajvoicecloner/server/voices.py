"""Persistent voice library.

Voices live under ``<data_dir>/voices/<voice_id>/`` with a ``voice.json``
metadata file and an optional ``sample.wav`` reference recording.

Categories:
- ``cloned``   - created from a user audio sample (instant voice cloning)
- ``designed`` - defined purely by a text description (RajVoiceCloner2 voice design)
- ``premade``  - built-in designed voices shipped with the server
"""

import json
import shutil
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path

VOICE_META_FILENAME = "voice.json"
VOICE_SAMPLE_FILENAME = "sample.wav"

# Spoken once to mint a reference sample for description-only voices, so they
# keep one consistent identity (and benefit from prompt caching) afterwards.
# Kept short on purpose: the reference audio is re-attended on every
# generation, so a long seed sample slows down all later synthesis.
SEED_SAMPLE_TRANSCRIPT = "Hello there, thanks for calling today. I'm happy to help you."
TELUGU_CINEMA_SEED_SAMPLE_TRANSCRIPT = (
    "నమస్కారం. మీతో మాట్లాడటం నాకు చాలా ఆనందంగా ఉంది. మన మాటల్లో నమ్మకం, ఆప్యాయత, ధైర్యం ఉండాలి."
)

PREMADE_VOICES: list[dict] = [
    {
        "voice_id": "premade-aria",
        "name": "Aria",
        "category": "premade",
        "description": "A warm, expressive young female voice with an American accent, friendly and conversational",
        "labels": {"gender": "female", "accent": "american", "use_case": "conversational"},
    },
    {
        "voice_id": "premade-marcus",
        "name": "Marcus",
        "category": "premade",
        "description": "A deep, confident middle-aged male voice with a British accent, perfect for narration",
        "labels": {"gender": "male", "accent": "british", "use_case": "narration"},
    },
    {
        "voice_id": "premade-luna",
        "name": "Luna",
        "category": "premade",
        "description": "A soft, soothing female voice speaking slowly and calmly, ideal for meditation and audiobooks",
        "labels": {"gender": "female", "accent": "american", "use_case": "audiobook"},
    },
    {
        "voice_id": "premade-rex",
        "name": "Rex",
        "category": "premade",
        "description": "An energetic, upbeat young male voice with fast pacing, great for ads and promos",
        "labels": {"gender": "male", "accent": "american", "use_case": "advertisement"},
    },
    {
        "voice_id": "premade-nova",
        "name": "Nova",
        "category": "premade",
        "description": "A crisp, professional female news anchor voice, articulate and neutral",
        "labels": {"gender": "female", "accent": "neutral", "use_case": "news"},
    },
    {
        "voice_id": "premade-veer-telugu",
        "name": "Veer Telugu",
        "category": "premade",
        "description": (
            "A distinct original young Telugu male cinematic voice, not an imitation of any real person. "
            "Natural human studio-quality delivery with warm chest resonance, smooth breath control, "
            "clear Telugu diction, expressive emotional phrasing, confident heroic presence, subtle smile, "
            "realistic pauses, and a modern Hyderabad-influenced cadence. The voice should sound grounded, "
            "charismatic, intimate, and believable, never robotic, exaggerated, or announcer-like."
        ),
        "labels": {
            "gender": "male",
            "accent": "telugu",
            "language": "telugu",
            "use_case": "cinematic",
        },
    },
]


def seed_sample_transcript(voice_id: str) -> str:
    if voice_id == "premade-veer-telugu":
        return TELUGU_CINEMA_SEED_SAMPLE_TRANSCRIPT
    return SEED_SAMPLE_TRANSCRIPT


@dataclass
class Voice:
    voice_id: str
    name: str
    category: str  # "premade" | "cloned" | "designed"
    description: str | None = None
    transcript: str | None = None
    labels: dict[str, str] = field(default_factory=dict)
    created_at_unix: int = field(default_factory=lambda: int(time.time()))
    sample_path: str | None = None  # absolute path to sample.wav, if any

    def to_public_dict(self) -> dict:
        data = asdict(self)
        data.pop("sample_path", None)
        data["preview_url"] = f"/v1/voices/{self.voice_id}/sample" if self.sample_path else None
        return data


class VoiceLibrary:
    def __init__(self, voices_dir: Path):
        self.voices_dir = voices_dir
        self.voices_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    def list_voices(self) -> list[Voice]:
        voices = [self._premade(meta) for meta in PREMADE_VOICES]
        for voice_dir in sorted(self.voices_dir.iterdir()):
            if not voice_dir.is_dir():
                continue
            meta_path = voice_dir / VOICE_META_FILENAME
            if not meta_path.is_file():
                continue
            voices.append(self._load(meta_path))
        return voices

    def get(self, voice_id: str) -> Voice | None:
        for meta in PREMADE_VOICES:
            if meta["voice_id"] == voice_id:
                return self._premade(meta)
        meta_path = self.voices_dir / voice_id / VOICE_META_FILENAME
        if meta_path.is_file():
            return self._load(meta_path)
        return None

    def _premade(self, meta: dict) -> Voice:
        """Premade voices are defined in code; only their seed sample lives on disk."""
        voice = Voice(**meta)
        sample = self.voices_dir / voice.voice_id / VOICE_SAMPLE_FILENAME
        if sample.is_file():
            voice.sample_path = str(sample)
            voice.transcript = seed_sample_transcript(voice.voice_id)
        return voice

    def attach_seed_sample(self, voice: Voice, wav_src_path: Path, transcript: str | None = None) -> Voice:
        """Pin a description-only voice to a generated reference recording."""
        voice_dir = self.voices_dir / voice.voice_id
        voice_dir.mkdir(parents=True, exist_ok=True)
        sample_path = voice_dir / VOICE_SAMPLE_FILENAME
        shutil.copyfile(wav_src_path, sample_path)
        voice.sample_path = str(sample_path)
        voice.transcript = transcript or SEED_SAMPLE_TRANSCRIPT
        if voice.category != "premade":  # premade metadata stays in code
            self._save(voice)
        return voice

    def add(
        self,
        name: str,
        *,
        description: str | None = None,
        transcript: str | None = None,
        labels: dict[str, str] | None = None,
        sample_src_path: Path | None = None,
    ) -> Voice:
        category = "cloned" if sample_src_path is not None else "designed"
        voice_id = uuid.uuid4().hex[:20]
        voice_dir = self.voices_dir / voice_id
        voice_dir.mkdir(parents=True, exist_ok=True)

        sample_path = None
        if sample_src_path is not None:
            sample_path = str(voice_dir / VOICE_SAMPLE_FILENAME)
            shutil.copyfile(sample_src_path, sample_path)

        voice = Voice(
            voice_id=voice_id,
            name=name,
            category=category,
            description=description,
            transcript=transcript,
            labels=labels or {},
            sample_path=sample_path,
        )
        self._save(voice)
        return voice

    def update(self, voice: Voice) -> None:
        if voice.category == "premade":
            raise ValueError("Premade voices cannot be edited")
        self._save(voice)

    def delete(self, voice_id: str) -> bool:
        voice_dir = self.voices_dir / voice_id
        if not (voice_dir / VOICE_META_FILENAME).is_file():
            return False
        shutil.rmtree(voice_dir)
        return True

    # ------------------------------------------------------------------ #
    def _save(self, voice: Voice) -> None:
        meta_path = self.voices_dir / voice.voice_id / VOICE_META_FILENAME
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(asdict(voice), ensure_ascii=False, indent=2), encoding="utf-8")

    def _load(self, meta_path: Path) -> Voice:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        sample_path = data.get("sample_path")
        if sample_path and not Path(sample_path).is_file():
            migrated_sample = meta_path.parent / VOICE_SAMPLE_FILENAME
            if migrated_sample.is_file():
                data["sample_path"] = str(migrated_sample)
        return Voice(**data)
