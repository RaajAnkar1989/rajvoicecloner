from pydantic import BaseModel, Field


class VoiceSettings(BaseModel):
    """ElevenLabs-compatible voice settings, mapped onto RajVoiceCloner controls.

    ``stability`` maps to CFG guidance (higher stability -> stronger guidance),
    ``quality`` maps to diffusion timesteps. Native fields override the mapping.
    """

    stability: float = Field(default=0.5, ge=0.0, le=1.0)
    similarity_boost: float = Field(default=0.75, ge=0.0, le=1.0)
    style: float = Field(default=0.0, ge=0.0, le=1.0)
    use_speaker_boost: bool = True
    speed: float = Field(default=1.0, ge=0.5, le=2.0)

    # Native RajVoiceCloner overrides (optional)
    cfg_value: float | None = Field(default=None, ge=0.1, le=10.0)
    inference_timesteps: int | None = Field(default=None, ge=1, le=100)

    def resolved_cfg(self) -> float:
        if self.cfg_value is not None:
            return self.cfg_value
        return round(1.0 + self.stability * 2.0, 2)  # 0..1 -> 1.0..3.0

    def resolved_timesteps(self) -> int:
        if self.inference_timesteps is not None:
            return self.inference_timesteps
        return 10


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10_000)
    model_id: str = "rajvoice"
    voice_settings: VoiceSettings = Field(default_factory=VoiceSettings)
    # Extra control instruction, e.g. "whispering", "excited tone" (RajVoiceCloner2)
    control: str | None = None
    normalize: bool = False
    denoise: bool = True


class VoiceDesignRequest(BaseModel):
    voice_description: str = Field(min_length=1, max_length=1_000)
    text: str = Field(default="Hello! I am your newly designed voice. I hope you like how I sound.", max_length=2_000)
    voice_settings: VoiceSettings = Field(default_factory=VoiceSettings)


class VoiceResponse(BaseModel):
    voice_id: str
    name: str
    description: str | None = None
    category: str  # "premade" | "cloned" | "designed"
    labels: dict[str, str] = Field(default_factory=dict)
    preview_url: str | None = None
    transcript: str | None = None
    created_at_unix: int | None = None
    settings: VoiceSettings | None = None


class VoicesListResponse(BaseModel):
    voices: list[VoiceResponse]


class AddVoiceResponse(BaseModel):
    voice_id: str
    requires_verification: bool = False


class ModelResponse(BaseModel):
    model_id: str
    name: str
    description: str
    can_do_text_to_speech: bool = True
    can_do_voice_conversion: bool = False
    can_use_style: bool = True
    can_be_finetuned: bool = True
    languages: list[dict[str, str]] = Field(default_factory=list)


class AgentCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    prompt: str = Field(default="", max_length=4_000, description="Persona/greeting spoken at call start")
    questions: list[str] = Field(min_length=1, max_length=100)
    voice_id: str
    closing: str | None = Field(default=None, max_length=1_000)
    call_voice: str = Field(default="", description="Realtime voice for live calls ('' = auto)")


class AgentResponse(BaseModel):
    agent_id: str
    name: str
    prompt: str
    questions: list[str]
    voice_id: str
    closing: str
    call_voice: str = ""
    created_at_unix: int


class SettingsUpdateRequest(BaseModel):
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_api_key: str | None = None  # empty string clears the key
    default_voice_id: str | None = None  # empty string clears the default


class SettingsResponse(BaseModel):
    llm_base_url: str
    llm_model: str
    llm_has_api_key: bool
    llm_available: bool
    default_voice_id: str | None


class HistoryItemResponse(BaseModel):
    history_item_id: str
    voice_id: str | None
    voice_name: str | None
    text: str
    date_unix: int
    character_count: int
    content_type: str
    state: str = "created"


class HistoryListResponse(BaseModel):
    history: list[HistoryItemResponse]
