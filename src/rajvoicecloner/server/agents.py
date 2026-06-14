"""Voice agents: scripted questionnaire callers.

An agent is defined by a persona/greeting prompt, an ordered questionnaire,
and a voice from the library. A call session walks the caller through the
questions: the agent speaks, the caller answers by voice, answers are
transcribed and stored. Completed sessions are persisted to disk.
"""

import json
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .llm import LLMClient

logger = logging.getLogger("rajvoicecloner.server")

AGENT_META_FILENAME = "agent.json"

ACKNOWLEDGEMENTS = ["Got it.", "Thanks.", "Okay, noted.", "Understood.", "Alright."]

DEFAULT_CLOSING = "That was my last question. Thank you so much for your time. Goodbye!"


@dataclass
class Agent:
    agent_id: str
    name: str
    prompt: str  # persona / greeting, spoken at the start of the call
    questions: list[str]
    voice_id: str
    closing: str = DEFAULT_CLOSING
    call_voice: str = ""  # realtime (Kokoro) voice for live calls; "" = auto
    created_at_unix: int = field(default_factory=lambda: int(time.time()))


@dataclass
class Answer:
    question: str
    transcript: str
    answered_at_unix: int = field(default_factory=lambda: int(time.time()))


@dataclass
class CallSession:
    session_id: str
    agent_id: str
    answers: list[Answer] = field(default_factory=list)
    next_question_index: int = 0
    finished: bool = False
    started_at_unix: int = field(default_factory=lambda: int(time.time()))
    # Smart (LLM-driven) mode state
    smart: bool = False
    messages: list[dict] = field(default_factory=list)
    asked_index: int = -1  # 0-based index of the question currently being asked

    def to_public_dict(self, agent: Agent) -> dict:
        return {
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "agent_name": agent.name,
            "smart": self.smart,
            "finished": self.finished,
            "started_at_unix": self.started_at_unix,
            "questions_total": len(agent.questions),
            "questions_answered": len(self.answers),
            "answers": [asdict(a) for a in self.answers],
        }


class AgentStore:
    """Disk-backed agent definitions under ``<data_dir>/agents/<agent_id>/``."""

    def __init__(self, agents_dir: Path):
        self.agents_dir = agents_dir
        self.agents_dir.mkdir(parents=True, exist_ok=True)

    def list_agents(self) -> list[Agent]:
        agents = []
        for agent_dir in sorted(self.agents_dir.iterdir()):
            meta = agent_dir / AGENT_META_FILENAME
            if meta.is_file():
                agents.append(Agent(**json.loads(meta.read_text(encoding="utf-8"))))
        return agents

    def get(self, agent_id: str) -> Agent | None:
        meta = self.agents_dir / agent_id / AGENT_META_FILENAME
        if not meta.is_file():
            return None
        return Agent(**json.loads(meta.read_text(encoding="utf-8")))

    def add(
        self,
        *,
        name: str,
        prompt: str,
        questions: list[str],
        voice_id: str,
        closing: str | None = None,
        call_voice: str = "",
    ) -> Agent:
        agent = Agent(
            agent_id=uuid.uuid4().hex[:20],
            name=name,
            prompt=prompt,
            questions=questions,
            voice_id=voice_id,
            closing=closing or DEFAULT_CLOSING,
            call_voice=call_voice,
        )
        meta = self.agents_dir / agent.agent_id / AGENT_META_FILENAME
        meta.parent.mkdir(parents=True, exist_ok=True)
        meta.write_text(json.dumps(asdict(agent), ensure_ascii=False, indent=2), encoding="utf-8")
        return agent

    def update(
        self,
        agent: Agent,
        *,
        name: str,
        prompt: str,
        questions: list[str],
        voice_id: str,
        closing: str | None = None,
        call_voice: str = "",
    ) -> Agent:
        agent.name = name
        agent.prompt = prompt
        agent.questions = questions
        agent.voice_id = voice_id
        agent.closing = closing or DEFAULT_CLOSING
        agent.call_voice = call_voice
        meta = self.agents_dir / agent.agent_id / AGENT_META_FILENAME
        meta.write_text(json.dumps(asdict(agent), ensure_ascii=False, indent=2), encoding="utf-8")
        return agent

    def delete(self, agent_id: str) -> bool:
        agent_dir = self.agents_dir / agent_id
        if not (agent_dir / AGENT_META_FILENAME).is_file():
            return False
        for child in sorted(agent_dir.rglob("*"), reverse=True):
            child.unlink() if child.is_file() else child.rmdir()
        agent_dir.rmdir()
        return True

    def save_session(self, session: CallSession, agent: Agent) -> None:
        sessions_dir = self.agents_dir / agent.agent_id / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        path = sessions_dir / f"{session.session_id}.json"
        path.write_text(
            json.dumps(session.to_public_dict(agent), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def list_saved_sessions(self, agent_id: str) -> list[dict]:
        sessions_dir = self.agents_dir / agent_id / "sessions"
        if not sessions_dir.is_dir():
            return []
        sessions = [json.loads(p.read_text(encoding="utf-8")) for p in sessions_dir.glob("*.json")]
        sessions.sort(key=lambda s: s.get("started_at_unix", 0), reverse=True)
        return sessions


_TAG_RE = re.compile(r"\[\s*(?:Q\s*(\d+)|DONE)\s*\]", re.IGNORECASE)


def _build_system_prompt(agent: Agent) -> str:
    numbered = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(agent.questions))
    persona = agent.prompt.strip() or "You are a friendly, professional caller."
    return (
        f"You are '{agent.name}', a voice agent on a live phone call.\n"
        f"Persona and goal: {persona}\n\n"
        f"You must work through this questionnaire with the caller, one question at a time, in order:\n"
        f"{numbered}\n\n"
        "Rules:\n"
        "- You are SPEAKING out loud: plain conversational sentences only. No markdown, no lists, "
        "no emojis, no stage directions. Keep every reply under 25 words.\n"
        "- React briefly and naturally to what the caller just said before continuing.\n"
        "- If an answer is unclear, off-topic, or incomplete, ask ONE short follow-up about the same "
        "question, then move on either way.\n"
        "- Start EVERY reply with a tag: [Qn] where n is the number of the question you are asking in "
        "this reply, or [DONE] when all questions are answered and you are wrapping up.\n"
        "- When you use [DONE], thank the caller and say goodbye in one short sentence."
    )


def _parse_tagged_reply(reply: str) -> tuple[int | None, bool, str]:
    """Return (question_number_1based, is_done, speech_text)."""
    match = _TAG_RE.search(reply)
    question_num = None
    is_done = False
    if match:
        if match.group(1) is not None:
            question_num = int(match.group(1))
        else:
            is_done = True
    speech = _TAG_RE.sub("", reply)
    # LLMs sometimes emit extra stage directions in brackets; never speak those.
    speech = re.sub(r"[\[\(][^\]\)]*[\]\)]", "", speech)
    speech = re.sub(r"\s+", " ", speech).strip()
    return question_num, is_done, speech


class SessionManager:
    """In-memory live call sessions. Finished sessions are persisted via AgentStore.

    When a local LLM is reachable, calls run in "smart" mode: the LLM drives the
    conversation (reactions, follow-ups, transitions) while the questionnaire
    keeps it on rails via the [Qn]/[DONE] tag protocol. Without an LLM, calls
    fall back to the scripted ask-acknowledge-ask flow.
    """

    def __init__(self, store: AgentStore, llm: LLMClient | None = None):
        self.store = store
        self.llm = llm
        self._sessions: dict[str, CallSession] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    def start(self, agent: Agent) -> tuple[CallSession, str]:
        """Create a session and return it with the agent's opening utterance.

        The opening is always deterministic (greeting + first question) so the
        call is picked up immediately, without waiting on an LLM round-trip.
        When an LLM is reachable, the session is seeded so smart mode takes
        over from the caller's first reply, and the LLM is warmed up in the
        background while the greeting is being spoken.
        """
        session = CallSession(session_id=uuid.uuid4().hex[:20], agent_id=agent.agent_id)
        with self._lock:
            self._sessions[session.session_id] = session

        if not agent.questions:
            session.finished = True
            self.store.save_session(session, agent)
            return session, agent.closing

        opening = self._scripted_open(session, agent)

        if self.llm is not None and self.llm.available():
            session.smart = True
            session.messages = [
                {"role": "system", "content": _build_system_prompt(agent)},
                {"role": "user", "content": "(The call just connected. Greet the caller and ask the first question.)"},
                {"role": "assistant", "content": f"[Q1] {opening}"},
            ]
            self._warm_llm_async()

        return session, opening

    def get(self, session_id: str) -> CallSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def record_answer(self, session: CallSession, agent: Agent, transcript: str) -> str:
        """Store the caller's answer and return the agent's next utterance."""
        if session.smart:
            try:
                return self._smart_reply(session, agent, transcript)
            except Exception as exc:
                logger.warning("LLM turn failed (%s); continuing in scripted mode", exc)
                session.smart = False
                session.next_question_index = session.asked_index + 1
        return self._scripted_reply(session, agent, transcript)

    def end(self, session: CallSession, agent: Agent) -> None:
        if not session.finished:
            session.finished = True
            self.store.save_session(session, agent)

    # ------------------------------------------------------------------ #
    # Scripted mode
    # ------------------------------------------------------------------ #
    def _scripted_open(self, session: CallSession, agent: Agent) -> str:
        # Long prompts are persona instructions, not greetings: don't read them out.
        greeting = agent.prompt.strip()
        if not greeting or len(greeting) > 220:
            greeting = f"Hi! This is {agent.name}."
        session.asked_index = 0
        session.next_question_index = 1
        return f"{greeting} {agent.questions[0]}"

    def _scripted_reply(self, session: CallSession, agent: Agent, transcript: str) -> str:
        answered_index = session.next_question_index - 1
        question = agent.questions[answered_index] if 0 <= answered_index < len(agent.questions) else ""
        session.answers.append(Answer(question=question, transcript=transcript))

        if session.next_question_index < len(agent.questions):
            ack = ACKNOWLEDGEMENTS[answered_index % len(ACKNOWLEDGEMENTS)]
            utterance = f"{ack} {agent.questions[session.next_question_index]}"
            session.asked_index = session.next_question_index
            session.next_question_index += 1
        else:
            utterance = agent.closing
            session.finished = True
            self.store.save_session(session, agent)
        return utterance

    # ------------------------------------------------------------------ #
    # Smart (LLM) mode
    # ------------------------------------------------------------------ #
    def _warm_llm_async(self) -> None:
        """Load the LLM into memory while the greeting plays, off the request path."""

        def warm():
            try:
                self.llm.chat([{"role": "user", "content": "hi"}], max_tokens=1)
            except Exception as exc:
                logger.debug("LLM warm-up failed: %s", exc)

        threading.Thread(target=warm, daemon=True).start()

    def _smart_reply(self, session: CallSession, agent: Agent, transcript: str) -> str:
        session.messages.append({"role": "user", "content": transcript})
        reply = self.llm.chat(session.messages, max_tokens=80)
        question_num, is_done, speech = _parse_tagged_reply(reply)
        if not speech:
            raise ValueError("LLM returned an empty reply")
        session.messages.append({"role": "assistant", "content": reply})
        # Cap history so very long calls don't grow unbounded.
        if len(session.messages) > 60:
            session.messages = session.messages[:1] + session.messages[-50:]

        self._record_smart_answer(session, agent, transcript)

        if is_done:
            session.finished = True
            self.store.save_session(session, agent)
        elif question_num is not None:
            session.asked_index = min(question_num - 1, len(agent.questions) - 1)
        return speech

    def _record_smart_answer(self, session: CallSession, agent: Agent, transcript: str) -> None:
        """Attach the transcript to the question that was being asked; merge follow-ups."""
        index = max(0, min(session.asked_index, len(agent.questions) - 1))
        question = agent.questions[index]
        if session.answers and session.answers[-1].question == question:
            session.answers[-1].transcript += f" {transcript}"
        else:
            session.answers.append(Answer(question=question, transcript=transcript))
