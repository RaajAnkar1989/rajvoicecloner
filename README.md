# RajVoiceCloner

RajVoiceCloner is a self-hosted AI voice cloning and text-to-speech studio. It includes a browser-based voice workspace, a local REST API, voice design, instant voice cloning, generation history, and live voice agents.

## Features

- Text-to-speech playground with streaming playback
- Voice cloning from uploaded or recorded audio samples
- Voice design from natural-language descriptions
- Persistent voice library and generation history
- Live voice agents with questionnaire flows
- Local LLM integration for smarter agent conversations
- ElevenLabs-style REST endpoints for easier integration

## Quick Start

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
rajvoicecloner serve --host 127.0.0.1 --port 8001
```

Open:

```text
http://127.0.0.1:8001
```

For this workspace, you can also run directly without installing:

```bash
PYTHONPATH=src venv/bin/python -m rajvoicecloner.cli serve --host 127.0.0.1 --port 8001
```

## API Example

```bash
curl -X POST "http://127.0.0.1:8001/v1/text-to-speech/premade-aria?output_format=wav_48000" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello from RajVoiceCloner!", "voice_settings": {"stability": 0.5}}' \
  --output speech.wav
```

## Data

Voices, agents, settings, and history are stored under:

```text
~/.rajvoicecloner/studio
```

You can override this with:

```bash
RAJVOICECLONER_DATA_DIR=/path/to/data rajvoicecloner serve
```

## Safety

Only clone voices you own or have permission to use. Do not use generated voices for impersonation, fraud, or deceptive content.
