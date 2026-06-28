# Human Preference Tracking Conversational Agent

An experimental conversational agent that adapts its response style from implicit user feedback.  
The project combines a latent preference aligner, LLM-based reward estimation, a stateful conversation loop, and a FastAPI dashboard for observing the alignment process.

This is not a production chatbot. It is a small research-style prototype for exploring how an AI companion could learn style preferences without asking users to rate every response.

## Why It Exists

Most chat applications treat personalization as a prompt problem: write a better system prompt, add some memory, and hope the model behaves. This project tests a different question:

> Can the assistant maintain a compact latent model of the user's preferred interaction style, then update that model from natural follow-up messages?

The system converts conversational reactions into a reward signal, updates a low-dimensional preference subspace, and surfaces the internal learning state in a web dashboard.

## Core Ideas

- **Latent preference tracking**: responses are conditioned by a sampled style vector from a learned latent subspace.
- **Implicit reward estimation**: an LLM reads the user's next natural response and estimates satisfaction in `[-1, 1]`.
- **Adaptive dimensionality**: when recent reward is poor and residual signal is strong, the aligner can expand its latent subspace.
- **Operational visibility**: reward history, current `k`, token usage, style hints, and dimension expansion events are exposed through a dashboard.
- **Two interfaces**: a CLI demo for quick iteration and a FastAPI + WebSocket UI for interactive debugging.

## Architecture

```text
User message
    |
    v
ConversationSession
    |
    |-- previous pending action + new user reaction
    |       -> LLM reward estimator
    |       -> LatentAligner.update_with_sample()
    |
    |-- sample next latent action vector
    |       -> LLM actor prompt with style_code
    |       -> assistant reply
    |
    v
FastAPI / CLI response
    |
    v
Dashboard telemetry: reward, k, tokens, w_hat preview, dim events
```

## Project Structure

```text
.
├── latent_aligner.py      # Online latent preference model and subspace expansion
├── llm_bridge.py          # LLM actor + reward estimator bridge
├── session_core.py        # Stateful conversation loop shared by CLI and API
├── run_simulation.py      # Offline simulation with synthetic users
├── run_llm_online.py      # CLI demo with a real LLM
├── web_server.py          # FastAPI API, WebSocket, and static frontend
├── web_frontend/
│   └── index.html         # Debug dashboard
├── config.py              # Experiment parameters
├── user_env.py            # Synthetic user environment for simulation
└── requirements.txt
```

## Quick Start

### 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
```

Fill in `DEEPSEEK_API_KEY` or another OpenAI-compatible key. Real keys should stay in `.env`, which is ignored by git.

### 3. Run The Synthetic Simulation

```bash
python run_simulation.py
```

This runs the latent aligner against a synthetic user preference vector. It is the fastest way to inspect the math loop without calling an LLM.

### 4. Run The CLI Demo

```bash
python run_llm_online.py
```

Each user message becomes both an input for the next assistant response and, when applicable, a natural reaction used to estimate reward for the previous turn.

### 5. Run The Web Dashboard

```bash
uvicorn web_server:app --reload --port 8000
```

Open `http://127.0.0.1:8000`.

The dashboard shows:

- live conversation
- latest and historical reward
- current latent dimension `k`
- token usage
- dimension expansion events
- preview of the learned preference vector

## API

### `POST /api/chat`

```json
{
  "message": "I prefer shorter answers with concrete next steps."
}
```

Returns the assistant response, debug information, current stats, and recent conversation tail.

### `GET /api/state`

Returns current telemetry without sending a new message.

### `WS /ws/state`

Pushes dashboard state updates over WebSocket.

## What This Demonstrates

- Turning vague product ideas about "personalized AI companions" into an inspectable prototype.
- Building an LLM application loop with explicit state, reward estimation, logs, and a debug surface.
- Separating model behavior into actor generation, reward judgment, and latent preference update modules.
- Designing AI features so failure cases are observable instead of hidden inside prompt text.

## Limitations

- Reward estimation is model-dependent and can be noisy.
- The latent style vector is intentionally abstract; dimensions are not directly human-interpretable.
- The current implementation is single-session and in-memory.
- This is a prototype for experimentation, not a hardened multi-user service.

## Next Steps

- Add deterministic tests for `LatentAligner`.
- Persist sessions and preference state by `user_id`.
- Add an evaluation script over scripted conversation traces.
- Support multiple reward models and compare judge stability.
- Add screenshots or short recordings of the dashboard.
