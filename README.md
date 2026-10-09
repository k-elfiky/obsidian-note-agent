# Obsidian Research Note Agent

A CrewAI-based agentic workflow that researches a topic on the web and produces a validated, ready-to-use Markdown note saved directly into your Obsidian vault.

## How it works

```
plan → research → write → validate → (revise → validate)* → save
```

1. **Research Planner** breaks the topic into a structured research plan (fundamentals → advanced).
2. **Web Researcher** executes that plan using Tavily search/extract, prioritizing authoritative sources.
3. **Note Writer** turns the research report into an Obsidian-ready Markdown note.
4. **Note Validator** checks the note for accuracy, completeness, clarity, technical quality, and Obsidian formatting, returning a structured `PASS`/`FAIL` report with specific issues.
5. If the validator returns `FAIL`, the **Note Writer** revises the note and it's re-validated. This loop repeats up to `MAX_REVISION_ATTEMPTS` times.
6. **Obsidian Agent** saves the final Markdown note into your vault under an `AI Notes` folder, without overwriting existing notes of the same name.

## Files

| File | Purpose |
|---|---|
| `obsidian_research_agent.py` | The full pipeline: config, tools, agents, tasks, retry/validation loop, and CLI entrypoint. |
| `requirements.txt` | Python dependencies. |
| `.env.example` | Template for required environment variables / secrets. |

## Setup

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment variables

Copy the example file and fill in your own values:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|---|---|---|
| `TAVILY_API_KEY` | Yes | API key for Tavily search/extract. |
| `OBSIDIAN_VAULT_PATH` | Yes | Absolute path to your Obsidian vault. Notes are saved under `<vault>/AI Notes/`. |
| `LLM_MODEL` | Yes | Model identifier passed to CrewAI's `LLM`. Use a paid, stable model in production — avoid free-tier/rate-limited models. |
| `OPENROUTER_API_KEY` | If using an OpenRouter model | API key for OpenRouter. |
| `GROQ_API_KEY` | If using a `groq/...` model | API key for Groq. |
| `AGENTOPS_API_KEY` | No | Enables AgentOps tracing/observability if set. The run works fine without it. |
| `MAX_REVISION_ATTEMPTS` | No (default `2`) | Max times the note is sent back for revision after a `FAIL` validation. |
| `RUN_TIMEOUT_SECONDS` | No (default `600`) | Hard timeout for the entire pipeline run. |
| `LOG_LEVEL` | No (default `INFO`) | Standard Python logging level. |

Never commit `.env` — only `.env.example` should go into version control.

### 3. Run

```bash
python obsidian_research_agent.py --topic "What is NLP"
```

On success, the script prints a confirmation message with the path of the saved note, e.g.:

```
Note saved successfully: /path/to/vault/AI Notes/What is NLP.md
```

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Success. |
| `1` | Unhandled error during the pipeline run (see logs). |
| `2` | Invalid configuration or invalid `--topic` input. |
| `3` | Run exceeded `RUN_TIMEOUT_SECONDS`. |

## Design notes / known limitations

- **Validation loop, not a single pass.** The validator's output is parsed as structured JSON (`status`, `issues`) rather than matched as a raw string, so the pipeline can reliably branch on it. If the note still fails after `MAX_REVISION_ATTEMPTS`, the pipeline logs a warning and saves the best available draft rather than failing the whole run — adjust this in `run_pipeline()` if you'd rather halt and escalate instead.
- **No concurrency control on saves.** Two runs writing a note with the same title at the same time could still race past the no-clobber check. Add a lock or queue if you expect concurrent runs.
- **Model choice matters.** Free-tier/rate-limited models are fine for testing but not recommended for production — use a model with a stable rate limit and uptime guarantee.
- **No persistent eval suite yet.** Before relying on this in production, run it against a fixed set of topics periodically and spot-check output quality, ideally with the validator's criteria reused as an offline eval rubric.

## Extending

- Swap `save_to_obsidian` for a different destination (Notion, a database, a git-backed notes repo) by replacing the tool and the `obsidian_agent`/`save_task` definitions — the rest of the pipeline is destination-agnostic.
- Add more validator criteria by editing `note_validation_task`'s description; the `ValidationReport` schema doesn't need to change.
- To process multiple topics in one run, loop over `run_pipeline(topic, cfg)` in `main()`, or batch them through a queue/worker if running at scale.
