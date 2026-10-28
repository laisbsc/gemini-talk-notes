# gemini-talk-notes

Turn a YouTube talk into validated notes with [Gemini](https://ai.google.dev/) and
[Pydantic AI](https://pydantic.dev/docs/ai/): chapters with timestamp links, a summary,
takeaways, and a documentation link for every tool mentioned.

It runs in two steps, so validation retries never re-watch the video:

1. **Watch** (`gemini-4-argon`): Gemini watches the talk with agentic video processing and
   writes lenient raw notes. No output retries.
2. **Enrich** (`gemini-3.5-flash-lite`): a cheaper, text-only step turns the raw notes into
   strict `TalkNotes`, finds docs links with Google Search, and retries automatically when
   a validation rule fails.

Read the full walkthrough in **[TUTORIAL.md](TUTORIAL.md)**.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/). uv installs Python 3.14 if you don't have it.

```bash
uv sync
cp .env.example .env   # then add your GEMINI_API_KEY
uv run --env-file .env talk-notes "https://www.youtube.com/watch?v=wXjZdrS3LqA" --save-raw talk.raw.json
```

Options:

| Flag | What it does |
|------|--------------|
| `--save-raw FILE` | Save step 1's raw notes. |
| `--from-raw FILE` | Skip step 1 and enrich saved raw notes. |
| `--out FILE` | Write the notes to a file instead of stdout. |
| `--json` | Output JSON instead of Markdown. |

Set `TALK_NOTES_WATCH_MODEL` or `TALK_NOTES_ENRICH_MODEL` to use different models.

## Project layout

```
src/talk_notes/
├── schemas.py    # RawNotes (step 1) and TalkNotes (step 2), with validators
├── agents.py     # the watcher and enricher agents
├── pipeline.py   # watch(), enrich() and take_notes()
└── cli.py        # the talk-notes command
tests/            # offline tests with fake models, no API key needed
evals/            # Pydantic Evals for step 2, run on saved raw notes
```

## Development

```bash
uv run pytest                    # offline tests
uv run ruff format . && uv run ruff check .
uv run pyright                   # strict type checking
uv run --env-file .env evals/run_evals.py   # live evals, needs GEMINI_API_KEY
```

CI runs formatting, linting, type checking and tests on every push and pull request.
