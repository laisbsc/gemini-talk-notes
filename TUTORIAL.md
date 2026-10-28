# Video notes agent with Gemini and Pydantic AI

[Pydantic AI](https://pydantic.dev/docs/ai/) is a Python agent framework from the team
behind Pydantic. You describe what you want back from the model as a Pydantic model, and
Pydantic AI validates every response against it. When a response fails validation, Pydantic
AI sends the error back to the model and asks it to try again, without any retry code of
your own.

This guide shows you how to build an app that turns a YouTube talk into validated notes:
chapters with clickable timestamps, a short summary, takeaways, and a documentation link for
every tool the speaker mentions. It uses Gemini's video understanding to watch the talk and
Google Search to find the documentation links.

The app will:

1. Watch a public YouTube video with Gemini, using agentic video processing.
2. Validate and enrich the notes on a cheaper Gemini model, retrying automatically when the
   output breaks a rule, such as a chapter that starts after the video ends.
3. Test the whole pipeline offline, and evaluate the second step with Pydantic Evals.

The sample video is Samuel Colvin's PyAI Conf 2026 talk,
[Controlling the wild: Monty, from tool calling to computer use](https://www.youtube.com/watch?v=wXjZdrS3LqA).

## Why two steps? {#two-steps}

When structured output fails validation, the usual fix is to retry. But a retry repeats the
whole request. If that request includes a 20-minute video, every retry asks Gemini to watch
the talk again, and pays for it again.

So this app splits the work by cost:

| Step | Model | Input | Output schema | Output retries |
|------|-------|-------|---------------|----------------|
| 1. Watch | `gemini-4-argon` | The video | `RawNotes`, lenient | 0 |
| 2. Enrich | `gemini-3.5-flash-lite` | Step 1's notes, as text | `TalkNotes`, strict | 3 |

Step 1 is expensive, so its schema is deliberately loose: plain strings, no validators,
nothing that should ever fail. Step 2 is cheap and text-only, so that's where the strict
rules live. When one of them fails, Pydantic AI sends the error to the step 2 model and it
fixes its answer from the raw notes. The video is watched exactly once.

The general rule: **validate where retries are cheap.**

## Prerequisites {#prerequisites}

To complete this guide, you'll need:

* A Gemini API key. You can create one for free in
  [Google AI Studio](https://aistudio.google.com/apikey).
* [uv](https://docs.astral.sh/uv/getting-started/installation/), which also installs
  Python 3.14 for you.

## Setup {#setup}

Create a project and add the dependencies:

```bash
uv init --package --python 3.14 talk-notes
cd talk-notes
uv add "pydantic-ai-slim[google]"
uv add --dev pydantic-evals pytest pytest-asyncio
```

Create a `.env` file in the project root with your API key, and keep it out of git:

```bash
echo "GEMINI_API_KEY=your-api-key-here" > .env
echo ".env" >> .gitignore
```

Every command in this guide passes the file to uv with `--env-file .env`.

The finished code lives in four files under `src/talk_notes/`:

```
src/talk_notes/
├── schemas.py    # what each step returns
├── agents.py     # the two agents
├── pipeline.py   # runs step 1, then step 2
└── cli.py        # the talk-notes command
```

## Define the output schemas {#schemas}

Start with what you want back. In `src/talk_notes/schemas.py`, first add two helpers: a
timestamp type, and a function that turns `'4:05'` into seconds.

```python
import re
from typing import Annotated, Self

from pydantic import BaseModel, Field, HttpUrl, UrlConstraints, model_validator

TIMESTAMP_PATTERN = r'^(\d+:)?[0-5]?\d:[0-5]\d$'

Timestamp = Annotated[
    str,
    Field(pattern=TIMESTAMP_PATTERN, description="Start time as 'M:SS' or 'H:MM:SS'.", examples=['4:05', '1:02:30']),
]
HttpsUrl = Annotated[HttpUrl, UrlConstraints(allowed_schemes=['https'])]


def to_seconds(timestamp: str) -> int:
    """Convert '4:05' or '1:02:30' to seconds."""
    if not re.fullmatch(TIMESTAMP_PATTERN, timestamp):
        raise ValueError(f'Not a timestamp: {timestamp!r}')
    seconds = 0
    for part in timestamp.split(':'):
        seconds = seconds * 60 + int(part)
    return seconds
```

Next, the step 1 schema. It's all plain strings on purpose, so the expensive call never fails
validation. Field descriptions become part of the JSON schema Gemini receives, so use them to
tell the model what you expect.

```python
class RawChapter(BaseModel):
    start: str = Field(description='When this part starts, as shown in the video player, e.g. "4:05".')
    title: str
    notes: str = Field(description='What is said and shown in this part, including any code on screen.')


class RawNotes(BaseModel):
    title: str
    speaker: str | None = None
    duration: str = Field(description='Total length of the video, e.g. "28:45".')
    chapters: list[RawChapter]
    tools_mentioned: list[str] = Field(
        default=[], description='Libraries, frameworks, products and services the speaker mentions.'
    )
```

Then the step 2 schema, where the rules live:

* Every chapter start must look like a timestamp.
* Chapters must be in order, with no two at the same time.
* Every documentation link must be an `https` URL.
* There must be between one and five takeaways.

```python
class Chapter(BaseModel):
    start: Timestamp
    title: str = Field(max_length=80)
    summary: str = Field(description='One or two sentences.')

    @property
    def seconds(self) -> int:
        return to_seconds(self.start)


class ToolRef(BaseModel):
    name: str
    docs_url: HttpsUrl = Field(description='Official documentation URL, found with Google Search.')
    mentioned_for: str = Field(description='Why the speaker brought it up, in one sentence.')


class TalkNotes(BaseModel):
    title: str
    speaker: str | None = None
    tldr: str = Field(description='Two or three sentences a busy engineer would read.')
    chapters: list[Chapter] = Field(min_length=1)
    tools: list[ToolRef] = []
    takeaways: list[str] = Field(min_length=1, max_length=5)

    @model_validator(mode='after')
    def chapters_in_order(self) -> Self:
        starts = [chapter.seconds for chapter in self.chapters]
        if starts != sorted(set(starts)):
            raise ValueError(
                f'Chapter start times must be strictly increasing, got {[c.start for c in self.chapters]}.'
            )
        return self
```

The error message matters. When validation fails, this exact text goes back to the model,
so write it the way you'd explain the problem to a colleague.

The [full file](src/talk_notes/schemas.py) also has a `to_markdown()` method that renders the
notes with timestamp links into the video.

## Step 1: watch the video {#watch}

In `src/talk_notes/agents.py`, create the first agent. `NativeOutput` asks Gemini to return
JSON that matches the schema directly, using Gemini's structured output feature.

```python
import os
from dataclasses import dataclass

from pydantic_ai import Agent, ModelRetry, NativeOutput, RunContext, WebSearchTool
from pydantic_ai.capabilities import NativeTool

from .schemas import RawNotes, TalkNotes

WATCH_MODEL = os.getenv('TALK_NOTES_WATCH_MODEL', 'google:gemini-4-argon')
ENRICH_MODEL = os.getenv('TALK_NOTES_ENRICH_MODEL', 'google:gemini-3.5-flash-lite')

watcher = Agent(
    WATCH_MODEL,
    name='watcher',
    output_type=NativeOutput(RawNotes),
    instructions=(
        'You are taking notes on a technical talk for engineers who could not attend. '
        'Watch the whole video. Split it into chapters at each change of topic, using the '
        'timestamps shown in the video player. In each chapter, note what is said and any code '
        'or commands shown on screen. List every library, framework and service mentioned.'
    ),
    retries={'output': 0},
    defer_model_check=True,
)
```

A few things to note:

* `retries={'output': 0}` is the first half of the two-step design. If Gemini's answer
  somehow fails to parse, the run fails fast instead of watching the talk again.
* `defer_model_check=True` lets you import the module without an API key, which the offline
  tests rely on.
* Both model names can be overridden with environment variables, so you can try a cheaper
  model for step 1, such as `google:gemini-3.8-flash`.

## Step 2: validate and enrich {#enrich}

The second agent runs on a cheaper model and only ever sees text. It also gets Google Search
as a native tool, to find each tool's documentation. On Gemini 3 and later models, native
tools and structured output work together in a single request.

```python
@dataclass(frozen=True)
class EnrichDeps:
    duration_seconds: int | None
    """Length of the video, used to catch chapters that start after it ends."""


enricher = Agent(
    ENRICH_MODEL,
    name='enricher',
    deps_type=EnrichDeps,
    output_type=NativeOutput(TalkNotes),
    instructions=(
        'You turn raw notes from a technical talk into clean, accurate notes. '
        'Merge chapters that cover the same topic and keep each summary short. '
        'Use only what the raw notes say: never invent content. '
        'For each tool mentioned, search the web for its official documentation and link to it.'
    ),
    capabilities=[NativeTool(WebSearchTool())],
    retries={'output': 3},
    defer_model_check=True,
)
```

Some rules need information from outside the output. Knowing whether a chapter starts after
the video ends requires the video's length. Pass it in as a
[dependency](https://pydantic.dev/docs/ai/core-concepts/dependencies/) and check it in an
output validator. Raising `ModelRetry` sends your message back to the model, just like a
schema error:

```python
@enricher.output_validator
def chapters_inside_video(ctx: RunContext[EnrichDeps], notes: TalkNotes) -> TalkNotes:
    """Reject chapters that start after the video ends, and tell the model how to fix them."""
    duration = ctx.deps.duration_seconds
    if duration is None:
        return notes
    if late := [chapter.start for chapter in notes.chapters if chapter.seconds >= duration]:
        raise ModelRetry(
            f'Chapters starting at {late} begin after the video ends ({duration} seconds). '
            'Correct their start times from the raw notes, or drop them.'
        )
    return notes
```

`retries={'output': 3}` is the second half of the design. Step 2 gets several attempts,
because each one is a small text-only request. If all of them fail, the run raises
`UnexpectedModelBehavior` instead of returning bad data.

## Connect the two steps {#pipeline}

In `src/talk_notes/pipeline.py`, pass the video to step 1 as a `VideoUrl`. Pydantic AI sends
YouTube URLs to Gemini directly, without downloading the video. The `media_processing`
setting turns on [agentic video processing](https://ai.google.dev/gemini-api/docs/generate-content/video-understanding),
which lets Gemini look closely at the parts of a long video it needs.

```python
from dataclasses import dataclass

from pydantic_ai import RunUsage, VideoUrl

from .agents import EnrichDeps, enricher, watcher
from .schemas import RawNotes, TalkNotes, to_seconds


@dataclass(frozen=True)
class TalkNotesResult:
    raw: RawNotes
    notes: TalkNotes
    watch_usage: RunUsage
    enrich_usage: RunUsage


async def watch(video_url: str) -> tuple[RawNotes, RunUsage]:
    """Step 1: Gemini watches the video and writes raw notes."""
    video = VideoUrl(url=video_url, vendor_metadata={'media_processing': 'AGENTIC'})
    result = await watcher.run(['Take notes on this talk.', video])
    return result.output, result.usage


async def enrich(raw: RawNotes) -> tuple[TalkNotes, RunUsage]:
    """Step 2: a cheaper model validates and enriches the raw notes, retrying on errors."""
    try:
        duration = to_seconds(raw.duration)
    except ValueError:
        duration = None
    result = await enricher.run(raw.model_dump_json(indent=2), deps=EnrichDeps(duration_seconds=duration))
    return result.output, result.usage


async def take_notes(video_url: str) -> TalkNotesResult:
    raw, watch_usage = await watch(video_url)
    notes, enrich_usage = await enrich(raw)
    return TalkNotesResult(raw=raw, notes=notes, watch_usage=watch_usage, enrich_usage=enrich_usage)
```

Converting the duration happens in plain Python, not in the model. Anything you can compute
reliably in code, compute in code, and leave the model the parts that need judgment.

The [`cli.py`](src/talk_notes/cli.py) file wraps this in a `talk-notes` command. Register it in
`pyproject.toml`:

```toml
[project.scripts]
talk-notes = "talk_notes.cli:main"
```

## Run the app {#run}

Run the pipeline on the sample talk, and save step 1's raw notes so you can reuse them:

```bash
uv run --env-file .env talk-notes "https://www.youtube.com/watch?v=wXjZdrS3LqA" \
    --save-raw talk.raw.json --out notes.md
```

The command prints the token usage of each step, so you can see the difference in cost:

```
step 1 (watch): 1 request(s), ... input / ... output tokens
raw notes saved to talk.raw.json
step 2 (enrich): 2 request(s), ... input / ... output tokens
notes written to notes.md
```

If step 2 shows more than one request, a validation rule caught a problem and the model fixed
it. Open `notes.md` to see the result: every chapter timestamp links to that moment in the
video.

## Iterate on step 2 without re-watching {#iterate}

Because step 1's output is saved, you can change step 2's instructions, rules or model and
re-run only the cheap part:

```bash
uv run --env-file .env talk-notes "https://www.youtube.com/watch?v=wXjZdrS3LqA" \
    --from-raw talk.raw.json
```

To watch the retry loop at work, try the deliberately messy fixture in the repo. Its chapters
are out of order, two of them start at the same time, and one starts after the video ends:

```bash
uv run --env-file .env talk-notes "https://www.youtube.com/watch?v=wXjZdrS3LqA" \
    --from-raw evals/fixtures/synthetic_messy.json
```

Step 2 will usually need more than one request here, and the final notes will still pass
every rule.

## Test without calling Gemini {#test}

You don't need an API key to test the pipeline. Pydantic AI's `FunctionModel` lets you
replace Gemini with a function that returns whatever you choose, and `agent.override()` swaps
it in for a test. This test checks the core promise of the design: a bad step 2 answer is
retried, and the video is still watched only once.

```python
async def test_step_2_retries_without_rewatching_the_video(raw_notes: RawNotes) -> None:
    watch_model, watch_calls = scripted_model(raw_notes.model_dump())
    enrich_model, enrich_calls = scripted_model(
        talk_notes_json('0:00', '25:00'),  # starts after the 20:00 video ends
        talk_notes_json('0:00', '5:30'),
    )

    with watcher.override(model=watch_model), enricher.override(model=enrich_model):
        result = await take_notes(VIDEO_URL)

    assert len(watch_calls) == 1
    assert len(enrich_calls) == 2
    assert 'begin after the video ends' in feedback_text(enrich_calls[1])
```

`scripted_model` is a small helper in [`tests/conftest.py`](tests/conftest.py). It returns a
`FunctionModel` that replies with each payload in turn and records every request. The
conftest also sets `models.ALLOW_MODEL_REQUESTS = False`, so a test that accidentally calls
the real API fails immediately.

```bash
uv run pytest
```

## Evaluate step 2 {#evaluate}

Tests check that the code behaves. Evals check that the model does a good job.
[Pydantic Evals](https://pydantic.dev/docs/ai/evals/evals/) runs a task against a dataset of
cases and scores each output.

Because step 1 is saved to disk, you can evaluate step 2 on its own, as often as you like,
without paying for video again. The eval in [`evals/run_evals.py`](evals/run_evals.py) turns
every JSON file in `evals/fixtures/` into a case and applies three evaluators:

* `ChaptersInsideVideo`: a custom check that no chapter starts after the video ends.
* `ToolsLinked`: a custom score for how many of the mentioned tools got a documentation link.
* `LLMJudge`: Gemini reads the raw notes and the final notes, and checks nothing was invented.

```python
dataset = Dataset[RawNotes, TalkNotes, None](
    name='enrich',
    cases=[
        Case(name=path.stem, inputs=RawNotes.model_validate_json(path.read_text()))
        for path in sorted(FIXTURES.glob('*.json'))
    ],
    evaluators=[
        ChaptersInsideVideo(),
        ToolsLinked(),
        LLMJudge(
            rubric='Every chapter summary and takeaway is supported by the raw notes. Nothing is invented.',
            include_input=True,
            model=JUDGE_MODEL,
        ),
    ],
)
```

Add the real talk as a case, then run the eval:

```bash
cp talk.raw.json evals/fixtures/monty-talk.json
uv run --env-file .env evals/run_evals.py
```

Pydantic Evals prints a table with each case's scores and assertions. Run it again after
changing step 2, and you'll see whether your change made the notes better or worse.

## Going further {#going-further}

* **Check citations against search results.** An output validator can read the run's
  messages through `ctx.messages`, including the sources Google Search returned. You could
  reject any `docs_url` that didn't come from a search result.
* **Search your own documents.** Swap `WebSearchTool` for `FileSearchTool` to ground step 2
  in a Gemini File Search store instead of the web.
* **Trace each step.** Pydantic AI is instrumented with OpenTelemetry, so you can send traces
  of both steps to any OpenTelemetry backend.
* **Process a playlist.** `enrich()` is a plain async function, so you can run step 2 for many
  saved talks concurrently with `asyncio.gather`.

## Further resources {#further-resources}

* [Pydantic AI documentation](https://pydantic.dev/docs/ai/)
* [Pydantic AI Google model guide](https://pydantic.dev/docs/ai/models/google/)
* [Structured output in Pydantic AI](https://pydantic.dev/docs/ai/core-concepts/output/)
* [Retries in Pydantic AI](https://pydantic.dev/docs/ai/core-concepts/retries/)
* [Pydantic Evals](https://pydantic.dev/docs/ai/evals/evals/)
* [Gemini video understanding](https://ai.google.dev/gemini-api/docs/generate-content/video-understanding)
