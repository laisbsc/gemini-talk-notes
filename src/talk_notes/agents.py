"""The two agents behind the pipeline.

1. `watcher` watches the video with Gemini's agentic video processing and writes raw notes.
   It is the expensive step, so it gets no output retries: a retry would re-watch the talk.
2. `enricher` turns raw notes into strict `TalkNotes` on a cheaper model, using Google Search
   to find documentation links. Validation failures are sent back to the model to fix,
   and each retry only costs a small text-only request.
"""

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
