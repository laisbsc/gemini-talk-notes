"""The pipeline, end to end, with fake models standing in for Gemini.

The key behaviour under test: when step 2 output fails validation, only step 2 is retried.
The video is watched exactly once.
"""

import pytest
from pydantic_ai import UnexpectedModelBehavior, VideoUrl
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart

from talk_notes import RawNotes, take_notes
from talk_notes.agents import enricher, watcher
from talk_notes.pipeline import enrich

from .conftest import scripted_model, talk_notes_json

VIDEO_URL = 'https://www.youtube.com/watch?v=abc'


def feedback_text(messages: list[ModelMessage]) -> str:
    """Everything the agent sent back to the model in its latest request."""
    last = messages[-1]
    assert isinstance(last, ModelRequest)
    return '\n'.join(str(getattr(part, 'content', '')) for part in last.parts)


async def test_watch_sends_the_video_with_agentic_processing(raw_notes: RawNotes) -> None:
    watch_model, watch_calls = scripted_model(raw_notes.model_dump())
    enrich_model, _ = scripted_model(talk_notes_json('0:00', '5:30'))

    with watcher.override(model=watch_model), enricher.override(model=enrich_model):
        result = await take_notes(VIDEO_URL)

    prompt = watch_calls[0][-1]
    assert isinstance(prompt, ModelRequest)
    user_prompt = next(part for part in prompt.parts if isinstance(part, UserPromptPart))
    video = next(item for item in user_prompt.content if isinstance(item, VideoUrl))
    assert video.url == VIDEO_URL
    assert video.vendor_metadata == {'media_processing': 'AGENTIC'}
    assert result.raw == raw_notes
    assert [chapter.start for chapter in result.notes.chapters] == ['0:00', '5:30']


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
    assert result.enrich_usage.requests == 2


async def test_schema_errors_are_fed_back_to_the_model(raw_notes: RawNotes) -> None:
    enrich_model, enrich_calls = scripted_model(talk_notes_json('5:30', '0:00'), talk_notes_json('0:00', '5:30'))

    with enricher.override(model=enrich_model):
        notes, _ = await enrich(raw_notes)

    assert 'strictly increasing' in feedback_text(enrich_calls[1])
    assert [chapter.start for chapter in notes.chapters] == ['0:00', '5:30']


async def test_step_2_gives_up_after_its_retry_budget(raw_notes: RawNotes) -> None:
    enrich_model, enrich_calls = scripted_model(talk_notes_json('0:00', '25:00'))

    with enricher.override(model=enrich_model), pytest.raises(UnexpectedModelBehavior, match='output retries'):
        await enrich(raw_notes)

    assert len(enrich_calls) == 4  # first attempt plus 3 retries


async def test_unparseable_duration_skips_the_duration_check(raw_notes: RawNotes) -> None:
    raw = raw_notes.model_copy(update={'duration': 'about twenty minutes'})
    enrich_model, enrich_calls = scripted_model(talk_notes_json('0:00', '25:00'))

    with enricher.override(model=enrich_model):
        notes, _ = await enrich(raw)

    assert len(enrich_calls) == 1
    assert notes.chapters[-1].start == '25:00'
