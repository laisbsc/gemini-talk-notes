"""Glue the two steps together: watch once, then validate and enrich as often as needed."""

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
