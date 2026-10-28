"""Data models for the two pipeline steps.

`RawNotes` is what Gemini writes while watching the video. It is deliberately lenient:
the call that produces it is the expensive one, so nothing here should ever fail
validation and trigger a retry that re-watches the talk.

`TalkNotes` is the final, strict result. Its validators run against a cheap, text-only
call, where a retry costs a fraction of watching the video again.
"""

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


def link_at(video_url: str, seconds: int) -> str:
    """A link that opens the video at the given second."""
    separator = '&' if '?' in video_url else '?'
    return f'{video_url}{separator}t={seconds}s'


# Step 1: raw notes, written while watching the video.


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


# Step 2: the final notes, validated and enriched.


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

    def to_markdown(self, video_url: str) -> str:
        lines = [f'# {self.title}']
        if self.speaker:
            lines.append(f'*{self.speaker}*')
        lines += ['', self.tldr, '', '## Chapters', '']
        lines += [f'- [{c.start}]({link_at(video_url, c.seconds)}) **{c.title}**: {c.summary}' for c in self.chapters]
        if self.tools:
            lines += ['', '## Tools mentioned', '']
            lines += [f'- [{t.name}]({t.docs_url}): {t.mentioned_for}' for t in self.tools]
        lines += ['', '## Takeaways', '']
        lines += [f'- {takeaway}' for takeaway in self.takeaways]
        return '\n'.join(lines) + '\n'
