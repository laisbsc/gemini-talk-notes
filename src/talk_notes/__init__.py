"""Turn a YouTube talk into validated notes with Gemini and Pydantic AI."""

from .pipeline import TalkNotesResult, enrich, take_notes, watch
from .schemas import Chapter, RawChapter, RawNotes, TalkNotes, ToolRef

__all__ = [
    'Chapter',
    'RawChapter',
    'RawNotes',
    'TalkNotes',
    'TalkNotesResult',
    'ToolRef',
    'enrich',
    'take_notes',
    'watch',
]
