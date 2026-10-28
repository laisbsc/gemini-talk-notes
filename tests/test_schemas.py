import pytest
from pydantic import ValidationError

from talk_notes import TalkNotes
from talk_notes.schemas import link_at, to_seconds

from .conftest import talk_notes_json


@pytest.mark.parametrize(('timestamp', 'seconds'), [('0:00', 0), ('4:05', 245), ('59:59', 3599), ('1:02:30', 3750)])
def test_to_seconds(timestamp: str, seconds: int) -> None:
    assert to_seconds(timestamp) == seconds


@pytest.mark.parametrize('timestamp', ['', '4', '4:5', '4:60', 'about 4 minutes'])
def test_to_seconds_rejects_bad_input(timestamp: str) -> None:
    with pytest.raises(ValueError, match='Not a timestamp'):
        to_seconds(timestamp)


def test_chapters_must_be_in_order() -> None:
    with pytest.raises(ValidationError, match='strictly increasing'):
        TalkNotes.model_validate(talk_notes_json('5:30', '0:00'))


def test_duplicate_chapter_starts_are_rejected() -> None:
    with pytest.raises(ValidationError, match='strictly increasing'):
        TalkNotes.model_validate(talk_notes_json('0:00', '0:00'))


def test_docs_url_must_be_https() -> None:
    payload = talk_notes_json('0:00')
    payload['tools'][0]['docs_url'] = 'http://pydantic.dev/docs/ai/'
    with pytest.raises(ValidationError, match='URL scheme'):
        TalkNotes.model_validate(payload)


def test_to_markdown_links_chapters_to_the_video() -> None:
    notes = TalkNotes.model_validate(talk_notes_json('0:00', '5:30'))
    markdown = notes.to_markdown('https://www.youtube.com/watch?v=abc')
    assert '[5:30](https://www.youtube.com/watch?v=abc&t=330s)' in markdown
    assert '[Pydantic AI](https://pydantic.dev/docs/ai/)' in markdown


def test_link_at_without_query_string() -> None:
    assert link_at('https://youtu.be/abc', 90) == 'https://youtu.be/abc?t=90s'
