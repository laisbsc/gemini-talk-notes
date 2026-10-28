import json
from typing import Any

import pytest
from pydantic_ai import models
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.profiles import ModelProfile

from talk_notes import RawChapter, RawNotes

# Fail loudly if a test ever tries to reach a real model.
models.ALLOW_MODEL_REQUESTS = False

Calls = list[list[ModelMessage]]


def scripted_model(*replies: dict[str, Any]) -> tuple[FunctionModel, Calls]:
    """A fake model that returns each reply in turn, and records every request it receives."""
    calls: Calls = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append(messages)
        reply = replies[min(len(calls), len(replies)) - 1]
        return ModelResponse(parts=[TextPart(json.dumps(reply))])

    return FunctionModel(respond, profile=ModelProfile(supports_json_schema_output=True)), calls


@pytest.fixture
def raw_notes() -> RawNotes:
    return RawNotes(
        title='Typed agents in Python',
        speaker='Ada Example',
        duration='20:00',
        chapters=[
            RawChapter(start='0:00', title='Intro', notes='Why agents need types.'),
            RawChapter(start='5:30', title='Demo', notes='Builds an agent with Pydantic AI.'),
        ],
        tools_mentioned=['Pydantic AI'],
    )


def talk_notes_json(*starts: str) -> dict[str, Any]:
    """A valid `TalkNotes` payload with chapters at the given start times."""
    return {
        'title': 'Typed agents in Python',
        'speaker': 'Ada Example',
        'tldr': 'Types make agents easier to trust.',
        'chapters': [{'start': start, 'title': f'Part {i}', 'summary': '...'} for i, start in enumerate(starts)],
        'tools': [
            {
                'name': 'Pydantic AI',
                'docs_url': 'https://pydantic.dev/docs/ai/',
                'mentioned_for': 'Building the demo agent.',
            }
        ],
        'takeaways': ['Validate model output.'],
    }
