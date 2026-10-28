"""Evaluate step 2 against saved raw notes, without re-watching any video.

Every `*.json` file in `evals/fixtures/` becomes a case. The two synthetic fixtures are
made up: one clean, one deliberately messy (chapters out of order, a duplicate start time
and a chapter after the video ends). Add real cases with:

    uv run --env-file .env talk-notes <youtube-url> --save-raw evals/fixtures/my-talk.json

Run with:

    uv run --env-file .env evals/run_evals.py
"""

from dataclasses import dataclass
from pathlib import Path

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, LLMJudge

from talk_notes import RawNotes, TalkNotes, enrich
from talk_notes.schemas import to_seconds

FIXTURES = Path(__file__).parent / 'fixtures'
JUDGE_MODEL = 'google:gemini-3.8-flash'


@dataclass
class ChaptersInsideVideo(Evaluator[RawNotes, TalkNotes]):
    """Every chapter starts before the video ends."""

    def evaluate(self, ctx: EvaluatorContext[RawNotes, TalkNotes]) -> bool:
        duration = to_seconds(ctx.inputs.duration)
        return all(chapter.seconds < duration for chapter in ctx.output.chapters)


@dataclass
class ToolsLinked(Evaluator[RawNotes, TalkNotes]):
    """Share of the tools in the raw notes that ended up with a docs link."""

    def evaluate(self, ctx: EvaluatorContext[RawNotes, TalkNotes]) -> float:
        mentioned = {name.casefold() for name in ctx.inputs.tools_mentioned}
        if not mentioned:
            return 1.0
        linked = {tool.name.casefold() for tool in ctx.output.tools}
        return len(mentioned & linked) / len(mentioned)


async def enrich_task(raw: RawNotes) -> TalkNotes:
    notes, _ = await enrich(raw)
    return notes


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


if __name__ == '__main__':
    report = dataset.evaluate_sync(enrich_task)
    report.print(include_input=False, include_output=False)
