"""Command line entry point: `talk-notes <youtube-url>`."""

import argparse
import asyncio
import sys
from pathlib import Path

from pydantic_ai import RunUsage

from .pipeline import enrich, watch
from .schemas import RawNotes


def describe(step: str, usage: RunUsage) -> str:
    return f'{step}: {usage.requests} request(s), {usage.input_tokens:,} input / {usage.output_tokens:,} output tokens'


async def run(args: argparse.Namespace) -> None:
    if args.from_raw:
        raw = RawNotes.model_validate_json(args.from_raw.read_text())
        print(f'step 1 skipped: loaded raw notes from {args.from_raw}', file=sys.stderr)
    else:
        raw, watch_usage = await watch(args.video_url)
        print(describe('step 1 (watch)', watch_usage), file=sys.stderr)
        if args.save_raw:
            args.save_raw.write_text(raw.model_dump_json(indent=2))
            print(f'raw notes saved to {args.save_raw}', file=sys.stderr)

    notes, enrich_usage = await enrich(raw)
    print(describe('step 2 (enrich)', enrich_usage), file=sys.stderr)

    output = notes.model_dump_json(indent=2) if args.json else notes.to_markdown(args.video_url)
    if args.out:
        args.out.write_text(output)
        print(f'notes written to {args.out}', file=sys.stderr)
    else:
        print(output)


def main() -> None:
    parser = argparse.ArgumentParser(prog='talk-notes', description='Turn a YouTube talk into validated notes.')
    parser.add_argument('video_url', help='A public YouTube URL.')
    parser.add_argument('--save-raw', type=Path, metavar='FILE', help='Save step 1 raw notes to FILE.')
    parser.add_argument('--from-raw', type=Path, metavar='FILE', help='Skip step 1 and enrich raw notes from FILE.')
    parser.add_argument('--out', type=Path, metavar='FILE', help='Write the notes to FILE instead of stdout.')
    parser.add_argument('--json', action='store_true', help='Output JSON instead of Markdown.')
    asyncio.run(run(parser.parse_args()))
