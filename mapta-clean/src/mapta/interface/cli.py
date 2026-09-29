"""Command-line entry point.

Parsing, logging and printing live here; the scan itself is the application
layer's job.
"""

import argparse
import asyncio
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from ..application import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT, ScanRequest, default_registry
from ..application.scanning import summarise
from ..config import Settings, load_dotenv
from ..domain import ConfigurationError, MaptaError, ScanOutcome, Target, targets_from_lines
from .container import build_scan_service

__all__ = ["build_parser", "main"]

logger = logging.getLogger("mapta")

EXIT_OK = 0
EXIT_SCAN_FAILED = 1
EXIT_BAD_CONFIG = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mapta",
        description=(
            "MAPTA - autonomous multi-agent penetration testing. "
            "Only run this against targets you are authorised to test."
        ),
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "-t", "--target",
        action="append",
        metavar="URL",
        help="Target URL to scan. Repeat the flag to scan several targets in parallel.",
    )
    source.add_argument(
        "-f", "--targets-file",
        default="targets.txt",
        metavar="PATH",
        help="File with one target URL per line ('#' starts a comment). Default: targets.txt",
    )
    parser.add_argument(
        "-p", "--prompt",
        default=None,
        help="Scan instruction template. '{target_url}' is replaced with each target. "
             "Defaults to $USER_PROMPT or the built-in full-scan prompt.",
    )
    parser.add_argument(
        "-s", "--system-prompt",
        default=None,
        help="System prompt for the main agent. Defaults to $SYSTEM_PROMPT or the built-in one.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Directory for reports and usage logs. Defaults to $OUTPUT_DIR or scan-results.",
    )
    parser.add_argument(
        "-r", "--max-rounds",
        type=int,
        default=None,
        help="Maximum tool-execution rounds for the main agent (0 = unlimited). Default: 100",
    )
    parser.add_argument(
        "-c", "--max-concurrency",
        type=int,
        default=None,
        help="Maximum scans running at once (0 = unlimited). Default: 0",
    )
    parser.add_argument(
        "-m", "--model",
        default=None,
        help="OpenRouter model slug (see https://openrouter.ai/models). "
             "Defaults to $MODEL, else openai/gpt-5.",
    )
    parser.add_argument(
        "--reasoning",
        choices=["high", "medium", "low", "none"],
        default=None,
        help="Reasoning effort. Use 'none' for models that reject a reasoning field.",
    )
    parser.add_argument(
        "--log-file",
        default="scan_usage.log",
        help="Path to the run log. Default: scan_usage.log",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Log at DEBUG level.",
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="Print the generated tool schemas and exit (no API key required).",
    )
    return parser


def configure_logging(log_file: str, *, verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


def resolve_targets(args: argparse.Namespace) -> list[Target]:
    """Explicit --target flags win over the targets file."""
    if args.target:
        return [Target(url) for url in args.target]

    path = Path(args.targets_file)
    if not path.exists():
        raise ConfigurationError(
            f"No targets given: pass --target URL, or create {path} with one URL per line."
        )
    targets = targets_from_lines(path.read_text(encoding="utf-8").splitlines())
    if not targets:
        raise ConfigurationError(f"No valid targets found in {path}.")
    return targets


def apply_overrides(settings: Settings, args: argparse.Namespace) -> Settings:
    """Fold CLI flags into the environment-derived settings."""
    from dataclasses import replace

    scan = settings.scan
    if args.system_prompt is not None:
        scan = replace(scan, system_prompt=args.system_prompt)
    if args.prompt is not None:
        scan = replace(scan, user_prompt=args.prompt)
    if args.max_rounds is not None:
        scan = replace(scan, max_rounds=args.max_rounds)
    if args.max_concurrency is not None:
        scan = replace(scan, max_concurrency=args.max_concurrency)

    reports = settings.reports
    if args.output_dir is not None:
        reports = replace(reports, output_dir=args.output_dir)

    return replace(settings, scan=scan, reports=reports)


def print_summary(outcomes: Sequence[ScanOutcome]) -> None:
    counts = summarise(outcomes)
    print("\nScan summary")
    print(f"  Targets:    {counts['targets']}")
    print(f"  Completed:  {counts['completed']}")
    print(f"  Failed:     {counts['failed']}")
    print(f"  Model calls:{counts['model_calls']:>4}")
    for outcome in outcomes:
        if outcome.succeeded:
            print(f"  ✔ {outcome.target.url} -> {outcome.report_path}")
        else:
            print(f"  ✘ {outcome.target.url}: {outcome.error}")


async def run(settings: Settings, targets: Sequence[Target]) -> list[ScanOutcome]:
    container = build_scan_service(settings)
    request = ScanRequest(
        targets=tuple(targets),
        prompt_template=settings.scan.user_prompt or DEFAULT_USER_PROMPT,
        max_concurrency=settings.scan.max_concurrency,
    )
    try:
        return await container.scans.run(request)
    finally:
        await container.aclose()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_tools:
        specs = [spec.as_dict() for spec in default_registry().specs()]
        print(json.dumps(specs, indent=2))
        return EXIT_OK

    load_dotenv()
    configure_logging(args.log_file, verbose=args.verbose)

    try:
        settings = apply_overrides(
            Settings.from_env(model=args.model, reasoning=args.reasoning),
            args,
        )
        targets = resolve_targets(args)
    except MaptaError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_BAD_CONFIG

    prompt = settings.scan.user_prompt or DEFAULT_USER_PROMPT
    if "{target_url}" not in prompt:
        print(
            "[warn] the scan prompt does not contain '{target_url}'; the agent will not be "
            "told which target to scan.",
            file=sys.stderr,
        )
    if not (settings.scan.system_prompt or DEFAULT_SYSTEM_PROMPT):  # pragma: no cover
        print("[warn] empty system prompt", file=sys.stderr)

    print(f"Provider: OpenRouter | model: {settings.llm.model}")
    print(f"Sandbox:  {settings.sandbox.provider}")
    print(f"Scanning {len(targets)} target(s)")

    try:
        outcomes = asyncio.run(run(settings, targets))
    except MaptaError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_BAD_CONFIG
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print("\nInterrupted.", file=sys.stderr)
        return EXIT_SCAN_FAILED

    print_summary(outcomes)
    return EXIT_OK if all(outcome.succeeded for outcome in outcomes) else EXIT_SCAN_FAILED
