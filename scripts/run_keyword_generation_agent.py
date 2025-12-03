"""Generate indicator keywords via the OpenAI Agents SDK."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import sys

# Ensure project root is on PYTHONPATH when executed via `uv run python ...`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.keyword_generation_agent import (
    KeywordGenerationAgent,
    KeywordGenerationError,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate sparse keywords for indicators using the OpenAI Agents SDK",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("country_indicators.json"),
        help="Path to the source indicator JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save the enriched JSON output (defaults to updating the input file)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of indicators to process (useful for smoke tests)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the model used by the keyword agent",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output file",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging verbosity",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate keywords even if an indicator already has them",
    )
    return parser.parse_args()


def load_indicators(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with path.open("r", encoding="utf-8") as src:
        return json.load(src)


def write_payload(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_suffix = path.suffix + ".tmp"
    temp_path = path.with_suffix(temp_suffix) if temp_suffix else path.with_name(path.name + ".tmp")
    with temp_path.open("w", encoding="utf-8") as sink:
        json.dump(payload, sink, indent=2, ensure_ascii=False)
    temp_path.replace(path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger.info(
        "Starting keyword generation | input=%s output=%s limit=%s model_override=%s",
        args.input,
        args.output or args.input,
        args.limit,
        args.model,
    )
    payload = load_indicators(args.input)
    indicators: List[Dict[str, Any]] = payload.get("indicators", [])

    if not indicators:
        logger.error("No indicators found in the input payload: %s", args.input)
        raise SystemExit("No indicators found in the input payload.")

    target_path = args.output or args.input
    if target_path.exists() and not args.overwrite:
        logger.error(
            "Output file exists and overwrite is disabled: %s",
            target_path,
        )
        raise SystemExit(
            f"Output file {target_path} already exists. Pass --overwrite to replace it."
        )

    agent = KeywordGenerationAgent(model=args.model)

    processed = 0
    skipped = 0
    for idx, indicator in enumerate(indicators, start=1):
        if args.limit is not None and processed >= args.limit:
            logger.info("Limit reached (%s). Stopping early.", args.limit)
            break
        indicator_name = indicator.get("indicator_name", f"indicator_{idx}")

        existing_keywords = indicator.get("keywords")
        if (
            not args.force
            and isinstance(existing_keywords, list)
            and existing_keywords
            and all(isinstance(k, str) for k in existing_keywords)
        ):
            logger.info(
                "[%s/%s] Skipping %s (already has %s keywords)",
                idx,
                len(indicators),
                indicator_name,
                len(existing_keywords),
            )
            skipped += 1
            continue
        logger.info(
            "[%s/%s] Generating keywords for %s",
            idx,
            len(indicators),
            indicator_name,
        )

        try:
            result = agent.generate_keywords(indicator)
        except KeywordGenerationError as exc:
            logger.exception(
                "Keyword generation failed for %s", indicator_name,
            )
            raise SystemExit(f"Failed to generate keywords for '{indicator_name}': {exc}") from exc

        indicator["keywords"] = result.keywords
        logger.debug("Keywords for %s: %s", indicator_name, result.keywords)
        processed += 1

        payload["keyword_generation"] = {
            "keyword_count": agent.keyword_count,
            "model": agent.model,
            "prompt_version": agent.prompt_version,
            "processed": processed,
            "skipped": skipped,
        }
        payload["total_indicators"] = len(indicators)
        write_payload(target_path, payload)
        logger.debug("Persisted progress for %s", indicator_name)

    if processed == 0:
        logger.warning("No indicators processed; output file left untouched.")
    else:
        logger.info(
            "Keyword generation complete. %s indicators updated at %s",
            processed,
            target_path,
        )


if __name__ == "__main__":
    main()
