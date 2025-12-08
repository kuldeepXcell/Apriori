"""Keyword generation preprocess step and CLI."""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from agents import Agent, Runner
from pydantic import BaseModel, Field, ValidationError

from app.config import paths
from app.config.settings import settings


class KeywordGenerationError(RuntimeError):
    """Raised when the agent response cannot be parsed or validated."""


class KeywordGenerationResult(BaseModel):
    """Structured payload returned after generating keywords for an indicator."""

    indicator_name: str
    normalized_indicator_name: str
    keywords: List[str] = Field(..., description="List of generated keywords", min_length=1)


@dataclass
class KeywordGenerationAgent:
    """Agent wrapper that produces search keywords for financial indicators."""

    keyword_count: int = 15
    model: str | None = None
    prompt_version: str = "v1"

    def __post_init__(self) -> None:
        # Initialize the agent with configured or default model.
        self.model = self.model or settings.KEYWORD_AGENT_MODEL
        self._agent = Agent(
            name="Financial Indicator Keyword Agent",
            instructions=self._build_instructions(),
            model=self.model,
        )

    def _build_instructions(self) -> str:
        """Construct the system prompt for the agent."""

        return (
            "You are an elite financial research assistant assigned to design the sparse "
            "keyword layer for a hybrid (dense + sparse) RAG system. Users phrase their "
            "queries almost exactly like the provided `question`, so at least 70% of the "
            "keywords must mirror or paraphrase that question text while still anchoring "
            "on the indicator name, definition and common abbreviations (GDP, CPI, etc.). "
            f"Always return EXACTLY {self.keyword_count} distinct keywords per indicator."
            "\n\nGuidelines:"
            "\n- Think in literal search queries: 1-4 word snippets that a user would type when asking the provided question."
            "\n- Force coverage across three buckets:"
            "\n  • Question Alignment (>=70%): paraphrases of the question, using the same verbs/nouns."
            "\n  • Indicator Anchors: official names, normalized names, and abbreviations."
            "\n  • Intent Boosters: short phrases highlighting context from definition/application."
            "\n- Mix singular/plural, abbreviation/full form, and alternative word order, but avoid punctuation and long clauses."
            "\n- Never repeat exact strings or add numbering/bullets."
            "\n- Return JSON only, matching this schema exactly:"
            "\n  {\"indicator_name\": string, \"normalized_indicator_name\": string, \"keywords\": [string, ...]}"
            "\n- Ensure the keywords array contains exactly the requested count."
        )

    def generate_keywords(self, indicator_data: Dict[str, Any]) -> KeywordGenerationResult:
        """Run the agent synchronously for a single indicator."""

        if "indicator_name" not in indicator_data:
            raise KeywordGenerationError("Indicator metadata must include 'indicator_name'.")

        formatted_context = self._format_indicator_context(indicator_data)
        result = Runner.run_sync(self._agent, input=formatted_context)
        output_text = result.final_output_as(str)
        parsed = self._parse_agent_response(output_text, indicator_data)

        if len(parsed.keywords) != self.keyword_count:
            raise KeywordGenerationError(
                f"Agent returned {len(parsed.keywords)} keywords for {parsed.indicator_name}; "
                f"expected {self.keyword_count}."
            )

        cleaned_keywords = [kw.strip() for kw in parsed.keywords]
        logging.info("Generated %s keywords for %s", len(cleaned_keywords), parsed.indicator_name)
        logging.debug("Keywords for %s: %s", parsed.indicator_name, cleaned_keywords)
        return KeywordGenerationResult(
            indicator_name=parsed.indicator_name,
            normalized_indicator_name=parsed.normalized_indicator_name,
            keywords=cleaned_keywords,
        )

    def _format_indicator_context(self, indicator_data: Dict[str, Any]) -> str:
        """Turn indicator metadata into a deterministic context payload for the agent."""

        parts = [
            "You must study the following indicator metadata and respond per instructions.",
            f"Prompt version: {self.prompt_version}",
            f"Target keyword count: {self.keyword_count}",
            "---\nIndicator Metadata:",
        ]

        for key in [
            "subsection",
            "subsubsection",
            "indicator_name",
            "normalized_indicator_name",
            "definition",
            "question",
            "application_context",
        ]:
            value = indicator_data.get(key)
            if value:
                parts.append(f"{key}: {value}")

        return "\n".join(parts)

    def _parse_agent_response(
        self, raw_output: str, fallback_indicator: Dict[str, Any]
    ) -> KeywordGenerationResult:
        """Extract JSON from the raw agent text and validate it."""

        cleaned = self._strip_code_fences(raw_output.strip())
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logging.exception("JSON decoding failed for agent output: %s", raw_output)
            raise KeywordGenerationError("Unable to decode keyword JSON from agent output") from exc

        data.setdefault("indicator_name", fallback_indicator.get("indicator_name", ""))
        data.setdefault("normalized_indicator_name", fallback_indicator.get("normalized_indicator_name", ""))

        try:
            return KeywordGenerationResult(**data)
        except ValidationError as exc:
            logging.exception("Keyword payload failed validation: %s", data)
            raise KeywordGenerationError("Agent response failed validation") from exc

    @staticmethod
    def _strip_code_fences(raw_output: str) -> str:
        """Remove markdown fences and isolate the JSON payload."""

        if raw_output.startswith("```") and raw_output.endswith("```"):
            raw_output = raw_output.strip("`").strip()

        fenced_match = re.search(r"```json(.*?)```", raw_output, re.DOTALL | re.IGNORECASE)
        if fenced_match:
            return fenced_match.group(1).strip()

        generic_match = re.search(r"```(.*?)```", raw_output, re.DOTALL)
        if generic_match:
            return generic_match.group(1).strip()

        return raw_output


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    default_input = paths.DATA_DIR / "country_indicators.json"
    parser = argparse.ArgumentParser(
        description="Generate sparse keywords for indicators using the OpenAI Agents SDK",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help="Path to the source indicator JSON file",
    )
    parser.add_argument(
        "--sheet",
        required=True,
        help="Sheet name within the payload to target (e.g., 'Country')",
    )
    parser.add_argument(
        "--range",
        dest="row_range",
        required=True,
        help="Inclusive start, exclusive end indicator indices (e.g., 0:10 or 5:).",
    )
    parser.add_argument(
        "--keywords",
        type=int,
        default=15,
        help="Number of keywords to generate per indicator",
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
        help="Regenerate keywords even if an indicator already has them",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def parse_range(range_arg: str, max_len: int) -> Tuple[int, int]:
    """Convert a CLI range string into start/end indices."""

    parts = range_arg.split(":")
    if len(parts) != 2:
        raise SystemExit("Range must be in the form start:end (exclusive end).")

    start_str, end_str = parts
    try:
        start = int(start_str) if start_str else 0
        end = int(end_str) if end_str else max_len
    except ValueError as exc:
        raise SystemExit("Range bounds must be integers.") from exc

    if start < 0 or end < 0 or start >= end:
        raise SystemExit(f"Invalid range {range_arg}; ensure 0 <= start < end.")
    if start >= max_len:
        raise SystemExit(f"Range start {start} exceeds indicator length {max_len}.")

    return start, min(end, max_len)


def load_payload(path: Path) -> Dict[str, Any]:
    """Load the indicator payload from disk."""

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    with path.open("r", encoding="utf-8") as src:
        return json.load(src)


def write_payload(path: Path, payload: Dict[str, Any]) -> None:
    """Atomically persist the JSON payload."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as sink:
        json.dump(payload, sink, indent=2, ensure_ascii=False)
    temp_path.replace(path)


def select_sheet(payload: Dict[str, Any], sheet_name: str) -> Dict[str, Any]:
    """Locate the target sheet by name."""

    for sheet in payload.get("sheets", []):
        if sheet.get("sheet_name") == sheet_name:
            return sheet
    raise SystemExit(f"Sheet '{sheet_name}' not found in payload.")


def assert_overwrite_policy(indicators: List[Dict[str, Any]], overwrite: bool, sheet: str, start: int, end: int) -> None:
    """Fail fast if keywords already exist and overwrite is false."""

    conflicted = []
    for idx, indicator in enumerate(indicators, start=start):
        existing = indicator.get("keywords")
        if isinstance(existing, list) and existing:
            conflicted.append(f"{indicator.get('indicator_name', 'unknown')}@{idx}")
    if conflicted and not overwrite:
        raise SystemExit(
            f"Keywords already present for {len(conflicted)} indicators in sheet '{sheet}' "
            f"range {start}:{end}: {', '.join(conflicted)}. Pass --overwrite to regenerate."
        )


def append_run_record(sheet: Dict[str, Any], record: Dict[str, Any]) -> None:
    """Append a run record to the sheet for traceability."""

    sheet.setdefault("keyword_generation_runs", []).append(record)


def main() -> None:
    """CLI entrypoint to generate keywords for a slice of indicators."""

    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    payload = load_payload(args.input)
    sheet = select_sheet(payload, args.sheet)
    indicators: List[Dict[str, Any]] = sheet.get("indicators", [])
    if not indicators:
        raise SystemExit(f"No indicators found for sheet '{args.sheet}'.")

    start, end = parse_range(args.row_range, len(indicators))
    target_indicators = indicators[start:end]
    if not target_indicators:
        raise SystemExit("Selected range produced no indicators to process.")

    assert_overwrite_policy(target_indicators, args.overwrite, args.sheet, start, end)
    agent = KeywordGenerationAgent(keyword_count=args.keywords, model=args.model)

    processed = 0
    # Generate and persist keywords for each indicator in the requested slice.
    for idx, indicator in enumerate(target_indicators, start=start):
        indicator_name = indicator.get("indicator_name", f"indicator_{idx}")
        logging.info("Generating keywords for %s (index %s)", indicator_name, idx)
        try:
            result = agent.generate_keywords(indicator)
        except KeywordGenerationError as exc:
            logging.error("Keyword generation failed for %s: %s", indicator_name, exc)
            raise SystemExit(f"Failed to generate keywords for '{indicator_name}': {exc}") from exc

        indicator["keywords"] = result.keywords
        processed += 1
        write_payload(args.input, payload)
        logging.debug("Persisted keywords for %s", indicator_name)

    # Record the run metadata for traceability.
    run_record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "sheet": args.sheet,
        "range": f"{start}:{end}",
        "keyword_count": agent.keyword_count,
        "model": agent.model,
        "processed": processed,
        "overwrite": args.overwrite,
    }
    append_run_record(sheet, run_record)
    write_payload(args.input, payload)
    logging.info("Keyword generation complete for %s indicators.", processed)


if __name__ == "__main__":
    main()

