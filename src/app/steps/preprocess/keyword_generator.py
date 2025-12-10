"""Keyword generation preprocess step for indicators before ingestion."""

from __future__ import annotations
import json
import logging
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Field, ValidationError

# Ensure project src/ is on sys.path when executed as a script.
ROOT_DIR = Path(__file__).resolve().parents[4]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app.adapters import llm
from app.core.logging import ModuleName, get_logger, setup_logging

# Resolve project root relative to this file (src/app/steps/preprocess/..).
DEFAULT_INPUT_PATH = ROOT_DIR / "data" / "country_indicators.json"
DEFAULT_SHEET_NAME = "Country"
DEFAULT_RANGE = "0:1"  # inclusive start, exclusive end
DEFAULT_KEYWORD_COUNT = 15
DEFAULT_OVERWRITE = True
DEFAULT_KEYWORD_MODEL = "gpt-5-nano"

logger = get_logger(__name__)


class KeywordGenerationError(RuntimeError):
    """Raised when keyword generation or parsing fails."""


class KeywordGenerationResult(BaseModel):
    indicator_name: str
    normalized_indicator_name: str
    keywords: List[str] = Field(..., description="Generated keywords", min_length=1)


@dataclass
class KeywordGenerator:
    """Agent wrapper that produces search keywords for financial indicators."""

    keyword_count: int = DEFAULT_KEYWORD_COUNT
    model: str = DEFAULT_KEYWORD_MODEL

    def __post_init__(self) -> None:
        # model is fixed per user instruction; still allow override via initializer/CLI
        self.model = self.model or DEFAULT_KEYWORD_MODEL

    def _system_prompt(self) -> str:
        """Construct the system prompt for the agent."""
        return (
            "You are an elite financial research assistant assigned to design the sparse "
            "keyword layer for a hybrid (dense + sparse) RAG system. Users phrase their "
            "queries almost exactly like the provided `question`, so at least 70% of the "
            "keywords must mirror or paraphrase that question text while still anchoring "
            "on the indicator name, definition and common abbreviations (GDP, CPI, etc.). "
            f"Always return EXACTLY {self.keyword_count} distinct keywords per indicator."
            "\n\nGuidelines:"
            "\n- Think in literal search queries: 1-4 word snippets a user would type."
            "\n- Force coverage across three buckets:"
            "\n  • Question Alignment (>=70%): paraphrases of the question."
            "\n  • Indicator Anchors: official names, normalized names, abbreviations."
            "\n  • Intent Boosters: short context phrases from definition/application."
            "\n- Mix singular/plural, abbreviation/full form; avoid punctuation and long clauses."
            "\n- Never repeat exact strings or add numbering/bullets."
            "\n- Return JSON only, matching this schema exactly:"
            "\n  {\"indicator_name\": string, \"normalized_indicator_name\": string, \"keywords\": [string, ...]}"
            "\n- Ensure the keywords array contains exactly the requested count."
        )

    def _format_indicator_context(self, indicator: Dict[str, Any]) -> str:
        """Serialize indicator metadata for the agent."""
        parts = [
            "You must study the following indicator metadata and respond per instructions.",
            "---",
            "Indicator Metadata:",
        ]
        for key in [
            "subsection",
            "subsubsection",
            "indicator_name",
            "definition",
            "question",
            "application_context",
        ]:
            value = indicator.get(key)
            if value:
                parts.append(f"{key}: {value}")
        return "\n".join(parts)

    @staticmethod
    def _strip_code_fences(raw_output: str) -> str:
        """Remove markdown fences and isolate JSON payload."""
        raw_output = raw_output.strip()
        if raw_output.startswith("```") and raw_output.endswith("```"):
            raw_output = raw_output.strip("`").strip()
        fenced_match = re.search(r"```json(.*?)```", raw_output, re.DOTALL | re.IGNORECASE)
        if fenced_match:
            return fenced_match.group(1).strip()
        generic_match = re.search(r"```(.*?)```", raw_output, re.DOTALL)
        if generic_match:
            return generic_match.group(1).strip()
        return raw_output

    def _parse_agent_response(
        self, raw_output: str, fallback_indicator: Dict[str, Any]
    ) -> KeywordGenerationResult:
        """Extract JSON from the agent text and validate it."""
        cleaned = self._strip_code_fences(raw_output)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.exception(
                "JSON decoding failed for agent output", extra={"module_name": ModuleName.PREPROCESS}
            )
            raise KeywordGenerationError("Unable to decode keyword JSON from agent output") from exc

        data.setdefault("indicator_name", fallback_indicator.get("indicator_name", ""))
        data.setdefault("normalized_indicator_name", fallback_indicator.get("normalized_indicator_name", ""))

        try:
            return KeywordGenerationResult(**data)
        except ValidationError as exc:
            logger.exception(
                "Keyword payload failed validation", extra={"module_name": ModuleName.PREPROCESS}
            )
            raise KeywordGenerationError("Agent response failed validation") from exc

    def generate_keywords(self, indicator: Dict[str, Any]) -> KeywordGenerationResult:
        """Run the agent synchronously for a single indicator."""
        if "indicator_name" not in indicator:
            raise KeywordGenerationError("Indicator metadata must include 'indicator_name'.")

        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": self._format_indicator_context(indicator)},
        ]
        response = llm.chat(messages=messages, model=self.model, temperature=1)
        output_text = response.choices[0].message.content if response.choices else ""

        parsed = self._parse_agent_response(output_text, indicator)
        if len(parsed.keywords) != self.keyword_count:
            raise KeywordGenerationError(
                f"Agent returned {len(parsed.keywords)} keywords for {parsed.indicator_name}; "
                f"expected {self.keyword_count}."
            )
        cleaned_keywords = [kw.strip() for kw in parsed.keywords]
        logger.info(
            "Generated keywords",
            extra={
                "module_name": ModuleName.PREPROCESS,
                "indicator": parsed.indicator_name,
                "count": len(cleaned_keywords),
            },
        )
        logger.debug(
            "Keywords detail",
            extra={
                "module_name": ModuleName.PREPROCESS,
                "indicator": parsed.indicator_name,
                "keywords": cleaned_keywords,
            },
        )
        return KeywordGenerationResult(
            indicator_name=parsed.indicator_name,
            normalized_indicator_name=parsed.normalized_indicator_name,
            keywords=cleaned_keywords,
        )


def load_payload(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return json.loads(path.read_text())


def write_payload(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    temp_path.replace(path)


def select_sheet(payload: Dict[str, Any], sheet_name: str) -> Dict[str, Any]:
    for sheet in payload.get("sheets", []):
        if sheet.get("sheet_name") == sheet_name:
            return sheet
    raise SystemExit(f"Sheet '{sheet_name}' not found in payload.")


def parse_range(range_arg: str, max_len: int) -> Tuple[int, int]:
    parts = range_arg.split(":")
    if len(parts) != 2:
        raise SystemExit("Range must be start:end (exclusive end).")
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


def assert_overwrite_policy(indicators: List[Dict[str, Any]], overwrite: bool, sheet: str, start: int, end: int) -> None:
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
    sheet.setdefault("keyword_generation_runs", []).append(record)


def generate_for_file(
    input_path: Path,
    sheet_name: str,
    range_arg: str,
    keyword_count: int,
    overwrite: bool,
    model: str | None = None,
) -> int:
    payload = load_payload(input_path)
    sheet = select_sheet(payload, sheet_name)
    indicators: List[Dict[str, Any]] = sheet.get("indicators", [])
    if not indicators:
        raise SystemExit(f"No indicators found for sheet '{sheet_name}'.")

    start, end = parse_range(range_arg, len(indicators))
    target = indicators[start:end]
    if not target:
        raise SystemExit("Selected range produced no indicators to process.")

    assert_overwrite_policy(target, overwrite, sheet_name, start, end)
    generator = KeywordGenerator(keyword_count=keyword_count, model=model)

    processed = 0
    for idx, indicator in enumerate(target, start=start):
        name = indicator.get("indicator_name", f"indicator_{idx}")
        logger.info(
            "Generating keywords",
            extra={
                "module_name": ModuleName.PREPROCESS,
                "indicator": name,
                "index": idx,
                "sheet": sheet_name,
            },
        )
        try:
            result = generator.generate_keywords(indicator)
        except KeywordGenerationError as exc:
            logger.error(
                "Keyword generation failed",
                extra={
                    "module_name": ModuleName.PREPROCESS,
                    "indicator": name,
                    "error": str(exc),
                },
            )
            raise SystemExit(f"Failed to generate keywords for '{name}': {exc}") from exc

        indicator["keywords"] = result.keywords
        processed += 1
        write_payload(input_path, payload)
        logger.debug(
            "Persisted keywords",
            extra={"module_name": ModuleName.PREPROCESS, "indicator": name, "index": idx},
        )

    run_record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "sheet": sheet_name,
        "range": f"{start}:{end}",
        "keyword_count": keyword_count,
        "model": generator.model,
        "processed": processed,
        "overwrite": overwrite,
    }
    append_run_record(sheet, run_record)
    write_payload(input_path, payload)
    logger.info(
        "Keyword generation complete",
        extra={
            "module_name": ModuleName.PREPROCESS,
            "sheet": sheet_name,
            "processed": processed,
        },
    )
    return processed


def main() -> None:
    """Run keyword generation with in-file defaults (no CLI arguments)."""
    setup_logging()  # defaults to INFO unless overridden by LOG_LEVEL env
    generate_for_file(
        input_path=DEFAULT_INPUT_PATH,
        sheet_name=DEFAULT_SHEET_NAME,
        range_arg=DEFAULT_RANGE,
        keyword_count=DEFAULT_KEYWORD_COUNT,
        overwrite=DEFAULT_OVERWRITE,
        model=DEFAULT_KEYWORD_MODEL,
    )


if __name__ == "__main__":
    main()

