"""Keyword generation agent built on the OpenAI Agents SDK."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List

from agents import Agent, Runner
from pydantic import BaseModel, Field, ValidationError

from app.core.config import settings

logger = logging.getLogger(__name__)


class KeywordGenerationError(RuntimeError):
    """Raised when the agent response cannot be parsed into the expected schema."""


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
        self.model = self.model or settings.KEYWORD_AGENT_MODEL
        self._agent = Agent(
            name="Financial Indicator Keyword Agent",
            instructions=self._build_instructions(),
            model=self.model,
        )

    def _build_instructions(self) -> str:
        """Constructs the system prompt for the agent."""

        return (
            "You are an elite financial research assistant assigned to design the sparse "
            "keyword layer for a hybrid (dense + sparse) RAG system. Users phrase their "
            "queries almost exactly like the provided `question`, so at least 70% of the "
            "keywords must mirror or paraphrase that question text while still anchoring "
            "on the indicator name, definition and common abbreviations (GDP, CPI, etc.). "
            f"Always return EXACTLY {self.keyword_count} distinct keywords per indicator."
            "\n\nGuidelines:"
            "\n- Think in literal search queries: 1-4 word snippets that a user would type when asking the provided question (e.g., 'how fast gdp growing')."
            "\n- Force coverage across three buckets:"
            "\n  • Question Alignment (>=70%): paraphrases of the question, using the same verbs/nouns (growth, rate, pace, size, etc.)."
            "\n  • Indicator Anchors: official names, normalized names, and abbreviations."
            "\n  • Intent Boosters: short phrases highlighting context from definition/application (e.g., 'constant prices', 'current dollars')."
            "\n- Mix singular/plural, abbreviation/full form, and alternative word order, but avoid punctuation and long descriptive clauses."
            "\n- Never repeat exact strings or add numbering/bullets."
            "\n- Return JSON only, matching this schema exactly:"
            "\n  {\"indicator_name\": string, \"normalized_indicator_name\": string, \"keywords\": [string, ...]}"
            "\n- Ensure the keywords array contains exactly the requested count."
            "\nExample for 'Nominal GDP' question 'How large is an economy … without adjusting for inflation?':"
            "\n  ['nominal gdp', 'gdp current prices', 'economy size current dollars', 'gdp without inflation', 'size of economy question', 'current dollar output', ...]"
        )

    def generate_keywords(self, indicator_data: Dict[str, Any]) -> KeywordGenerationResult:
        """Runs the agent synchronously for a single indicator."""

        if "indicator_name" not in indicator_data:
            raise KeywordGenerationError("Indicator metadata must include 'indicator_name'.")

        formatted_context = self._format_indicator_context(indicator_data)
        logger.debug("Generating keywords for %s", indicator_data.get("indicator_name"))

        result = Runner.run_sync(self._agent, input=formatted_context)
        output_text = result.final_output_as(str)
        parsed = self._parse_agent_response(output_text, indicator_data)

        if len(parsed.keywords) != self.keyword_count:
            raise KeywordGenerationError(
                f"Agent returned {len(parsed.keywords)} keywords for {parsed.indicator_name}; "
                f"expected {self.keyword_count}."
            )

        cleaned_keywords = [kw.strip() for kw in parsed.keywords]
        logger.info(
            "Generated %s keywords for %s using %s",
            len(cleaned_keywords),
            parsed.indicator_name,
            self.model,
        )
        logger.debug("Keywords for %s: %s", parsed.indicator_name, cleaned_keywords)
        return KeywordGenerationResult(
            indicator_name=parsed.indicator_name,
            normalized_indicator_name=parsed.normalized_indicator_name,
            keywords=cleaned_keywords,
        )

    def _format_indicator_context(self, indicator_data: Dict[str, Any]) -> str:
        """Turns indicator metadata into a deterministic context payload for the agent."""

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
        """Extracts JSON from the raw agent text and validates it."""

        cleaned = self._strip_code_fences(raw_output.strip())
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.exception("JSON decoding failed for agent output: %s", raw_output)
            raise KeywordGenerationError(
                "Unable to decode keyword JSON from agent output"
            ) from exc

        data.setdefault("indicator_name", fallback_indicator.get("indicator_name", ""))
        data.setdefault(
            "normalized_indicator_name", fallback_indicator.get("normalized_indicator_name", "")
        )

        try:
            return KeywordGenerationResult(**data)
        except ValidationError as exc:
            logger.exception("Keyword payload failed validation: %s", data)
            raise KeywordGenerationError("Agent response failed validation") from exc

    @staticmethod
    def _strip_code_fences(raw_output: str) -> str:
        """Removes markdown fences and isolates the JSON payload."""

        if raw_output.startswith("```") and raw_output.endswith("```"):
            raw_output = raw_output.strip("`").strip()

        fenced_match = re.search(r"```json(.*?)```", raw_output, re.DOTALL | re.IGNORECASE)
        if fenced_match:
            return fenced_match.group(1).strip()

        generic_match = re.search(r"```(.*?)```", raw_output, re.DOTALL)
        if generic_match:
            return generic_match.group(1).strip()

        return raw_output
