from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from langchain.agents import create_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase

from app.adapters.llm import create_chat_model
from app.adapters.postgres import build_connection_uri
from app.config.settings import settings
from app.core.logging import ModuleName, get_logger

logger = get_logger(__name__)

BASE_SYSTEM_PROMPT = """You are a helpful SQL analyst working over a Postgres database.
- Use the SQLDatabaseToolkit tools: sql_db_list_tables, sql_db_schema, sql_db_query_checker, sql_db_query.
- Default to listing tables first, but when indicator context supplies the exact tables you may skip straight to sql_db_schema for those tables.
- Always inspect schemas with sql_db_schema before issuing sql_db_query.
- Use sql_db_query_checker to validate every SQL query before execution.
- Limit results to 100 rows unless the user explicitly asks for more.
- Prefer returning tidy result sets that are easy to turn into charts (one column for x-axis/category, one for value, plus optional grouping column).
- When writing SQL, alias output columns using snake_case, e.g., SELECT year, avg(value) AS avg_value ...
- Use double quotes for all JSON keys and string values in the final response.

Final response format (return JSON ONLY, no prose, no Markdown fences):
{{
  "insights": "<2-3 bullet sentences summarizing the finding>",
  "table_rows": [
    {{"<x_field>": "...", "<y_field>": ..., "<group_field>": "... optional ..."}}
  ],
  "chart_schema": {{
    "x_field": "<column name for horizontal axis>",
    "y_field": "<column name for value axis>",
    "group_field": "<optional grouping column or null>",
    "default_chart_type": "<line|bar|table> (default to line unless user asks otherwise>",
    "title": "<concise chart title>",
    "subtitle": "<contextual subtitle or empty string>",
    "legend_title": "<label for legend, or empty string>",
    "legend_position": "<top|bottom|left|right>",
    "axis_titles": {{"x": "<x axis label>", "y": "<y axis label>"}},
    "show_legend": true,
    "show_data_labels": false,
    "show_gridlines": true
  }}
}}
- Provide at most 200 table_rows. Populate chart_schema defaults if the question does not specify them.

Indicator context:
{indicator_context}

If multiple indicators are provided, decide which table(s) best answer the question and query them accordingly.
If a question requests a comparison, join or union the relevant tables by shared dimensions (e.g., year, geography) when possible.
Explain any assumptions inside the final answer through the insights field only."""


def _format_sample_table(sample_table: Sequence[Sequence[Any]] | None) -> str:
    if not sample_table:
        return "    sample table: not provided"
    header = sample_table[0]
    rows = sample_table[1:]
    if not rows:
        return f"    columns: {', '.join(map(str, header))}"
    first_row = rows[0]
    preview = ", ".join(f"{col}={val}" for col, val in zip(header, first_row))
    return f"    columns: {', '.join(map(str, header))}\n    sample row: {preview}"


def _format_indicator_context(indicators: Sequence[Mapping[str, Any]]) -> str:
    sections: list[str] = []
    for indicator in indicators:
        metadata = indicator.get("metadata", {})
        normalized = metadata.get("normalized_indicator_name") or indicator.get(
            "normalized_indicator_name"
        )
        indicator_name = metadata.get("indicator_name", normalized)
        question = metadata.get("question")
        definition = metadata.get("definition")
        table_hint = metadata.get("sheet_signature", {}).get("sample_table")
        section_lines = [f"- {indicator_name} (table: {normalized})"]
        if question:
            section_lines.append(f"    question: {question}")
        if definition:
            section_lines.append(f"    definition: {definition}")
        section_lines.append(_format_sample_table(table_hint))
        sections.append("\n".join(section_lines))
    return "\n".join(sections) if sections else "  (no indicator context provided)"


def create_sql_agent(
    indicators: Sequence[Mapping[str, Any]],
    *,
    extra_instructions: str | None = None,
) -> Any:
    """
    Build a LangChain SQL agent primed with indicator metadata.

    Args:
        indicators: Sequence of Qdrant-style documents containing metadata + sheet_signature.
        extra_instructions: Optional string appended to the system prompt.

    Returns:
        A Runnable agent that accepts {"input": "..."} payloads.
    """
    db = SQLDatabase.from_uri(build_connection_uri())
    llm = create_chat_model(model=settings.sql_agent_model, temperature=0.0)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
    indicator_context = _format_indicator_context(indicators)
    logger.debug(
        "Initializing SQL agent with %s indicator(s)",
        len(indicators),
        extra={"module_name": ModuleName.ADAPTER},
    )
    system_prompt = BASE_SYSTEM_PROMPT.format(indicator_context=indicator_context)
    if extra_instructions:
        system_prompt = f"{system_prompt}\n\nAdditional guidance:\n{extra_instructions}"
    agent = create_agent(llm, tools, system_prompt=system_prompt)
    return agent


def _extract_output_text(agent_result: Any) -> str:
    if isinstance(agent_result, str):
        return agent_result
    if isinstance(agent_result, Mapping):
        messages = agent_result.get("messages")
        if isinstance(messages, Sequence) and messages:
            for msg in reversed(messages):
                content = getattr(msg, "content", None)
                if isinstance(content, str) and content.strip():
                    return content
                if isinstance(content, list):
                    # Some LangChain messages use list-of-chunks structure
                    text_chunks = [c.get("text") for c in content if isinstance(c, Mapping) and c.get("text")]
                    if text_chunks:
                        return "\n".join(text_chunks)
        for key in ("output", "final_output", "result", "response"):
            if key in agent_result:
                return str(agent_result[key])
    return str(agent_result)


def parse_agent_response(agent_result: Any) -> dict[str, Any]:
    """
    Attempt to coerce the agent output into a JSON dict.
    """
    raw_text = _extract_output_text(agent_result).strip()
    if raw_text.startswith("```"):
        # Strip optional markdown fences like ```json ... ```
        stripped = raw_text.strip("`").strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
        raw_text = stripped
    # Try full string first
    try:
        parsed = json.loads(raw_text)
        logger.debug("SQL agent returned valid JSON payload", extra={"module_name": ModuleName.ADAPTER})
        return parsed
    except json.JSONDecodeError as exc:
        logger.warning(
            "Direct JSON parse failed: %s | raw=%s",
            exc,
            raw_text[:500],
            extra={"module_name": ModuleName.ADAPTER},
        )
    # Fallback: look for the largest JSON object inside the text
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = raw_text[start : end + 1]
        try:
            parsed = json.loads(snippet)
            logger.debug(
                "Recovered JSON by slicing response", extra={"module_name": ModuleName.ADAPTER}
            )
            return parsed
        except json.JSONDecodeError as exc:
            logger.error(
                "Snippet JSON parse failed: %s | snippet=%s",
                exc,
                snippet[:500],
                extra={"module_name": ModuleName.ADAPTER},
            )
    raise ValueError(f"Agent response was not valid JSON: {raw_text}")


def run_sql_agent(
    query: str,
    indicators: Sequence[Mapping[str, Any]],
    *,
    extra_instructions: str | None = None,
) -> dict[str, Any]:
    """
    Convenience wrapper that instantiates the agent and runs a single query.
    """
    logger.info(
        "Running SQL agent for query=%s indicators=%s",
        query,
        [i.get("metadata", {}).get("normalized_indicator_name") for i in indicators],
        extra={"module_name": ModuleName.ADAPTER},
    )
    agent = create_sql_agent(indicators, extra_instructions=extra_instructions)
    result = agent.invoke({"input": query})
    return parse_agent_response(result)
