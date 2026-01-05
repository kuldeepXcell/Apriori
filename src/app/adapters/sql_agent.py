from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from langchain.agents import create_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from pydantic import ValidationError, create_model, ConfigDict

from app.adapters.llm import create_chat_model
from app.adapters.postgres import build_connection_uri
from app.adapters.sql_agent_models import SQLAgentResponse
from app.adapters.sql_agent_prompts import build_system_prompt
from app.config.settings import settings
from app.core.logging import ModuleName, get_logger

logger = get_logger(__name__)

def _format_indicator_context(indicators: Sequence[Mapping[str, Any]]) -> str:
    sections: list[str] = []
    for indicator in indicators:
        metadata = indicator.get("metadata", {})
        normalized = metadata.get("normalized_indicator_name") or indicator.get("normalized_indicator_name")
        indicator_name = metadata.get("indicator_name", normalized)
        definition = metadata.get("definition")
        application_context = metadata.get("application_context")
        file_info = metadata.get("file_info") or {}
        sheet_dimensions = file_info.get("sheet_dimensions") or []
        num_sheets = file_info.get("number_of_sheets")
        if not isinstance(num_sheets, int) and isinstance(sheet_dimensions, list):
            num_sheets = len(sheet_dimensions)

        table_name = normalized if num_sheets == 1 else None
        section_lines = [f"- {indicator_name}"]
        if definition:
            section_lines.append(f"    definition: {definition}")
        if application_context:
            section_lines.append(f"    application_context: {application_context}")
        if table_name:
            section_lines.append(f"    table_name: {table_name}")
        if isinstance(num_sheets, int):
            section_lines.append(f"    number_of_sheets: {num_sheets}")
        if isinstance(sheet_dimensions, list) and sheet_dimensions:
            section_lines.append("    sheets:")
            for sheet in sheet_dimensions:
                if not isinstance(sheet, Mapping):
                    continue
                sheet_name = sheet.get("sheet_name", "")
                rows = sheet.get("rows", "")
                columns = sheet.get("columns", "")
                sheet_table = sheet_name if num_sheets and num_sheets > 1 else table_name
                section_lines.append(
                    f"      - name: {sheet_name} | table: {sheet_table} | rows: {rows} | columns: {columns}"
                )
        sections.append("\n".join(section_lines))
    return "\n".join(sections) if sections else "  (no indicator context provided)"

def _make_tool_strict(tool: Any) -> Any:
    """
    Patch a LangChain tool to be compatible with OpenAI's strict mode.
    OpenAI requires 'additionalProperties: false' and all properties to be in 'required'.
    """
    if not hasattr(tool, "args_schema") or tool.args_schema is None:
        return tool

    original_schema = tool.args_schema

    # Create a new model where all fields are required and extra properties are forbidden
    fields = {}
    for name, field in original_schema.model_fields.items():
        # (type, ...) means the field is required
        fields[name] = (field.annotation, ...)

    new_schema = create_model(
        original_schema.__name__,
        __config__=ConfigDict(extra="forbid"),
        **fields
    )
    tool.args_schema = new_schema
    return tool


def create_sql_agent(
    indicators: Sequence[Mapping[str, Any]],
    *,
    user_question: str | None = None,
    extra_instructions: str | None = None,
) -> Any:
    """
    Build a LangChain SQL agent primed with indicator metadata.

    Args:
        indicators: Sequence of Qdrant-style documents containing metadata.
        user_question: Raw question typed in the Streamlit UI; injected into the system prompt.
        extra_instructions: Optional string appended to the system prompt.

    Returns:
        A Runnable agent that accepts {"input": "..."} payloads.
    """
    db = SQLDatabase.from_uri(build_connection_uri())
    llm = create_chat_model(model=settings.sql_agent_model, temperature=0.0)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = [_make_tool_strict(t) for t in toolkit.get_tools()]

    indicator_context = _format_indicator_context(indicators)
    logger.debug(
        "Initializing SQL agent with %s indicator(s)",
        len(indicators),
        extra={"module_name": ModuleName.ADAPTER},
    )
    system_prompt = build_system_prompt(indicator_context, user_question=user_question)
    if extra_instructions:
        system_prompt = f"{system_prompt}\n\nAdditional guidance:\n{extra_instructions}"
    agent = create_agent(
        llm,
        tools,
        system_prompt=system_prompt,
        response_format=SQLAgentResponse,
    )
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


def parse_agent_response(agent_result: Any) -> SQLAgentResponse:
    """Parse and validate the agent output into a SQLAgentResponse."""
    if isinstance(agent_result, SQLAgentResponse):
        return agent_result

    structured_candidate: Any | None = None
    if isinstance(agent_result, Mapping):
        structured_candidate = agent_result.get("structured_response")
    if structured_candidate is not None:
        try:
            response = (
                structured_candidate
                if isinstance(structured_candidate, SQLAgentResponse)
                else SQLAgentResponse.model_validate(structured_candidate)
            )
            logger.debug(
                "SQL agent returned structured_response payload",
                extra={"module_name": ModuleName.ADAPTER},
            )
            return response
        except ValidationError as exc:
            logger.error(
                "Structured response validation failed: %s",
                exc,
                extra={"module_name": ModuleName.ADAPTER},
            )
            raise ValueError(f"Agent response schema mismatch: {exc}") from exc

    raw_text = _extract_output_text(agent_result).strip()
    if raw_text.startswith("```"):
        # Strip optional markdown fences like ```json ... ```
        stripped = raw_text.strip("`").strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
        raw_text = stripped
    # Try full string first
    parsed: dict[str, Any] | None = None
    try:
        parsed = json.loads(raw_text)
        logger.debug("SQL agent returned valid JSON payload", extra={"module_name": ModuleName.ADAPTER})
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
            logger.debug("Recovered JSON by slicing response", extra={"module_name": ModuleName.ADAPTER})
        except json.JSONDecodeError as exc:
            logger.error(
                "Snippet JSON parse failed: %s | snippet=%s",
                exc,
                snippet[:500],
                extra={"module_name": ModuleName.ADAPTER},
            )
    if parsed is None:
        raise ValueError(f"Agent response was not valid JSON: {raw_text}")
    try:
        response = SQLAgentResponse.model_validate(parsed)
        logger.debug("Validated SQL agent response via Pydantic", extra={"module_name": ModuleName.ADAPTER})
        return response
    except ValidationError as exc:
        logger.error("Structured response validation failed: %s", exc, extra={"module_name": ModuleName.ADAPTER})
        raise ValueError(f"Agent response schema mismatch: {exc}") from exc


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
    agent = create_sql_agent(
        indicators,
        user_question=query,
        extra_instructions=extra_instructions,
    )
    result = agent.invoke(
        {"input": query},
        config={
            "tags": ["feedback-agent"],
            "metadata": {"component": "feedback-agent"},
        },
    )
    structured = parse_agent_response(result)
    payload = structured.model_dump()
    logger.debug(
        "SQL agent structured output: %s",
        payload,
        extra={
            "module_name": ModuleName.ADAPTER,
            "structured_output": payload,
        },
    )
    return payload
