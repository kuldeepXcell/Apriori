from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChartSchema(BaseModel):
    """Visualization schema that the SQL agent must populate."""

    model_config = ConfigDict(extra="ignore")

    x_field: str = Field(..., description="Column name for the horizontal axis")
    y_field: str = Field(..., description="Column name for the measured value axis")
    group_field: str | None = Field(
        default=None, description="Optional grouping column for multi-series charts"
    )
    title: str = Field(default="", description="Primary chart title")
    subtitle: str = Field(default="", description="Short subtitle or context")
    legend_title: str = Field(default="", description="Label displayed above the legend; mandatory")
    axis_titles: dict[str, str] = Field(
        default_factory=lambda: {"x": "", "y": ""},
        description="Axis labels with keys x and y",
    )
    footnote_left: str = Field(
        default="", description="Left-aligned footnote (definition or methodology)"
    )
    footnote_right: str = Field(
        default="", description="Right-aligned footnote (e.g., last updated date)"
    )

    @field_validator("axis_titles")
    @classmethod
    def ensure_axis_keys(cls, value: dict[str, str]) -> dict[str, str]:
        """Guarantee x/y keys exist so the UI can fall back gracefully."""

        return {"x": value.get("x", ""), "y": value.get("y", "")}


class SQLAgentResponse(BaseModel):
    """Structured payload returned by the SQL agent."""

    model_config = ConfigDict(extra="ignore")

    insights: str = Field(
        ...,
        description="2-3 bullet-style sentences with the analytical takeaway.",
    )
    table_rows: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Up to 200 rows from the executed SQL query.",
    )
    chart_schema: ChartSchema
    data_quality_notes: str | None = Field(
        default=None,
        description="Optional note about missing data, NULL handling, or outliers.",
    )

    @field_validator("table_rows")
    @classmethod
    def clamp_table_rows(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Trim overly large responses to keep UI rendering predictable."""

        if len(value) > 200:
            return value[:200]
        return value
