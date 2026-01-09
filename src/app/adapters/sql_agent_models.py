from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AxisSpec(BaseModel):
    """Axis metadata for chart rendering."""

    model_config = ConfigDict(extra="ignore")

    key: str = Field(..., description="Field name used for the axis")
    label: str = Field(default="", description="Human-friendly axis label")


class YAxisSpec(BaseModel):
    """Vertical axis label metadata."""

    model_config = ConfigDict(extra="ignore")

    label: str = Field(default="", description="Human-friendly axis label")


class SeriesSpec(BaseModel):
    """Series metadata for chart rendering."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(..., description="Series display name")
    key: str = Field(..., description="Field name in data rows for the series values")


class ChartSpec(BaseModel):
    """Visualization schema that the SQL agent must populate."""

    model_config = ConfigDict(extra="ignore")

    title: str = Field(default="", description="Primary chart title")
    x: AxisSpec
    y: YAxisSpec = Field(default_factory=YAxisSpec)
    series: list[SeriesSpec] = Field(
        default_factory=list,
        description="Series definitions mapping to keys in the data rows",
    )

    @field_validator("series")
    @classmethod
    def ensure_series(cls, value: list[SeriesSpec]) -> list[SeriesSpec]:
        """Require at least one series for chart rendering."""

        if not value:
            raise ValueError("chart.series must include at least one series")
        return value


class TableColumnSpec(BaseModel):
    """Column metadata for table rendering."""

    model_config = ConfigDict(extra="ignore")

    key: str = Field(..., description="Field name in the data rows")
    label: str = Field(default="", description="Column header label")


class TableSpec(BaseModel):
    """Table schema describing which fields to render."""

    model_config = ConfigDict(extra="ignore")

    columns: list[TableColumnSpec] = Field(default_factory=list)


class SQLAgentResponse(BaseModel):
    """Structured payload returned by the SQL agent."""

    model_config = ConfigDict(extra="ignore")

    insights: str = Field(
        ...,
        description="2-3 bullet-style sentences with the analytical takeaway.",
    )
    data: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Up to 200 data rows used for charting and tables.",
    )
    chart: ChartSpec
    table: TableSpec
    data_quality_notes: str | None = Field(
        default=None,
        description="Optional note about missing data, NULL handling, or outliers.",
    )

    @field_validator("data")
    @classmethod
    def clamp_data_rows(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Trim overly large responses to keep UI rendering predictable."""

        if len(value) > 200:
            return value[:200]
        return value

    @model_validator(mode="after")
    def validate_data_keys(self) -> "SQLAgentResponse":
        """Ensure every data row contains keys needed for chart and table rendering."""

        required_keys = {self.chart.x.key}
        required_keys.update({series.key for series in self.chart.series})
        required_keys.update({column.key for column in self.table.columns})

        missing = set()
        for row in self.data:
            row_keys = set(row.keys())
            missing.update(required_keys - row_keys)

        if missing:
            raise ValueError(
                "data rows missing required keys: " + ", ".join(sorted(missing))
            )
        return self
