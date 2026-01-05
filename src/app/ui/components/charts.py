"""
Shared chart helpers for the Streamlit UI.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import altair as alt
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from .chart_constants import DEFAULT_COLOR_PALETTE, DEFAULT_CHART_WIDTH, DEFAULT_BAR_GUTTER

def _resolve_schema(df: pd.DataFrame, chart_schema: Mapping[str, Any]) -> dict[str, Any]:
    infer = {
        "x_field": chart_schema.get("x_field") or (df.columns[0] if not df.empty else None),
        "y_field": chart_schema.get("y_field")
        or (df.columns[1] if df.shape[1] > 1 else None),
        "group_field": chart_schema.get("group_field"),
    }
    if infer["x_field"] is None or infer["y_field"] is None:
        raise ValueError("Chart schema must supply at least x_field and y_field")
    infer["title"] = chart_schema.get("title", "")
    infer["subtitle"] = chart_schema.get("subtitle", "")
    infer["legend_title"] = chart_schema.get("legend_title")
    infer["axis_titles"] = chart_schema.get("axis_titles") or {}
    return infer


def build_chart(
    df: pd.DataFrame,
    chart_schema: Mapping[str, Any],
    *,
    chart_type: str,
    colors: Sequence[str],
    font: str,
    legend_title: str,
    legend_position: str,
    show_legend: bool,
    title: str,
    subtitle: str,
    title_anchor: str,
    title_orient: str,
    axis_titles: Mapping[str, str],
    show_gridlines: bool,
    show_data_labels: bool,
) -> alt.Chart:
    """
    Build an Altair chart from a dataframe + schema.
    """
    resolved = _resolve_schema(df, chart_schema)
    x_field = resolved["x_field"]
    y_field = resolved["y_field"]
    group_field = resolved.get("group_field")

    palette = list(colors) if colors else list(DEFAULT_COLOR_PALETTE)

    x_enc = alt.X(
        x_field,
        title=axis_titles.get("x") or resolved["axis_titles"].get("x") or x_field,
        axis=alt.Axis(grid=show_gridlines),
    )
    y_enc = alt.Y(
        y_field,
        title=axis_titles.get("y") or resolved["axis_titles"].get("y") or y_field,
        axis=alt.Axis(grid=show_gridlines),
    )

    chart_data = df.copy()
    legend_field = group_field
    legend_label = legend_title or resolved["legend_title"] or (group_field or y_field)

    if not legend_field:
        legend_field = "__series"
        chart_data[legend_field] = legend_label or "series"
    elif show_legend and legend_field not in chart_data.columns:
        chart_data[legend_field] = chart_data[group_field]

    # Determine ordering for highlight
    x_values = chart_data[x_field]
    if is_datetime64_any_dtype(x_values):
        sort_key = pd.Series(x_values, index=chart_data.index)
    else:
        sort_key = pd.to_numeric(x_values, errors="coerce")
        if sort_key.isna().all():
            try:
                # format="mixed" avoids repeated fallback parsing warnings on heterogeneous strings
                sort_key = pd.to_datetime(x_values, errors="coerce", format="mixed")
            except TypeError:  # Older pandas versions (<2.0) lack format="mixed"
                sort_key = pd.to_datetime(x_values, errors="coerce")
    if sort_key.isna().all():
        sort_key = pd.RangeIndex(len(chart_data))
    else:
        sort_key = sort_key.ffill().bfill().fillna(0)
    chart_data["_sort_key"] = sort_key
    chart_data["_is_last"] = chart_data.groupby(legend_field)["_sort_key"].transform(
        lambda series: series == series.max()
    )

    base = alt.Chart(chart_data).encode(x=x_enc, y=y_enc)

    if legend_field:
        legend = (
            alt.Legend(
                title=legend_label,
                orient=legend_position,
                labelColor="#334155",
                titleColor="#0f172a",
                titleFontWeight="bold",
            )
            if show_legend
            else None
        )
        base = base.encode(color=alt.Color(legend_field, scale=alt.Scale(range=palette), legend=legend))
    else:
        base = base.encode(color=alt.value(palette[0]))

    chart_type = chart_type.lower()
    if chart_type == "line":
        chart = base.mark_line(
            interpolate="monotone",
            strokeWidth=2.5,
            strokeDash=[5, 5],
            point=alt.OverlayMarkDef(filled=True, size=60, opacity=1)
        )
        highlight_points = alt.Chart(chart_data).transform_filter(
            alt.datum._is_last
        ).mark_point(
            filled=True,
            size=100,
            color=palette[0],
            opacity=1,
        ).encode(x=x_enc, y=y_enc, color=alt.Color(legend_field, legend=None, scale=alt.Scale(range=palette)))
        
        chart = chart + highlight_points
    elif chart_type == "bar":
        # Calculate bar width based on chart width and number of entries
        num_entries = len(chart_data[x_field].unique())
        if num_entries > 0:
            # bar_width = (total_width - (num_gutters * gutter_size)) / num_entries
            # We use num_entries + 1 for gutters to have padding on both ends
            available_width = DEFAULT_CHART_WIDTH - (num_entries + 1) * DEFAULT_BAR_GUTTER
            calculated_width = max(2, available_width / num_entries)
            # Cap the width so it doesn't look too chunky with 1-2 bars
            bar_size = min(calculated_width, 60)
        else:
            bar_size = 20
            
        chart = base.mark_bar(
            cornerRadiusTopLeft=4, 
            cornerRadiusTopRight=4, 
            opacity=0.9,
            size=bar_size
        )
    else:
        chart = base.mark_line(point=True)

    if show_data_labels:
        label_mark = alt.Chart(chart_data).mark_text(
            align="left" if chart_type == "line" else "center",
            baseline="bottom",
            dx=8 if chart_type == "line" else 0,
            dy=-5,
            font=font,
            color="#0f172a",
        ).encode(
            x=x_enc,
            y=y_enc,
            text=alt.Text(y_field, format=",.2f"),
        )
        if legend_field:
            label_mark = label_mark.encode(color=alt.Color(legend_field, legend=None, scale=alt.Scale(range=palette)))
        chart = chart + label_mark

    title_params = None
    if title:
        title_params = alt.TitleParams(
            text=title,
            subtitle=subtitle or None,
            anchor=title_anchor,
            orient=title_orient,
            color="#0f172a",
            subtitleColor="#64748b",
            fontSize=18,
            subtitleFontSize=14,
            dy=0,
            dx=60,  # Align title with y-axis (compensating for left padding)
        )

    chart = (
        chart.properties(
            width=DEFAULT_CHART_WIDTH,
            height=400,
            title=title_params,
            autosize=alt.AutoSizeParams(type="fit", contains="padding"),
        )
        .configure(background="#ffffff", padding={"left": 80, "top": 90, "right": 30, "bottom": 20})
        .configure_view(fill="#ffffff", stroke="transparent")
        .configure_axis(
            labelFont=font,
            titleFont=font,
            labelColor="#64748b",
            titleColor="#334155",
            titleFontWeight="bold",
            gridColor="#f1f5f9",
            domainColor="#cbd5f5",
            tickColor="#cbd5f5",
            labelFontSize=11,
            titleFontSize=13,
        )
        .configure_legend(
            labelFont=font,
            titleFont=font,
            labelColor="#475569",
            titleColor="#0f172a",
        )
        .configure_title(
            font=font,
            subtitleFont=font,
            color="#0f172a",
            subtitleColor="#475569",
        )
    )

    return chart
