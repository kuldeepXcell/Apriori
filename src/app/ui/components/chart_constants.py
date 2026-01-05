"""Centralized visualization constants for the Streamlit UI."""

from __future__ import annotations

# Default brand colors (Apriori red + Apriori grey)
APRIORI_RED = "#BB271A"
APRIORI_GREY = "#D9D9D9"

# Default palette leans entirely on the primary + neutral tones from the spec sheet
DEFAULT_COLOR_PALETTE: list[str] = [APRIORI_RED, APRIORI_GREY]

COLOR_PALETTES: dict[str, list[str]] = {
    "Apriori Red": [APRIORI_RED, APRIORI_GREY],
    "Apriori Grey": [APRIORI_GREY, APRIORI_RED],
    "Sunset": ["#f5b700", "#f18701", "#f25f5c"],
    "Emerald": ["#0b8457", "#42b883", "#9fd356"],
    "Mono": ["#4f5d75", "#bfc0c0", "#ef8354"],
}

DEFAULT_FONT = "Inter"
DEFAULT_FONT_SIZE = 12
AVAILABLE_FONTS = [
    DEFAULT_FONT,
    "Roboto",
    "Source Sans Pro",
    "Work Sans",
    "Montserrat",
]

# Display defaults (LLM-independent)
DEFAULT_CHART_TYPE = "line"
DEFAULT_LEGEND_POSITION = "top"
DEFAULT_SHOW_LEGEND = True
DEFAULT_SHOW_DATA_LABELS = False
DEFAULT_SHOW_GRIDLINES = True
DEFAULT_CHART_WIDTH = 700
DEFAULT_BAR_GUTTER = 4

LEGEND_POSITIONS = ["top", "bottom", "left", "right"]
TITLE_ANCHORS = ["start", "middle", "end"]
TITLE_ORIENTS = ["top", "bottom"]
