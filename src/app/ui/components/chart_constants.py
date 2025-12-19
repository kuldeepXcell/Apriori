"""Centralized visualization constants for the Streamlit UI."""

from __future__ import annotations

# Default palette leans on Apriori's red + grey brand colors
DEFAULT_COLOR_PALETTE: list[str] = ["#D93025", "#9AA0A6", "#5F6368"]

COLOR_PALETTES: dict[str, list[str]] = {
    "Apriori Default": DEFAULT_COLOR_PALETTE,
    "Apriori Blue": ["#1f77b4", "#3da5d9", "#125e8a"],
    "Sunset": ["#f5b700", "#f18701", "#f25f5c"],
    "Emerald": ["#0b8457", "#42b883", "#9fd356"],
    "Mono": ["#4f5d75", "#bfc0c0", "#ef8354"],
}

DEFAULT_FONT = "Helvetica"
DEFAULT_FONT_SIZE = 12
AVAILABLE_FONTS = [
    DEFAULT_FONT,
    "Inter",
    "Roboto",
    "Source Sans Pro",
    "Work Sans",
    "Montserrat",
]

LEGEND_POSITIONS = ["top", "bottom", "left", "right"]
TITLE_ANCHORS = ["start", "middle", "end"]
TITLE_ORIENTS = ["top", "bottom"]
