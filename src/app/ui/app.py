from pathlib import Path
import sys
import time
import threading
import io
from typing import Any, Optional

# Project root (two levels above src/)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
STATIC_DIR = PROJECT_ROOT / "static"
LOGO_PATH = STATIC_DIR / "AprioriFullLogo.png"
FAVICON_PATH = STATIC_DIR / "AprioriFavicon.png"
SRC_ROOT = PROJECT_ROOT / "src"
src_str = str(SRC_ROOT)
if src_str in sys.path:
    sys.path.remove(src_str)
# Ensure package imports resolve when running as a script with Streamlit
sys.path.insert(0, src_str)

import pandas as pd
import streamlit as st
from PIL import Image
import vl_convert as vlc

from app.adapters.sql_agent import run_sql_agent
from app.adapters.feedback import save_indicator_feedback, update_chart_feedback
from app.adapters.supabase_storage import upload_chart_image
from app.config.settings import settings
from app.core.logging import ModuleName, get_logger, setup_logging
from app.core.langsmith import configure_langsmith
from app.steps.retrieval import baseline_hybrid_retrieval as retrieval
from app.ui.components import charts as chart_utils
from app.ui.components.vega_snapshot import capture_chart_png as capture_chart_png_from_browser
from app.ui.components.chart_constants import (
    AVAILABLE_FONTS,
    COLOR_PALETTES,
    DEFAULT_FONT,
    DEFAULT_LEGEND_POSITION,
    DEFAULT_SHOW_DATA_LABELS,
    DEFAULT_SHOW_GRIDLINES,
    DEFAULT_SHOW_LEGEND,
    LEGEND_POSITIONS,
    TITLE_ANCHORS,
    TITLE_ORIENTS,
)


EXPORT_FONT = "Arial"


def format_duration(seconds: float, precision: int = 1) -> str:
    """Format duration in seconds to m s if > 60s."""
    if seconds < 60:
        return f"{seconds:.{precision}f}s"
    minutes = int(seconds // 60)
    rem_seconds = seconds % 60
    return f"{minutes}m {rem_seconds:.{precision}f}s"


def _prepare_chart_for_export(chart):
    """Force fonts to the bundled family so backend exports stay consistent."""
    font = EXPORT_FONT
    return (
        chart.configure_axis(labelFont=font, titleFont=font)
        .configure_legend(labelFont=font, titleFont=font)
        .configure_title(font=font, subtitleFont=font)
        .configure_text(font=font)
    )


def build_chart_export_spec(chart) -> Optional[dict[str, Any]]:
    """Return a Vega-Lite spec ready for export so heavy conversions can be deferred."""
    try:
        export_chart = _prepare_chart_for_export(chart)
        return export_chart.to_dict(format="vega-lite")
    except Exception:  # pragma: no cover - defensive
        logger.exception("Chart spec build failed", extra={"module_name": ModuleName.UI})
        return None


def export_chart_png(spec: dict[str, Any]) -> bytes | None:
    """Render a Vega-Lite spec to PNG bytes using vl-convert (heavy; call sparingly)."""
    try:
        return vlc.vegalite_to_png(spec, scale=2)
    except Exception:  # pragma: no cover - defensive
        logger.exception("Chart export failed", extra={"module_name": ModuleName.UI})
        return None


def compress_chart_image(png_bytes: bytes, *, max_width: int = 1400, quality: int = 85) -> tuple[bytes, str, str]:
    """Compress a PNG screenshot to a JPEG with optional resizing."""
    image = Image.open(io.BytesIO(png_bytes))
    image = image.convert("RGB")
    if image.width > max_width:
        ratio = max_width / float(image.width)
        new_height = int(image.height * ratio)
        image = image.resize((max_width, new_height))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", optimize=True, quality=quality)
    return buffer.getvalue(), "image/jpeg", "jpg"

# Page configuration
st.set_page_config(
    page_title="Data Visualization | Apriori",
    page_icon=str(FAVICON_PATH) if FAVICON_PATH.exists() else None,
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "Get Help": "https://www.aprioriconsultants.com/",
        "About": "This is a simple application designed to Identify Indicators from the user's question.",
    },
)

# Respect env-driven log level (LOG_LEVEL) via settings.log_level
setup_logging(settings.log_level)
configure_langsmith(settings)
logger = get_logger(__name__)

# Custom CSS for styling
st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #f5f7fb;
        color: #1f2a37;
    }
    .stAppHeader {
        background-color: #ffffff !important;
        border-bottom: 1px solid #e5e7eb;
    }
    .main-header {
        text-align: left;
        padding: 1.5rem 0 0.5rem;
    }
    .company-logo {
        font-size: 48px;
        margin-bottom: 10px;
    }
    .app-title {
        font-size: 20px;
        color: #6b7280;
        letter-spacing: 0.08em;
        margin-bottom: 0.75rem;
    }
    .question-card {
        background: #ffffff;
        border-radius: 18px;
        padding: 1.5rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 15px 35px rgba(15, 23, 42, 0.08);
        margin-bottom: 1.4rem;
    }
    .section-divider {
        height: 1px;
        width: 100%;
        background: linear-gradient(90deg, transparent, #d3dae6, transparent);
        margin: 1.2rem 0;
    }
    .button-spacer {
        height: 1.6rem;
    }
    .stTextInput > label {
        font-size: 16px;
        font-weight: 500;
        color: #1f2a37;
    }
    .stCheckbox label {
        white-space: nowrap;
        color: #1f2a37;
    }
    div[role="radiogroup"] {
        gap: 0.6rem;
    }
    div[role="radiogroup"] > label {
        border: 1px solid #d1d5db;
        border-radius: 999px;
        padding: 4px 12px;
        margin-bottom: 6px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: #ffffff;
        color: #1f2a37;
    }
    div[role="radiogroup"] > label:nth-child(1) {
        display: none;
    }
    div[role="radiogroup"] > label:nth-child(2) {
        border-color: #34a853;
        color: #1f7f3e;
    }
    div[role="radiogroup"] > label:nth-child(3) {
        border-color: #fbbc05;
        color: #b45309;
    }
    div[role="radiogroup"] > label:nth-child(4) {
        border-color: #ea4335;
        color: #b91c1c;
    }
    .indicator-row {
        padding: 1.25rem;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        background: #ffffff;
        box-shadow: 0 12px 24px rgba(15, 23, 42, 0.08);
        margin-bottom: 1rem;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .indicator-row:hover {
        border-color: #f97316;
        box-shadow: 0 16px 30px rgba(249, 115, 22, 0.15);
    }
    .indicator-meta-line {
        margin-top: 0.75rem;
        font-size: 0.85rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #6b7280;
    }
    .indicator-meta-line span {
        margin-right: 0.75rem;
    }
    .indicator-meta-line .meta-accent {
        color: #f97316;
    }
    .indicator-meta-line .meta-llm {
        color: #1f7f3e;
    }
    .sidebar-badge {
        background: #f3f4f6;
        padding: 0.4rem 0.75rem;
        border-radius: 10px;
        font-size: 0.85rem;
        margin-top: 0.5rem;
        border: 1px solid #e5e7eb;
        color: #1f2a37;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header section
st.markdown('<div class="main-header">', unsafe_allow_html=True)
if LOGO_PATH.exists():
    st.image(str(LOGO_PATH), output_format="PNG", width=360)
else:
    st.warning(f"Logo not found at {LOGO_PATH}")
st.markdown('<div class="app-title">Data Visualization</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

col_question, col_button = st.columns([4, 1])
question = col_question.text_input("Ask a question", placeholder="Type your question here...")
with col_button:
    st.markdown('<div class="button-spacer"></div>', unsafe_allow_html=True)
    identify_clicked = st.button("Identify Indicators", type="primary", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# Initialize session state for storing responses
if "results" not in st.session_state:
    st.session_state.results = []
if "sql_agent_running" not in st.session_state:
    st.session_state.sql_agent_running = False
if "feedback_id" not in st.session_state:
    st.session_state.feedback_id = None
if "chart_spec" not in st.session_state:
    st.session_state.chart_spec = None
if "chart_feedback_notes" not in st.session_state:
    st.session_state.chart_feedback_notes = ""
if "chart_capture_trigger" not in st.session_state:
    st.session_state.chart_capture_trigger = 0

# Sidebar weights (sum enforced to 1 inside retrieval)
st.sidebar.subheader("Hybrid weights")
weight_def = st.sidebar.slider("Definition weight", 0.0, 1.0, 0.25, 0.05)
weight_q = st.sidebar.slider("Question weight", 0.0, 1.0, 0.25, 0.05)
weight_ctx = st.sidebar.slider("Context weight", 0.0, 1.0, 0.25, 0.05)
weight_kw = st.sidebar.slider("Keywords weight", 0.0, 1.0, 0.25, 0.05)
weights = {
    "definition": weight_def,
    "question": weight_q,
    "context": weight_ctx,
    "keywords": weight_kw,
}
total_weight = sum(weights.values())
st.sidebar.markdown(
    f'<div class="sidebar-badge"><strong>Total weight:</strong> {total_weight:.2f} (target <= 1.0)</div>',
    unsafe_allow_html=True,
)
if total_weight > 1.0:
    st.sidebar.error("Reduce weights so the total is <= 1.0.")

use_llm_rerank = st.sidebar.checkbox(
    "Use LLM reranker",
    value=st.session_state.get("use_llm_rerank", False),
    help="Let the reranker rescore the blended retrieval results.",
)


def _checkbox_key(item: dict) -> str:
    payload = item.get("payload") or {}
    identifier = item.get("id") or payload.get("normalized_indicator_name") or payload.get("indicator_name") or ""
    if not identifier:
        identifier = str(len(payload))
    return f"indicator_select_{identifier}"


def _feedback_key(item: dict) -> str:
    payload = item.get("payload") or {}
    identifier = item.get("id") or payload.get("normalized_indicator_name") or payload.get("indicator_name") or ""
    if not identifier:
        identifier = str(len(payload))
    return f"indicator_feedback_{identifier}"


def _order_results_for_display(results: list[dict], use_llm_rerank: bool) -> list[dict]:
    if not use_llm_rerank:
        return results
    selected = [r for r in results if r.get("selected_by_llm")]
    selected.sort(key=lambda r: r.get("llm_rank", float("inf")))
    non_selected = [r for r in results if not r.get("selected_by_llm")]
    return selected + non_selected


def _initialize_result_state(results: list[dict]) -> None:
    for item in results:
        st.session_state[_checkbox_key(item)] = False
        st.session_state[_feedback_key(item)] = "Select..."


def collect_selected_indicators(results: list[dict]) -> list[dict]:
    selections: list[dict] = []
    for item in results:
        key = _checkbox_key(item)
        if st.session_state.get(key):
            payload = item.get("payload") or {}
            selections.append(
                {
                    "normalized_indicator_name": payload.get("normalized_indicator_name"),
                    "metadata": payload,
                    "id": item.get("id") or payload.get("normalized_indicator_name"),
                }
            )
    return selections


def collect_indicator_feedback(results: list[dict]) -> list[dict]:
    label_map = {
        "\u2705": "Directly related",
        "\u26A0": "Indirectly related",
        "\u274C": "Not related",
    }
    entries: list[dict] = []
    for item in results:
        payload = item.get("payload") or {}
        label_choice = st.session_state.get(_feedback_key(item), "Select...")
        indicator_entry = {
            "id": item.get("id"),
            "indicator_name": payload.get("indicator_name"),
            "normalized_indicator_name": payload.get("normalized_indicator_name"),
            "score": item.get("score"),
            "selected_by_llm": item.get("selected_by_llm", False),
            "llm_rank": item.get("llm_rank"),
            "label": label_map.get(label_choice) if label_choice != "Select..." else None,
        }
        entries.append(indicator_entry)
    return entries


def _render_indicator_block(item: dict, idx: int, any_llm_selected: bool) -> None:
    payload = item.get("payload", {}) or {}
    name = payload.get("indicator_name") or "Unknown indicator"
    question_text = payload.get("question") or ""
    definition = payload.get("definition") or ""
    app_ctx = payload.get("application_context") or ""
    sheet = payload.get("sheet_name") or ""
    score = item.get("score", 0)
    source_scores = item.get("source_scores") or {}
    selected_by_llm = item.get("selected_by_llm", False)
    llm_rank = item.get("llm_rank")

    title = f"{idx}. {name}"

    checkbox_col, expander_col = st.columns([0.15, 0.85])
    with checkbox_col:
        st.checkbox("Select", key=_checkbox_key(item))
    with expander_col:
        with st.expander(title, expanded=False):
            if question_text:
                st.markdown(f"- Question: {question_text}")
            if definition:
                st.markdown(f"- Definition: {definition}")
            if app_ctx:
                st.markdown(f"- Application context: {app_ctx}")
            if sheet:
                st.markdown(f"- Sheet: {sheet}")
            score_cols = st.columns(4)
            score_labels = [
                ("definition", "Definition"),
                ("question", "Question"),
                ("context", "Context"),
                ("keywords", "Keywords"),
            ]
            for col, (score_key, label) in zip(score_cols, score_labels):
                score_val = source_scores.get(score_key, 0.0)
                col.metric(f"{label} score", f"{score_val:.4f}")
    meta_bits: list[str] = [f'<span>Score <span class="meta-accent">{score:.3f}</span></span>']
    if selected_by_llm:
        rank_label = f"LLM rank {llm_rank}" if llm_rank else "LLM selected"
        meta_bits.append(f'<span class="meta-llm">{rank_label}</span>')
    elif any_llm_selected:
        meta_bits.append("<span>Weighted blend result</span>")
    if sheet:
        meta_bits.append(f"<span>Sheet {sheet}</span>")
    if app_ctx and not question_text:
        meta_bits.append(f"<span>{app_ctx}</span>")
    st.markdown(f'<div class="indicator-meta-line">{"".join(meta_bits)}</div>', unsafe_allow_html=True)

    st.radio(
        "Relevance",
        [
            "Select...",
            "\u2705",
            "\u26A0",
            "\u274C",
        ],
        key=_feedback_key(item),
        label_visibility="collapsed",
        horizontal=True,
    )


def render_results(results: list[dict], use_llm_rerank: bool) -> None:
    if not results:
        st.info("No indicators found.")
        return
    st.markdown("### Top indicators")
    ordered = _order_results_for_display(results, use_llm_rerank)

    any_llm_selected = any(r.get("selected_by_llm") for r in results)
    if use_llm_rerank:
        if not any_llm_selected:
            st.caption("LLM reranker returned no selections; showing weighted results.")
    else:
        st.caption("LLM reranker disabled; showing weighted results.")

    for i in range(0, len(ordered), 2):
        row_col1, row_col2 = st.columns(2)
        if i < len(ordered):
            with row_col1:
                _render_indicator_block(ordered[i], i + 1, any_llm_selected)
                st.markdown("</div>", unsafe_allow_html=True)
        if i + 1 < len(ordered):
            with row_col2:
                _render_indicator_block(ordered[i + 1], i + 2, any_llm_selected)
                st.markdown("</div>", unsafe_allow_html=True)


def render_chart_designer(chart_payload: dict, df: pd.DataFrame) -> None:
    if df.empty:
        st.warning("The SQL agent returned no rows to visualize.")
        return

    st.session_state.chart_spec = None
    schema = chart_payload.get("chart_schema") or {}
    insights = chart_payload.get("insights")
    if insights:
        st.markdown("### Insights")
        st.write(insights)

    data_quality_notes = chart_payload.get("data_quality_notes")

    with st.sidebar:
        st.markdown("### Chart Configuration")
        palette_names = list(COLOR_PALETTES.keys())
        palette_name = st.selectbox("Color palette", palette_names, index=0)
        base_palette = COLOR_PALETTES[palette_name]
        primary_color = st.color_picker("Primary color", base_palette[0])
        colors = [primary_color] + base_palette[1:]

        font_choice = st.selectbox(
            "Font family",
            AVAILABLE_FONTS,
            index=max(0, AVAILABLE_FONTS.index(DEFAULT_FONT)) if DEFAULT_FONT in AVAILABLE_FONTS else 0,
        )

        show_legend = st.checkbox("Show legend", value=DEFAULT_SHOW_LEGEND)
        legend_title = st.text_input("Legend title", schema.get("legend_title", ""))
        legend_position_default = DEFAULT_LEGEND_POSITION
        legend_position = st.selectbox(
            "Legend position",
            LEGEND_POSITIONS,
            index=LEGEND_POSITIONS.index(legend_position_default),
        )

        title_text = st.text_input("Title", schema.get("title", ""))
        subtitle_text = st.text_input("Subtitle", schema.get("subtitle", ""))
        title_anchor_default = TITLE_ANCHORS[0]
        if title_anchor_default not in TITLE_ANCHORS:
            title_anchor_default = TITLE_ANCHORS[0]
        title_anchor = st.selectbox(
            "Title alignment",
            TITLE_ANCHORS,
            index=TITLE_ANCHORS.index(title_anchor_default),
        )
        title_orient_default = TITLE_ORIENTS[0]
        if title_orient_default not in TITLE_ORIENTS:
            title_orient_default = TITLE_ORIENTS[0]
        title_orient = st.selectbox(
            "Title position",
            TITLE_ORIENTS,
            index=TITLE_ORIENTS.index(title_orient_default),
        )

        axis_titles = schema.get("axis_titles") or {}
        axis_x_title = st.text_input("X-axis title", axis_titles.get("x", schema.get("x_field", "x")))
        axis_y_title = st.text_input("Y-axis title", axis_titles.get("y", schema.get("y_field", "value")))

        show_data_labels = st.checkbox(
            "Show data labels",
            value=DEFAULT_SHOW_DATA_LABELS,
            help="Displays labels above bars or points.",
        )
        show_gridlines = st.checkbox(
            "Show gridlines",
            value=DEFAULT_SHOW_GRIDLINES,
        )

    tabs = st.tabs(["Line", "Bar", "Table"])
    try:
        with tabs[0]:
            line_chart = chart_utils.build_chart(
                df,
                schema,
                chart_type="Line",
                colors=colors,
                font=font_choice,
                legend_title=legend_title,
                legend_position=legend_position,
                show_legend=show_legend,
                title=title_text,
                subtitle=subtitle_text,
                title_anchor=title_anchor,
                title_orient=title_orient,
                axis_titles={"x": axis_x_title, "y": axis_y_title},
                show_gridlines=show_gridlines,
                show_data_labels=show_data_labels,
            )
            st.altair_chart(line_chart, width="stretch")
            spec = build_chart_export_spec(line_chart)
            if spec:
                st.session_state.chart_spec = spec
        with tabs[1]:
            bar_chart = chart_utils.build_chart(
                df,
                schema,
                chart_type="Bar",
                colors=colors,
                font=font_choice,
                legend_title=legend_title,
                legend_position=legend_position,
                show_legend=show_legend,
                title=title_text,
                subtitle=subtitle_text,
                title_anchor=title_anchor,
                title_orient=title_orient,
                axis_titles={"x": axis_x_title, "y": axis_y_title},
                show_gridlines=show_gridlines,
                show_data_labels=show_data_labels,
            )
            st.altair_chart(bar_chart, width="stretch")
        with tabs[2]:
            st.dataframe(df)
    except Exception as exc:  # pragma: no cover - UI resilience
        logger.exception("Chart rendering failed", extra={"module_name": ModuleName.UI})
        st.error(f"Unable to render chart: {exc}")

    if data_quality_notes:
        st.markdown("### Data quality notes")
        st.info(data_quality_notes)
# Process button
if identify_clicked:
    if question:
        if total_weight > 1.0:
            st.error("Total weight must be <= 1.0. Please adjust sliders.")
        else:
            with st.status("Searching indicators...") as status:
                start_time = time.perf_counter()
                res: dict[str, Any] = {"results": None, "error": None, "done": False}

                def search_task():
                    try:
                        res["results"] = retrieval.search(
                            question=question,
                            weights=weights,
                            top_k=15,
                            use_llm_rerank=use_llm_rerank,
                        )
                    except Exception as e:
                        res["error"] = e
                    finally:
                        res["done"] = True

                thread = threading.Thread(target=search_task)
                thread.start()

                while not res["done"]:
                    elapsed = time.perf_counter() - start_time
                    status.update(label=f"Searching indicators... {format_duration(elapsed)}")
                    time.sleep(0.1)

                if res["error"]:
                    status.update(label="Retrieval failed", state="error")
                    logger.exception("Retrieval failed", extra={"module_name": ModuleName.UI})
                    st.error(f"Retrieval failed: {res['error']}")
                else:
                    elapsed = time.perf_counter() - start_time
                    search_payload = res["results"] or {}
                    st.session_state.results = search_payload.get("results", [])
                    st.session_state.use_llm_rerank = use_llm_rerank
                    _initialize_result_state(st.session_state.results)
                    st.session_state.pop("chart_payload", None)
                    st.session_state.pop("chart_dataframe", None)
                    st.session_state.chart_spec = None
                    st.session_state.chart_feedback_notes = ""
                    st.session_state.feedback_id = None
                    status.update(label=f"Found indicators in {format_duration(elapsed, 2)}", state="complete")
    else:
        st.warning("Please ask a question first!")

# Output section
if st.session_state.results:
    # Render indicators in two columns
    render_results(st.session_state.results, st.session_state.get("use_llm_rerank", False))

    # Submit button below the indicators
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            "Generate chart & insights",
            type="primary",
            width="stretch",
            disabled=st.session_state.sql_agent_running,
        ):
            selected_indicators = collect_selected_indicators(st.session_state.results)
            if not question:
                st.warning("Please provide a question so the SQL agent knows what to answer.")
            elif not selected_indicators:
                st.warning("Select at least one indicator to proceed.")
            else:
                st.session_state.sql_agent_running = True
                with st.status("Querying SQL agent...") as status:
                    start_time = time.perf_counter()
                    res: dict[str, Any] = {"payload": None, "error": None, "done": False}

                    def agent_task():
                        try:
                            res["payload"] = run_sql_agent(question, selected_indicators)
                        except Exception as e:
                            res["error"] = e
                        finally:
                            res["done"] = True

                    thread = threading.Thread(target=agent_task)
                    thread.start()

                    while not res["done"]:
                        elapsed = time.perf_counter() - start_time
                        status.update(label=f"Querying SQL agent... {format_duration(elapsed)}")
                        time.sleep(0.1)

                    if res["error"]:
                        status.update(label="SQL agent failed", state="error")
                        logger.exception("SQL agent failed", extra={"module_name": ModuleName.UI})
                        st.error(f"SQL agent failed: {res['error']}")
                    else:
                        elapsed = time.perf_counter() - start_time
                        agent_payload = res["payload"]
                        rows = agent_payload.get("table_rows") or []
                        df = pd.DataFrame(rows)
                        st.session_state.chart_payload = agent_payload
                        st.session_state.chart_dataframe = df
                        status.update(label=f"SQL query complete in {format_duration(elapsed, 2)}", state="complete")
                    st.session_state.sql_agent_running = False
    with col2:
        if st.button("Submit Feedback", width="stretch"):
            ordered_results = _order_results_for_display(
                st.session_state.results, st.session_state.get("use_llm_rerank", False)
            )
            if not question:
                st.warning("Please provide a question so the feedback can be tied to it.")
            else:
                indicators_payload = collect_indicator_feedback(ordered_results)
                payload = {
                    "query_text": question,
                    "reranker_used": st.session_state.get("use_llm_rerank", False),
                    "indicators": indicators_payload,
                }
                with st.spinner("Saving feedback..."):
                    try:
                        feedback_id = save_indicator_feedback(payload)
                    except Exception as exc:  # pragma: no cover - UI resilience
                        logger.exception("Feedback capture failed", extra={"module_name": ModuleName.UI})
                        st.error(f"Feedback capture failed: {exc}")
                    else:
                        st.session_state.feedback_id = feedback_id
                        st.success("Feedback saved.")

if st.session_state.get("chart_payload") and st.session_state.get("chart_dataframe") is not None:
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("## Visualization")
    render_chart_designer(st.session_state.chart_payload, st.session_state.chart_dataframe)
    st.markdown("### Feedback on chart & insights")
    st.caption("Share optional notes about the generated visualization. Submit indicator feedback first to unlock this step.")
    st.text_area(
        "Feedback notes",
        key="chart_feedback_notes",
        placeholder="What worked well? Anything missing or incorrect in the chart/insights?",
    )
    can_submit_chart = bool(st.session_state.feedback_id)
    if not can_submit_chart:
        st.info("Submit indicator feedback to generate a feedback id before sharing chart comments.")
    if st.button("Submit chart feedback", disabled=not can_submit_chart):
        if not st.session_state.feedback_id:
            st.warning("Submit indicator feedback first.")
        else:
            chart_spec = st.session_state.get("chart_spec")
            if not chart_spec:
                st.warning("Chart snapshot unavailable. Re-run the chart generation first.")
            else:
                # Trigger capture
                st.session_state.chart_capture_trigger += 1
    
    # Handle capture component (renders when trigger > 0)
    if st.session_state.chart_capture_trigger > 0 and st.session_state.get("chart_spec"):
        capture_key = f"chart_capture_{st.session_state.chart_capture_trigger}"
        
        with st.spinner("Capturing chart from browser..."):
            image_bytes = capture_chart_png_from_browser(
                st.session_state.chart_spec,
                key=capture_key,
                height=600
            )
            
        # If we got bytes back, proceed with upload
        if image_bytes:
            try:
                compressed_bytes, content_type, extension = compress_chart_image(image_bytes)
                image_url = upload_chart_image(
                    st.session_state.feedback_id,
                    compressed_bytes,
                    content_type=content_type,
                    extension=extension,
                )
                update_chart_feedback(
                    st.session_state.feedback_id,
                    chart_notes=st.session_state.chart_feedback_notes.strip() or None,
                    chart_image_url=image_url,
                )
                st.success("Chart feedback saved.")
                # Reset trigger
                st.session_state.chart_capture_trigger = 0
            except Exception as exc:
                logger.exception("Chart feedback save failed", extra={"module_name": ModuleName.UI})
                st.error(f"Chart feedback failed: {exc}")
                st.session_state.chart_capture_trigger = 0
