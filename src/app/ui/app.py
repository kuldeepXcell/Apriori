from pathlib import Path
import sys
import time
import threading

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

from app.adapters.sql_agent import run_sql_agent
from app.config.settings import settings
from app.core.logging import ModuleName, get_logger, setup_logging
from app.core.langsmith import configure_langsmith
from app.steps.retrieval import baseline_hybrid_retrieval as retrieval
from app.ui.components import charts as chart_utils

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
    .main-header {
        text-align: left;
        padding: 20px 0;
    }
    .company-logo {
        font-size: 48px;
        margin-bottom: 10px;
    }
    .app-title {
        font-size: 20px;
        color: #666;
        margin-bottom: 30px;
    }
    .output-box {
        background-color: #010011;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        min-height: 100px;
        border-left: 4px solid #1f77b4;
    }
    .stTextInput > label {
        font-size: 16px;
        font-weight: 500;
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

st.markdown("---")

# Initialize session state for storing responses
if "results" not in st.session_state:
    st.session_state.results = []

# Input section
question = st.text_input("Ask a question", placeholder="Type your question here...")

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
st.sidebar.markdown(f"**Total weight:** {total_weight:.2f} (must be ≤ 1.0)")
if total_weight > 1.0:
    st.sidebar.error("Reduce weights so the total is ≤ 1.0.")

use_llm_rerank = st.sidebar.checkbox("Use LLM reranker", value=st.session_state.get("use_llm_rerank", False))


def _checkbox_key(item: dict) -> str:
    payload = item.get("payload") or {}
    identifier = item.get("id") or payload.get("normalized_indicator_name") or payload.get("indicator_name") or ""
    if not identifier:
        identifier = str(len(payload))
    return f"indicator_select_{identifier}"


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


def render_results(results: list[dict], use_llm_rerank: bool) -> None:
    if not results:
        st.info("No indicators found.")
        return
    st.markdown("### Top indicators")
    selected = [r for r in results if r.get("selected_by_llm")]
    selected.sort(key=lambda r: r.get("llm_rank", float("inf")))
    non_selected = [r for r in results if not r.get("selected_by_llm")]
    ordered = selected + non_selected

    any_llm_selected = bool(selected)
    if use_llm_rerank:
        if not any_llm_selected:
            st.caption("LLM reranker returned no selections; showing weighted results.")
    else:
        st.caption("LLM reranker disabled; showing weighted results.")

    # Render indicators in rows of two columns each
    for i in range(0, len(ordered), 2):
        # Create a row with two columns
        row_col1, row_col2 = st.columns(2)

        # First indicator in this row (left column)
        if i < len(ordered):
            with row_col1:
                item = ordered[i]
                idx = i + 1
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

                title = f"{idx}. {name} (score: {score:.3f})"
                if selected_by_llm:
                    rank_label = f"rank {llm_rank}" if llm_rank else "selected"
                    title += f" | LLM {rank_label}"

                # Checkbox stays visible; expander holds details.
                checkbox_col, expander_col = st.columns([0.15, 0.85])  # Small column for checkbox, larger for expander
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
                        if selected_by_llm:
                            st.caption("Selected by LLM reranker")
                        elif any_llm_selected:
                            st.caption("Not selected by LLM reranker")

        # Second indicator in this row (right column)
        if i + 1 < len(ordered):
            with row_col2:
                item = ordered[i + 1]
                idx = i + 2
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

                title = f"{idx}. {name} (score: {score:.3f})"
                if selected_by_llm:
                    rank_label = f"rank {llm_rank}" if llm_rank else "selected"
                    title += f" | LLM {rank_label}"

                # Checkbox stays visible; expander holds details.
                checkbox_col, expander_col = st.columns([0.15, 0.85])  # Small column for checkbox, larger for expander
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
                        if selected_by_llm:
                            st.caption("Selected by LLM reranker")
                        elif any_llm_selected:
                            st.caption("Not selected by LLM reranker")


def render_chart_designer(chart_payload: dict, df: pd.DataFrame) -> None:
    if df.empty:
        st.warning("The SQL agent returned no rows to visualize.")
        return

    schema = chart_payload.get("chart_schema") or {}
    insights = chart_payload.get("insights")
    if insights:
        st.markdown("### Insights")
        st.write(insights)

    data_quality_notes = chart_payload.get("data_quality_notes")

    with st.sidebar:
        st.markdown("### Chart Configuration")
        default_chart = schema.get("default_chart_type", "line").title()
        palette_names = list(chart_utils.COLOR_PALETTES.keys())
        palette_name = st.selectbox("Color palette", palette_names, index=0)
        base_palette = chart_utils.COLOR_PALETTES[palette_name]
        primary_color = st.color_picker("Primary color", base_palette[0])
        colors = [primary_color] + base_palette[1:]

        font_choice = st.selectbox("Font family", chart_utils.DEFAULT_FONTS, index=0)

        show_legend_default = schema.get("show_legend", bool(schema.get("group_field")))
        show_legend = st.checkbox("Show legend", value=show_legend_default)
        legend_title = st.text_input("Legend title", schema.get("legend_title", ""))
        legend_position_default = schema.get("legend_position", "top")
        if legend_position_default not in chart_utils.LEGEND_POSITIONS:
            legend_position_default = "top"
        legend_position = st.selectbox(
            "Legend position",
            chart_utils.LEGEND_POSITIONS,
            index=chart_utils.LEGEND_POSITIONS.index(legend_position_default),
        )

        title_text = st.text_input("Title", schema.get("title", ""))
        subtitle_text = st.text_input("Subtitle", schema.get("subtitle", ""))
        title_anchor_default = schema.get("title_anchor", chart_utils.TITLE_ANCHORS[0])
        if title_anchor_default not in chart_utils.TITLE_ANCHORS:
            title_anchor_default = chart_utils.TITLE_ANCHORS[0]
        title_anchor = st.selectbox(
            "Title alignment",
            chart_utils.TITLE_ANCHORS,
            index=chart_utils.TITLE_ANCHORS.index(title_anchor_default),
        )
        title_orient_default = schema.get("title_orient", chart_utils.TITLE_ORIENTS[0])
        if title_orient_default not in chart_utils.TITLE_ORIENTS:
            title_orient_default = chart_utils.TITLE_ORIENTS[0]
        title_orient = st.selectbox(
            "Title position",
            chart_utils.TITLE_ORIENTS,
            index=chart_utils.TITLE_ORIENTS.index(title_orient_default),
        )

        axis_titles = schema.get("axis_titles") or {}
        axis_x_title = st.text_input("X-axis title", axis_titles.get("x", schema.get("x_field", "x")))
        axis_y_title = st.text_input("Y-axis title", axis_titles.get("y", schema.get("y_field", "value")))

        show_data_labels = st.checkbox(
            "Show data labels",
            value=schema.get("show_data_labels", False),
            help="Displays labels above bars or points.",
        )
        show_gridlines = st.checkbox(
            "Show gridlines",
            value=schema.get("show_gridlines", True),
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
if st.button("Identify Indicators", type="primary", width="stretch"):
    if question:
        if total_weight > 1.0:
            st.error("Total weight must be ≤ 1.0. Please adjust sliders.")
        else:
            with st.status("Searching indicators...") as status:
                start_time = time.perf_counter()
                res = {"results": None, "error": None, "done": False}

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
                    status.update(label=f"Searching indicators... {elapsed:.1f}s")
                    time.sleep(0.1)

                if res["error"]:
                    status.update(label="Retrieval failed", state="error")
                    logger.exception("Retrieval failed", extra={"module_name": ModuleName.UI})
                    st.error(f"Retrieval failed: {res['error']}")
                else:
                    elapsed = time.perf_counter() - start_time
                    st.session_state.results = res["results"]
                    st.session_state.use_llm_rerank = use_llm_rerank
                    st.session_state.pop("chart_payload", None)
                    st.session_state.pop("chart_dataframe", None)
                    status.update(label=f"Found indicators in {elapsed:.2f}s", state="complete")
    else:
        st.warning("Please ask a question first!")

# Output section
if st.session_state.results:
    # Render indicators in two columns
    render_results(st.session_state.results, st.session_state.get("use_llm_rerank", False))

    # Submit button below the indicators
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])  # Center the button
    with col2:
        if st.button("Submit Selection", type="primary", width="stretch"):
            selected_indicators = collect_selected_indicators(st.session_state.results)
            if not question:
                st.warning("Please provide a question so the SQL agent knows what to answer.")
            elif not selected_indicators:
                st.warning("Select at least one indicator to proceed.")
            else:
                with st.status("Querying SQL agent...") as status:
                    start_time = time.perf_counter()
                    res = {"payload": None, "error": None, "done": False}

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
                        status.update(label=f"Querying SQL agent... {elapsed:.1f}s")
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
                        status.update(label=f"SQL query complete in {elapsed:.2f}s", state="complete")

if st.session_state.get("chart_payload") and st.session_state.get("chart_dataframe") is not None:
    st.markdown("---")
    st.markdown("## Visualization")
    render_chart_designer(st.session_state.chart_payload, st.session_state.chart_dataframe)
