from pathlib import Path
import sys

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

import streamlit as st

from app.config.settings import settings
from app.core.logging import ModuleName, get_logger, setup_logging
from app.core.langsmith import configure_langsmith
from app.steps.retrieval import baseline_hybrid_retrieval as retrieval

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
                selected_by_llm = item.get("selected_by_llm", False)
                llm_rank = item.get("llm_rank")

                title = f"{idx}. {name} (score: {score:.3f})"
                if selected_by_llm:
                    rank_label = f"rank {llm_rank}" if llm_rank else "selected"
                    title += f" | LLM {rank_label}"

                # Checkbox stays visible; expander holds details.
                checkbox_col, expander_col = st.columns([0.15, 0.85])  # Small column for checkbox, larger for expander
                with checkbox_col:
                    st.checkbox("Select", key=f"select_{idx}_{item.get('id', name)}")
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
                selected_by_llm = item.get("selected_by_llm", False)
                llm_rank = item.get("llm_rank")

                title = f"{idx}. {name} (score: {score:.3f})"
                if selected_by_llm:
                    rank_label = f"rank {llm_rank}" if llm_rank else "selected"
                    title += f" | LLM {rank_label}"

                # Checkbox stays visible; expander holds details.
                checkbox_col, expander_col = st.columns([0.15, 0.85])  # Small column for checkbox, larger for expander
                with checkbox_col:
                    st.checkbox("Select", key=f"select_{idx}_{item.get('id', name)}")
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
                        if selected_by_llm:
                            st.caption("Selected by LLM reranker")
                        elif any_llm_selected:
                            st.caption("Not selected by LLM reranker")


# Process button
if st.button("Identify Indicators", type="primary", use_container_width=True):
    if question:
        if total_weight > 1.0:
            st.error("Total weight must be ≤ 1.0. Please adjust sliders.")
        else:
            with st.spinner("Searching indicators..."):
                try:
                    results = retrieval.search(
                        question=question,
                        weights=weights,
                        top_k=15,
                        use_llm_rerank=use_llm_rerank,
                    )
                    st.session_state.results = results
                    st.session_state.use_llm_rerank = use_llm_rerank
                except Exception as exc:  # pragma: no cover - UI path
                    logger.exception("Retrieval failed", extra={"module_name": ModuleName.UI})
                    st.error(f"Retrieval failed: {exc}")
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
        if st.button("Submit Selection", type="primary", use_container_width=True):
            # TODO: Add submit functionality here
            st.info("Submit functionality will be implemented here")

