"""Streamlit UI for running configured pipelines side-by-side."""

import streamlit as st

from app.config import paths
from app.config.settings import settings
from app.pipeline.pipeline_registry import pipeline_registry
from app.pipeline.models import MultivectorWeights, SearchHit, SearchResponse


def _render_results_column(
    container,
    title: str,
    response: SearchResponse,
    show_generated_def: bool = False,
) -> None:
    """Render a single column of results with reranker-first ordering."""
    with container:
        st.subheader(f"📌 {title}")
        if show_generated_def and response.generated_definition:
            st.info(f"HyDe definition:\n\n{response.generated_definition}")

        if response.error:
            st.error(f"❌ Error: {response.error}")
            return
        if not response.hits:
            st.info("ℹ️ No relevant indicators found.")
            return

        reranked = response.reranked or []
        if reranked:
            st.markdown(f"### 🧠 LLM reranker picks ({len(reranked)})")
            _render_hit_list(reranked, key_prefix=f"{title}-rerank")
        else:
            st.info("LLM reranker did not return any reordered items.")

        remaining = _remaining_hits(response.hits, reranked)
        if remaining:
            st.divider()
            st.markdown(f"### 📂 Other retrieved (from initial 15) ({len(remaining)})")
            _render_hit_list(remaining, key_prefix=f"{title}-others")

        st.button("Submit Selection", key=f"{title}-submit")


def _render_hit_list(hits: list[SearchHit], key_prefix: str) -> None:
    for idx, hit in enumerate(hits):
        indicator = hit.indicator
        checkbox_key = f"{key_prefix}-chk-{idx}-{indicator.name}"
        checked = st.checkbox(
            f"{indicator.name} ({hit.score:.2f})",
            key=checkbox_key,
        )
        with st.expander(f"Details: {indicator.name}"):
            if indicator.definition:
                st.markdown(f"📝 **Definition:** {indicator.definition}")
            if indicator.application_context:
                st.markdown(f"🏛️ **Application:** {indicator.application_context}")
            if indicator.keywords:
                st.markdown(f"🏷️ **Keywords:** {', '.join(indicator.keywords)}")
            if indicator.question:
                st.markdown(f"❓ **Question:** {indicator.question}")
            if hit.relevance_reason:
                st.caption(f"💡 {hit.relevance_reason}")
            st.caption(f"Selected: {checked}")


def _remaining_hits(all_hits: list[SearchHit], reranked: list[SearchHit]) -> list[SearchHit]:
    rerank_names = {
        h.indicator.normalized_name or h.indicator.name for h in reranked
    }
    remaining: list[SearchHit] = []
    for hit in all_hits:
        name = hit.indicator.normalized_name or hit.indicator.name
        if name not in rerank_names:
            remaining.append(hit)
    return remaining


def run_app() -> None:
    """Launch the Streamlit UI."""
    paths.ensure_directories()

    st.set_page_config(
        page_title="Financial Indicator RAG - Multi-Model Comparison",
        layout="wide",
    )

    st.title("🔍 Financial Indicator Search")
    st.markdown("Compare results across different embedding models and LLMs in parallel.")

    all_pipelines = pipeline_registry.get_all_pipelines()
    active_pipeline = all_pipelines[0] if all_pipelines else None

    # Sidebar for configuration and debug
    with st.sidebar:
        st.header("⚙️ Configuration")
        if not settings.OPENAI_API_KEY:
            st.error("❌ OpenAI API Key not found in .env")
        else:
            st.success("✅ OpenAI API Key loaded")

        if not settings.QDRANT_API_KEY and "localhost" not in settings.QDRANT_URL:
            st.warning("⚠️ Qdrant API Key missing (might be needed for cloud)")

        st.divider()
        st.header("📊 Active Pipeline")
        if active_pipeline:
            st.markdown(f"**Name:** {active_pipeline.config.name}")
            st.markdown(f"**Embedding:** `{active_pipeline.config.embedding_model or 'N/A'}`")
            st.markdown(f"**LLM:** `{active_pipeline.config.llm_model or 'N/A'}`")
            st.markdown(
                f"**Collection:** `{active_pipeline.config.collection_name or 'N/A'}`"
            )
        else:
            st.warning("No pipelines found. Add configs under `config/pipelines/`.")

        st.divider()
        st.header("🎚️ Multivector Weights")
        w_application = st.slider("Application weight", 0.0, 1.0, 0.33, 0.01)
        w_context = st.slider("Context weight", 0.0, 1.0, 0.33, 0.01)
        w_question = st.slider("Question weight", 0.0, 1.0, 0.33, 0.01)

    # Main Search Interface
    st.divider()
    query = st.text_input(
        "🔎 Enter your query:",
        placeholder="e.g., GDP growth in developing countries",
    )

    if st.button("🚀 Search", type="primary") or query:
        if not query:
            st.warning("Please enter a query.")
        elif not active_pipeline:
            st.warning("No pipeline found. Add configs under `config/pipelines/`.")
        else:
            st.info("Running multivector and HyDe searches...")
            weights = MultivectorWeights(
                application=w_application, context=w_context, question=w_question
            )

            with st.spinner("Searching Qdrant..."):
                try:
                    bundle = active_pipeline.search(query, weights=weights)
                    exec_time = active_pipeline.execution_time
                except Exception as exc:  # pragma: no cover - defensive UI catch
                    bundle = None
                    exec_time = 0.0
                    st.error(f"❌ Error while searching: {exc}")

            if bundle:
                st.success("✅ Search complete!")
                st.caption(f"Execution time: {exec_time:.2f}s")
                st.divider()

                col_mv, col_hyde = st.columns(2)
                _render_results_column(
                    col_mv,
                    title="Multivector",
                    response=bundle.multivector,
                    show_generated_def=False,
                )
                _render_results_column(
                    col_hyde,
                    title="HyDe",
                    response=bundle.hyde,
                    show_generated_def=True,
                )

    st.divider()
    st.caption("💡 Tip: Add more pipelines in `config/pipelines/` to compare different models.")
    st.caption("Run locally: `streamlit run main.py`")


if __name__ == "__main__":
    run_app()


