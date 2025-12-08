"""Streamlit UI for running configured pipelines side-by-side."""

from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st

from app.config import paths
from app.config.settings import settings
from app.pipeline.pipeline_registry import pipeline_registry


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
        st.header("📊 Active Pipelines")
        st.write(f"**{len(all_pipelines)} pipelines loaded:**")
        for pipeline in all_pipelines:
            with st.expander(f"🔧 {pipeline.config.name}"):
                st.markdown(f"**Description:** {pipeline.config.description or 'N/A'}")
                st.markdown(f"**Embedding:** `{pipeline.config.embedding_model or 'N/A'}`")
                st.markdown(f"**LLM:** `{pipeline.config.llm_model or 'N/A'}`")
                st.markdown(
                    f"**Collection:** `{pipeline.config.collection_name or 'N/A'}`"
                )

    # Main Search Interface
    st.divider()
    query = st.text_input(
        "🔎 Enter your query:",
        placeholder="e.g., GDP growth in developing countries",
    )

    if st.button("🚀 Search All Pipelines", type="primary") or query:
        if not query:
            st.warning("Please enter a query.")
        elif not all_pipelines:
            st.warning("No pipelines found. Add configs under `config/pipelines/`.")
        else:
            st.info(f"Running {len(all_pipelines)} pipelines in parallel...")

            pipeline_results = {}

            def run_pipeline(pipeline):
                """Helper function to run a single pipeline."""
                try:
                    results = pipeline.search(query)
                    return pipeline.config.name, results, pipeline.execution_time, None
                except Exception as exc:  # pragma: no cover - defensive UI catch
                    return pipeline.config.name, [], 0.0, str(exc)

            # Use ThreadPoolExecutor for parallel execution
            with st.spinner("Searching..."):
                with ThreadPoolExecutor(max_workers=max(len(all_pipelines), 1)) as exe:
                    futures = {exe.submit(run_pipeline, p): p for p in all_pipelines}

                    for future in as_completed(futures):
                        name, results, exec_time, error = future.result()
                        pipeline_results[name] = {
                            "results": results,
                            "execution_time": exec_time,
                            "error": error,
                        }

            # Display results in columns
            st.success("✅ Search complete!")
            st.divider()

            cols = st.columns(len(all_pipelines))

            for idx, pipeline in enumerate(all_pipelines):
                name = pipeline.config.name
                data = pipeline_results.get(name, {})
                results = data.get("results", [])
                exec_time = data.get("execution_time", 0.0)
                error = data.get("error")

                with cols[idx]:
                    st.subheader(f"📌 {name.upper()}")
                    st.caption(
                        f"{pipeline.config.embedding_model or 'N/A'} + "
                        f"{pipeline.config.llm_model or 'N/A'}"
                    )
                    st.metric("Execution Time", f"{exec_time:.2f}s")

                    if error:
                        st.error(f"❌ Error: {error}")
                    elif not results:
                        st.info("ℹ️ No relevant indicators found.")
                    else:
                        st.success(f"✅ Found {len(results)} indicators")

                        for res in results:
                            indicator = getattr(res, "indicator", None)
                            score = getattr(res, "score", None)
                            relevance = getattr(res, "relevance_reason", None)

                            title = getattr(indicator, "name", "Unknown")
                            definition = getattr(indicator, "definition", None)
                            keywords = getattr(indicator, "keywords", None)

                            with st.expander(
                                f"**{title}** ({score:.2f})" if score is not None else f"**{title}**"
                            ):
                                if definition:
                                    st.markdown(f"📝 **Definition:** {definition}")
                                if keywords:
                                    st.markdown(
                                        f"🏷️ **Keywords:** {', '.join(keywords)}"
                                    )
                                if relevance:
                                    st.caption(f"💡 {relevance}")

    st.divider()
    st.caption(
        "💡 Tip: Add more pipelines in `config/pipelines/` to compare different models."
    )


if __name__ == "__main__":
    run_app()


