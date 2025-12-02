import streamlit as st
from app.pipelines.pipeline_registry import pipeline_registry
from app.core.config import settings
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

st.set_page_config(page_title="Financial Indicator RAG - Multi-Model Comparison", layout="wide")

st.title("🔍 Financial Indicator Search")
st.markdown("Compare results across different embedding models and LLMs in parallel.")

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
    
    all_pipelines = pipeline_registry.get_all_pipelines()
    st.write(f"**{len(all_pipelines)} pipelines loaded:**")
    for pipeline in all_pipelines:
        with st.expander(f"🔧 {pipeline.config.name}"):
            st.markdown(f"**Description:** {pipeline.config.description}")
            st.markdown(f"**Embedding:** `{pipeline.config.embedding_model}`")
            st.markdown(f"**LLM:** `{pipeline.config.llm_model}`")
            st.markdown(f"**Collection:** `{pipeline.config.collection_name}`")

# Main Search Interface
st.divider()
query = st.text_input("🔎 Enter your query:", placeholder="e.g., GDP growth in developing countries")

if st.button("🚀 Search All Pipelines", type="primary") or query:
    if not query:
        st.warning("Please enter a query.")
    else:
        st.info(f"Running {len(all_pipelines)} pipelines in parallel...")
        
        # Execute all pipelines in parallel
        pipeline_results = {}
        
        def run_pipeline(pipeline):
            """Helper function to run a single pipeline."""
            try:
                results = pipeline.search(query)
                return pipeline.config.name, results, pipeline.execution_time, None
            except Exception as e:
                return pipeline.config.name, [], 0.0, str(e)
        
        # Use ThreadPoolExecutor for parallel execution
        with st.spinner("Searching..."):
            with ThreadPoolExecutor(max_workers=len(all_pipelines)) as executor:
                futures = {executor.submit(run_pipeline, p): p for p in all_pipelines}
                
                for future in as_completed(futures):
                    name, results, exec_time, error = future.result()
                    pipeline_results[name] = {
                        "results": results,
                        "execution_time": exec_time,
                        "error": error
                    }
        
        # Display results in columns
        st.success("✅ Search complete!")
        st.divider()
        
        # Create columns for side-by-side comparison
        cols = st.columns(len(all_pipelines))
        
        for idx, pipeline in enumerate(all_pipelines):
            name = pipeline.config.name
            data = pipeline_results.get(name, {})
            results = data.get("results", [])
            exec_time = data.get("execution_time", 0.0)
            error = data.get("error")
            
            with cols[idx]:
                st.subheader(f"📌 {name.upper()}")
                st.caption(f"{pipeline.config.embedding_model} + {pipeline.config.llm_model}")
                st.metric("Execution Time", f"{exec_time:.2f}s")
                
                if error:
                    st.error(f"❌ Error: {error}")
                elif not results:
                    st.info("ℹ️ No relevant indicators found.")
                else:
                    st.success(f"✅ Found {len(results)} indicators")
                    
                    for res in results:
                        with st.expander(f"**{res.indicator.name}** ({res.score:.2f})"):
                            st.markdown(f"📝 **Definition:** {res.indicator.definition}")
                            if res.indicator.keywords:
                                st.markdown(f"🏷️ **Keywords:** {', '.join(res.indicator.keywords)}")
                            if res.relevance_reason:
                                st.caption(f"💡 {res.relevance_reason}")

# Footer
st.divider()
st.caption("💡 Tip: Add more pipelines in `app/configs/default_pipelines.py` to compare different models.")
