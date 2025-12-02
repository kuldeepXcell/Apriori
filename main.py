import streamlit as st
from app.services.retrieval import retrieval_service
from app.core.config import settings

st.set_page_config(page_title="Financial Indicator RAG", layout="wide")

st.title("Financial Indicator Search")
st.markdown("Search for financial indicators using natural language.")

# Sidebar for debug/config
with st.sidebar:
    st.header("Configuration")
    if not settings.OPENAI_API_KEY:
        st.error("OpenAI API Key not found in .env")
    else:
        st.success("OpenAI API Key loaded")
        
    if not settings.QDRANT_API_KEY and "localhost" not in settings.QDRANT_URL:
         st.warning("Qdrant API Key missing (might be needed for cloud)")

# Main Search Interface
query = st.text_input("Enter your query:", placeholder="e.g., GDP growth in developing countries")

if st.button("Search") or query:
    if not query:
        st.warning("Please enter a query.")
    else:
        with st.spinner("Searching and analyzing..."):
            try:
                results = retrieval_service.search_indicators(query)
                
                if not results:
                    st.info("No relevant indicators found.")
                else:
                    st.success(f"Found {len(results)} relevant indicators.")
                    
                    for res in results:
                        with st.expander(f"{res.indicator.name} (Relevance: {res.score:.2f})"):
                            st.markdown(f"**Definition:** {res.indicator.definition}")
                            if res.indicator.keywords:
                                st.markdown(f"**Keywords:** {', '.join(res.indicator.keywords)}")
                            if res.relevance_reason:
                                st.markdown(f"**Reason:** {res.relevance_reason}")
                                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
