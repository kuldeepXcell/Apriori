import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Data Visualization | Apriori",
    page_icon="static/AprioriFavicon.png",
    layout="centered",
    menu_items={
        'Get Help': 'https://www.aprioriconsultants.com/',
        'About': "This is a simple application designed to Identify Indicators from the user's question."
    }
)

# Initialize DB connection.
db_conn = st.connection("postgresql", type="sql")

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        text-align: center;
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
""", unsafe_allow_html=True)

# Header section
st.markdown("""
    <div class="main-header">
        <div class="company-logo">
            <img src="app/static/AprioriFullLogo.png" alt="Apriori Logo">
        </div>
        <div class="app-title">Data Visualization</div>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# Initialize session state for storing responses
if 'response' not in st.session_state:
    st.session_state.response = ""

# Input section
question = st.text_input("Ask a question", placeholder="Type your question here...")


# Your chatbot logic function
def process_question(que):
    """
    Replace this function with your actual chatbot logic.
    This is just a simple example that echoes the question.
    """
    # Example logic - replace with your implementation
    response = f"You asked: '{que}'\n\n"
    response += "This is where your chatbot response would appear. "
    response += "Integrate your AI model, API calls, or logic here."

    # Example: Simple rule-based responses
    question_lower = que.lower()
    if "hello" in question_lower or "hi" in question_lower:
        response = "Hello! How can I assist you today?"
    elif "help" in question_lower:
        response = "I'm here to help! You can ask me questions about our services, products, or general inquiries."
    elif "?" in que:
        response = f"That's a great question! Regarding '{que}', let me provide you with a detailed answer based on my knowledge base."

    return response


# Process button
if st.button("Identify Indicators", type="primary", use_container_width=True):
    if question:
        with st.spinner("Processing your question..."):
            # YOUR LOGIC HERE
            # Replace this with your actual chatbot logic
            response = process_question(question)
            st.session_state.response = response
    else:
        st.warning("Please ask a question first!")

# Output section
if st.session_state.response:
    st.markdown("### Response:")
    st.markdown(f'<div class="output-box">{st.session_state.response}</div>',
                unsafe_allow_html=True)
