import streamlit as st
from sqlalchemy import text
import json

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


def init_db():
    with db_conn.session as session:
        session.execute(text("CREATE EXTENSION IF NOT EXISTS pgcrypto;"))
        session.execute(text("""
        CREATE TABLE IF NOT EXISTS query_traces 
        (
          trace_id UUID PRIMARY KEY DEFAULT gen_random_uuid(), 
          query_text TEXT NOT NULL, embedding_model TEXT,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now(), 
          reranker_used BOOLEAN NOT NULL DEFAULT FALSE, 
          reranker_model TEXT, reranker_prompt TEXT, 
          quadrant_collection TEXT,
          reranker_selected JSONB NOT NULL DEFAULT '[]'::jsonb, 
          reranker_rejected JSONB NOT NULL DEFAULT '[]'::jsonb, 
          user_checked JSONB NOT NULL DEFAULT '[]'::jsonb,
          user_unchecked JSONB NOT NULL DEFAULT '[]'::jsonb,
          indicator_catalog JSONB NOT NULL, user_suggested JSONB,
          retrieval_latency_ms INTEGER, reranker_latency_ms INTEGER, 
          precision_at_k NUMERIC(5,4), recall_at_k NUMERIC(5,4),
          reranker_notes JSONB
        );"""))

        # Consumer: Consumer Confidence Index Score Table
        session.execute(text("""
        Drop TABLE IF EXISTS consumer_confidence_index_score;
        CREATE TABLE consumer_confidence_index_score 
        (
            id SERIAL PRIMARY KEY,
            country_name VARCHAR(100) NOT NULL UNIQUE,
            last NUMERIC(10, 2),
            previous NUMERIC(10, 2),
            reference VARCHAR(20),
            unit VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );"""))

        # Consumer: Insert data into Consumer Confidence Index Score Table
        session.execute(text("""
        INSERT INTO consumer_confidence_index_score (country_name, last, previous, reference, unit) VALUES
            ('Albania', -26.1, -25.1, 'Oct/25', 'points'),
            ('Argentina', 46.04, 42.32, 'Nov/25', 'points'),
            ('Australia', 104, 92.1, 'Nov/25', 'points'),
            ('Austria', -17.5, -17.7, 'Nov/25', 'points'),
            ('Belgium', 2, 0, 'Nov/25', 'points'),
            ('Bulgaria', -25.1, -21.1, 'Dec/25', 'points'),
            ('Canada', 46, 47.8, 'Oct/25', 'points'),
            ('Cape Verde', 13, 13, 'Jun/25', 'points'),
            ('Chile', 37.9, 41.3, 'Oct/25', 'points'),
            ('China', 89.6, 89.2, 'Sep/25', 'points'),
            ('Colombia', 13.6, 1.6, 'Oct/25', '%'),
            ('Croatia', -10.5, -9.8, 'Nov/25', 'points'),
            ('Cyprus', -13.6, -15.8, 'Nov/25', 'points'),
            ('Czech Republic', 112, 107, 'Nov/25', 'points'),
            ('Denmark', -20.1, -19.5, 'Nov/25', 'points'),
            ('Ecuador', 37.22, 38.35, 'Sep/25', 'points'),
            ('Estonia', -28.2, -30.4, 'Nov/25', 'points'),
            ('Euro Area', -14.2, -14.2, 'Nov/25', 'points'),
            ('European Union', -13.6, -13.5, 'Nov/25', 'points'),
            ('Faroe Islands', -17, -12, 'Jun/25', 'points'),
            ('Finland', -6.5, -7.6, 'Nov/25', 'points'),
            ('France', 89, 90, 'Nov/25', 'points'),
            ('Georgia', -24.9, -19.5, 'Oct/24', 'points'),
            ('Germany', -23.2, -24.1, 'Dec/25', 'points'),
            ('Greece', -50.6, -47.6, 'Nov/25', 'points'),
            ('Hungary', -27.2, -25.6, 'Nov/25', 'points'),
            ('Iceland', 73.5, 84.1, 'Oct/25', 'points'),
            ('India', 96.9, 96.5, 'Sep/25', 'points'),
            ('Indonesia', 121, 115, 'Oct/25', 'points'),
            ('Ireland', 61, 59.9, 'Nov/25', 'points'),
            ('Israel', -12.36, -25.21, 'Oct/25', 'points'),
            ('Italy', 95, 97.6, 'Nov/25', 'points'),
            ('Japan', 35.8, 35.3, 'Oct/25', 'points'),
            ('Kyrgyzstan', 42.8, 39.2, 'Jun/25', 'points'),
            ('Latvia', -7.4, -8.1, 'Nov/25', 'points'),
            ('Lithuania', 2, 2, 'Nov/25', 'points'),
            ('Luxembourg', -6.1, -10, 'Nov/25', 'points'),
            ('Malaysia', 127, 141, 'Mar/25', 'points'),
            ('Malta', 8.4, 4.4, 'Nov/25', 'points'),
            ('Mexico', 46.1, 46.4, 'Oct/25', 'points'),
            ('Morocco', 53.6, 54.6, 'Sep/25', 'points'),
            ('Netherlands', -21, -27, 'Nov/25', 'points'),
            ('New Zealand', 90.9, 91.2, 'Sep/25', 'points'),
            ('Norway', -3.7, -4.5, 'Dec/25', 'points'),
            ('Pakistan', 40, 37.7, 'Oct/25', 'points'),
            ('Paraguay', 48.32, 49.86, 'Oct/25', 'points'),
            ('Philippines', -9.8, -14, 'Sep/25', 'points'),
            ('Poland', -9.9, -10.9, 'Nov/25', 'points'),
            ('Portugal', -15.2, -15.9, 'Nov/25', 'points'),
            ('Romania', -29.7, -30.5, 'Nov/25', 'points'),
            ('Russia', -9, -8, 'Sep/25', 'points'),
            ('Slovakia', -24.6, -22.8, 'Nov/25', 'points'),
            ('Slovenia', -23, -25, 'Nov/25', 'points'),
            ('South Africa', -13, -10, 'Sep/25', 'points'),
            ('South Korea', 112, 110, 'Nov/25', 'points'),
            ('Spain', 81.5, 82.9, 'Sep/25', 'points'),
            ('Sweden', 96.1, 96.8, 'Nov/25', 'points'),
            ('Switzerland', -37, -37, 'Oct/25', 'points'),
            ('Taiwan', 64.65, 63.96, 'Nov/25', 'points'),
            ('Thailand', 51.9, 50.7, 'Oct/25', 'points'),
            ('Turkey', 85, 83.6, 'Nov/25', 'points'),
            ('Ukraine', 75.1, 81.7, 'Sep/25', 'points'),
            ('United Kingdom', -19, -17, 'Nov/25', 'points'),
            ('United States', 51, 53.6, 'Nov/25', 'points');
        """))

        # Consumer: Television Viewership Table
        session.execute(text("""
        Drop TABLE IF EXISTS television_viewership;
        CREATE TABLE television_viewership 
        (
            id SERIAL PRIMARY KEY,
            year INTEGER NOT NULL UNIQUE,
            viewership_billion_amas BIGINT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );"""))

        # Consumer: Insert data into Television Viewership Table
        session.execute(text("""
        INSERT INTO television_viewership (year, viewership_billion_amas) VALUES
            (2018, 1604),
            (2019, 1614),
            (2020, 1731),
            (2021, 1591),
            (2022, 1474),
            (2023, 1508),
            (2024, 1530);
        """))

        # Country: Air Quality Index Major Indian Cities Table
        session.execute(text("""
        Drop TABLE IF EXISTS air_quality_index_major_indian_cities;
        CREATE TABLE air_quality_index_major_indian_cities 
        (
            id SERIAL PRIMARY KEY,
            state VARCHAR(100),
            city VARCHAR(100),
            year_2022_good INTEGER,
            year_2022_satisfactory INTEGER,
            year_2022_moderate INTEGER,
            year_2022_poor INTEGER,
            year_2022_very_poor INTEGER,
            year_2022_severe INTEGER,
            year_2023_good INTEGER,
            year_2023_satisfactory INTEGER,
            year_2023_moderate INTEGER,
            year_2023_poor INTEGER,
            year_2023_very_poor INTEGER,
            year_2023_severe INTEGER,
            year_2024_good INTEGER,
            year_2024_satisfactory INTEGER,
            year_2024_moderate INTEGER,
            year_2024_poor INTEGER,
            year_2024_very_poor INTEGER,
            year_2024_severe INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );"""))

        # Country: Insert data into Air Quality Index Major Indian Cities Table
        session.execute(text("""
        INSERT INTO air_quality_index_major_indian_cities (state, city, year_2022_good, year_2022_satisfactory, year_2022_moderate, year_2022_poor, year_2022_very_poor, year_2022_severe, year_2023_good, year_2023_satisfactory, year_2023_moderate, year_2023_poor, year_2023_very_poor, year_2023_severe, year_2024_good, year_2024_satisfactory, year_2024_moderate, year_2024_poor, year_2024_very_poor, year_2024_severe) VALUES
            ('Maharashtra', 'Mumbai', 21, 152, 148, 42, 1, 0, 5, 180, 145, 35, 0, 0, 82, 137, 145, 2, 0, 0),
            ('Delhi', 'Delhi', 3, 65, 95, 130, 66, 6, 1, 60, 145, 77, 67, 15, 0, 66, 143, 70, 70, 17),
            ('Karnataka', 'Bengaluru', 36, 263, 65, 0, 0, 0, 69, 259, 37, 0, 0, 0, 65, 253, 48, 0, 0, 0),
            ('West Bengal', 'Kolkata', 93, 123, 80, 63, 5, 0, 93, 116, 110, 44, 2, 0, 93, 139, 92, 42, 0, 0),
            ('Tamil Nadu', 'Chennai', 70, 252, 38, 4, 0, 0, 32, 276, 54, 1, 0, 0, 52, 275, 37, 2, 0, 0),
            ('Telangana', 'Hyderabad', 47, 184, 133, 0, 0, 0, 5, 283, 77, 0, 0, 0, 29, 284, 53, 0, 0, 0),
            ('Gujarat', 'Ahmedabad', 10, 141, 167, 29, 0, 0, 3, 158, 201, 3, 0, 0, 12, 119, 231, 4, 0, 0),
            ('Maharashtra', 'Pune', 9, 71, 127, 5, 2, 0, 1, 119, 210, 15, 1, 0, 52, 137, 174, 3, 0, 0),
            ('Gujarat', 'Surat', NULL, NULL, NULL, NULL, NULL, NULL, 29, 70, 113, 56, 7, 0, 56, 184, 13, 25, 1, 0),
            ('Rajasthan', 'Jaipur', 33, 99, 204, 27, 1, 0, 8, 112, 203, 37, 5, 0, 18, 97, 191, 59, 1, 0);
        """))

        # Country: House Price to Income Ratio Table
        session.execute(text("""
        Drop TABLE IF EXISTS house_price_to_income_ratio;
        CREATE TABLE house_price_to_income_ratio 
        (
            id SERIAL PRIMARY KEY,
            ref_area VARCHAR(10),
            country_name VARCHAR(100),
            price_to_income_ratio_2021_q4 NUMERIC(10, 3),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );"""))

        # Country: Insert data into House Price to Income Ratio Table
        session.execute(text("""
        INSERT INTO house_price_to_income_ratio (ref_area, country_name, price_to_income_ratio_2021_q4) VALUES
            ('ARE', 'United Arab Emirates', 76.541),
            ('AUS', 'Australia', 112.602),
            ('AUT', 'Austria', 127.281),
            ('BEL', 'Belgium', 104.803),
            ('BGR', 'Bulgaria', 99.682),
            ('BRA', 'Brazil', 84.713),
            ('CAN', 'Canada', 133.704),
            ('CHE', 'Switzerland', 113.631),
            ('CHL', 'Chile', 115.619),
            ('CHN', 'China', 82.677),
            ('COL', 'Colombia', 111.211),
            ('CYP', 'Cyprus', 84.711),
            ('CZE', 'Czechia', 153.946),
            ('DEU', 'Germany', 137.182),
            ('DNK', 'Denmark', 109.328),
            ('ESP', 'Spain', 116.753),
            ('EST', 'Estonia', 104.367),
            ('FIN', 'Finland', 92.106),
            ('FRA', 'France', 111.338),
            ('GBR', 'United Kingdom', 113.047),
            ('GRC', 'Greece', 109.951),
            ('HKG', 'Hong Kong SAR, China', 113.687),
            ('HRV', 'Croatia', 107.702),
            ('HUN', 'Hungary', 132.804),
            ('IDN', 'Indonesia', 79.741),
            ('IND', 'India', 75.002),
            ('IRL', 'Ireland', 99.531),
            ('ISL', 'Iceland', 139.756),
            ('ISR', 'Israel', 105.519),
            ('ITA', 'Italy', 93.272),
            ('JPN', 'Japan', 114.443),
            ('KOR', 'Korea, Rep.', 106.187),
            ('LTU', 'Lithuania', 108.935),
            ('LUX', 'Luxembourg', 140.086),
            ('LVA', 'Latvia', 117.947),
            ('MAR', 'Morocco', 89.07),
            ('MEX', 'Mexico', 116.589),
            ('MKD', 'North Macedonia', 92.561),
            ('MLT', 'Malta', 111.069),
            ('MYS', 'Malaysia', 91.418),
            ('NLD', 'Netherlands', 141.059),
            ('NOR', 'Norway', 87.121),
            ('NZL', 'New Zealand', 152.392),
            ('PER', 'Peru', 75.585),
            ('PHL', 'Philippines', 92.833),
            ('POL', 'Poland', 106.104),
            ('PRT', 'Portugal', 143.903),
            ('ROU', 'Romania', 77.996),
            ('RUS', 'Russian Federation', 76.94),
            ('SGP', 'Singapore', 93.64),
            ('SRB', 'Serbia', 84.459),
            ('SVK', 'Slovak Republic', 131.488),
            ('SVN', 'Slovenia', 116.098),
            ('SWE', 'Sweden', 110.433),
            ('THA', 'Thailand', 96.406),
            ('TUR', 'Turkiye', 89.994),
            ('USA', 'United States', 126.006),
            ('ZAF', 'South Africa', 96.639);
        """))

        session.commit()


# Initialize the database and tables
init_db()

# if st.button("Insert sample trace & show table"):
#     # Insert a sample row using bound parameters (avoids SQLAlchemy param parsing issues)
#     with db_conn.session as session:
#         session.execute(
#             text(
#                 """
#                 INSERT INTO query_traces (
#                   query_text,
#                   embedding_model,
#                   reranker_used,
#                   reranker_model,
#                   reranker_prompt,
#                   quadrant_collection,
#                   reranker_selected,
#                   reranker_rejected,
#                   user_checked,
#                   user_unchecked,
#                   indicator_catalog,
#                   user_suggested,
#                   reranker_notes
#                 ) VALUES (
#                   :query_text,
#                   :embedding_model,
#                   :reranker_used,
#                   :reranker_model,
#                   :reranker_prompt,
#                   :quadrant_collection,
#                   :reranker_selected,
#                   :reranker_rejected,
#                   :user_checked,
#                   :user_unchecked,
#                   :indicator_catalog,
#                   :user_suggested,
#                   :reranker_notes
#                 );
#                 """
#             ),
#             {
#                 "query_text": "What are the key indicators that signal a fiscal policy tightening in Europe?",
#                 "embedding_model": "text-embedding-3-small",
#                 "reranker_used": True,
#                 "reranker_model": "gpt-5-nano",
#                 "reranker_prompt": "...reranker prompt here...",
#                 "quadrant_collection": "europe-macro-quadrant-v1",
#                 "reranker_selected": json.dumps([
#                     {"indicator": "core_inflation", "rank": 1, "score": 0.94}
#                 ]),
#                 "reranker_rejected": json.dumps([
#                     {"indicator": "headline_inflation", "rank": 2, "score": 0.72}
#                 ]),
#                 "user_checked": json.dumps([
#                     {
#                         "indicator": "core_inflation",
#                         "clicked_at": "2025-12-16T12:34:56Z",
#                     }
#                 ]),
#                 "user_unchecked": json.dumps([
#                     {"indicator": "headline_inflation"}
#                 ]),
#                 "indicator_catalog": json.dumps([
#                     {"indicator": "core_inflation"},
#                     {"indicator": "headline_inflation"},
#                 ]),
#                 "user_suggested": json.dumps(["credit_growth"]),
#                 "reranker_notes": json.dumps(
#                     {"llm_tokens_used": 1234, "notes": "first experiment"}
#                 ),
#             },
#         )
#         session.commit()
#
#     # Now read back and display the latest rows
#     df = db_conn.query(
#         "SELECT * FROM query_traces ORDER BY created_at DESC LIMIT 50;",
#         ttl=0,  # no caching so you see the new row immediately
#     )
#     st.dataframe(df)

with st.expander("Country and Consumer Data"):
    if st.button("Consumer: Show Consumer Confidence Index Score Table"):
        df = db_conn.query(
            "SELECT * FROM consumer_confidence_index_score ORDER BY created_at DESC;",
            ttl=0,
        )
        st.dataframe(df)

    if st.button("Consumer: Show Television Viewership Table"):
        df = db_conn.query(
            "SELECT * FROM television_viewership ORDER BY created_at DESC;",
            ttl=0,
        )
        st.dataframe(df)

    if st.button("Country: Show Air Quality Index Major Indian Cities Table"):
        df = db_conn.query(
            "SELECT * FROM air_quality_index_major_indian_cities ORDER BY created_at DESC;",
            ttl=0,
        )
        st.dataframe(df)

    if st.button("Country: Show House Price to Income Ratio Table"):
        df = db_conn.query(
            "SELECT * FROM house_price_to_income_ratio ORDER BY created_at DESC;",
            ttl=0,
        )
        st.dataframe(df)

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
