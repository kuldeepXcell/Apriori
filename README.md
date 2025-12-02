# Data Visualization | Apriori

This is a simple **Streamlit-based application** designed to showcase basic data visualization functionality and an
interactive chatbot interface. The app answers user queries and provides a clean, modern interface for interacting with
data.

For the larger production system (moderation → hybrid retrieval → reranking), see `docs/architecture-plan.md` for the
proposed architecture and module layout.

## Features

- **Customizable Data Visualization:** A dedicated space to implement data visualization, which can be added to or
  modified as needed.
- **Interactive Chatbot:** Users can ask questions, and the app processes and returns responses based on a set of
  predefined rules.
- **Responsive Design:** The layout adapts to various screen sizes for optimal viewing on different devices.

## Requirements

- **Python:** 3.11+
- **Streamlit:** 1.x+

## Installation

To get started with this project, follow the steps below to clone the repository and install the necessary dependencies.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/apriori-data-visualization.git
   cd apriori-data-visualization
   ```

2. **Set up a Python environment and install required libraries:**
   ```bash
   uv sync
   ```

## Running the Application

After installing the necessary dependencies, you can launch the application using the following command:

```bash
streamlit run main.py
```

This will launch the Streamlit app in your default web browser. You can start interacting with the chatbot and data
visualizations directly.

Note: STREAMLIT_SERVER_ENABLE_STATIC_SERVING=true
