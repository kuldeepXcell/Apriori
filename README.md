# Data Visualization | Apriori

This is a simple **Streamlit-based application** designed to showcase basic data visualization functionality and an
interactive chatbot interface. The app answers user queries and provides a clean, modern interface for interacting with
data.

For the larger production system (moderation → hybrid retrieval → reranking), see `PROJECT_STRUCTURE.md` for the current
layout and how to extend it.

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

## CLI: Keyword Generator

  - Defaults (edit in `src/app/steps/preprocess/keyword_generator.py`):
    - input: `data/country_indicators.json`
    - sheet: `Country`
    - range: `0:25` (inclusive start, exclusive end)
    - keywords per indicator: `15`
    - overwrite existing: `False`
  - Command (uses defaults; no args needed):
    ```bash
    PYTHONPATH=./src uv run python -m app.steps.preprocess.keyword_generator
    ```

## CLI: Ingestion Pipeline

  ```bash
  PYTHONPATH=./src uv run python -m app.ingest --pipeline baseline_hybrid
  ```

## Pipelines and preprocessing

- All pipelines start from `data/country_indicators.json` (future versions may change text but keep the same structure).
- Pipeline YAML configs live under `config/pipelines/`. Add new variants there to toggle if/else flow in the orchestrator.
- Shared preprocessing hooks will live under `src/app/steps/` (per-pipeline customizations can be added later if needed).
