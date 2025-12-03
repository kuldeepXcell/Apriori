# Retrieval Evaluation Metrics

When validating whether the system returns the correct financial indicator(s) for a user question, track both **retrieval-stage** and **answer-stage** metrics. The example query below illustrates how each metric applies to our data.

> Example query: *"How fast is an economy expanding or contracting from one year to the next after adjusting for inflation?"*  
> Expected indicator: **GDP Growth rate / GDP growth (annual %)**

## Retrieval Metrics
- **Recall@k / Hit Rate** – For each query, check if the expected indicator appears in the top *k* retrieved items. If GDP Growth rate is inside the top 3 candidates, Recall@3 = 1; otherwise 0. Average across all benchmark queries.
- **Mean Reciprocal Rank (MRR)** – Uses the inverse rank of the correct indicator. If GDP Growth rate is ranked 1st, the score is 1; if 3rd, it is 1/3. Averaging MRR highlights whether the correct indicator consistently appears near the top.
- **nDCG@k** – Supports multiple relevant indicators (e.g., "real GDP" and "GDP growth" might both help). Assign relevance grades, discount lower ranks, and normalize. Useful when several indicators partially answer a query.
- **Query Coverage** – Percentage of indicators that ever appear in the retrieved set for the benchmark queries. If only 80 of 96 indicators appear, coverage is ~83%, signaling missing keywords or metadata.
- **Keyword–Question Overlap** – For each indicator, compute token overlap between generated keywords and its `question`. A low overlap for GDP Growth rate would indicate the keyword prompt needs improvement because sparse search won’t match the query phrasing.

## Answer Quality Metrics
- **Answer Accuracy / F1** – After LLM generation, compare responses against reference answers. Example: verify the answer states the correct annual percentage growth and cites the GDP Growth rate indicator.
- **Faithfulness / Groundedness** – Manually or with LLM-as-judge, check that the answer only uses facts from the retrieved indicator definition or application context (no hallucinated values).
- **Latency & Cost** – Track time and token usage per query for each pipeline (baseline vs. fast) to ensure the improved keywords don’t introduce unacceptable delays.

## How to Use Them
1. Build a small benchmark set of realistic questions covering all subsections (at least 20–30 entries).
2. For each pipeline run, log the top *k* retrieved indicators and compute Recall@k, MRR, nDCG, coverage, and keyword–question overlap.
3. Run the full RAG workflow for the same queries, then grade answers for accuracy and faithfulness.
4. Compare metrics before/after changes (e.g., updated keyword prompt). For example, if Recall@5 for the GDP Growth rate query improves from 0 to 1 and the overlap score increases, you can attribute gains to better keyword alignment.

Tracking these metrics ensures that when a user asks "How fast is GDP growing?", the system reliably retrieves **GDP Growth rate** (and related indicators) and produces grounded answers.
