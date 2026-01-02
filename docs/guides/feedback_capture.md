# Feedback capture schema

This describes a minimal, single-table schema for capturing per-indicator feedback after each query.

## Table: indicator_feedback

| column | type | intent |
| --- | --- | --- |
| `feedback_id` | `UUID PRIMARY KEY DEFAULT gen_random_uuid()` | Unique row id. |
| `query_text` | `TEXT NOT NULL` | The user question. |
| `created_at` | `TIMESTAMPTZ NOT NULL DEFAULT now()` | Capture timestamp. |
| `reranker_used` | `BOOLEAN NOT NULL DEFAULT FALSE` | Whether the LLM reranker was enabled. |
| `retrieved_indicators` | `JSONB NOT NULL DEFAULT '[]'` | Ordered list of the 15 retrieved indicators. If reranked, include `llm_rank` in each item. |
| `user_feedback` | `JSONB NOT NULL DEFAULT '[]'` | Ordered list of labels per indicator (`Directly related`, `Indirectly related`, `Not related`). |

## JSON structure

`retrieved_indicators` example:

```json
[
  {
    "id": "point_123",
    "indicator_name": "Consumer Price Index (CPI)",
    "normalized_indicator_name": "inflation_consumer_price_index",
    "score": 0.9123,
    "selected_by_llm": true,
    "llm_rank": 1
  }
]
```

`user_feedback` example:

```json
[
  {
    "id": "point_123",
    "normalized_indicator_name": "inflation_consumer_price_index",
    "indicator_name": "Consumer Price Index (CPI)",
    "label": "Directly related"
  }
]
```

## Suggested DDL

```sql
CREATE TABLE indicator_feedback (
  feedback_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  query_text TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  reranker_used BOOLEAN NOT NULL DEFAULT FALSE,
  retrieved_indicators JSONB NOT NULL DEFAULT '[]',
  user_feedback JSONB NOT NULL DEFAULT '[]'
);
```
