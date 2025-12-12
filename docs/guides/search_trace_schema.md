# Search trace schema

This schema captures everything we can derive from a single question/indicator round trip without introducing sessions or users, which keeps it compatible with the current `structure.md` layout (place new reference material under `docs/guides/`). The focus is on a row per query plus the ordered signal lists that you want for fine-tuning later.

## Goals

1. Record what question was asked and when, along with which reranker (if any) participated.
2. Save the reranker’s ordered picks/rejections for all 15 indicators so we can replay the decision path.
3. Capture the checkbox signals the UI is collecting (checked vs unchecked indicators).
4. Keep payloads query-able without denormalizing indicator metadata, while keeping order intact for later analysis.

## Table: `search_trace`

| column | type | intent | example |
| --- | --- | --- | --- |
| `trace_id` | `UUID PRIMARY KEY DEFAULT gen_random_uuid()` | Unique, auto-generated row id. | `d2f5c9fa-4ade-4726-9648-e8bb7a97c91c` |
| `query_text` | `TEXT NOT NULL` | The user’s original question. | `What are the key indicators that signal a fiscal policy tightening in Europe?` |
| `embedding_model` | `TEXT` | Embedding model used for this search so we can tie back vector signatures to a model release. | `text-embedding-3-small` |
| `created_at` | `TIMESTAMPTZ NOT NULL DEFAULT now()` | When the search/hit occurred. | `(auto-generated via DEFAULT now())` |
| `reranker_used` | `BOOLEAN NOT NULL DEFAULT FALSE` | Flag for reranker execution. | `TRUE` |
| `reranker_model` | `TEXT` | Which reranker model/routing was used (nullable when reranker was skipped). | `gpt-5-nano` |
| `reranker_prompt` | `TEXT` | Prompt that was passed to the reranker (helps us understand context during fine-tuning). | _truncated prompt string_ |
| `quadrant_collection` | `TEXT` | The Qdrant collection/quadrant name that stored the embeddings for this query’s retrieval. | `europe-macro-quadrant-v1` |
| `reranker_selected` | `JSONB NOT NULL DEFAULT '[]'` | Ordered array of objects for the indicators the reranker picked. Each item is `{ indicator: <normalized_name>, rank: <int>, score?: <float> }`. PostgreSQL keeps array order, and JSONB lets us contain the entire 15-indicator universe while still indexing with the `@>` / GIN pattern described in the ScaleGrid guide on using JSONB effectively. | see **Reranker-selected ordering** table below |
| `reranker_rejected` | `JSONB NOT NULL DEFAULT '[]'` | Same shape as above, but for the indicators the reranker deliberately placed after the selected set (order matters because you want to know which indicator was dropped first). | see **Reranker-rejected ordering** table below |
| `user_checked` | `JSONB NOT NULL DEFAULT '[]'` | Ordered (or timestamped) list of indicators the user explicitly affirmed in the checkbox UI. Each element can be `{ indicator: ..., clicked_at: <ts?> }`. | see **User checkbox accepted** table below |
| `user_unchecked` | `JSONB NOT NULL DEFAULT '[]'` | Ordered list of checkbox indicators the user left unchecked. | see **User checkbox skipped** table below |
| `indicator_catalog` | `JSONB NOT NULL` | Static snapshot of the 15 indicators you used for that query (names, normalized forms, sheet provenance). Keeps the record self-contained without needing another join, and lets you audit expansions later. | see **Indicator catalog snapshot** table below |
| `user_suggested` | `JSONB` | Optional list of normalized indicator names the user thought should be considered but weren’t in the 15-item set; helpful for feature engineering/fine-tuning. | see **User suggested indicators** table below |
| `reranker_notes` | `JSONB` | Any debugging metadata (LLM response text, reasoning tokens, etc.). `jsonb` also supports the `IS JSON` checks documented in PostgreSQL’s JSON reference so this data stays validated when stored. | see **Reranker notes** table below |

### Indexing & querying notes

- GIN indexes on `reranker_selected` / `user_checked` allow fast lookups by indicator name (`CREATE INDEX ON search_trace USING gin (reranker_selected jsonb_path_ops);`). The PostgreSQL docs for JSON functions confirm JSONB stays ordered when you store arrays, so storing ordered arrays aligns with the `jsonb_object/jsonb_object` patterns from the official doc.
- You can add a partial index on `reranker_used` if you expect only some queries to run through the reranker.
- For faster filtering by indicator names without scanning arrays, you can also maintain a materialized view or derived table that explodes each JSON array into rows (e.g., using `jsonb_array_elements`), but this schema gives you an approachable starting point for fine-tuning data dumps.

Once you confirm this direction, the next steps are to add migrations (e.g., in `config/` if you keep SQL there) and expose a Python model/serializer under `src/app/` (likely `models.py` or a new `schema/` module) that writes to this table after the pipeline completes.

## Example entry

```json
{
  "trace_id": "d2f5c9fa-4ade-4726-9648-e8bb7a97c91c",
  "query_text": "What are the key indicators that signal a fiscal policy tightening in Europe?",
  "embedding_model": "text-embedding-3-small",
  "created_at": "2024-08-23T14:02:12.713Z",
  "quadrant_collection": "europe-macro-quadrant-v1",
  "reranker_used": true,
  "reranker_model": "gpt-5-nano",
  "reranker_prompt": "You are a reranker ... choose the indicators most relevant to fiscal tightening in Europe.",
  "reranker_selected": [
    { "indicator": "interest_rate_real", "rank": 1, "score": 0.92 },
    { "indicator": "inflation_consumer_price_index", "rank": 2, "score": 0.87 },
    { "indicator": "federal_reserve_policy_bias", "rank": 3, "score": 0.82 }
  ],
  "reranker_rejected": [
    { "indicator": "gdp_growth_real", "rank": 4, "score": 0.67 },
    { "indicator": "employment_rate", "rank": 5, "score": 0.61 }
  ],
  "user_checked": [
    { "indicator": "interest_rate_real", "clicked_at": "2024-08-23T14:05:00Z" },
    { "indicator": "inflation_consumer_price_index", "clicked_at": "2024-08-23T14:05:03Z" }
  ],
  "user_unchecked": [
    { "indicator": "gdp_growth_real" }
  ],
  "indicator_catalog": {
    "source_sheet": "country_indicators",
    "indicators": [
      { "name": "Interest rate, Real (2015 USD)", "normalized": "interest_rate_real" },
      { "name": "Consumer Price Index (CPI)", "normalized": "inflation_consumer_price_index" }
    ]
  },
  "user_suggested": [
    "bond_yield_spread_10y_2y",
    "central_bank_balance_sheet_change"
  ],
  "reranker_notes": {
    "llm_response": "I selected real rates and inflation because they directly reflect tightening ...",
    "tokens": 198
  }
}
```
