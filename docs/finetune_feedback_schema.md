# Fine-Tuning Feedback Capture (Postgres, Single Table)

Target: one row per question/run; keep common fields explicit, approach specifics in JSON.

Table: `finetune_events`
- `id` UUID PRIMARY KEY DEFAULT `gen_random_uuid()`
- `created_at` TIMESTAMPTZ DEFAULT `now()`
- `question_text` TEXT NOT NULL
- `approach` TEXT NOT NULL  _(text enum in code; e.g., `multi_named_vectors`, `hyde`)_
- `final_selected_indicators` TEXT[]  _(normalized names; store as comma-separated input and cast to array)_
- `correct_retrieved_indicators` TEXT[]  _(comma-separated input → array)_
- `incorrect_retrieved_indicators` TEXT[]  _(comma-separated input → array)_
- `correct_suggested_indicators` TEXT[]  _(comma-separated input → array)_
- `approach_data` JSONB NULL  _(shape depends on approach; see examples)_
- `final_latency` INT ( in seconds )

Suggested `approach_data` shapes
- `multi_named_vectors`: {"weights":{"application":0.5,"context":0.3,"definition":0.2},"embedding_model":"...","reranker_model":"...","reranker_prompt":"...","intermediate_topk":["ind_a","ind_b"]}
- `hyde`: {"prompt":"...","generated_definition":"...","intermediate_topk":["ind_x","ind_y"]}