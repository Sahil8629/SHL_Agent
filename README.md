# SHL Assessment Recommender

Stateless conversational agent over the SHL Individual Test Solutions catalog.

## Run locally

```bash
pip install -r requirements.txt --break-system-packages
export ANTHROPIC_API_KEY=sk-ant-...
cd app
uvicorn main:app --reload --port 8000
```

Test:
```bash
curl localhost:8000/health
curl -X POST localhost:8000/chat -H "Content-Type: application/json" -d '{
  "messages": [{"role": "user", "content": "We are hiring a senior Java developer who works with stakeholders"}]
}'
```

## Architecture (see app/ for each piece)

- `catalog.py` — loads + normalizes the scraped catalog JSON (handles the raw-newline
  parse issue, maps full category names to single-letter test_type codes).
- `retriever.py` — hybrid search: BM25 (with domain stopword filtering) + optional
  semantic embeddings (sentence-transformers, degrades gracefully to BM25-only if
  the model can't be downloaded), plus hard metadata filters and fuzzy name lookup
  for the compare flow.
- `agent.py` — orchestration: builds the retrieval query from the whole conversation,
  detects explicit "compare X and Y" requests, reconstructs any previously-recommended
  items by scanning prior assistant replies (necessary because the stateless API never
  echoes back a `recommendations` array, only `content` text), assembles the candidate
  pool, makes ONE LLM call with forced structured output, and validates the result
  (every returned URL must exist in the catalog, array capped at 10 items).
- `llm_client.py` — thin wrapper using Anthropic tool-forcing for guaranteed-schema
  output. Swap providers by editing only this file.
- `prompts.py` — system prompt encoding scope limits, refusal rules, and the four
  conversational behaviors (clarify / recommend / refine / compare).
- `schemas.py` — exact pydantic models matching the required API contract.
- `main.py` — FastAPI app, `/health` + `/chat`.
- `test_agent_mock.py` — orchestration test with a mock LLM (no API key needed) —
  verifies grounding, history-reconstruction, and compare-detection logic in isolation
  from LLM quality.

## Key design decisions worth knowing for the interview

1. **One LLM call per turn**, not multi-agent — the 30s timeout and 8-turn cap make
   extra LLM hops expensive for no real benefit at this catalog size (377 items).
2. **No vector DB** — BM25 + optional local embeddings over an in-memory list is enough
   for 377 items and keeps every ranking decision inspectable/explainable.
3. **Grounding is enforced in code, not just prompted** — `_validate()` strips any
   recommendation whose URL isn't in the catalog, so hallucination can't reach the
   response even if the model tries.
4. **History-reconstruction problem**: the `/chat` request schema only sends back
   `{role, content}` — never the structured `recommendations` array. So refine/compare
   on turn N+1 only works if turn N's `reply` text actually named the items. This is
   handled two ways: (a) the system prompt explicitly instructs the model to name every
   recommended item in `reply`, and (b) `_extract_previously_named_items()` scans prior
   assistant text for catalog names as a safety net and feeds them back into the
   candidate pool.
5. **Stopword filtering matters more than it looks** — raw BM25 let generic recruiting
   words ("hiring", "who", "with") outrank the actual skill keyword ("java") on some
   queries; a small domain stopword list fixed this materially (see git history / dev
   notes for the before/after).
