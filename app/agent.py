"""
Single-agent orchestration. One LLM call per turn (matches the 30s budget and
the 8-turn cap comfortably). Everything else here is deterministic Python:
retrieval, history parsing, and output validation.
"""
from __future__ import annotations

import re
from typing import Any

from app.prompts import SYSTEM_PROMPT
from app.retriever import CatalogRetriever, SearchFilters

MAX_CANDIDATES = 30

# Patterns that suggest the user is asking to compare two specific named items.
_COMPARE_PATTERNS = [
    r"difference between (.+?) and (.+?)[\?\.]?$",
    r"compare (.+?) (?:and|vs\.?|versus) (.+?)[\?\.]?$",
    r"(.+?) vs\.? (.+?)[\?\.]?$",
]


def _last_user_message(messages: list[dict[str, str]]) -> str:
    for m in reversed(messages):
        if m["role"] == "user":
            return m["content"]
    return ""


def _build_weighted_queries(messages: list[dict[str, str]]) -> list[tuple[str, float]]:
    """
    Full conversation (all user turns) gets base weight; the latest user turn
    is ALSO searched separately at extra weight, so a newly-added constraint
    is biased toward without letting it drown out earlier context via naive
    text repetition (see retriever.multi_query_search docstring for why that
    approach was wrong).
    """
    user_turns = [m["content"] for m in messages if m["role"] == "user"]
    if not user_turns:
        return []
    full_history = " ".join(user_turns)
    latest = user_turns[-1]
    return [(full_history, 0.6), (latest, 0.4)]


def _detect_compare_targets(text: str, retriever: CatalogRetriever) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    lowered = text.lower()
    for pattern in _COMPARE_PATTERNS:
        m = re.search(pattern, lowered)
        if m:
            for raw_name in m.groups():
                item = retriever.get_by_fuzzy_name(raw_name.strip())
                if item and item not in found:
                    found.append(item)
            break
    return found


def _extract_previously_named_items(
    messages: list[dict[str, str]], catalog: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    Reconstruct which catalog items are already 'on the table' by scanning
    prior assistant replies for catalog item names. This is what makes REFINE
    possible on a stateless API -- the recommendations array from a previous
    turn is not echoed back to us, only the reply text is.
    """
    assistant_text = " ".join(m["content"] for m in messages if m["role"] == "assistant")
    if not assistant_text:
        return []
    found = []
    lowered = assistant_text.lower()
    for item in catalog:
        name_lower = item["name"].strip().lower()
        if len(name_lower) >= 3 and name_lower in lowered:
            found.append(item)
    return found


def _format_candidate_pool(items: list[dict[str, Any]]) -> str:
    lines = []
    for item in items:
        langs = ", ".join(item["languages"][:5]) if item["languages"] else "—"
        levels = ", ".join(item["job_levels"][:6]) if item["job_levels"] else "—"
        desc = item["description"][:400]
        lines.append(
            f"- name: {item['name']}\n"
            f"  url: {item['url']}\n"
            f"  test_type: {item['test_type']}\n"
            f"  job_levels: {levels}\n"
            f"  languages: {langs}\n"
            f"  duration: {item['duration'] or '—'}\n"
            f"  description: {desc}"
        )
    return "\n".join(lines) if lines else "(no candidates retrieved)"


class RecommenderAgent:
    def __init__(self, catalog: list[dict[str, Any]], retriever: CatalogRetriever, llm_client):
        self.catalog = catalog
        self.retriever = retriever
        self.llm = llm_client
        self._valid_urls = {item["url"] for item in catalog}

    def handle_turn(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        last_user = _last_user_message(messages)

        # 1. Retrieval: main query + any explicit compare targets + anything
        #    already committed earlier in the conversation (so it stays visible
        #    to the model for refine/compare turns without a second retrieval).
        query_weights = _build_weighted_queries(messages)
        main_candidates = self.retriever.multi_query_search(query_weights, top_k=MAX_CANDIDATES)

        compare_targets = _detect_compare_targets(last_user, self.retriever)
        previously_named = _extract_previously_named_items(messages, self.catalog)

        candidate_pool = _dedupe(main_candidates + compare_targets + previously_named)

        # 2. Build the turn's user-facing prompt content for the model.
        candidate_block = _format_candidate_pool(candidate_pool)
        history_for_llm = [{"role": m["role"], "content": m["content"]} for m in messages]
        history_for_llm.append(
            {
                "role": "user",
                "content": (
                    "[SYSTEM CONTEXT — not shown to the end user]\n"
                    "CANDIDATE POOL for this turn (choose recommendations only from here):\n"
                    f"{candidate_block}"
                ),
            }
        )

        # 3. One LLM call, forced structured output.
        raw = self.llm.respond(SYSTEM_PROMPT, history_for_llm)

        # 4. Validate & enforce grounding — never trust the model's URLs blindly.
        return self._validate(raw)

    def _validate(self, raw: dict[str, Any]) -> dict[str, Any]:
        reply = str(raw.get("reply", "")).strip() or "Could you tell me a bit more about the role?"
        end_of_conversation = bool(raw.get("end_of_conversation", False))

        recs = raw.get("recommendations") or []
        clean_recs = []
        for r in recs:
            url = str(r.get("url", "")).strip()
            if url in self._valid_urls:
                clean_recs.append(
                    {
                        "name": str(r.get("name", "")).strip(),
                        "url": url,
                        "test_type": str(r.get("test_type", "")).strip(),
                    }
                )
        clean_recs = clean_recs[:10]  # hard cap per spec

        return {
            "reply": reply,
            "recommendations": clean_recs,
            "end_of_conversation": end_of_conversation,
        }


def _dedupe(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for item in items:
        if item["url"] in seen:
            continue
        seen.add(item["url"])
        out.append(item)
    return out
