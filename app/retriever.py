"""
Hybrid retrieval over the SHL catalog.

Design choice: with ~377 items we don't need a vector DB. We keep everything
in memory and combine:
  1. BM25 lexical scoring        -> catches exact skill/tool keywords ("AWS", "SQL")
  2. Optional semantic embeddings -> catches paraphrases ("works well with stakeholders")
  3. Hard metadata filters        -> job_level / test_type / language, applied only
                                     when the caller is confident about that constraint

Everything here is deterministic and inspectable: for any result we can say
exactly which score(s) put it there. That matters both for grounding
(the agent may only recommend from what this module returns) and for being
able to explain/defend the design in an interview.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9\+\.#]+")

# Generic recruiting/English filler that appears in almost every query and in
# many catalog descriptions, drowning out the actual skill/role signal (e.g.
# "hiring" alone was outranking "java" for a "hiring a Java developer" query).
_STOPWORDS = {
    "a", "an", "the", "we", "are", "is", "our", "for", "of", "to", "and", "or",
    "who", "with", "works", "work", "working", "need", "needs", "looking",
    "hire", "hiring", "recruit", "recruiting", "screen", "screening",
    "candidate", "candidates", "role", "position", "job", "this", "that",
    "in", "on", "at", "as", "be", "will", "would", "should", "can", "have",
}


def _tokenize(text: str) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


@dataclass
class SearchFilters:
    job_level: Optional[str] = None          # must match one of catalog's job_levels values
    test_types: Optional[list[str]] = None   # e.g. ["P"] to force personality-only
    language: Optional[str] = None           # e.g. "Spanish"
    exclude_names: list[str] = field(default_factory=list)  # lowercase names to exclude


class CatalogRetriever:
    def __init__(self, catalog: list[dict[str, Any]], use_semantic: bool = True):
        self.catalog = catalog
        self._corpus_tokens = [_tokenize(item["searchable_text"]) for item in catalog]
        self.bm25 = BM25Okapi(self._corpus_tokens)

        self.use_semantic = use_semantic
        self._embedder = None
        self._doc_embeddings = None
        if use_semantic:
            self._init_semantic()

    # ---------- semantic layer (optional, degrades gracefully) ----------

    def _init_semantic(self) -> None:
        """
        Try to load a local sentence-transformers model. If it's not available
        (no internet to huggingface.co, package not installed, etc.) we silently
        fall back to BM25-only search rather than crashing the service --
        lexical-only search is still a fully working, defensible fallback.
        """
        try:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            texts = [item["searchable_text"] for item in self.catalog]
            self._doc_embeddings = self._embedder.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )
        except Exception as e:  # noqa: BLE001 - intentional broad fallback
            print(f"[retriever] semantic search disabled ({e}); using BM25-only.")
            self._embedder = None
            self._doc_embeddings = None

    def _semantic_scores(self, query: str) -> Optional[np.ndarray]:
        if self._embedder is None:
            return None
        q_emb = self._embedder.encode([query], normalize_embeddings=True)[0]
        # embeddings are already normalized -> dot product == cosine similarity
        return self._doc_embeddings @ q_emb

    # ---------- lexical layer ----------

    def _bm25_scores(self, query: str) -> np.ndarray:
        return np.array(self.bm25.get_scores(_tokenize(query)))

    # ---------- public API ----------

    def search(
        self,
        query: str,
        top_k: int = 25,
        filters: Optional[SearchFilters] = None,
        bm25_weight: float = 0.5,
    ) -> list[dict[str, Any]]:
        """
        Return up to top_k catalog items ranked by a blended score.
        `query` should be a compact representation of everything known so far
        (role, skills, constraints) -- not just the last raw user message.
        """
        return self.multi_query_search([(query, 1.0)], top_k, filters, bm25_weight)

    def multi_query_search(
        self,
        weighted_queries: list[tuple[str, float]],
        top_k: int = 25,
        filters: Optional[SearchFilters] = None,
        bm25_weight: float = 0.5,
    ) -> list[dict[str, Any]]:
        """
        Combine multiple queries with explicit weights (e.g. full conversation
        history at weight 0.6 + just the latest message at weight 0.4, to bias
        toward the newest constraint without letting it drown out earlier
        context). This is score-level fusion, NOT string concatenation --
        repeating words inside a single query string double/triple-counts them
        in BM25's per-term summation and distorts results disproportionately
        for whichever query happened to have more words. Combining separately
        computed, separately normalized score arrays avoids that entirely.
        """
        n = len(self.catalog)
        combined = np.zeros(n)

        for text, weight in weighted_queries:
            if not text.strip():
                continue
            bm25_norm = _safe_normalize(self._bm25_scores(text))
            semantic_scores = self._semantic_scores(text)
            if semantic_scores is not None:
                sem_norm = _safe_normalize(semantic_scores)
                query_score = bm25_weight * bm25_norm + (1 - bm25_weight) * sem_norm
            else:
                query_score = bm25_norm
            combined += weight * query_score

        order = np.argsort(combined)[::-1]

        results = []
        for idx in order:
            item = self.catalog[idx]
            if not _passes_filters(item, filters):
                continue
            results.append({**item, "_score": float(combined[idx])})
            if len(results) >= top_k:
                break
        return results

    def get_by_exact_name(self, name: str) -> Optional[dict[str, Any]]:
        target = name.strip().lower()
        for item in self.catalog:
            if item["name"].strip().lower() == target:
                return item
        return None

    def get_by_fuzzy_name(self, name: str, threshold: float = 0.6) -> Optional[dict[str, Any]]:
        """Fallback for the 'compare' flow when the user's phrasing doesn't
        exactly match the catalog name (e.g. "OPQ32r" instead of the full
        "Occupational Personality Questionnaire OPQ32r"). Short user phrases are
        almost always a *subset* of the official name, so we score on how much
        of the query is covered by the candidate name (recall-style), and treat
        a direct substring hit as an instant match -- Jaccard alone punishes
        short abbreviations too harshly against long official names."""
        exact = self.get_by_exact_name(name)
        if exact:
            return exact

        query_norm = name.strip().lower()
        query_tokens = set(_tokenize(name))
        if not query_tokens:
            return None

        best_item, best_score = None, 0.0
        for item in self.catalog:
            item_name_lower = item["name"].strip().lower()

            # Direct substring in either direction is a strong, unambiguous signal
            # (e.g. "OPQ32r" is literally inside "Occupational Personality Questionnaire OPQ32r").
            if query_norm in item_name_lower or item_name_lower in query_norm:
                return item

            item_tokens = set(_tokenize(item["name"]))
            if not item_tokens:
                continue
            # Recall of the query against this candidate's tokens -- how much of
            # what the user typed is actually present in this catalog name.
            coverage = len(query_tokens & item_tokens) / len(query_tokens)
            if coverage > best_score:
                best_score, best_item = coverage, item
        return best_item if best_score >= threshold else None


def _safe_normalize(scores: np.ndarray) -> np.ndarray:
    max_val = scores.max() if scores.size else 0.0
    if max_val <= 0:
        return np.zeros_like(scores)
    return scores / max_val


def _passes_filters(item: dict[str, Any], filters: Optional[SearchFilters]) -> bool:
    if filters is None:
        return True
    if item["name"].strip().lower() in filters.exclude_names:
        return False
    if filters.job_level and filters.job_level not in item["job_levels"]:
        return False
    if filters.test_types:
        item_codes = set(item["test_type"].split(","))
        if not item_codes.intersection(filters.test_types):
            return False
    if filters.language:
        # languages list can be empty for some report-only products; don't
        # over-filter those out just because language metadata is missing.
        if item["languages"] and filters.language not in item["languages"]:
            return False
    return True


if __name__ == "__main__":
    from catalog import load_catalog

    cat = load_catalog()
    retriever = CatalogRetriever(cat, use_semantic=True)
    for r in retriever.search("senior Java developer works with stakeholders", top_k=5):
        print(f"{r['_score']:.3f}  {r['name']}  [{r['test_type']}]")
