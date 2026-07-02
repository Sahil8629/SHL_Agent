"""
Loads the SHL product catalog JSON and normalizes it into a consistent,
code-friendly shape. This is the single source of truth every other module
(retriever, validator, agent) reads from -- nothing else touches the raw file.
"""
import json
import os
from typing import Any

# Full category name -> single-letter code, matching the codes used in
# SHL's own catalog UI and in the sample conversation transcripts (K, P, A, ...).
KEY_TO_CODE = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}

CATALOG_PATH = os.path.join(os.path.dirname(__file__), "catalog_raw.json")


def _codes_for(keys: list[str]) -> str:
    codes = [KEY_TO_CODE.get(k, "?") for k in keys]
    return ",".join(codes) if codes else "-"


def load_catalog(path: str = CATALOG_PATH) -> list[dict[str, Any]]:
    """
    Load the raw scraped catalog and return a list of normalized records.
    Each record carries both the original fields (for grounding / URLs) and a
    few derived fields used purely for search (searchable_text, test_type_codes).
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f, strict=False)

    catalog = []
    for item in raw:
        name = (item.get("name") or "").strip()
        description = (item.get("description") or "").strip()
        keys = item.get("keys") or []
        job_levels = item.get("job_levels") or []
        languages = item.get("languages") or []
        duration = (item.get("duration") or "").strip()
        url = (item.get("link") or "").strip()

        if not name or not url:
            # Anything without a name or URL can never be safely recommended
            # (we cannot ground it). Skip rather than risk a broken/hallucinated link.
            continue

        test_type_codes = _codes_for(keys)

        # A single blob used for lexical (BM25) and semantic (embedding) search.
        # Repeating the name gives it a bit more weight in BM25 term frequency.
        searchable_text = " ".join(
            [
                name,
                name,
                description,
                " ".join(keys),
                " ".join(job_levels),
            ]
        )

        catalog.append(
            {
                "entity_id": item.get("entity_id"),
                "name": name,
                "url": url,
                "description": description,
                "keys": keys,  # full category names, e.g. ["Knowledge & Skills"]
                "test_type": test_type_codes,  # e.g. "K" or "K,S"
                "job_levels": job_levels,
                "languages": languages,
                "duration": duration,
                "searchable_text": searchable_text,
            }
        )

    return catalog


# Convenience index by exact lowercase name, used by the "compare" flow to do
# a fast, precise lookup instead of a fuzzy semantic search.
def build_name_index(catalog: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {item["name"].strip().lower(): item for item in catalog}


def build_url_index(catalog: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {item["url"].strip(): item for item in catalog}


if __name__ == "__main__":
    cat = load_catalog()
    print(f"Loaded {len(cat)} catalog items")
    print(cat[0])
