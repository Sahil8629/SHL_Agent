"""
Validates agent.py's orchestration (retrieval, history-reconstruction,
grounding validation) without calling a real LLM -- useful for local dev
without burning API credits, and to sanity check before wiring a real key.
"""
import sys

sys.path.insert(0, ".")

from app.agent import RecommenderAgent, _build_weighted_queries, _extract_previously_named_items
from app.catalog import load_catalog
from app.retriever import CatalogRetriever


class MockLLM:
    """Pretends to be the LLM: picks the top candidates and echoes their names
    into the reply text, so we can verify grounding + history-reconstruction."""

    def respond(self, system_prompt, messages):
        # find the candidate pool block we injected
        candidate_block = messages[-1]["content"]
        names_urls = []
        for line in candidate_block.splitlines():
            line = line.strip()
            if line.startswith("- name:"):
                names_urls.append({"name": line.split("- name:")[1].strip(), "url": None, "test_type": None})
            elif line.startswith("url:") and names_urls:
                names_urls[-1]["url"] = line.split("url:")[1].strip()
            elif line.startswith("test_type:") and names_urls:
                names_urls[-1]["test_type"] = line.split("test_type:")[1].strip()

        top3 = names_urls[:3]
        reply = "Here are some assessments: " + ", ".join(t["name"] for t in top3)
        return {
            "reply": reply,
            "recommendations": top3,
            "end_of_conversation": False,
        }


def run():
    catalog = load_catalog()
    retriever = CatalogRetriever(catalog, use_semantic=False)  # BM25-only for this sandbox
    agent = RecommenderAgent(catalog, retriever, MockLLM())

    print("=== Turn 1: vague-ish query ===")
    messages = [{"role": "user", "content": "We're hiring a senior Java developer who works with stakeholders"}]
    result = agent.handle_turn(messages)
    print(result["reply"])
    print("recs:", [r["name"] for r in result["recommendations"]])
    assert all(r["url"] for r in result["recommendations"]), "grounding failed: missing url"

    print("\n=== Turn 2: refine (simulate prior assistant reply carrying names) ===")
    messages.append({"role": "assistant", "content": result["reply"]})
    messages.append({"role": "user", "content": "Actually also add AWS and Docker skills tests"})
    prev_named = _extract_previously_named_items(messages, catalog)
    print("Reconstructed previously-named items from history:", [i["name"] for i in prev_named])
    assert len(prev_named) > 0, "history reconstruction failed"

    print("\n=== Compare detection ===")
    from agent import _detect_compare_targets
    targets = _detect_compare_targets("what is the difference between OPQ32r and Global Skills Assessment", retriever)
    print("Compare targets found:", [t["name"] for t in targets])
    assert len(targets) == 2, f"expected 2 compare targets, got {len(targets)}"

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    run()
