"""
LIVE DEMO — runs the REAL pipeline (retriever.py, agent.py's helper functions,
grounding validation) with a simple RULE-BASED brain standing in for the LLM.

This is NOT a fake demo: retrieval scores, compare-target detection, and
history-reconstruction are all the real functions from agent.py / retriever.py.
Only the "what should I say / decide" step is replaced by simple if/else rules
instead of Claude actually reasoning about it -- because no API key is
available in this sandbox. Swap DemoBrain for the real LLMClient and nothing
else changes.
"""
import re
import sys

sys.path.insert(0, ".")

import app.agent as agent_module
from app.agent import RecommenderAgent, MAX_CANDIDATES
from app.catalog import load_catalog
from app.retriever import CatalogRetriever

REFINE_WORDS = {"add", "also", "drop", "remove", "instead", "replace", "actually"}
CONFIRM_WORDS = {"confirm", "confirmed", "perfect", "good", "great", "works", "yes", "thanks", "sounds"}
OFFTOPIC_PATTERNS = [
    r"legally required", r"job posting", r"write.*(email|letter) to reject",
    r"salary negotiation", r"interview questions to ask",
]


class DemoBrain:
    """Rule-based stand-in for the LLM. Reuses the REAL agent.py helper
    functions so compare-detection and history-reconstruction are genuine."""

    def __init__(self, retriever: CatalogRetriever):
        self.retriever = retriever

    def respond(self, system_prompt, conversation_messages, max_tokens=1024):
        # last item in conversation_messages is the injected "candidate pool" context
        # (added by agent.handle_turn) -- strip it to get the real chat history.
        real_history = conversation_messages[:-1]
        candidate_block = conversation_messages[-1]["content"]
        last_user = real_history[-1]["content"]
        last_user_lower = last_user.lower()
        user_turn_count = sum(1 for m in real_history if m["role"] == "user")

        candidates = _parse_candidate_block(candidate_block)

        # --- Rule 0: off-topic / legal ---
        if any(re.search(p, last_user_lower) for p in OFFTOPIC_PATTERNS):
            return {
                "reply": "That's outside what I can advise on -- I help with SHL assessment "
                         "selection, not legal or general hiring-process questions. "
                         "Happy to keep going on the assessment side though.",
                "recommendations": [],
                "end_of_conversation": False,
            }

        # --- Rule 1: compare ---
        compare_targets = agent_module._detect_compare_targets(last_user, self.retriever)
        if len(compare_targets) == 2:
            a, b = compare_targets
            reply = (
                f"**{a['name']}**: {a['description'][:200]}...\n\n"
                f"**{b['name']}**: {b['description'][:200]}...\n\n"
                f"Grounded in each item's actual catalog description, not general knowledge."
            )
            previously_named = agent_module._extract_previously_named_items(
                real_history, self.retriever.catalog
            )
            return {
                "reply": reply,
                "recommendations": [_to_rec(i) for i in previously_named],
                "end_of_conversation": False,
            }

        # --- Rule 2: refine (only if something was already committed) ---
        previously_named = agent_module._extract_previously_named_items(
            real_history, self.retriever.catalog
        )
        if previously_named and any(w in last_user_lower for w in REFINE_WORDS):
            updated = list(previously_named)
            # crude "drop X" handling
            for item in list(updated):
                if any(
                    tok in last_user_lower
                    for tok in item["name"].lower().split()
                    if len(tok) > 3
                ) and ("drop" in last_user_lower or "remove" in last_user_lower):
                    updated.remove(item)
            # crude "add X" handling: pull in top new candidate not already present
            existing_urls = {i["url"] for i in updated}
            for c in candidates:
                if c["url"] not in existing_urls and len(updated) < 8:
                    updated.append(c)
                    existing_urls.add(c["url"])
                    break
            reply = "Updated shortlist: " + ", ".join(i["name"] for i in updated)
            end = any(w in last_user_lower for w in CONFIRM_WORDS)
            return {
                "reply": reply,
                "recommendations": [_to_rec(i) for i in updated],
                "end_of_conversation": end,
            }

        # --- Rule 3: clarify on the very first turn ---
        if user_turn_count <= 1:
            return {
                "reply": "Happy to help narrow that down -- what's the seniority level, "
                         "and are there specific must-have skills for this role?",
                "recommendations": [],
                "end_of_conversation": False,
            }

        # --- Rule 4: pure confirmation with an existing shortlist -> close out, don't regenerate ---
        if previously_named and any(w in last_user_lower for w in CONFIRM_WORDS) and not any(
            w in last_user_lower for w in REFINE_WORDS
        ):
            reply = "Confirmed. Final shortlist: " + ", ".join(i["name"] for i in previously_named)
            return {
                "reply": reply,
                "recommendations": [_to_rec(i) for i in previously_named],
                "end_of_conversation": True,
            }

        # --- Rule 5: recommend (fresh) ---
        top = candidates[:4]
        reply = "Based on what you've shared, here's a shortlist: " + ", ".join(i["name"] for i in top)
        end = any(w in last_user_lower for w in CONFIRM_WORDS)
        return {
            "reply": reply,
            "recommendations": [_to_rec(i) for i in top],
            "end_of_conversation": end,
        }


def _to_rec(item):
    return {"name": item["name"], "url": item["url"], "test_type": item["test_type"]}


def _parse_candidate_block(block):
    items, current = [], {}
    for line in block.splitlines():
        line = line.strip()
        if line.startswith("- name:"):
            if current:
                items.append(current)
            current = {"name": line.split("- name:")[1].strip()}
        elif line.startswith("url:"):
            current["url"] = line.split("url:")[1].strip()
        elif line.startswith("test_type:"):
            current["test_type"] = line.split("test_type:")[1].strip()
        elif line.startswith("description:"):
            current["description"] = line.split("description:")[1].strip()
    if current:
        items.append(current)
    return items


def print_turn(n, user_msg, result):
    print(f"\n{'='*70}\nTURN {n}")
    print(f"USER: {user_msg}")
    print(f"{'-'*70}")
    print(f"AGENT REPLY: {result['reply']}")
    print(f"RECOMMENDATIONS ({len(result['recommendations'])} items):")
    for r in result["recommendations"]:
        print(f"   - {r['name']}  [{r['test_type']}]  {r['url']}")
    print(f"END_OF_CONVERSATION: {result['end_of_conversation']}")


def run_demo():
    catalog = load_catalog()
    retriever = CatalogRetriever(catalog, use_semantic=False)  # BM25-only in this sandbox
    brain = DemoBrain(retriever)
    agent = RecommenderAgent(catalog, retriever, brain)

    messages = []

    # Turn 1: vague-ish request -> should CLARIFY
    messages.append({"role": "user", "content": "We're hiring a senior Java developer."})
    result = agent.handle_turn(messages)
    print_turn(1, messages[-1]["content"], result)
    messages.append({"role": "assistant", "content": result["reply"]})

    # Turn 2: enough detail now -> should RECOMMEND
    messages.append({"role": "user", "content": "5 years experience, backend-focused, needs SQL and AWS knowledge."})
    result = agent.handle_turn(messages)
    print_turn(2, messages[-1]["content"], result)
    messages.append({"role": "assistant", "content": result["reply"]})

    # Turn 3: refine -> should ADD something, keep prior items
    messages.append({"role": "user", "content": "Actually also add a Docker test."})
    result = agent.handle_turn(messages)
    print_turn(3, messages[-1]["content"], result)
    messages.append({"role": "assistant", "content": result["reply"]})

    # Turn 4: compare -> should answer from catalog descriptions
    messages.append({"role": "user", "content": "What's the difference between Core Java (Advanced Level) (New) and SQL (New)?"})
    result = agent.handle_turn(messages)
    print_turn(4, messages[-1]["content"], result)
    messages.append({"role": "assistant", "content": result["reply"]})

    # Turn 5: off-topic -> should REFUSE
    messages.append({"role": "user", "content": "Are we legally required to test all candidates for this role?"})
    result = agent.handle_turn(messages)
    print_turn(5, messages[-1]["content"], result)
    messages.append({"role": "assistant", "content": result["reply"]})

    # Turn 6: confirm -> should end conversation
    messages.append({"role": "user", "content": "Understood. That shortlist works, thanks."})
    result = agent.handle_turn(messages)
    print_turn(6, messages[-1]["content"], result)


if __name__ == "__main__":
    run_demo()
