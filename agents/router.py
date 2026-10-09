import re
from enum import Enum
from typing import Dict, Any


class RetrievalRoute(str, Enum):
    GRAPH_ONLY = "GRAPH_ONLY"          # Pure relationship / interaction queries
    HYBRID_ONLY = "HYBRID_ONLY"        # Specific textual / dosage / mechanism questions
    FUSED_BOTH = "FUSED_BOTH"          # High-risk, multi-drug, or comprehensive queries


# Keywords that signal multi-drug relationships or ontology edges
RELATIONSHIP_PATTERNS = [
    r"\b(interact(s|ion|ions)? with|combine with|take with|together with)\b",
    r"\b(what drugs?|which medications?|what conditions?|what diseases?)\b",
    r"\b(contraindicated in|contraindications? in)\b",
    r"\b(treats?|causes?|leads? to)\b",
]

# Keywords that signal specific textual facts or deep dosage guidelines
TEXTUAL_PATTERNS = [
    r"\b(dose|dosage|how to take|administration|storage|how supplied)\b",
    r"\b(clinical trials?|mechanism of action|pharmacokinetics|inr target)\b",
]


class QueryRouter:
    def __init__(self):
        self.rel_regex = [re.compile(p, re.IGNORECASE) for p in RELATIONSHIP_PATTERNS]
        self.text_regex = [re.compile(p, re.IGNORECASE) for p in TEXTUAL_PATTERNS]

    def route_query(self, query: str, risk_level: str = "LOW") -> Dict[str, Any]:
        """
        Determines the optimal retrieval strategy for a clinical question.
        """
        # Rule 1: High-risk queries always get full dual-engine verification
        if risk_level == "HIGH":
            return {
                "route": RetrievalRoute.FUSED_BOTH,
                "reason": "HIGH clinical risk tier requires dual verification (Hybrid + Graph)."
            }

        # Rule 2: Check for explicit relationship/ontology patterns
        is_relationship = any(p.search(query) for p in self.rel_regex)
        is_textual = any(p.search(query) for p in self.text_regex)

        if is_relationship and not is_textual:
            return {
                "route": RetrievalRoute.GRAPH_ONLY,
                "reason": "Query is a pure relationship/interaction question best served by Knowledge Graph."
            }

        if is_textual and not is_relationship:
            return {
                "route": RetrievalRoute.HYBRID_ONLY,
                "reason": "Query requests specific dosage or clinical text details best served by Hybrid Search."
            }

        # Rule 3: Complex or blended questions get both
        return {
            "route": RetrievalRoute.FUSED_BOTH,
            "reason": "Query contains blended or general intent; using Fused Retrieval for maximum context."
        }


if __name__ == "__main__":
    router = QueryRouter()

    test_scenarios = [
        ("What drugs interact with Metformin?", "LOW"),
        ("What is the starting dosage and administration for Warfarin?", "LOW"),
        ("Can a patient take Warfarin and Aspirin together?", "HIGH"),
        ("Tell me about Aspirin.", "LOW")
    ]

    print("=" * 65)
    print("TESTING RETRIEVAL QUERY ROUTER")
    print("=" * 65)

    for query, risk in test_scenarios:
        decision = router.route_query(query, risk_level=risk)
        print(f"\nQuery:      '{query}'")
        print(f"Risk Tier:  {risk}")
        print(f"Route:      👉 {decision['route'].value}")
        print(f"Rationale:  {decision['reason']}")