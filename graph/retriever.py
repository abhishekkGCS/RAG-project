import os
import json
from typing import List, Dict, Any, Optional
import networkx as nx

GRAPH_PATH = os.path.join("data", "processed", "clinical_graph.json")


class ClinicalGraphRetriever:
    def __init__(self, graph_path: str = GRAPH_PATH):
        """
        Loads the pre-built NetworkX directed graph from disk.
        """
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"Knowledge graph not found at {graph_path}. Run extractor first!")

        with open(graph_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.graph: nx.DiGraph = nx.node_link_graph(data)
        print(f"[✓] Clinical Graph loaded: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

    def find_mentioned_entities(self, query: str) -> List[str]:
        """
        Finds nodes in the graph that match entities mentioned in the query.
        """
        query_lower = query.lower()
        matched = []
        for node in self.graph.nodes():
            if node.lower() in query_lower:
                matched.append(node)
        return matched

    def get_entity_subgraph(self, entity_name: str, max_depth: int = 1) -> List[Dict[str, str]]:
        """
        Retrieves all outgoing and incoming relationships for a clinical entity.
        """
        triples = []
        if entity_name not in self.graph:
            return triples

        # Outgoing edges: (Entity) -> [RELATION] -> (Target)
        for _, target, data in self.graph.out_edges(entity_name, data=True):
            triples.append({
                "source": entity_name,
                "relation": data.get("relation", "CONNECTED_TO"),
                "target": target,
                "direction": "outgoing"
            })

        # Incoming edges: (Source) -> [RELATION] -> (Entity)
        for source, _, data in self.graph.in_edges(entity_name, data=True):
            triples.append({
                "source": source,
                "relation": data.get("relation", "CONNECTED_TO"),
                "target": entity_name,
                "direction": "incoming"
            })

        return triples

    def search_graph(self, query: str) -> Dict[str, Any]:
        """
        Extracts entities from the query, traverses the graph, and returns structured triples.
        """
        entities = self.find_mentioned_entities(query)
        all_triples = []

        for ent in entities:
            subgraph_triples = self.get_entity_subgraph(ent)
            all_triples.extend(subgraph_triples)

        # Format as human-readable facts for LLM context
        facts = []
        for t in all_triples:
            facts.append(f"({t['source']}) --[{t['relation']}]--> ({t['target']})")

        return {
            "query": query,
            "matched_entities": entities,
            "triples": all_triples,
            "formatted_facts": facts
        }


if __name__ == "__main__":
    retriever = ClinicalGraphRetriever()

    test_queries = [
        "What conditions does Metformin treat?",
        "What are the risks or adverse effects of Warfarin?"
    ]

    for q in test_queries:
        print("\n" + "=" * 50)
        print(f"QUERY: '{q}'")
        res = retriever.search_graph(q)
        print(f"Matched Entities: {res['matched_entities']}")
        print("Graph Triples Found:")
        if res["formatted_facts"]:
            for fact in res["formatted_facts"]:
                print(f"  • {fact}")
        else:
            print("  (No direct graph edges found)")