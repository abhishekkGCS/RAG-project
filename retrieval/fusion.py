# fusion.py
from typing import List, Dict, Any, Optional
from collections import defaultdict

from retrieval.vector_store import ClinicalVectorStore
from retrieval.bm25 import ClinicalBM25Search


class HybridRetriever:
    def __init__(self, k_constant: int = 60):
        """
        Combines Dense Vector Search and BM25 Lexical Search via RRF.
        k_constant: smoothing factor (standard = 60)
        """
        print("[*] Initializing Hybrid Retriever (Dense + BM25)...")
        self.vector_store = ClinicalVectorStore()
        self.bm25_searcher = ClinicalBM25Search()
        self.k_constant = k_constant
        print("[✓] Hybrid Retriever ready!")

    def search_hybrid(
        self,
        query: str,
        top_k: int = 5,
        dense_candidates: int = 15,
        bm25_candidates: int = 15,
        drug_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes both retrievers and fuses results using Reciprocal Rank Fusion (RRF).
        """
        # 1. Fetch top candidates from Dense Vector Search
        dense_results = self.vector_store.search_vector(
            query=query,
            top_k=dense_candidates,
            drug_filter=drug_filter
        )

        # 2. Fetch top candidates from BM25 Lexical Search
        bm25_results = self.bm25_searcher.search_bm25(
            query=query,
            top_k=bm25_candidates
        )

        # 3. Compute RRF Scores
        # We identify chunks uniquely by their text hash or (source, page, snippet)
        rrf_scores = defaultdict(float)
        chunk_lookup = {}

        # Add Dense ranks
        for rank, item in enumerate(dense_results, start=1):
            key = (item["source"], item["page"], item["text"][:100])
            rrf_scores[key] += 1.0 / (self.k_constant + rank)
            chunk_lookup[key] = item

        # Add BM25 ranks
        for rank, item in enumerate(bm25_results, start=1):
            key = (item["source"], item["page"], item["text"][:100])
            rrf_scores[key] += 1.0 / (self.k_constant + rank)
            if key not in chunk_lookup:
                chunk_lookup[key] = item

        # 4. Sort all items by fused RRF score descending
        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)

        final_results = []
        for rank, key in enumerate(sorted_keys[:top_k], start=1):
            chunk_data = chunk_lookup[key].copy()
            chunk_data["rrf_rank"] = rank
            chunk_data["rrf_score"] = round(rrf_scores[key], 6)
            final_results.append(chunk_data)

        return final_results


if __name__ == "__main__":
    hybrid = HybridRetriever()
    test_query = "What is the INR target range for mechanical heart valves with Warfarin?"
    print(f"\n" + "=" * 55)
    print(f"HYBRID QUERY: '{test_query}'")
    print("=" * 55)
    fused_results = hybrid.search_hybrid(test_query, top_k=3)
    for r in fused_results:
        print(f"\n[Hybrid Rank #{r['rrf_rank']}] (RRF Score: {r['rrf_score']})")
        print(f"Drug:    {r['drug']} | Section: {r['section']} | Page: {r['page']}")
        print(f"Source:  {r['source']}")
        print(f"Content: {r['text'][:200]}...")