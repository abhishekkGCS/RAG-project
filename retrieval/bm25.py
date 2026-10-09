import os
import json
import re
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi


def tokenize_clinical_text(text: str) -> List[str]:
    """
    Tokenizes clinical text: splits on whitespace and punctuation while
    preserving alphanumeric tokens and hyphenated terms.
    """
    # Lowercase and split into alphanumeric tokens
    tokens = re.findall(r'\b[a-zA-Z0-9_\-\.]+\b', text.lower())
    return tokens


class ClinicalBM25Search:
    def __init__(self, chunks_file: str = os.path.join("data", "processed", "chunks.json")):
        """
        Builds an in-memory BM25 index over all processed chunks.
        """
        if not os.path.exists(chunks_file):
            raise FileNotFoundError(f"Chunks file not found: {chunks_file}")

        with open(chunks_file, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        print(f"[*] Building BM25 index over {len(self.chunks)} chunks...")
        
        # Tokenize all chunks for BM25
        self.corpus_tokens = [
            tokenize_clinical_text(chunk["text"]) for chunk in self.chunks
        ]
        
        self.bm25 = BM25Okapi(self.corpus_tokens)
        print("[✓] BM25 index built successfully!")

    def search_bm25(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs lexical keyword search using BM25.
        """
        query_tokens = tokenize_clinical_text(query)
        if not query_tokens:
            return []

        # Get raw BM25 scores across all chunks
        scores = self.bm25.get_scores(query_tokens)

        # Pair scores with their original chunk indices and sort descending
        scored_pairs = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        results = []
        for rank, (chunk_idx, score) in enumerate(scored_pairs, start=1):
            chunk = self.chunks[chunk_idx]
            results.append({
                "rank": rank,
                "score": round(float(score), 4),
                "text": chunk["text"],
                "drug": chunk["metadata"]["drug"],
                "section": chunk["metadata"]["section"],
                "page": chunk["metadata"]["page"],
                "source": chunk["metadata"]["source"]
            })

        return results


if __name__ == "__main__":
    bm25_searcher = ClinicalBM25Search()
    
    # Test query with exact medical / numerical terms
    test_query = "What is the INR target range and dosage for mechanical heart valves?"
    print(f"\n" + "=" * 50)
    print(f"BM25 SEARCH QUERY: '{test_query}'")
    print("=" * 50)

    results = bm25_searcher.search_bm25(test_query, top_k=3)
    for r in results:
        print(f"\n[BM25 Rank #{r['rank']}] (Score: {r['score']})")
        print(f"Drug:    {r['drug']} | Section: {r['section']} | Page: {r['page']}")
        print(f"Content: {r['text'][:200]}...")