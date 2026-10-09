import os
import json
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)

from retrieval.embeddings import EmbeddingPipeline

COLLECTION_NAME = "pharma_chunks"
QDRANT_STORAGE_PATH = "qdrant_db"


class ClinicalVectorStore:
    def __init__(self, storage_path: str = QDRANT_STORAGE_PATH):
        """
        Initializes an embedded local Qdrant database stored on disk.
        """
        print(f"[*] Initializing local Qdrant client at: {storage_path}")
        self.client = QdrantClient(path=storage_path)
        self.embedder = EmbeddingPipeline()

    def build_collection_from_chunks(
        self,
        chunks_file: str = os.path.join("data", "processed", "chunks.json"),
        recreate: bool = True
    ):
        """
        Reads processed chunks from Day 1, computes embeddings, and stores
        both vectors and clinical payloads in Qdrant.
        """
        if not os.path.exists(chunks_file):
            raise FileNotFoundError(f"Processed chunks not found at: {chunks_file}. Run Day 1 pipeline first!")

        with open(chunks_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        print(f"[*] Loaded {len(chunks)} chunks from {chunks_file}")

        # Check if collection exists
        collections = [col.name for col in self.client.get_collections().collections]
        if COLLECTION_NAME in collections and recreate:
            print(f"[*] Recreating collection: '{COLLECTION_NAME}'...")
            self.client.delete_collection(COLLECTION_NAME)

        if COLLECTION_NAME not in collections or recreate:
            # We use Cosine distance matching sentence-transformers normalized vectors
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self.embedder.dimension,
                    distance=Distance.COSINE
                )
            )
            print(f"[✓] Created collection: '{COLLECTION_NAME}' (dim={self.embedder.dimension})")

        # Extract texts for batch embedding
        texts = [chunk["text"] for chunk in chunks]
        print(f"[*] Computing embeddings for {len(texts)} chunks...")
        vectors = self.embedder.embed_batch(texts)

        # Build Qdrant points with complete metadata payload
        points = []
        for idx, (chunk, vec) in enumerate(zip(chunks, vectors)):
            points.append(
                PointStruct(
                    id=idx,
                    vector=vec,
                    payload={
                        "text": chunk["text"],
                        "drug": chunk["metadata"]["drug"],
                        "section": chunk["metadata"]["section"],
                        "page": chunk["metadata"]["page"],
                        "source": chunk["metadata"]["source"]
                    }
                )
            )

        print(f"[*] Uploading {len(points)} points to Qdrant...")
        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        print(f"[✓] Successfully indexed {len(points)} chunks into Qdrant!")

    def search_vector(
        self,
        query: str,
        top_k: int = 5,
        drug_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs semantic vector search with optional clinical metadata filtering.
        """
        query_vec = self.embedder.embed_text(query)

        # Optional Qdrant payload filter (e.g. drug == "Warfarin")
        query_filter = None
        if drug_filter:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="drug",
                        match=MatchValue(value=drug_filter)
                    )
                ]
            )

        # Run vector similarity search
        search_results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec,
            query_filter=query_filter,
            limit=top_k
        ).points

        formatted_results = []
        for res in search_results:
            formatted_results.append({
                "score": round(res.score, 4),
                "text": res.payload["text"],
                "drug": res.payload["drug"],
                "section": res.payload["section"],
                "page": res.payload["page"],
                "source": res.payload["source"]
            })

        return formatted_results


if __name__ == "__main__":
    # Test indexing and search
    store = ClinicalVectorStore()
    
    # 1. Build and index all 446 chunks
    store.build_collection_from_chunks()

    # 2. Test semantic query
    test_query = "What are the adverse effects and bleeding risks of Warfarin?"
    print(f"\n" + "=" * 50)
    print(f"SEARCH QUERY: '{test_query}'")
    print("=" * 50)

    results = store.search_vector(test_query, top_k=3)
    for i, r in enumerate(results, start=1):
        print(f"\n[Rank #{i}] (Similarity Score: {r['score']})")
        print(f"Drug:    {r['drug']} | Section: {r['section']} | Page: {r['page']}")
        print(f"Source:  {r['source']}")
        print(f"Content: {r['text'][:220]}...")