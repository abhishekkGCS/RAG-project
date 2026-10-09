import os
from typing import List, Union
from sentence_transformers import SentenceTransformer

# Specify your default local model folder path here
# e.g., "models/all-MiniLM-L6-v2" or the absolute path where you saved it
DEFAULT_LOCAL_MODEL_PATH = os.path.join("models", "all-MiniLM-L6-v2")


class EmbeddingPipeline:
    def __init__(self, model_path_or_name: str = DEFAULT_LOCAL_MODEL_PATH):
        """
        Initializes the embedding model using a local folder path.
        
        Args:
            model_path_or_name: Local directory path containing downloaded weights,
                                or the Hugging Face model identifier as fallback.
        """
        if os.path.isdir(model_path_or_name):
            print(f"[*] Loading model from LOCAL DIRECTORY: {os.path.abspath(model_path_or_name)}")
        else:
            print(f"[!] Local path '{model_path_or_name}' not found as a directory. Attempting fallback: {model_path_or_name}")

        # SentenceTransformer accepts local folder paths directly
        self.model = SentenceTransformer(model_path_or_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"[✓] Model loaded successfully. Vector dimension: {self.dimension}")

    def embed_text(self, text: str) -> List[float]:
        """
        Generates a vector embedding for a single text query.
        """
        vector = self.model.encode(text, convert_to_numpy=True)
        return vector.tolist()

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generates embeddings for a batch of text chunks.
        """
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return vectors.tolist()


if __name__ == "__main__":
    # Test loading from your local path:
    # Change "models/all-MiniLM-L6-v2" below to your exact local model folder path if different
    my_local_path = os.path.join("models", "all-MiniLM-L6-v2")
    
    # If your model is in another folder (e.g., C:/Users/.../all-MiniLM-L6-v2), update my_local_path
    pipeline = EmbeddingPipeline(model_path_or_name=my_local_path)
    
    sample = "Warfarin interacts with Aspirin."
    vec = pipeline.embed_text(sample)
    print(f"\nEmbedded query: '{sample}'")
    print(f"Vector dimension: {len(vec)}")
    print(f"First 5 vector values: {vec[:5]}")