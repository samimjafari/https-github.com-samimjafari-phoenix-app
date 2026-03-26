import os
import sys
import numpy as np

try:
    import faiss
    print("--- Termux FAISS instance successfully loaded! ---")
except ImportError:
    print("--- FAISS-CPU not found. Please run './build_faiss_termux.sh' first. ---")
    sys.exit(1)

class TermuxVectorStore:
    """Advanced, Termux-optimized Vector Store wrapper for MemGPT/Letta integration."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.ids = []

    def add_vectors(self, vectors: np.ndarray, ids: list):
        """Adds a batch of vectors (numpy array) with their respective IDs."""
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {vectors.shape[1]}")

        # Ensure vectors are float32 (required by faiss)
        vectors = vectors.astype('float32')
        self.index.add(vectors)
        self.ids.extend(ids)

    def search(self, query_vector: np.ndarray, top_k: int = 5):
        """Searches for the top_k most similar vectors for a given query."""
        if query_vector.shape[1] != self.dimension:
            raise ValueError(f"Query dimension mismatch: expected {self.dimension}, got {query_vector.shape[1]}")

        query_vector = query_vector.astype('float32')
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:
                results.append({
                    "id": self.ids[idx],
                    "score": float(dist)
                })
        return results

    def save_index(self, filepath: str):
        """Saves the index to a file on disk."""
        faiss.write_index(self.index, filepath)
        print(f"Index saved to {filepath}")

    def load_index(self, filepath: str):
        """Loads a saved index from disk."""
        if os.path.exists(filepath):
            self.index = faiss.read_index(filepath)
            print(f"Index loaded from {filepath}")
        else:
            print(f"Index file {filepath} not found.")

# Example usage for MemGPT/Letta routing:
# from termux_vector_store import TermuxVectorStore
# vector_store = TermuxVectorStore(dimension=384)
