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
        # We need an ID map for deletions and lookups
        self.id_map = {}
        self.next_id = 0

    def add_vectors(self, vectors: np.ndarray, metadata_ids: list):
        """Adds a batch of vectors (numpy array) with their respective metadata IDs."""
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {vectors.shape[1]}")

        # Ensure vectors are float32 (required by faiss)
        vectors = vectors.astype('float32')
        self.index.add(vectors)

        # Track the mapping of internal faiss index positions to metadata IDs
        for metadata_id in metadata_ids:
            self.id_map[self.next_id] = metadata_id
            self.next_id += 1

    def remove_vectors(self, metadata_id: str):
        """Removes vectors from the index by their metadata ID."""
        # Note: IndexFlatL2 doesn't support direct removal, so we rebuild the index.
        # This is efficient for small-medium Termux vector stores.
        ids_to_keep = [pos for pos, m_id in self.id_map.items() if m_id != metadata_id]

        if len(ids_to_keep) == len(self.id_map):
            return # Nothing to delete

        # Reconstruct index
        new_index = faiss.IndexFlatL2(self.dimension)
        new_id_map = {}
        new_next_id = 0

        for old_pos in ids_to_keep:
            vec = self.index.reconstruct(old_pos)
            new_index.add(np.expand_dims(vec, axis=0))
            new_id_map[new_next_id] = self.id_map[old_pos]
            new_next_id += 1

        self.index = new_index
        self.id_map = new_id_map
        self.next_id = new_next_id
        print(f"Removed vectors for ID {metadata_id}. Rebuilt index with {len(self.id_map)} vectors.")

    def search(self, query_vector: np.ndarray, top_k: int = 5):
        """Searches for the top_k most similar vectors for a given query."""
        if query_vector.shape[1] != self.dimension:
            raise ValueError(f"Query dimension mismatch: expected {self.dimension}, got {query_vector.shape[1]}")

        query_vector = query_vector.astype('float32')
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx in self.id_map:
                results.append({
                    "id": self.id_map[idx],
                    "score": float(dist)
                })
        return results

    def save_index(self, filepath: str):
        """Saves the index to a file on disk."""
        faiss.write_index(self.index, filepath)
        # Also save the ID map as JSON
        import json
        with open(filepath + ".map.json", 'w') as f:
            json.dump(self.id_map, f)
        print(f"Index and ID map saved to {filepath}")

    def load_index(self, filepath: str):
        """Loads a saved index from disk."""
        if os.path.exists(filepath):
            self.index = faiss.read_index(filepath)
            import json
            map_path = filepath + ".map.json"
            if os.path.exists(map_path):
                with open(map_path, 'r') as f:
                    self.id_map = {int(k): v for k, v in json.load(f).items()}
                self.next_id = max(self.id_map.keys()) + 1 if self.id_map else 0
            print(f"Index loaded from {filepath}")
        else:
            print(f"Index file {filepath} not found.")

# Example usage for MemGPT/Letta routing:
# from termux_vector_store import TermuxVectorStore
# vector_store = TermuxVectorStore(dimension=384)
