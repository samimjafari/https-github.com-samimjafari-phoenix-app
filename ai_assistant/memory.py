
import sqlite3
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from ai_assistant.utils import encrypt_data, decrypt_data
import os

class SemanticMemory:
    def __init__(self, db_path: str, encryption_key: bytes, model_name: str = 'all-MiniLM-L6-v2'):
        self.db_path = db_path
        self.encryption_key = encryption_key
        self.model = SentenceTransformer(model_name)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                encrypted_content TEXT,
                embedding BLOB,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()

    def add_memory(self, text: str):
        """Adds a memory: encrypts text and stores its embedding."""
        encrypted_text = encrypt_data(text, self.encryption_key)
        embedding = self.model.encode(text)
        embedding_blob = embedding.tobytes()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO memories (encrypted_content, embedding) VALUES (?, ?)",
            (encrypted_text, embedding_blob)
        )
        conn.commit()
        conn.close()

    def search_memories(self, query: str, top_k: int = 5) -> list:
        """Searches for top_k similar memories using cosine similarity."""
        query_embedding = self.model.encode(query)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT encrypted_content, embedding FROM memories")
        results = cursor.fetchall()
        conn.close()

        if not results:
            return []

        similarities = []
        for encrypted_content, embedding_blob in results:
            memory_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            # Cosine similarity
            similarity = np.dot(query_embedding, memory_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(memory_embedding)
            )
            similarities.append((similarity, encrypted_content))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[0], reverse=True)

        top_memories = []
        for sim, enc_content in similarities[:top_k]:
            decrypted = decrypt_data(enc_content, self.encryption_key)
            top_memories.append(decrypted)

        return top_memories

    def get_all_memories(self) -> list:
        """Returns all decrypted memories."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT encrypted_content FROM memories")
        results = cursor.fetchall()
        conn.close()

        return [decrypt_data(row[0], self.encryption_key) for row in results]
