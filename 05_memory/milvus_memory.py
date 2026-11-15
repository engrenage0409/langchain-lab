"""Semantic memory retrieval via Ollama embeddings + Milvus.

The *canonical* memory record is stored in SQLite (memory_core.py).
Milvus holds a searchable vector index of its contents and memory IDs.
The vector index can be rebuilt; do not treat it as the only source of truth.
"""
from __future__ import annotations

import os

from langchain_ollama import OllamaEmbeddings
from pymilvus import MilvusClient

from memory_core import MemoryStore


class MilvusMemoryIndex:
    def __init__(
        self,
        uri: str | None = None,
        collection: str = "memory_lab_vectors",
        embedding_model: str = "nomic-embed-text",
    ):
        self.collection = collection
        self.embedding = OllamaEmbeddings(model=embedding_model)
        self.client = MilvusClient(uri=uri or os.environ.get("MILVUS_URI", "./data/memory_vectors.db"))
        if self.client.has_collection(collection):
            self.client.load_collection(collection_name=collection)
        else:
            dimensions = len(self.embedding.embed_query("memory embedding dimension probe"))
            self.client.create_collection(
                collection_name=collection,
                dimension=dimensions,
                metric_type="COSINE",
            )
            self.client.load_collection(collection_name=collection)

    def rebuild(self, store: MemoryStore) -> None:
        """Rebuild search index from non-expired SQLite memories.

        Rebuild avoids stale Milvus documents after TTL expiry or explicit forget.
        The collection is dedicated to this lab and can be safely recreated.
        """
        if self.client.has_collection(self.collection):
            self.client.drop_collection(collection_name=self.collection)
        memories = store.list_active()
        if not memories:
            dim = len(self.embedding.embed_query("empty memory index"))
            self.client.create_collection(
                collection_name=self.collection, dimension=dim, metric_type="COSINE"
            )
            self.client.load_collection(collection_name=self.collection)
            return
        texts = [m.content for m in memories]
        vectors = self.embedding.embed_documents(texts)
        self.client.create_collection(
            collection_name=self.collection,
            dimension=len(vectors[0]),
            metric_type="COSINE",
        )
        self.client.insert(
            collection_name=self.collection,
            data=[
                {
                    "id": index + 1,
                    "vector": vector,
                    "memory_id": memory.id,
                    "kind": memory.kind,
                    "text": memory.content,
                }
                for index, (memory, vector) in enumerate(zip(memories, vectors))
            ],
        )
        self.client.load_collection(collection_name=self.collection)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        hits = self.client.search(
            collection_name=self.collection,
            data=[self.embedding.embed_query(query)],
            limit=top_k,
            output_fields=["memory_id", "kind", "text"],
        )
        return [
            {
                "memory_id": hit["entity"]["memory_id"],
                "kind": hit["entity"]["kind"],
                "content": hit["entity"]["text"],
                "score": round(float(hit["distance"]), 4),
            }
            for hit in hits[0]
        ]

    def close(self):
        self.client.close()
