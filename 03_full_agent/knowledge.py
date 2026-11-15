"""Milvus-backed rule knowledge base for the multi-agent lab.

Ollama provides *semantic* embeddings; Milvus stores and searches them.
Requires: ollama pull nomic-embed-text
"""
from __future__ import annotations

import hashlib
import os

from langchain_ollama import OllamaEmbeddings
from pymilvus import MilvusClient


SEED_RULES = [
    ("WELD-001", "A required welding symbol must be present at its specified joint. A missing symbol must be reported as an annotation defect."),
    ("DIM-001", "A drawing dimension must match its explicitly stated design requirement. A 25 mm label against a 30 mm requirement is a dimensional inconsistency."),
    ("EVID-001", "Do not infer missing design facts. Findings must cite observed input facts or retrieved engineering rules."),
]


class RuleStore:
    def __init__(
        self,
        uri: str | None = None,
        collection: str = "agent_lab_rules",
        embedding_model: str = "nomic-embed-text",
    ) -> None:
        self.collection = collection
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.client = MilvusClient(
            uri=uri or os.environ.get("MILVUS_URI", "./data/agent_rules.db"),
            **({"token": os.environ["MILVUS_TOKEN"]} if os.environ.get("MILVUS_TOKEN") else {}),
        )

    def seed(self) -> None:
        """Create and populate the demo collection once, before parallel work."""
        if self.client.has_collection(self.collection):
            # An existing Milvus collection may be in the released state
            # after a previous process exits; reload before searching.
            self.client.load_collection(collection_name=self.collection)
            print(f"[Milvus] loaded existing collection: {self.collection}", flush=True)
            return
        vectors = self.embeddings.embed_documents([text for _, text in SEED_RULES])
        if not vectors:
            raise RuntimeError("Embedding model returned no vectors")
        self.client.create_collection(
            collection_name=self.collection,
            dimension=len(vectors[0]),
            metric_type="COSINE",
        )
        records = [
            {
                "id": int(hashlib.sha256(source.encode()).hexdigest()[:12], 16),
                "vector": vector,
                "source": source,
                "text": text,
            }
            for (source, text), vector in zip(SEED_RULES, vectors)
        ]
        self.client.insert(collection_name=self.collection, data=records)
        self.client.load_collection(collection_name=self.collection)
        print(f"[Milvus] seeded and loaded {len(records)} rules", flush=True)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        query_vector = self.embeddings.embed_query(query)
        hits = self.client.search(
            collection_name=self.collection,
            data=[query_vector],
            limit=top_k,
            output_fields=["text", "source"],
        )
        return [
            {
                "source": hit["entity"].get("source", "unknown"),
                "text": hit["entity"].get("text", ""),
                "score": round(float(hit.get("distance", 0.0)), 4),
            }
            for hit in hits[0]
        ]

    def close(self) -> None:
        self.client.close()
