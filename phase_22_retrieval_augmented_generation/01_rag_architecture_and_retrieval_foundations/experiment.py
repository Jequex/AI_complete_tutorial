#!/usr/bin/env python3
"""
Phase 22 - Topic 01: RAG Architecture and Retrieval Foundations
Experiment: Pure Python End-to-End Mini RAG Pipeline & Hallucination Elimination

This experiment demonstrates:
1. Ingestion and indexing of a non-parametric knowledge corpus (5 proprietary facts).
2. Pure Python dense/sparse vectorizer and VectorStore with cosine similarity search.
3. Parametric Memory generation (simulating base LLM without context -> hallucinations).
4. Augmented Prompt Construction and Grounded Generation (RAG pipeline).
5. Empirical verification of Hallucination Elimination and Source Attribution.

Zero external dependencies: pure standard Python 3.
"""

import math
import re
import sys
from typing import List, Dict, Tuple, Any, Optional


class SimpleEmbeddingModel:
    """
    Pure Python TF-IDF and subword feature embedding generator.
    Constructs normalized dense vectors capturing term importance across corpus.
    """
    def __init__(self):
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_count = 0
        self.stopwords = {
            "a", "an", "the", "is", "are", "was", "were", "in", "on", "at", 
            "to", "for", "of", "with", "by", "what", "which", "who", "whom", 
            "this", "that", "these", "those", "its", "it"
        }

    def _tokenize(self, text: str) -> List[str]:
        # Lowercase and extract alphanumeric word tokens
        words = re.findall(r"\b\w+\b", text.lower())
        return [w for w in words if w not in self.stopwords]

    def fit_corpus(self, documents: List[str]) -> None:
        """Computes IDF over documents in the corpus."""
        self.doc_count = len(documents)
        doc_freqs: Dict[str, int] = {}
        
        for doc in documents:
            tokens = set(self._tokenize(doc))
            for t in tokens:
                doc_freqs[t] = doc_freqs.get(t, 0) + 1
                if t not in self.vocabulary:
                    self.vocabulary[t] = len(self.vocabulary)
                    
        # Compute smooth IDF: log(1 + (N / (1 + df)))
        for t, df in doc_freqs.items():
            self.idf[t] = math.log(1.0 + (self.doc_count / (1.0 + df)))

    def encode(self, text: str) -> List[float]:
        """Encodes text into a normalized TF-IDF vector over current vocabulary."""
        tokens = self._tokenize(text)
        if not self.vocabulary:
            return []
            
        dim = len(self.vocabulary)
        vec = [0.0] * dim
        
        # Term frequencies
        tf: Dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
            
        for t, count in tf.items():
            if t in self.vocabulary:
                idx = self.vocabulary[t]
                idf_val = self.idf.get(t, 1.0)
                # Augmented TF: 0.5 + 0.5 * (count / max_count)
                vec[idx] = (1.0 + math.log(count)) * idf_val
                
        # L2 normalization
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0.0:
            vec = [x / norm for x in vec]
        return vec


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Computes dot product between two unit-normalized vectors."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    return max(-1.0, min(1.0, dot))


class DocumentChunk:
    """Represents an ingested chunk in non-parametric memory."""
    def __init__(self, doc_id: str, content: str, metadata: Dict[str, Any]):
        self.doc_id = doc_id
        self.content = content
        self.metadata = metadata
        self.embedding: Optional[List[float]] = None

    def __repr__(self) -> str:
        return f"DocumentChunk(id={self.doc_id!r}, content={self.content[:40]!r}...)"


class VectorStore:
    """In-memory vector database with exact cosine similarity search."""
    def __init__(self, embedding_model: SimpleEmbeddingModel):
        self.embedding_model = embedding_model
        self.chunks: List[DocumentChunk] = []

    def ingest(self, doc_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> DocumentChunk:
        chunk = DocumentChunk(
            doc_id=doc_id,
            content=content,
            metadata=metadata or {}
        )
        chunk.embedding = self.embedding_model.encode(content)
        self.chunks.append(chunk)
        return chunk

    def search(self, query: str, top_k: int = 1) -> List[Tuple[DocumentChunk, float]]:
        q_vec = self.embedding_model.encode(query)
        scored_chunks = []
        for chunk in self.chunks:
            if chunk.embedding is not None:
                sim = cosine_similarity(q_vec, chunk.embedding)
                scored_chunks.append((chunk, sim))
        
        # Sort descending by similarity score
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_k]


class MiniLLMGenerator:
    """
    Simulates parametric generation vs context-augmented generation.
    Demonstrates parametric hallucination vs grounded factual synthesis.
    """
    def __init__(self, parametric_knowledge: Dict[str, str]):
        self.parametric_knowledge = parametric_knowledge

    def generate_without_rag(self, query: str) -> Dict[str, Any]:
        """
        Parametric generation only (fixed pre-trained weights).
        When queried about domain-specific, proprietary, or recent information,
        the model must guess, resulting in plausible hallucinations.
        """
        # Parametric fallback guess
        hallucinated_answer = self.parametric_knowledge.get(
            query, 
            "The system operates according to standard industry estimates of approximately 300 Kelvin."
        )
        return {
            "mode": "parametric_only (no RAG)",
            "query": query,
            "answer": hallucinated_answer,
            "sources": [],
            "grounded": False
        }

    def generate_with_rag(self, query: str, retrieved_context: List[Tuple[DocumentChunk, float]]) -> Dict[str, Any]:
        """
        Augmented generation: Synthesizes response conditioned strictly on retrieved chunks.
        """
        if not retrieved_context:
            return {
                "mode": "rag_augmented",
                "query": query,
                "answer": "I do not have sufficient information in my knowledge base to answer this question.",
                "sources": [],
                "grounded": False
            }

        top_chunk, score = retrieved_context[0]
        # In a real LLM, the context is inserted into the prompt template:
        # Prompt: "Context: {top_chunk.content}\nQuestion: {query}\nAnswer:"
        # Here we extract and ground the answer directly from the chunk text:
        grounded_answer = f"According to verified project records: {top_chunk.content} (Retrieval Confidence: {score:.3f})"
        
        return {
            "mode": "rag_augmented",
            "query": query,
            "answer": grounded_answer,
            "sources": [top_chunk.doc_id],
            "confidence": score,
            "grounded": True,
            "chunk_content": top_chunk.content
        }


def run_experiment():
    print("=" * 80)
    print("PHASE 22 - TOPIC 01: RAG ARCHITECTURE & RETRIEVAL FOUNDATIONS EXPERIMENT")
    print("=" * 80)

    # 1. Initialize Vector Index and Ingest 5 Proprietary Facts
    embedding_model = SimpleEmbeddingModel()
    
    proprietary_facts = [
        ("FACT-001", "Project Chronos was initiated in September 2028 by Dr. Elena Vance to study tachyon resonance.", {"category": "history", "date": "2028-09"}),
        ("FACT-002", "The containment chamber for Project Chronos operates at a cryogenic temperature of 1.4 Kelvin using liquid helium-4.", {"category": "engineering", "system": "containment"}),
        ("FACT-003", "Project Chronos primary energy core produces 450 Megawatts using a modular thorium pebble-bed reactor.", {"category": "power", "system": "reactor"}),
        ("FACT-004", "The designated backup safety protocol for Chronos is known as Protocol Amber-9, triggered upon detecting flux fluctuations exceeding 12 percent.", {"category": "safety", "protocol": "Amber-9"}),
        ("FACT-005", "Project Chronos achieved its first stable temporal echo on November 14, 2029, with a duration of 42 milliseconds.", {"category": "milestone", "date": "2029-11-14"})
    ]

    # Pre-fit embedding vocabulary
    all_texts = [f[1] for f in proprietary_facts]
    embedding_model.fit_corpus(all_texts)
    vector_store = VectorStore(embedding_model)

    print("\n[Step 1] Ingesting non-parametric knowledge corpus (5 facts)...")
    for doc_id, content, meta in proprietary_facts:
        chunk = vector_store.ingest(doc_id, content, meta)
        print(f"  Indexed: [{doc_id}] -> {content[:65]}...")

    assert len(vector_store.chunks) == 5, "All 5 facts must be successfully indexed."
    print("✓ Vector store successfully populated with 5 non-parametric chunks.")

    # 2. Configure Parametric Memory Simulator (Simulating an unaugmented LLM)
    parametric_hallucinations = {
        "What is the operating temperature of the Chronos containment chamber?":
            "The Chronos containment chamber operates at a nominal ambient temperature of 295 Kelvin (22°C) via forced water cooling.",
        "What is the power output of the Chronos energy core?":
            "The Chronos core generates approximately 50 Megawatts using standard hydrogen fuel cells.",
        "What backup safety protocol is triggered when flux fluctuations exceed 12 percent?":
            "The emergency protocol is Protocol Blue Delta, which activates water sprinkler valves."
    }
    llm = MiniLLMGenerator(parametric_hallucinations)

    # 3. Test Queries: Compare Parametric Only vs RAG Augmented
    test_queries = [
        ("What is the operating temperature of the Chronos containment chamber?", "FACT-002", "1.4 Kelvin"),
        ("What is the power output of the Chronos energy core?", "FACT-003", "450 Megawatts"),
        ("What backup safety protocol is triggered when flux fluctuations exceed 12 percent?", "FACT-004", "Protocol Amber-9")
    ]

    print("\n[Step 2] Executing comparative evaluation (Parametric vs RAG Augmented)...")
    
    for query, expected_doc_id, expected_keyword in test_queries:
        print(f"\nQuery: '{query}'")
        
        # A. Parametric Only
        resp_param = llm.generate_without_rag(query)
        print(f"  ❌ Parametric Response (Hallucination):")
        print(f"     \"{resp_param['answer']}\"")
        assert expected_keyword not in resp_param["answer"], "Parametric model should hallucinate and lack private facts."

        # B. Retrieval Step
        retrieved = vector_store.search(query, top_k=1)
        assert len(retrieved) > 0, "Retrieval should return at least 1 candidate."
        best_chunk, sim_score = retrieved[0]
        
        print(f"  🔍 Retrieved Context Chunk:")
        print(f"     Doc ID: {best_chunk.doc_id} | Cosine Similarity: {sim_score:.4f}")
        print(f"     Snippet: \"{best_chunk.content}\"")
        assert best_chunk.doc_id == expected_doc_id, f"Expected {expected_doc_id}, retrieved {best_chunk.doc_id}"

        # C. RAG Generation Step
        resp_rag = llm.generate_with_rag(query, retrieved)
        print(f"  ✅ RAG Augmented Response (Grounded with Attribution):")
        print(f"     \"{resp_rag['answer']}\"")
        print(f"     Sources Cited: {resp_rag['sources']}")

        # Verification Assertions
        assert expected_keyword in resp_rag["answer"], f"Target fact '{expected_keyword}' must be present in RAG output."
        assert resp_rag["sources"] == [expected_doc_id], f"Source attribution must cite {expected_doc_id}."
        assert resp_rag["grounded"] is True, "RAG response must be marked as grounded."

    # 4. Ingestion of Dynamic Real-Time Update (Demonstrating Temporal Adaptability)
    print("\n[Step 3] Testing Temporal Update without Retraining Weights...")
    update_fact = ("FACT-006", "On March 1, 2030, Project Chronos upgraded its core output to 620 Megawatts via antimatter-assisted plasma injectors.", {"category": "update"})
    all_texts.append(update_fact[1])
    embedding_model.fit_corpus(all_texts)
    # Re-encode stored chunks with new vocabulary
    for c in vector_store.chunks:
        c.embedding = embedding_model.encode(c.content)
    vector_store.ingest(*update_fact)
    
    update_query = "What is the upgraded power output of Project Chronos after the March 2030 injection?"
    retrieved_update = vector_store.search(update_query, top_k=1)
    resp_update = llm.generate_with_rag(update_query, retrieved_update)
    
    print(f"Query: '{update_query}'")
    print(f"RAG Answer: \"{resp_update['answer']}\"")
    assert "620 Megawatts" in resp_update["answer"], "Updated fact must immediately be retrievable without weight updates."
    assert resp_update["sources"] == ["FACT-006"]
    print("✓ Temporal recency demonstrated: New knowledge is immediately accessible with zero parameter fine-tuning.")

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED: Topic 01 Mini RAG Pipeline fully validated (Exit Code 0).")
    print("=" * 80)


if __name__ == "__main__":
    run_experiment()
    sys.exit(0)
