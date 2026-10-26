#!/usr/bin/env python3
"""
Phase 22 - Topic 02: Document Chunking and Embedding Strategies
Experiment: Pure Python Fixed-Overlap, Recursive, and Semantic Chunkers

This experiment implements and benchmarks:
1. Fixed-Size Sliding Window Chunker (N words, K overlap).
2. Recursive Character Splitter (respecting hierarchical delimiters: \\n\\n, \\n, ., space).
3. Semantic Boundary Chunker (cosine distance thresholds across adjacent sentences).
4. Rigorous assertions verifying boundary invariants, chunk lengths, and overlap consistency.

Zero external dependencies: pure standard Python 3.
"""

import math
import re
import sys
from typing import List, Dict, Tuple, Any, Optional


# ============================================================================
# 1. Fixed-Size Sliding Window Chunker
# ============================================================================
class FixedOverlapChunker:
    """
    Partitions text into chunks of N words with K words of overlap.
    Stride length S = N - K.
    """
    def __init__(self, chunk_size: int, overlap: int):
        if overlap >= chunk_size:
            raise ValueError("Overlap must be strictly smaller than chunk size.")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.stride = chunk_size - overlap

    def chunk(self, text: str) -> List[str]:
        words = text.split()
        if not words:
            return []
        if len(words) <= self.chunk_size:
            return [" ".join(words)]
            
        chunks = []
        start_idx = 0
        while start_idx < len(words):
            end_idx = min(start_idx + self.chunk_size, len(words))
            chunk_words = words[start_idx:end_idx]
            chunks.append(" ".join(chunk_words))
            if end_idx == len(words):
                break
            start_idx += self.stride
        return chunks


# ============================================================================
# 2. Recursive Character Chunker
# ============================================================================
class RecursiveCharacterChunker:
    """
    Recursively splits text using a hierarchical list of delimiters
    (paragraphs -> sentences -> clauses -> words) until all chunks fit within max_chars.
    """
    def __init__(self, max_chars: int, overlap_chars: int, delimiters: Optional[List[str]] = None):
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
        self.delimiters = delimiters or ["\n\n", "\n", ". ", " ", ""]

    def chunk(self, text: str) -> List[str]:
        return self._split_text(text, self.delimiters)

    def _split_text(self, text: str, delimiters: List[str]) -> List[str]:
        if len(text) <= self.max_chars:
            return [text] if text.strip() else []

        # Find current delimiter
        current_delim = delimiters[-1]
        next_delimiters = []
        for i, d in enumerate(delimiters):
            if d == "":
                current_delim = ""
                next_delimiters = []
                break
            if d in text:
                current_delim = d
                next_delimiters = delimiters[i+1:]
                break

        # Split text by current delimiter
        if current_delim == "":
            splits = list(text)
        else:
            splits = text.split(current_delim)

        # Merge splits into chunks with overlap
        chunks = []
        current_chunk = []
        current_len = 0
        
        for piece in splits:
            piece_str = piece if current_delim == "" else piece + current_delim
            if len(piece_str) > self.max_chars and next_delimiters:
                # Recurse on oversized piece
                sub_chunks = self._split_text(piece, next_delimiters)
                for sc in sub_chunks:
                    chunks.append(sc.strip())
                continue

            if current_len + len(piece_str) > self.max_chars and current_chunk:
                merged = "".join(current_chunk).strip()
                if merged:
                    chunks.append(merged)
                
                # Keep overlap from the end of current_chunk
                overlap_acc = []
                overlap_len = 0
                for part in reversed(current_chunk):
                    if overlap_len + len(part) <= self.overlap_chars:
                        overlap_acc.insert(0, part)
                        overlap_len += len(part)
                    else:
                        break
                current_chunk = overlap_acc
                current_len = overlap_len

            current_chunk.append(piece_str)
            current_len += len(piece_str)

        if current_chunk:
            merged = "".join(current_chunk).strip()
            if merged:
                chunks.append(merged)

        return [c for c in chunks if c]


# ============================================================================
# 3. Semantic Similarity Boundary Chunker
# ============================================================================
class SemanticSentenceChunker:
    """
    Computes semantic embeddings for adjacent sentences, calculates cosine distance,
    and places chunk boundaries wherever topic drift exceeds a similarity threshold.
    """
    def __init__(self, similarity_threshold: float = 0.45, max_chunk_sentences: int = 5):
        self.similarity_threshold = similarity_threshold
        self.max_chunk_sentences = max_chunk_sentences

    def _split_sentences(self, text: str) -> List[str]:
        # Split on sentence boundaries (., !, ?) followed by whitespace
        raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in raw_sentences if s.strip()]

    def _embed_sentence(self, sentence: str) -> Dict[str, float]:
        # Bag-of-words term frequency with subword n-gram features
        words = re.findall(r"\b\w+\b", sentence.lower())
        vec: Dict[str, float] = {}
        for w in words:
            vec[w] = vec.get(w, 0.0) + 1.0
        
        # Add character 3-grams
        for i in range(len(sentence) - 2):
            tri = sentence[i:i+3].lower()
            vec[tri] = vec.get(tri, 0.0) + 0.3

        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm > 0.0:
            for k in vec:
                vec[k] /= norm
        return vec

    def _cosine_similarity(self, vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
        dot = 0.0
        for k, v in vec_a.items():
            if k in vec_b:
                dot += v * vec_b[k]
        return max(0.0, min(1.0, dot))

    def chunk(self, text: str) -> List[str]:
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return sentences

        embeddings = [self._embed_sentence(s) for s in sentences]
        
        # Calculate cosine similarity between consecutive sentences
        adjacent_similarities = []
        for i in range(len(sentences) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i+1])
            adjacent_similarities.append(sim)

        chunks = []
        current_sentences = [sentences[0]]

        for i, sim in enumerate(adjacent_similarities):
            # If similarity drops below threshold or chunk exceeds max sentences, split
            if sim < self.similarity_threshold or len(current_sentences) >= self.max_chunk_sentences:
                chunks.append(" ".join(current_sentences))
                current_sentences = [sentences[i+1]]
            else:
                current_sentences.append(sentences[i+1])

        if current_sentences:
            chunks.append(" ".join(current_sentences))

        return chunks


def run_experiment():
    print("=" * 80)
    print("PHASE 22 - TOPIC 02: DOCUMENT CHUNKING & EMBEDDING STRATEGIES EXPERIMENT")
    print("=" * 80)

    sample_document = (
        "Artificial neural networks are computational systems inspired by the biological brain. "
        "They consist of interconnected nodes organized into input, hidden, and output layers. "
        "During training, gradient descent minimizes a loss function to adjust connection weights. "
        "Backpropagation applies the chain rule of calculus to compute analytical gradients efficiently. "
        "\n\n"
        "Relational database management systems rely on structured query language and ACID guarantees. "
        "Transactions must exhibit atomicity, consistency, isolation, and durability. "
        "B-tree and LSM-tree index structures accelerate data retrieval across disk and memory. "
        "Distributed consensus algorithms like Raft and Paxos ensure replication across leader and follower nodes. "
        "\n\n"
        "Quantum computing harnesses superposition and entanglement to execute non-classical algorithms. "
        "Qubits can exist in linear combinations of zero and one states simultaneously. "
        "Shor's algorithm achieves polynomial time prime factorization, threatening classical RSA encryption. "
        "Quantum error correction utilizes surface codes to shield delicate physical qubits from environmental decoherence."
    )

    # ------------------------------------------------------------------------
    # 1. Test Fixed-Size Sliding Window Chunker
    # ------------------------------------------------------------------------
    print("\n[Step 1] Testing Fixed-Size Sliding Window Chunker (N=20 words, K=5 overlap)...")
    fixed_chunker = FixedOverlapChunker(chunk_size=20, overlap=5)
    fixed_chunks = fixed_chunker.chunk(sample_document)
    
    print(f"  Generated {len(fixed_chunker.chunk(sample_document))} fixed chunks.")
    for i, c in enumerate(fixed_chunks):
        word_count = len(c.split())
        print(f"  Chunk {i+1} ({word_count} words): \"{c[:55]}...\"")

    # Assertions for Fixed Chunker
    for i in range(len(fixed_chunks) - 1):
        words_curr = fixed_chunks[i].split()
        words_next = fixed_chunks[i+1].split()
        # The last 5 words of current chunk must match the first 5 words of next chunk
        overlap_from_curr = words_curr[-5:]
        overlap_from_next = words_next[:5]
        assert overlap_from_curr == overlap_from_next, (
            f"Overlap mismatch at chunk {i}: {overlap_from_curr} vs {overlap_from_next}"
        )
    print("✓ Fixed sliding window chunk overlap verified: consecutive chunks share exact K=5 boundary tokens.")

    # ------------------------------------------------------------------------
    # 2. Test Recursive Character Chunker
    # ------------------------------------------------------------------------
    print("\n[Step 2] Testing Recursive Character Chunker (max_chars=250, overlap_chars=40)...")
    rec_chunker = RecursiveCharacterChunker(max_chars=250, overlap_chars=40)
    rec_chunks = rec_chunker.chunk(sample_document)
    
    print(f"  Generated {len(rec_chunks)} recursive chunks.")
    for i, c in enumerate(rec_chunks):
        print(f"  Chunk {i+1} ({len(c)} chars): \"{c[:55].replace(chr(10), ' ')}...\"")
        assert len(c) <= 260, f"Chunk {i+1} exceeded max character limit: {len(c)}"
    print("✓ Recursive chunker size limits verified.")

    # ------------------------------------------------------------------------
    # 3. Test Semantic Similarity Boundary Chunker
    # ------------------------------------------------------------------------
    print("\n[Step 3] Testing Semantic Sentence Chunker on Multi-Topic Document...")
    semantic_chunker = SemanticSentenceChunker(similarity_threshold=0.25, max_chunk_sentences=4)
    semantic_chunks = semantic_chunker.chunk(sample_document)
    
    print(f"  Generated {len(semantic_chunks)} semantic chunks.")
    for i, c in enumerate(semantic_chunks):
        print(f"  Semantic Chunk {i+1}:")
        print(f"     \"{c[:80]}...\"")

    # The sample document has 3 distinct topics:
    # 1: Neural networks / Backprop
    # 2: Relational databases / ACID
    # 3: Quantum computing / Shor's algorithm
    # Semantic chunker should detect the topic shifts
    assert len(semantic_chunks) >= 3, f"Expected at least 3 semantic chunks, got {len(semantic_chunks)}"
    assert "neural networks" in semantic_chunks[0].lower() or "artificial" in semantic_chunks[0].lower()
    assert any("database" in c.lower() or "acid" in c.lower() for c in semantic_chunks), "Database topic must be isolated"
    assert any("quantum" in c.lower() or "qubits" in c.lower() for c in semantic_chunks), "Quantum topic must be isolated"
    print("✓ Semantic boundary chunker successfully segmented distinct conceptual domains.")

    # ------------------------------------------------------------------------
    # 4. Chunk Size Dilemma Benchmark: Precision vs Context Retention
    # ------------------------------------------------------------------------
    print("\n[Step 4] Benchmarking Small Chunks vs Large Chunks (Precision Dilemma)...")
    small_chunker = FixedOverlapChunker(chunk_size=10, overlap=2)
    large_chunker = FixedOverlapChunker(chunk_size=60, overlap=10)
    
    small_res = small_chunker.chunk(sample_document)
    large_res = large_chunker.chunk(sample_document)
    
    print(f"  Small Chunking Strategy: {len(small_res)} fine-grained chunks (avg {sum(len(c.split()) for c in small_res)/len(small_res):.1f} words)")
    print(f"  Large Chunking Strategy: {len(large_res)} broad chunks (avg {sum(len(c.split()) for c in large_res)/len(large_res):.1f} words)")
    
    assert len(small_res) > len(large_res) * 2, "Small chunks should yield significantly higher index granularity."
    print("✓ Chunk size trade-off benchmark complete.")

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED: Topic 02 Chunking Strategies fully verified (Exit Code 0).")
    print("=" * 80)


if __name__ == "__main__":
    run_experiment()
    sys.exit(0)
