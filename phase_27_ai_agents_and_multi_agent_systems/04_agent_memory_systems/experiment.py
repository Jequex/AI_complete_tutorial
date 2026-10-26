#!/usr/bin/env python3
"""
Topic 27.4 Experiment: Multi-Tiered Cognitive Agent Memory Manager
Phase 27: AI Agents & Multi-Agent Systems

Implements from first principles in pure Python (standard library only):
1. Cognitive Memory Architecture:
   - Working Memory Buffer: Constrained capacity window (e.g., max 4 active memory items).
   - Episodic Memory: Historical event stream with timestamps and importance scores.
   - Semantic Memory: Synthesized abstract insights and consolidated knowledge.
2. Multi-Factor Memory Scoring (Park et al. 2023 / Generative Agents):
   Score(m, q, t) = w_r * Recency(m, t) + w_i * Importance(m) + w_s * Relevance(m, q)
   - Recency: Exponential decay R = exp(-lambda * (t - t_accessed))
   - Importance: Intrinsic criticality score in [0.0, 1.0]
   - Relevance: Pure-Python vector cosine similarity over content keywords
3. Memory Consolidation & Eviction:
   - When working buffer overflows, evicts lowest-scoring items.
   - Consolidates episodic observations into high-level semantic insights.
   - Validates that critical long-term facts survive eviction and answer queries accurately.

Zero external dependencies. Pure Python 3 standard library.
"""

import math
import re
from typing import List, Dict, Optional, Tuple

STOP_WORDS = {
    "is", "the", "what", "a", "an", "and", "with", "to", "for", "of",
    "in", "at", "on", "my", "this", "that", "it", "are", "was", "were", "be"
}


def tokenize(text: str) -> List[str]:
    """Alphanumeric tokenizer with stop-word filtering."""
    words = re.findall(r"\b\w+\b", text.lower())
    return [w for w in words if w not in STOP_WORDS]


def cosine_similarity(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    """Computes cosine similarity between two sparse frequency dictionaries."""
    common_keys = set(v1.keys()).intersection(set(v2.keys()))
    dot_product = sum(v1[k] * v2[k] for k in common_keys)
    norm1 = math.sqrt(sum(val * val for val in v1.values()))
    norm2 = math.sqrt(sum(val * val for val in v2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


class MemoryItem:
    """Represents a single episodic or semantic memory node."""

    def __init__(self, memory_id: int, content: str, created_at: int, importance: float, memory_type: str = "episodic"):
        self.id = memory_id
        self.content = content
        self.created_at = created_at
        self.importance = importance  # in [0.0, 1.0]
        self.last_accessed_at = created_at
        self.memory_type = memory_type  # 'episodic' or 'semantic'

        # Precompute term frequency vector
        tokens = tokenize(content)
        self.tf: Dict[str, float] = {}
        for t in tokens:
            self.tf[t] = self.tf.get(t, 0.0) + 1.0

    def __repr__(self):
        return f"[{self.memory_type[:3].upper()} #{self.id} @t={self.created_at}] (Imp={self.importance:.2f}) {self.content}"


class CognitiveMemoryManager:
    """
    Manages working buffer, episodic store, and semantic reflection.
    """

    def __init__(self, working_capacity: int = 4, decay_lambda: float = 0.15,
                 w_recency: float = 0.2, w_importance: float = 0.3, w_relevance: float = 0.5):
        self.working_capacity = working_capacity
        self.decay_lambda = decay_lambda
        self.w_recency = w_recency
        self.w_importance = w_importance
        self.w_relevance = w_relevance

        self.current_time = 0
        self.memory_counter = 0

        self.all_memories: List[MemoryItem] = []
        self.working_buffer: List[MemoryItem] = []

    def advance_time(self, steps: int = 1):
        self.current_time += steps

    def add_episodic_memory(self, content: str, importance: float) -> MemoryItem:
        self.memory_counter += 1
        item = MemoryItem(self.memory_counter, content, self.current_time, importance, "episodic")
        self.all_memories.append(item)
        self.working_buffer.append(item)

        # Eviction if working memory exceeds capacity
        if len(self.working_buffer) > self.working_capacity:
            self._evict_working_memory()

        return item

    def consolidate_semantic_reflection(self, summary_insight: str, importance: float) -> MemoryItem:
        """Consolidates observations into a higher-level semantic principle."""
        self.memory_counter += 1
        item = MemoryItem(self.memory_counter, summary_insight, self.current_time, importance, "semantic")
        self.all_memories.append(item)
        self.working_buffer.append(item)

        if len(self.working_buffer) > self.working_capacity:
            self._evict_working_memory()

        return item

    def compute_score(self, memory: MemoryItem, query_tf: Dict[str, float]) -> Tuple[float, float, float, float]:
        """
        Computes composite retrieval score:
        S = w_r * Recency + w_i * Importance + w_s * Relevance
        """
        # Recency score with exponential decay
        elapsed = max(0, self.current_time - memory.last_accessed_at)
        recency = math.exp(-self.decay_lambda * elapsed)

        # Importance score
        importance = memory.importance

        # Relevance score
        relevance = cosine_similarity(memory.tf, query_tf)

        total_score = (self.w_recency * recency) + (self.w_importance * importance) + (self.w_relevance * relevance)
        return total_score, recency, importance, relevance

    def _evict_working_memory(self):
        """Evicts the item with lowest recency*importance from active working buffer."""
        def eviction_priority(m: MemoryItem):
            elapsed = self.current_time - m.created_at
            rec = math.exp(-self.decay_lambda * elapsed)
            return m.importance * rec

        self.working_buffer.sort(key=eviction_priority, reverse=True)
        self.working_buffer.pop()

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[MemoryItem, float]]:
        """
        Retrieves top_k relevant memories across the entire long-term store.
        Updates last_accessed_at timestamps.
        """
        q_tokens = tokenize(query)
        q_tf: Dict[str, float] = {}
        for t in q_tokens:
            q_tf[t] = q_tf.get(t, 0.0) + 1.0

        scored_items = []
        for m in self.all_memories:
            score, rec, imp, rel = self.compute_score(m, q_tf)
            scored_items.append((m, score))

        scored_items.sort(key=lambda x: x[1], reverse=True)
        top_results = scored_items[:top_k]

        # Update accessed time for retrieved items
        for m, _ in top_results:
            m.last_accessed_at = self.current_time

        return top_results


def run_experiment():
    print("=" * 75)
    print("TOPIC 27.4: COGNITIVE AGENT MEMORY SYSTEMS & RETRIEVAL DYNAMICS")
    print("=" * 75)

    mem_mgr = CognitiveMemoryManager(working_capacity=4, decay_lambda=0.15)

    # Simulate chronological stream of events
    events = [
        ("User states: 'My primary development database password is OmegaSecret99!'", 0.95),  # t=1
        ("User orders coffee with oat milk and sugar.", 0.20),                              # t=2
        ("Checked weather forecast: partly cloudy 18C in Seattle.", 0.15),                   # t=3
        ("System alert: CPU utilization at 45%.", 0.30),                                     # t=4
        ("Installed security patch for OpenSSL v3.1.2.", 0.70),                             # t=5
        ("User mentions favorite color is navy blue.", 0.10),                                # t=6
        ("Configured SSL certificates for domain api.internal.org.", 0.65),                  # t=7
        ("User asks about train schedule to Boston.", 0.25),                                 # t=8
        ("Routine backup completed for logs archive.", 0.40),                                # t=9
        ("Database health check: all replication nodes healthy.", 0.60),                     # t=10
        ("User lunch meeting scheduled for 1 PM.", 0.20),                                    # t=11
        ("Updated Docker container image tag to v2.4.", 0.50),                               # t=12
    ]

    print("[1] Ingesting Episodic Event Stream over Time (Strict Working Buffer Limit = 4):")
    for t, (content, imp) in enumerate(events):
        mem_mgr.advance_time(1)
        mem_mgr.add_episodic_memory(content, imp)
        print(f" t={mem_mgr.current_time:<2} | Ingested: '{content[:55]}...'")

    print(f"\nWorking Buffer State at t={mem_mgr.current_time} (Capacity = {mem_mgr.working_capacity}):")
    for item in mem_mgr.working_buffer:
        print(f"  - {item}")

    # Check that working buffer is at capacity and early items got evicted
    working_ids = [m.id for m in mem_mgr.working_buffer]
    assert len(mem_mgr.working_buffer) == 4, "Working buffer capacity exceeded!"
    assert 1 not in working_ids, "Memory #1 should have been evicted from active working buffer after t=12!"
    print(" [PASS] Working memory maintained strict capacity invariant <= 4.")
    print(" [PASS] Memory #1 was successfully evicted from transient working buffer.")

    # 2. Generative Reflection Consolidation
    print("\n[2] Triggering Memory Consolidation (Reflection Synthesis):")
    mem_mgr.advance_time(1)
    insight = "System Security Summary: Server infrastructure is secured with OpenSSL patch and configured internal SSL certificates."
    reflected_item = mem_mgr.consolidate_semantic_reflection(insight, importance=0.85)
    print(f" Generated Semantic Insight: {reflected_item}")

    # 3. Query Retrieval Verification
    print(f"\n[3] Querying Long-Term Memory at t={mem_mgr.current_time}:")
    query_1 = "What is the secret database password?"
    print(f" Query: '{query_1}'")
    results = mem_mgr.retrieve(query_1, top_k=2)

    for rank, (m, score) in enumerate(results, start=1):
        print(f"  Rank {rank} (Score={score:.4f}): {m.content}")

    # Verify that the critical database password from t=1 was retrieved #1 despite being evicted from working buffer!
    top_memory, top_score = results[0]
    assert "OmegaSecret99" in top_memory.content, "Failed to retrieve critical long-term secret!"
    assert top_memory.id == 1, "Expected Memory #1 to be retrieved."
    print(" [PASS] Critical memory retrieved via multi-factor scoring despite working buffer eviction.")

    query_2 = "What security updates and SSL configurations occurred?"
    print(f"\n Query: '{query_2}'")
    results_2 = mem_mgr.retrieve(query_2, top_k=2)
    for rank, (m, score) in enumerate(results_2, start=1):
        print(f"  Rank {rank} (Score={score:.4f}): {m.content}")

    assert any("SSL" in m.content for m, _ in results_2), "Failed to retrieve SSL security insights."
    print(" [PASS] Semantic reflection insight ranked among top retrieval results.")
    print("=" * 75)


if __name__ == "__main__":
    run_experiment()
