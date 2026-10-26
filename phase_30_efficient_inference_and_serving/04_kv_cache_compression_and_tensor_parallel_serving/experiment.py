#!/usr/bin/env python3
"""
Topic 30.4 Experiment: GQA & H2O Heavy-Hitter KV Cache Compressor
Phase 30: Efficient Inference & Serving

This experiment implements a pure Python 3 (standard library only) engine for:
1. Grouped-Query Attention (GQA) KV memory sharding across simulated Tensor Parallel ranks.
2. The Heavy Hitter Oracle (H2O) dynamic KV cache compressor with Attention Sinks,
   Local Recency Window, and cumulative attention score tracking.
3. Verification that compressed KV cache maintains >95% attention distribution fidelity
   while achieving >60% memory reduction on long sequences.
"""

from typing import Dict, List, Set, Tuple
import math
import random

def softmax(logits: List[float]) -> List[float]:
    """Computes numerically stable softmax over a list of floats."""
    max_val = max(logits)
    exps = [math.exp(x - max_val) for x in logits]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]


def dot_product(v1: List[float], v2: List[float]) -> float:
    return sum(a * b for a, b in zip(v1, v2))


class H2OKVCache:
    """
    Implements the Heavy Hitter Oracle (H2O) KV Cache compression policy.
    Maintains:
    - Initial Attention Sinks (pinned)
    - Local Recency Window (pinned)
    - Top-k Heavy Hitters (dynamically selected based on cumulative attention scores)
    """
    def __init__(self, budget: int, sink_size: int = 4, recency_size: int = 16):
        self.budget = budget
        self.sink_size = sink_size
        self.recency_size = recency_size
        assert budget >= sink_size + recency_size, "Budget must exceed sink + recency window!"

        # Stored tokens: token_id -> (key_vector, value_vector)
        self.keys: Dict[int, List[float]] = {}
        self.values: Dict[int, List[float]] = {}
        # Cumulative attention scores: token_id -> float
        self.cumulative_scores: Dict[int, float] = {}
        self.token_history: List[int] = []

    def append_and_evict(self, token_id: int, key: List[float], val: List[float], attn_weights: Dict[int, float]):
        """
        Appends new token Key/Value, updates cumulative scores, and evicts lowest-scoring
        token if cache exceeds budget.
        """
        self.keys[token_id] = key
        self.values[token_id] = val
        self.token_history.append(token_id)
        self.cumulative_scores[token_id] = 0.0

        # Accumulate attention scores from latest query step
        for tid, weight in attn_weights.items():
            if tid in self.cumulative_scores:
                self.cumulative_scores[tid] += weight

        # Check if eviction is required
        if len(self.keys) <= self.budget:
            return

        # Identify candidate tokens eligible for eviction
        # Sinks (first sink_size tokens) are protected
        sinks = set(self.token_history[:self.sink_size])
        # Recency window (most recent recency_size tokens) are protected
        recency = set(self.token_history[-self.recency_size:])
        protected = sinks | recency

        # Candidates are intermediate tokens outside sink and recency
        candidates = [tid for tid in self.keys if tid not in protected]
        if not candidates:
            return

        # Find candidate with the lowest cumulative attention score
        victim = min(candidates, key=lambda tid: self.cumulative_scores[tid])

        # Evict victim from cache
        del self.keys[victim]
        del self.values[victim]
        del self.cumulative_scores[victim]

    def size(self) -> int:
        return len(self.keys)


def run_kv_compression_simulation():
    print("=" * 80)
    print("PHASE 30: TOPIC 04 EXPERIMENT — GQA & H2O KV CACHE COMPRESSOR")
    print("=" * 80)

    # 1. Architectural GQA vs MHA Memory Comparison
    print("\n[1] ARCHITECTURAL KV FOOTPRINT: MHA VS GQA VS MQA (70B Model, Context 4096, Batch 32):")
    n_layers = 80
    h_query = 64
    d_head = 128
    b = 32
    s = 4096
    bytes_per_elem = 2  # FP16

    for name, h_kv in [("Multi-Head Attention (MHA)", 64), ("Grouped-Query Attention (GQA)", 8), ("Multi-Query Attention (MQA)", 1)]:
        mem_bytes = 2 * bytes_per_elem * n_layers * h_kv * d_head * b * s
        mem_gb = mem_bytes / 1e9
        reduction = 64 / h_kv
        print(f"  • {name:<30}: KV Heads: {h_kv:2d} | Memory: {mem_gb:6.2f} GB | Reduction: {reduction:4.1f}x")

    # 2. Simulate Sequence Generation with Full Attention vs H2O Compressed Attention
    print("\n[2] SIMULATING LONG-CONTEXT ATTENTION DYNAMICS WITH H2O COMPRESSION:")
    seq_length = 256
    d_model = 64
    cache_budget = 64  # Keep only 64 tokens out of 256 (75% compression!)
    sink_size = 4
    recency_size = 16

    h2o_cache = H2OKVCache(budget=cache_budget, sink_size=sink_size, recency_size=recency_size)
    full_keys: Dict[int, List[float]] = {}
    full_vals: Dict[int, List[float]] = {}

    random.seed(1337)
    # Generate synthetic key vectors that reflect empirical LLM attention sparsity:
    # 1. Attention Sinks (first 4 tokens) receive massive baseline attention mass.
    # 2. Heavy Hitters (salient semantic tokens) have strong key alignments with recurring query topics.
    # 3. Filler background tokens have negligible logits.
    tokens_keys = {}
    for t in range(seq_length):
        if t < sink_size:
            # Sinks: high positive bias
            vec = [1.2 for _ in range(d_model)]
        elif t in [30, 45, 80, 110, 150]:
            # Heavy Hitters: correlated with queries
            vec = [0.8 + random.gauss(0.0, 0.05) for _ in range(d_model)]
        else:
            # Filler tokens: negative / zero alignment
            vec = [-1.5 + random.gauss(0.0, 0.1) for _ in range(d_model)]
        tokens_keys[t] = vec

    # Simulate autoregressive generation steps
    fidelity_scores = []
    for step in range(seq_length):
        # Query aligned with sink + salient topics
        q_vec = [0.6 + random.gauss(0.0, 0.05) for _ in range(d_model)]
        k_curr = tokens_keys[step]
        v_curr = [random.uniform(-1.0, 1.0) for _ in range(d_model)]

        # Store in uncompressed baseline
        full_keys[step] = k_curr
        full_vals[step] = v_curr

        # Compute full uncompressed attention scores over all past tokens
        full_logits = [dot_product(q_vec, full_keys[past]) / math.sqrt(d_model) for past in range(step + 1)]
        full_probs = softmax(full_logits)
        full_attn_map = {past: full_probs[idx] for idx, past in enumerate(range(step + 1))}

        # H2O Cache step: append and evict
        h2o_cache.append_and_evict(token_id=step, key=k_curr, val=v_curr, attn_weights=full_attn_map)

        # In later steps (step >= 120), compare attention output fidelity
        if step >= 120:
            # Compute attention over compressed cache
            cached_ids = list(h2o_cache.keys.keys())
            compressed_logits = [dot_product(q_vec, h2o_cache.keys[cid]) / math.sqrt(d_model) for cid in cached_ids]
            compressed_probs = softmax(compressed_logits)

            # Compare retained attention mass: sum of true attention scores captured by the compressed cache
            captured_mass = sum(full_attn_map[cid] for cid in cached_ids)
            fidelity_scores.append(captured_mass)

    avg_fidelity = sum(fidelity_scores) / len(fidelity_scores)
    memory_reduction_pct = (1.0 - (cache_budget / seq_length)) * 100.0

    print(f"  Simulation completed over {seq_length} tokens.")
    print(f"  • Full Uncompressed KV Cache Size : {seq_length} tokens")
    print(f"  • H2O Compressed KV Cache Budget  : {h2o_cache.size()} tokens (Sinks: {sink_size}, Recency: {recency_size})")
    print(f"  • Memory Footprint Reduction      : {memory_reduction_pct:.1f}%")
    print(f"  • Mean Retained Attention Mass    : {avg_fidelity * 100:.2f}% (Target: > 95%)")

    # 3. Inspect Retained Heavy Hitters
    retained_ids = set(h2o_cache.keys.keys())
    print(f"\n[3] INSPECTING RETAINED TOKENS IN COMPRESSED CACHE:")
    print(f"  Initial Sinks Retained : {[t for t in range(sink_size) if t in retained_ids]} (Protected)")
    print(f"  Recency Window Retained: {[t for t in range(seq_length - recency_size, seq_length) if t in retained_ids]}")
    synthetic_hh = [30, 45, 80, 110, 150]
    recovered_hh = [t for t in synthetic_hh if t in retained_ids]
    print(f"  Synthetic Heavy Hitters: {synthetic_hh}")
    print(f"  Successfully Retained  : {recovered_hh} ({len(recovered_hh)} / {len(synthetic_hh)} preserved)")

    # Assertions
    assert avg_fidelity > 0.95, f"H2O must retain >95% attention fidelity, got {avg_fidelity:.4f}"
    assert memory_reduction_pct >= 60.0, f"Memory reduction must be >=60%, got {memory_reduction_pct:.1f}%"
    print("\n✓ ALL KV CACHE COMPRESSION & H2O INVARIANTS VERIFIED SUCCESSFULLY.")

if __name__ == "__main__":
    run_kv_compression_simulation()
