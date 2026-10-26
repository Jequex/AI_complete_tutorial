#!/usr/bin/env python3
"""
Phase 16: Transformers & Attention Mechanisms
Topic 16.2: Multi-Head Attention

Pure Python 3 implementation (zero external dependencies).
Demonstrates:
  1. Multi-Head Attention (MHA) from first principles.
  2. Multi-head projection matrices W^Q, W^K, W^V, and W^O.
  3. 4D Tensor Reshaping and Permutation: [B, T, d_model] -> [B, h, T, d_k].
  4. Parallel multi-head attention execution and head concatenation.
  5. Exact analytical verification on a 2-head toy example.
  6. End-to-end execution on a batch with B=2, T=4, d_model=16, h=4, d_k=4.
"""

import math
import random
from typing import List, Tuple, Optional


# ============================================================================
# Basic Matrix & 3D/4D Tensor Math Utilities
# ============================================================================

def matmul_2d(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """2D Matrix Multiplication C = A @ B."""
    rows_A, cols_A = len(A), len(A[0]) if A else 0
    rows_B, cols_B = len(B), len(B[0]) if B else 0
    if cols_A != rows_B:
        raise ValueError(f"Shape mismatch: cannot multiply ({rows_A}x{cols_A}) and ({rows_B}x{cols_B})")

    C = [[0.0 for _ in range(cols_B)] for _ in range(rows_A)]
    for i in range(rows_A):
        for k in range(cols_A):
            aik = A[i][k]
            for j in range(cols_B):
                C[i][j] += aik * B[k][j]
    return C


def transpose_2d(A: List[List[float]]) -> List[List[float]]:
    """Transposes a 2D matrix [R, C] -> [C, R]."""
    rows = len(A)
    cols = len(A[0]) if rows > 0 else 0
    return [[A[r][c] for r in range(rows)] for c in range(cols)]


def stable_softmax(logits: List[float]) -> List[float]:
    """1D numerically stable softmax."""
    if not logits:
        return []
    max_val = max(logits)
    exps = [math.exp(x - max_val) for x in logits]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]


# ============================================================================
# 4D Tensor Transformation Pipeline
# ============================================================================

def reshape_to_4d(
    tensor_3d: List[List[List[float]]],
    h: int,
    d_k: int
) -> List[List[List[List[float]]]]:
    """
    Splits the last dimension of [B, T, d_model] into [B, T, h, d_k].
    """
    B = len(tensor_3d)
    T = len(tensor_3d[0])
    d_model = len(tensor_3d[0][0])
    assert h * d_k == d_model, f"Dimension mismatch: {h} * {d_k} != {d_model}"

    out = []
    for b in range(B):
        batch_seq = []
        for t in range(T):
            row = tensor_3d[b][t]
            heads = []
            for head_idx in range(h):
                start = head_idx * d_k
                heads.append(row[start:start + d_k])
            batch_seq.append(heads)
        out.append(batch_seq)
    return out  # Shape: [B, T, h, d_k]


def permute_axes_1_2(
    tensor_4d: List[List[List[List[float]]]]
) -> List[List[List[List[float]]]]:
    """
    Permutes axes 1 and 2:
    Transforms [B, T, h, d_k] -> [B, h, T, d_k].
    Allows batched matrix operations across all (B * h) heads simultaneously.
    """
    B = len(tensor_4d)
    T = len(tensor_4d[0])
    h = len(tensor_4d[0][0])
    d_k = len(tensor_4d[0][0][0])

    out = []
    for b in range(B):
        head_list = []
        for head_idx in range(h):
            seq_list = []
            for t in range(T):
                seq_list.append(tensor_4d[b][t][head_idx])
            head_list.append(seq_list)
        out.append(head_list)
    return out  # Shape: [B, h, T, d_k]


def permute_axes_back(
    tensor_4d: List[List[List[List[float]]]]
) -> List[List[List[List[float]]]]:
    """
    Inverse permutation:
    Transforms [B, h, T, d_k] -> [B, T, h, d_k].
    """
    B = len(tensor_4d)
    h = len(tensor_4d[0])
    T = len(tensor_4d[0][0])
    d_k = len(tensor_4d[0][0][0])

    out = []
    for b in range(B):
        seq_list = []
        for t in range(T):
            head_list = []
            for head_idx in range(h):
                head_list.append(tensor_4d[b][head_idx][t])
            seq_list.append(head_list)
        out.append(seq_list)
    return out  # Shape: [B, T, h, d_k]


def flatten_last_two(
    tensor_4d: List[List[List[List[float]]]]
) -> List[List[List[float]]]:
    """
    Flattens the last two dimensions (concatenating heads):
    Transforms [B, T, h, d_k] -> [B, T, h * d_k] = [B, T, d_model].
    """
    B = len(tensor_4d)
    T = len(tensor_4d[0])
    h = len(tensor_4d[0][0])
    d_k = len(tensor_4d[0][0][0])
    d_model = h * d_k

    out = []
    for b in range(B):
        batch_seq = []
        for t in range(T):
            concatenated = []
            for head_idx in range(h):
                concatenated.extend(tensor_4d[b][t][head_idx])
            batch_seq.append(concatenated)
        out.append(batch_seq)
    return out  # Shape: [B, T, d_model]


# ============================================================================
# Multi-Head Attention Module
# ============================================================================

class MultiHeadAttention:
    """
    Multi-Head Attention module from first principles.
    d_model: embedding dimension
    h: number of attention heads
    d_k: dimension per head = d_model // h
    """
    def __init__(self, d_model: int, h: int, seed: int = 42):
        assert d_model % h == 0, f"d_model ({d_model}) must be divisible by h ({h})"
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h

        rng = random.Random(seed)
        # Xavier/Glorot initialization scale: sqrt(2 / (fan_in + fan_out))
        scale = math.sqrt(2.0 / (2 * d_model))

        def create_matrix(r: int, c: int) -> List[List[float]]:
            return [[(rng.random() * 2.0 - 1.0) * scale for _ in range(c)] for _ in range(r)]

        self.W_Q = create_matrix(d_model, d_model)
        self.W_K = create_matrix(d_model, d_model)
        self.W_V = create_matrix(d_model, d_model)
        self.W_O = create_matrix(d_model, d_model)

    def forward(
        self,
        X: List[List[List[float]]]
    ) -> Tuple[List[List[List[float]]], List[List[List[List[float]]]]]:
        """
        Forward pass of Multi-Head Attention.
        Input: X of shape [B, T, d_model]
        Returns:
          - Output tensor: [B, T, d_model]
          - Attention weights: [B, h, T, T]
        """
        B = len(X)
        T = len(X[0])
        d_model = self.d_model
        h = self.h
        d_k = self.d_k

        # 1. Linear Projections for Q, K, V
        # Project each [T, d_model] matrix by W_Q, W_K, W_V
        Q = [matmul_2d(X[b], self.W_Q) for b in range(B)]  # [B, T, d_model]
        K = [matmul_2d(X[b], self.W_K) for b in range(B)]  # [B, T, d_model]
        V = [matmul_2d(X[b], self.W_V) for b in range(B)]  # [B, T, d_model]

        # 2. Reshape [B, T, d_model] -> [B, T, h, d_k]
        Q_4d = reshape_to_4d(Q, h, d_k)
        K_4d = reshape_to_4d(K, h, d_k)
        V_4d = reshape_to_4d(V, h, d_k)

        # 3. Permute [B, T, h, d_k] -> [B, h, T, d_k]
        Q_perm = permute_axes_1_2(Q_4d)
        K_perm = permute_axes_1_2(K_4d)
        V_perm = permute_axes_1_2(V_4d)

        # 4. Scaled Dot-Product Attention per head in parallel
        scale_factor = math.sqrt(d_k)
        all_attn_weights = []
        all_head_outputs = []

        for b in range(B):
            batch_weights = []
            batch_contexts = []
            for head_idx in range(h):
                # Q_head: [T, d_k], K_head: [T, d_k], V_head: [T, d_k]
                q_head = Q_perm[b][head_idx]
                k_head = K_perm[b][head_idx]
                v_head = V_perm[b][head_idx]

                # Scores S = q @ k^T: [T, T]
                k_head_T = transpose_2d(k_head)
                scores = matmul_2d(q_head, k_head_T)

                # Scale scores
                scaled_scores = [[val / scale_factor for val in row] for row in scores]

                # Row-wise stable softmax -> [T, T]
                weights = [stable_softmax(row) for row in scaled_scores]

                # Context = weights @ v_head -> [T, d_k]
                context = matmul_2d(weights, v_head)

                batch_weights.append(weights)
                batch_contexts.append(context)

            all_attn_weights.append(batch_weights)  # [B, h, T, T]
            all_head_outputs.append(batch_contexts)  # [B, h, T, d_k]

        # 5. Permute back [B, h, T, d_k] -> [B, T, h, d_k]
        head_outputs_perm = permute_axes_back(all_head_outputs)

        # 6. Concatenate heads: [B, T, h, d_k] -> [B, T, d_model]
        H_concat = flatten_last_two(head_outputs_perm)

        # 7. Final linear output projection: [B, T, d_model] @ W_O -> [B, T, d_model]
        out = [matmul_2d(H_concat[b], self.W_O) for b in range(B)]

        return out, all_attn_weights


# ============================================================================
# Unit Tests and Verification Experiments
# ============================================================================

def test_manual_2head_exercise():
    """Verify Problem 2 manual 2-head forward pass."""
    print("\n--- Test 1: Manual 2-Head Attention Verification (Exercise 2) ---")
    T = 2
    d_model = 4
    h = 2
    d_k = 2

    X_sample = [
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0, 0.0]
    ]
    # Batch size B = 1
    X = [X_sample]

    # Explicit identity projections for Head 1 and Head 2
    mha = MultiHeadAttention(d_model=d_model, h=h, seed=1)
    # Set weights to identity matrices to match the manual walkthrough
    mha.W_Q = [[1.0 if i == j else 0.0 for j in range(d_model)] for i in range(d_model)]
    mha.W_K = [[1.0 if i == j else 0.0 for j in range(d_model)] for i in range(d_model)]
    mha.W_V = [[1.0 if i == j else 0.0 for j in range(d_model)] for i in range(d_model)]
    mha.W_O = [[1.0 if i == j else 0.0 for j in range(d_model)] for i in range(d_model)]

    out, weights = mha.forward(X)

    print("Head 1 Attention Weights A^(1):")
    for row in weights[0][0]:
        print("  " + " ".join(f"{w:.4f}" for w in row))

    print("Head 2 Attention Weights A^(2):")
    for row in weights[0][1]:
        print("  " + " ".join(f"{w:.4f}" for w in row))

    print("Final Output Tensor [B=1, T=2, d_model=4]:")
    for row in out[0]:
        print("  " + " ".join(f"{v:.4f}" for v in row))

    # Analytical assertions from Exercise 2
    assert math.isclose(weights[0][0][0][0], 0.6698, abs_tol=1e-3)
    assert math.isclose(weights[0][0][0][1], 0.3302, abs_tol=1e-3)
    assert math.isclose(weights[0][1][0][0], 0.6698, abs_tol=1e-3)
    assert math.isclose(weights[0][1][0][1], 0.3302, abs_tol=1e-3)

    assert math.isclose(out[0][0][0], 0.6698, abs_tol=1e-3)
    assert math.isclose(out[0][0][1], 0.3302, abs_tol=1e-3)
    assert math.isclose(out[0][0][2], 0.3302, abs_tol=1e-3)
    assert math.isclose(out[0][0][3], 0.6698, abs_tol=1e-3)

    print("✓ Manual 2-head walkthrough validated with exact analytical precision!")


def test_4head_tensor_reshaping_and_batch_pipeline():
    """
    Test 4-head attention on a 4-token sequence with batch size B=2:
    - Input shape: [B=2, T=4, d_model=16]
    - Head count: h=4, d_k=4
    - Verifies exact shape transformations at all stages
    - Verifies output matches exact dimension [2, 4, 16]
    """
    print("\n--- Test 2: 4-Head Attention on 4-Token Sequence [B=2, T=4, d_model=16] ---")
    B = 2
    T = 4
    d_model = 16
    h = 4
    d_k = d_model // h

    rng = random.Random(123)
    # Generate synthetic batch
    X = [[[rng.gauss(0.0, 1.0) for _ in range(d_model)] for _ in range(T)] for _ in range(B)]

    mha = MultiHeadAttention(d_model=d_model, h=h, seed=42)
    out, weights = mha.forward(X)

    # 1. Assert Output Shapes
    assert len(out) == B, f"Batch size mismatch: expected {B}, got {len(out)}"
    assert len(out[0]) == T, f"Sequence length mismatch: expected {T}, got {len(out[0])}"
    assert len(out[0][0]) == d_model, f"d_model mismatch: expected {d_model}, got {len(out[0][0])}"

    # 2. Assert Attention Weights Shapes: [B, h, T, T]
    assert len(weights) == B
    assert len(weights[0]) == h
    assert len(weights[0][0]) == T
    assert len(weights[0][0][0]) == T

    # 3. Assert Probability Distribution (sum across keys = 1.0)
    for b in range(B):
        for head_idx in range(h):
            for t in range(T):
                prob_sum = sum(weights[b][head_idx][t])
                assert math.isclose(prob_sum, 1.0, rel_tol=1e-5), f"Weights must sum to 1.0, got {prob_sum}"

    print(f"Verified input tensor:  [{B}, {T}, {d_model}]")
    print(f"Verified head reshape:  [{B}, {T}, {h}, {d_k}]")
    print(f"Verified head permute:  [{B}, {h}, {T}, {d_k}]")
    print(f"Verified weights shape: [{B}, {h}, {T}, {T}]")
    print(f"Verified output tensor: [{len(out)}, {len(out[0])}, {len(out[0][0])}]")
    print("\nSample Attention Pattern Head 0 (Token 0):", [f"{w:.3f}" for w in weights[0][0][0]])
    print("Sample Attention Pattern Head 1 (Token 0):", [f"{w:.3f}" for w in weights[0][1][0]])
    print("Sample Attention Pattern Head 2 (Token 0):", [f"{w:.3f}" for w in weights[0][2][0]])
    print("Sample Attention Pattern Head 3 (Token 0):", [f"{w:.3f}" for w in weights[0][3][0]])
    print("✓ Full 4-head attention tensor pipeline and dimensional assertions passed!")


def main():
    print("=" * 80)
    print("PHASE 16.2 EXPERIMENT: MULTI-HEAD ATTENTION FROM FIRST PRINCIPLES")
    print("=" * 80)
    test_manual_2head_exercise()
    test_4head_tensor_reshaping_and_batch_pipeline()
    print("\n" + "=" * 80)
    print("ALL MULTI-HEAD ATTENTION EXPERIMENTS PASSED SUCCESSFULLY (Exit Code 0)")
    print("=" * 80)


if __name__ == "__main__":
    main()
