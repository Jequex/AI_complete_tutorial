#!/usr/bin/env python3
"""
Phase 16: Transformers & Attention Mechanisms
Topic 16.4: Transformer Architecture & Causal Masking

Pure Python 3 implementation (zero external dependencies).
Demonstrates:
  1. Causal (Autoregressive) Masking: M_ij = -inf for j > i.
  2. Pre-Layer Normalization (Pre-LN).
  3. Position-Wise Feed-Forward Network (FFN / MLP) with GELU.
  4. Complete Pre-LN Transformer Decoder Block from scratch.
  5. Exact verification of Exercise 1 manual masked attention.
  6. Verification of exact zero attention weights for future positions (A_ij == 0 for j > i).
  7. Verification of residual stream gradient preservation across layers.
"""

import math
import random
from typing import List, Tuple, Optional


# ============================================================================
# Basic Matrix Math & Numerics
# ============================================================================

def matmul_2d(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """2D Matrix Multiplication C = A @ B."""
    rows_A, cols_A = len(A), len(A[0]) if A else 0
    rows_B, cols_B = len(B), len(B[0]) if B else 0
    if cols_A != rows_B:
        raise ValueError(f"Shape mismatch: ({rows_A}x{cols_A}) and ({rows_B}x{cols_B})")

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


def add_matrix(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Element-wise addition of two 2D matrices."""
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def gelu(x: float) -> float:
    """Gaussian Error Linear Unit (GELU) activation function."""
    return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x ** 3))))


# ============================================================================
# Layer Normalization (Pre-LN)
# ============================================================================

class LayerNorm:
    """Layer Normalization across token embedding dimension."""
    def __init__(self, d_model: int, eps: float = 1e-5):
        self.d_model = d_model
        self.eps = eps
        self.gamma = [1.0 for _ in range(d_model)]
        self.beta = [0.0 for _ in range(d_model)]

    def forward_1d(self, x: List[float]) -> List[float]:
        mean = sum(x) / self.d_model
        var = sum((v - mean) ** 2 for v in x) / self.d_model
        std = math.sqrt(var + self.eps)
        return [(x[i] - mean) / std * self.gamma[i] + self.beta[i] for i in range(self.d_model)]

    def forward_2d(self, matrix: List[List[float]]) -> List[List[float]]:
        return [self.forward_1d(row) for row in matrix]


# ============================================================================
# Position-Wise Feed-Forward Network (FFN / MLP)
# ============================================================================

class FeedForwardNetwork:
    """
    Position-Wise Feed-Forward Network:
    FFN(x) = GELU(x W_1 + b_1) W_2 + b_2
    """
    def __init__(self, d_model: int, d_ff: int, seed: int = 42):
        self.d_model = d_model
        self.d_ff = d_ff
        rng = random.Random(seed)
        scale1 = math.sqrt(2.0 / (d_model + d_ff))
        scale2 = math.sqrt(2.0 / (d_ff + d_model))

        self.W1 = [[(rng.random() * 2.0 - 1.0) * scale1 for _ in range(d_ff)] for _ in range(d_model)]
        self.b1 = [0.0 for _ in range(d_ff)]
        self.W2 = [[(rng.random() * 2.0 - 1.0) * scale2 for _ in range(d_model)] for _ in range(d_ff)]
        self.b2 = [0.0 for _ in range(d_model)]

    def forward(self, X: List[List[float]]) -> List[List[float]]:
        # X: [T, d_model]
        # 1. First linear layer: [T, d_model] @ W1 + b1 -> [T, d_ff]
        h1 = matmul_2d(X, self.W1)
        for row in h1:
            for j in range(self.d_ff):
                row[j] = gelu(row[j] + self.b1[j])

        # 2. Second linear layer: [T, d_ff] @ W2 + b2 -> [T, d_model]
        h2 = matmul_2d(h1, self.W2)
        for row in h2:
            for j in range(self.d_model):
                row[j] += self.b2[j]

        return h2


# ============================================================================
# Causal Masked Multi-Head Attention
# ============================================================================

def stable_masked_softmax(logits: List[float]) -> List[float]:
    """
    Numerically stable softmax handling -infinity values from causal masking:
    If x == -inf, exp(x - max) = 0.
    """
    # Find max finite value
    finite_vals = [x for x in logits if not math.isinf(x) and x < 1e20]
    if not finite_vals:
        return [1.0 / len(logits) for _ in logits]
    max_val = max(finite_vals)

    exps = []
    for x in logits:
        if x <= -1e20 or math.isinf(x):
            exps.append(0.0)
        else:
            exps.append(math.exp(x - max_val))

    sum_exps = sum(exps)
    if sum_exps == 0.0:
        return [0.0 for _ in logits]
    return [e / sum_exps for e in exps]


class CausalMultiHeadAttention:
    """Multi-Head Attention with upper-triangular causal masking."""
    def __init__(self, d_model: int, h: int, seed: int = 42):
        assert d_model % h == 0
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h

        rng = random.Random(seed)
        scale = math.sqrt(2.0 / (2 * d_model))
        def init_mat():
            return [[(rng.random() * 2.0 - 1.0) * scale for _ in range(d_model)] for _ in range(d_model)]

        self.W_Q = init_mat()
        self.W_K = init_mat()
        self.W_V = init_mat()
        self.W_O = init_mat()

    def forward(
        self,
        X: List[List[float]]
    ) -> Tuple[List[List[float]], List[List[List[float]]]]:
        """
        Forward pass for a single sequence X: [T, d_model].
        Returns:
          - Output: [T, d_model]
          - Attention Weights: [h, T, T]
        """
        T = len(X)
        d_k = self.d_k
        h = self.h
        scale_factor = math.sqrt(d_k)

        Q_full = matmul_2d(X, self.W_Q)
        K_full = matmul_2d(X, self.W_K)
        V_full = matmul_2d(X, self.W_V)

        all_head_weights = []
        head_outputs = []

        for head_idx in range(h):
            start = head_idx * d_k
            end = start + d_k

            Q_h = [row[start:end] for row in Q_full]
            K_h = [row[start:end] for row in K_full]
            V_h = [row[start:end] for row in V_full]

            # Unscaled scores: [T, T]
            scores = matmul_2d(Q_h, transpose_2d(K_h))

            # Apply Causal Mask: add -inf if j > i
            masked_weights = []
            for i in range(T):
                scaled_row = []
                for j in range(T):
                    if j > i:
                        scaled_row.append(-float("inf"))  # Future token!
                    else:
                        scaled_row.append(scores[i][j] / scale_factor)
                
                # Softmax with mask
                prob_row = stable_masked_softmax(scaled_row)
                masked_weights.append(prob_row)

            # Context: A @ V -> [T, d_k]
            context = matmul_2d(masked_weights, V_h)

            all_head_weights.append(masked_weights)
            head_outputs.append(context)

        # Concatenate heads: [T, h * d_k] = [T, d_model]
        concat_H = []
        for t in range(T):
            row = []
            for head_idx in range(h):
                row.extend(head_outputs[head_idx][t])
            concat_H.append(row)

        # Final projection: [T, d_model] @ W_O
        out = matmul_2d(concat_H, self.W_O)
        return out, all_head_weights


# ============================================================================
# Complete Transformer Decoder Block (Pre-LN)
# ============================================================================

class TransformerDecoderBlock:
    """
    Full Pre-LN Transformer Decoder Block:
    x1 = x + CausalMHA(LN1(x))
    x2 = x1 + FFN(LN2(x1))
    """
    def __init__(self, d_model: int, h: int, d_ff: int, seed: int = 42):
        self.d_model = d_model
        self.h = h
        self.d_ff = d_ff

        self.ln1 = LayerNorm(d_model)
        self.mha = CausalMultiHeadAttention(d_model, h, seed=seed)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model, d_ff, seed=seed + 1)

    def forward(
        self,
        X: List[List[float]]
    ) -> Tuple[List[List[float]], List[List[List[float]]]]:
        # 1. Pre-LN + Causal Attention + Residual
        norm_x1 = self.ln1.forward_2d(X)
        attn_out, weights = self.mha.forward(norm_x1)
        x1 = add_matrix(X, attn_out)

        # 2. Pre-LN + Feed-Forward + Residual
        norm_x2 = self.ln2.forward_2d(x1)
        ffn_out = self.ffn.forward(norm_x2)
        out = add_matrix(x1, ffn_out)

        return out, weights


# ============================================================================
# Unit Tests & Verification
# ============================================================================

def test_exercise_1_causal_mask():
    """Verify Exercise 1 manual causal mask calculation."""
    print("\n--- Test 1: Manual Causal Mask Calculation (Exercise 1) ---")
    S = [
        [2.0, 4.0, 1.0],
        [3.0, 1.0, 5.0],
        [0.0, 2.0, 2.0]
    ]
    T = len(S)
    masked_A = []
    for i in range(T):
        row = []
        for j in range(T):
            if j > i:
                row.append(-float("inf"))
            else:
                row.append(S[i][j])
        masked_A.append(stable_masked_softmax(row))

    print("Computed Causal Attention Weights Matrix A:")
    for row in masked_A:
        print("  " + " ".join(f"{v:.4f}" for v in row))

    # Analytical assertions from Exercise 1
    assert math.isclose(masked_A[0][0], 1.0000, abs_tol=1e-3)
    assert math.isclose(masked_A[0][1], 0.0000, abs_tol=1e-6)
    assert math.isclose(masked_A[0][2], 0.0000, abs_tol=1e-6)

    assert math.isclose(masked_A[1][0], 0.8808, abs_tol=1e-3)
    assert math.isclose(masked_A[1][1], 0.1192, abs_tol=1e-3)
    assert math.isclose(masked_A[1][2], 0.0000, abs_tol=1e-6)

    assert math.isclose(masked_A[2][0], 0.0634, abs_tol=1e-3)
    assert math.isclose(masked_A[2][1], 0.4683, abs_tol=1e-3)
    assert math.isclose(masked_A[2][2], 0.4683, abs_tol=1e-3)

    print("✓ Exercise 1 manual causal masking analytically verified!")


def test_transformer_decoder_block_and_zero_future_leakage():
    """
    Test full Transformer Decoder Block:
    - Verifies output shape is strictly [T, d_model]
    - Verifies strictly ZERO attention weight for all future tokens (j > i) across all heads
    - Verifies row sums equal 1.0
    """
    print("\n--- Test 2: Full Transformer Decoder Block Execution ---")
    T = 4
    d_model = 16
    h = 4
    d_ff = 64

    rng = random.Random(101)
    X = [[rng.gauss(0.0, 1.0) for _ in range(d_model)] for _ in range(T)]

    decoder_block = TransformerDecoderBlock(d_model=d_model, h=h, d_ff=d_ff, seed=777)
    out, weights = decoder_block.forward(X)

    # 1. Dimension checks
    assert len(out) == T and len(out[0]) == d_model, f"Expected [{T}, {d_model}], got [{len(out)}, {len(out[0])}]"
    assert len(weights) == h and len(weights[0]) == T and len(weights[0][0]) == T

    # 2. Strict Causal Zero Leakage Assertion
    for head_idx in range(h):
        for i in range(T):
            row_sum = sum(weights[head_idx][i])
            assert math.isclose(row_sum, 1.0, rel_tol=1e-5), f"Row sum must be 1.0, got {row_sum}"
            for j in range(T):
                if j > i:
                    assert weights[head_idx][i][j] == 0.0, f"Future position ({i}, {j}) must have weight 0.0!"

    print(f"Decoder Block Output Shape Verified: [{len(out)}, {len(out[0])}]")
    print(f"Head 0 Causal Attention Matrix:")
    for row in weights[0]:
        print("  " + " ".join(f"{w:6.3f}" for w in row))
    print("✓ Strict causal masking verified: future positions receive exact zero weight across all heads!")


def test_residual_gradient_preservation():
    """
    Verifies that the residual connection preserves identity gradient:
    d(out) / d(X) includes an identity term I, preventing vanishing gradients.
    """
    print("\n--- Test 3: Residual Stream Gradient Highway Verification ---")
    d_model = 8
    h = 2
    d_ff = 32
    T = 3

    decoder_block = TransformerDecoderBlock(d_model=d_model, h=h, d_ff=d_ff, seed=99)
    rng = random.Random(42)
    X = [[rng.gauss(0.0, 1.0) for _ in range(d_model)] for _ in range(T)]

    # Compute numerical finite-difference Jacobian of out[t][k] with respect to X[t][k]
    eps = 1e-5
    out_base, _ = decoder_block.forward(X)

    # Perturb X[0][0]
    X_perturbed = [row.copy() for row in X]
    X_perturbed[0][0] += eps
    out_perturbed, _ = decoder_block.forward(X_perturbed)

    d_out_d_x = (out_perturbed[0][0] - out_base[0][0]) / eps
    print(f"Empirical Gradient d(out[0][0]) / d(X[0][0]): {d_out_d_x:.5f}")

    # Because of Pre-LN residual stream: out = X + F_attn(X) + F_ffn(X)
    # d(out) / d(X) = 1.0 + dF_attn + dF_ffn.
    # The derivative must remain robustly positive and non-zero (> 0.5)
    assert d_out_d_x > 0.5, f"Expected non-vanishing gradient highway > 0.5, got {d_out_d_x}"
    print("✓ Residual stream gradient highway successfully validated!")


def main():
    print("=" * 80)
    print("PHASE 16.4 EXPERIMENT: TRANSFORMER ARCHITECTURE & CAUSAL MASKING")
    print("=" * 80)
    test_exercise_1_causal_mask()
    test_transformer_decoder_block_and_zero_future_leakage()
    test_residual_gradient_preservation()
    print("\n" + "=" * 80)
    print("ALL TRANSFORMER DECODER EXPERIMENTS PASSED SUCCESSFULLY (Exit Code 0)")
    print("=" * 80)


if __name__ == "__main__":
    main()
