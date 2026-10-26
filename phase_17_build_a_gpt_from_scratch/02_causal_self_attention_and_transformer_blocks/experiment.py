#!/usr/bin/env python3
"""
Phase 17 - Topic 02: Causal Self-Attention and Pre-LN Transformer Blocks
Complete pure Python 3 implementation with ZERO external dependencies.

Implements:
1. Pure Python Tensor arithmetic (matmul, batched matmul, transpose, softmax, layer norm).
2. Exact GELU (Gaussian Error Linear Unit) approximation.
3. Multi-Head Causal Self-Attention with lower-triangular causal masking (M_{ij} = -inf for j > i).
4. Feed-Forward Network (MLP) with 4x hidden expansion and GELU.
5. Pre-LayerNorm Transformer Block architecture with dual residual stream connections.
6. Rigorous causality test: proves mathematically and numerically that altering future tokens
   causes ZERO perturbation to earlier token representations.
"""

import math
import random
import sys
from typing import List, Tuple

# Set fixed seed for deterministic behavior
random.seed(42)


# =====================================================================
# 1. Pure Python Math & Tensor Primitives
# =====================================================================

def gelu(x: float) -> float:
    """GELU activation using standard tanh approximation."""
    return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x ** 3))))


def softmax_1d(vec: List[float]) -> List[float]:
    """Numerically stable 1D softmax."""
    max_val = max(vec)
    exp_vec = [math.exp(v - max_val) for v in vec]
    sum_exp = sum(exp_vec)
    return [v / sum_exp for v in exp_vec]


def matmul_2d(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Multiply matrix A [M x K] by matrix B [K x N] -> [M x N]."""
    M = len(A)
    K = len(A[0])
    N = len(B[0])
    # Transpose B for cache-friendly dot products
    B_T = [[B[r][c] for r in range(K)] for c in range(N)]
    out = [[0.0] * N for _ in range(M)]
    for i in range(M):
        row_A = A[i]
        for j in range(N):
            row_B_T = B_T[j]
            s = 0.0
            for k in range(K):
                s += row_A[k] * row_B_T[k]
            out[i][j] = s
    return out


def transpose_2d(A: List[List[float]]) -> List[List[float]]:
    """Transpose matrix A [M x N] -> [N x M]."""
    M = len(A)
    N = len(A[0])
    return [[A[r][c] for r in range(M)] for c in range(N)]


# =====================================================================
# 2. Layer Normalization
# =====================================================================

class LayerNorm:
    """Layer Normalization over the last hidden dimension."""
    def __init__(self, d_model: int, eps: float = 1e-5):
        self.d_model = d_model
        self.eps = eps
        self.gamma = [1.0] * d_model  # learnable scale
        self.beta = [0.0] * d_model   # learnable shift

    def forward(self, X: List[List[List[float]]]) -> List[List[List[float]]]:
        """X has shape [B, T, d_model]."""
        B = len(X)
        T = len(X[0])
        D = self.d_model
        out = [[[0.0] * D for _ in range(T)] for _ in range(B)]
        
        for b in range(B):
            for t in range(T):
                vec = X[b][t]
                mean = sum(vec) / D
                var = sum((v - mean) ** 2 for v in vec) / D
                std = math.sqrt(var + self.eps)
                for d in range(D):
                    out[b][t][d] = ((vec[d] - mean) / std) * self.gamma[d] + self.beta[d]
        return out


# =====================================================================
# 3. Multi-Head Causal Self-Attention
# =====================================================================

class CausalSelfAttention:
    """
    Multi-Head Causal Self-Attention with strict lower-triangular causal masking.
    Attention(Q, K, V) = softmax( (Q K^T) / sqrt(d_k) + M ) V
    where M_{i, j} = -inf when j > i (future positions).
    """
    def __init__(self, d_model: int, n_heads: int):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_k)
        
        # Projection weights [d_model, d_model] initialized with Xavier-like normal
        init_scale = 1.0 / math.sqrt(d_model)
        self.W_q = [[random.gauss(0.0, init_scale) for _ in range(d_model)] for _ in range(d_model)]
        self.W_k = [[random.gauss(0.0, init_scale) for _ in range(d_model)] for _ in range(d_model)]
        self.W_v = [[random.gauss(0.0, init_scale) for _ in range(d_model)] for _ in range(d_model)]
        self.W_o = [[random.gauss(0.0, init_scale) for _ in range(d_model)] for _ in range(d_model)]

    def forward(self, X: List[List[List[float]]]) -> Tuple[List[List[List[float]]], List[List[List[List[float]]]]]:
        """
        Input X: [B, T, d_model]
        Returns:
            out: [B, T, d_model]
            all_attn_weights: [B, n_heads, T, T] for inspection and causality verification
        """
        B = len(X)
        T = len(X[0])
        D = self.d_model
        H = self.n_heads
        d_k = self.d_k

        out = [[[0.0] * D for _ in range(T)] for _ in range(B)]
        all_attn_weights = [[[[0.0] * T for _ in range(T)] for _ in range(H)] for _ in range(B)]

        for b in range(B):
            # Compute Q, K, V projections for this batch sequence: [T, D]
            seq = X[b]
            Q_all = matmul_2d(seq, self.W_q)
            K_all = matmul_2d(seq, self.W_k)
            V_all = matmul_2d(seq, self.W_v)

            # Process each attention head independently
            head_outputs = [[[0.0] * d_k for _ in range(T)] for _ in range(H)]

            for h in range(H):
                head_offset = h * d_k
                # Slice Q, K, V for this head: [T, d_k]
                Q_h = [[Q_all[t][head_offset + i] for i in range(d_k)] for t in range(T)]
                K_h = [[K_all[t][head_offset + i] for i in range(d_k)] for t in range(T)]
                V_h = [[V_all[t][head_offset + i] for i in range(d_k)] for t in range(T)]

                # Compute Scaled Dot Product Attention with Causal Mask
                # S = Q_h * K_h^T [T, T]
                K_h_T = transpose_2d(K_h)
                scores = matmul_2d(Q_h, K_h_T)

                # Apply scale and causal mask
                attn_weights = [[0.0] * T for _ in range(T)]
                for i in range(T):
                    row = [0.0] * T
                    for j in range(T):
                        if j <= i:
                            row[j] = scores[i][j] * self.scale
                        else:
                            # Strict causal mask: future positions cannot be seen!
                            row[j] = -1e9
                    # Softmax row
                    attn_weights[i] = softmax_1d(row)

                all_attn_weights[b][h] = attn_weights

                # Context = attn_weights * V_h -> [T, d_k]
                context_h = matmul_2d(attn_weights, V_h)
                head_outputs[h] = context_h

            # Concatenate head outputs across heads: [T, D]
            concat_heads = [[0.0] * D for _ in range(T)]
            for t in range(T):
                for h in range(H):
                    head_offset = h * d_k
                    for i in range(d_k):
                        concat_heads[t][head_offset + i] = head_outputs[h][t][i]

            # Final linear projection: [T, D] * [D, D] -> [T, D]
            out[b] = matmul_2d(concat_heads, self.W_o)

        return out, all_attn_weights


# =====================================================================
# 4. Multi-Layer Perceptron (Feed-Forward Network)
# =====================================================================

class MLP:
    """
    Position-wise Feed-Forward Network:
    Linear(d_model -> 4*d_model) -> GELU -> Linear(4*d_model -> d_model)
    """
    def __init__(self, d_model: int):
        self.d_model = d_model
        d_ff = 4 * d_model
        init_scale_1 = 1.0 / math.sqrt(d_model)
        init_scale_2 = 1.0 / math.sqrt(d_ff)
        
        self.W_1 = [[random.gauss(0.0, init_scale_1) for _ in range(d_ff)] for _ in range(d_model)]
        self.b_1 = [0.0] * d_ff
        self.W_2 = [[random.gauss(0.0, init_scale_2) for _ in range(d_model)] for _ in range(d_ff)]
        self.b_2 = [0.0] * d_model

    def forward(self, X: List[List[List[float]]]) -> List[List[List[float]]]:
        """X has shape [B, T, d_model]."""
        B = len(X)
        T = len(X[0])
        D = self.d_model
        d_ff = 4 * D
        out = [[[0.0] * D for _ in range(T)] for _ in range(B)]

        for b in range(B):
            # First linear expansion: [T, D] * [D, 4D] -> [T, 4D]
            h1 = matmul_2d(X[b], self.W_1)
            # Add bias + apply GELU
            for t in range(T):
                for j in range(d_ff):
                    h1[t][j] = gelu(h1[t][j] + self.b_1[j])
            # Second projection: [T, 4D] * [4D, D] -> [T, D]
            h2 = matmul_2d(h1, self.W_2)
            # Add bias
            for t in range(T):
                for j in range(D):
                    out[b][t][j] = h2[t][j] + self.b_2[j]

        return out


# =====================================================================
# 5. Pre-LayerNorm Transformer Block
# =====================================================================

class TransformerBlock:
    """
    Pre-LN Transformer Block Architecture (Radford et al. 2019):
    x = x + CausalSelfAttention(LayerNorm_1(x))
    x = x + MLP(LayerNorm_2(x))
    """
    def __init__(self, d_model: int, n_heads: int):
        self.ln_1 = LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln_2 = LayerNorm(d_model)
        self.mlp = MLP(d_model)

    def forward(self, X: List[List[List[float]]]) -> Tuple[List[List[List[float]]], List[List[List[List[float]]]]]:
        B = len(X)
        T = len(X[0])
        D = len(X[0][0])

        # 1. Pre-LN + Causal Attention + Residual Stream
        normed_1 = self.ln_1.forward(X)
        attn_out, attn_weights = self.attn.forward(normed_1)
        
        # Residual connection 1: x_res1 = X + attn_out
        x_res1 = [[[X[b][t][d] + attn_out[b][t][d] for d in range(D)] for t in range(T)] for b in range(B)]

        # 2. Pre-LN + MLP + Residual Stream
        normed_2 = self.ln_2.forward(x_res1)
        mlp_out = self.mlp.forward(normed_2)

        # Residual connection 2: x_out = x_res1 + mlp_out
        x_out = [[[x_res1[b][t][d] + mlp_out[b][t][d] for d in range(D)] for t in range(T)] for b in range(B)]

        return x_out, attn_weights


# =====================================================================
# 6. Verification and Causality Leakage Test
# =====================================================================

def run_experiment():
    print("=" * 70)
    print("Phase 17 - Topic 02: Causal Self-Attention & Transformer Blocks")
    print("=" * 70)

    # Configuration
    batch_size = 2
    seq_len = 5
    d_model = 16
    n_heads = 4

    print(f"[*] Block Specs: B={batch_size}, T={seq_len}, d_model={d_model}, n_heads={n_heads}, d_k={d_model // n_heads}")

    block = TransformerBlock(d_model=d_model, n_heads=n_heads)

    # Generate synthetic input sequence: [B, T, d_model]
    X_original = [[[random.gauss(0.0, 1.0) for _ in range(d_model)] for _ in range(seq_len)] for _ in range(batch_size)]

    # 1. Forward Pass
    out_orig, attn_weights = block.forward(X_original)

    # Assert shape preservation
    assert len(out_orig) == batch_size
    assert len(out_orig[0]) == seq_len
    assert len(out_orig[0][0]) == d_model
    print("    [1] Output tensor shape matches input identically: [2, 5, 16] -> ✅ PASSED")

    # 2. Inspect Attention Mask Values
    print("\n" + "=" * 70)
    print("[*] Inspecting Causal Attention Matrix (Batch 0, Head 0):")
    print("=" * 70)
    head_weights = attn_weights[0][0]
    for i in range(seq_len):
        row_str = " ".join(f"{w:6.4f}" for w in head_weights[i])
        print(f"  Token {i} attends to: [{row_str}]")
        # Assert future weights are exactly 0.0
        for j in range(i + 1, seq_len):
            assert head_weights[i][j] == 0.0, f"Leakage detected! Token {i} attended to future token {j} with weight {head_weights[i][j]}"
    print("    All future upper-triangular attention weights are strictly 0.0 -> ✅ PASSED")

    # 3. Mathematical Proof of Causality: Future Perturbation Test
    print("\n" + "=" * 70)
    print("[*] Performing Future Perturbation / Non-Leakage Test:")
    print("=" * 70)
    # Create a copy of X_original and radically modify the very last token at position T - 1 (position 4)
    X_modified = [[[X_original[b][t][d] for d in range(d_model)] for t in range(seq_len)] for b in range(batch_size)]
    for d in range(d_model):
        X_modified[0][seq_len - 1][d] += 999.0  # massive perturbation at token t=4

    out_mod, _ = block.forward(X_modified)

    # Verify that for all positions t < seq_len - 1 (t = 0, 1, 2, 3), the output is EXACTLY unchanged!
    max_diff_early = 0.0
    for t in range(seq_len - 1):
        for d in range(d_model):
            diff = abs(out_orig[0][t][d] - out_mod[0][t][d])
            if diff > max_diff_early:
                max_diff_early = diff

    diff_last = sum(abs(out_orig[0][seq_len - 1][d] - out_mod[0][seq_len - 1][d]) for d in range(d_model))

    print(f"  Perturbation introduced at token t={seq_len - 1}: +999.0 in all dimensions.")
    print(f"  Maximum absolute difference in early tokens (t=0..{seq_len - 2}): {max_diff_early:.1e}")
    print(f"  Absolute difference in perturbed token (t={seq_len - 1})      : {diff_last:.4f}")

    assert max_diff_early < 1e-12, f"Causal violation! Earlier tokens were influenced by the future (max_diff={max_diff_early})"
    assert diff_last > 1.0, "Perturbation failed to take effect on the modified token itself."
    print("    Mathematical non-leakage confirmed: Earlier tokens have zero awareness of the future! -> ✅ PASSED")

    print("\n" + "=" * 70)
    print("✅ All Causal Self-Attention & Transformer Block tests passed successfully!")
    print("=" * 70)
    sys.exit(0)


if __name__ == "__main__":
    run_experiment()
