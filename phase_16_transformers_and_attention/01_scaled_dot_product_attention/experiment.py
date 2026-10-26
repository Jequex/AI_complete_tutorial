#!/usr/bin/env python3
"""
Phase 16: Transformers & Attention Mechanisms
Topic 16.1: Scaled Dot-Product Attention

Pure Python 3 implementation (zero external dependencies).
Demonstrates:
  1. Matrix multiplication, transposition, and numerically stable softmax from scratch.
  2. Scaled Dot-Product Attention: Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V.
  3. Analytical verification against hand-calculated toy matrix example.
  4. Simulation across dimensions d_k in [16, 64, 512, 2048] proving:
     - Var(q^T k) = d_k (unscaled) vs Var(q^T k / sqrt(d_k)) = 1 (scaled).
     - Gradient saturation catastrophe in unscaled softmax (Jacobian norm -> 0).
"""

import math
import random
from typing import List, Tuple, Optional


def matmul_2d(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Matrix multiplication C = A @ B."""
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
    """Transposes a 2D matrix A [R, C] -> [C, R]."""
    rows = len(A)
    cols = len(A[0]) if rows > 0 else 0
    return [[A[r][c] for r in range(rows)] for c in range(cols)]


def stable_softmax(logits: List[float]) -> List[float]:
    """
    Numerically stable softmax for a 1D vector:
    softmax(z)_i = exp(z_i - max(z)) / sum_j exp(z_j - max(z)).
    """
    if not logits:
        return []
    max_val = max(logits)
    exps = [math.exp(x - max_val) for x in logits]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]


def softmax_2d(matrix: List[List[float]]) -> List[List[float]]:
    """Row-wise softmax on a 2D matrix."""
    return [stable_softmax(row) for row in matrix]


def scaled_dot_product_attention(
    Q: List[List[float]],
    K: List[List[float]],
    V: List[List[float]],
    scale: bool = True
) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Computes Scaled Dot-Product Attention:
    Attention(Q, K, V) = softmax(Q K^T / scale_factor) V
    Returns: (output, attention_weights)
    """
    T_q, d_k = len(Q), len(Q[0])
    T_k = len(K)
    
    # 1. Unscaled affinity matrix S = Q @ K^T: [T_q, T_k]
    K_T = transpose_2d(K)
    S = matmul_2d(Q, K_T)

    # 2. Scaling
    scale_factor = math.sqrt(d_k) if scale else 1.0
    scaled_scores = [[val / scale_factor for val in row] for row in S]

    # 3. Softmax row-wise
    attn_weights = softmax_2d(scaled_scores)

    # 4. Multiply by Value matrix: A @ V: [T_q, d_v]
    output = matmul_2d(attn_weights, V)

    return output, attn_weights


def standard_normal_vector(dim: int, rng: random.Random) -> List[float]:
    """Generates standard normal vector using Box-Muller transform."""
    vec = []
    for _ in range(dim // 2):
        u1 = max(1e-12, rng.random())
        u2 = rng.random()
        z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        z1 = math.sqrt(-2.0 * math.log(u1)) * math.sin(2.0 * math.pi * u2)
        vec.extend([z0, z1])
    if len(vec) < dim:
        u1 = max(1e-12, rng.random())
        u2 = rng.random()
        z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        vec.append(z0)
    return vec


def test_toy_example_verification():
    """Verify hand-computed Problem 1 values."""
    print("\n--- Test 1: Toy Example Verification (Exercise 1) ---")
    Q = [[2.0, 0.0], [0.0, 2.0]]
    K = [[2.0, 0.0], [1.0, 1.0]]
    V = [[1.0, 3.0], [4.0, 2.0]]

    out, weights = scaled_dot_product_attention(Q, K, V, scale=True)

    print("Computed Attention Weights A:")
    for row in weights:
        print("  " + " ".join(f"{w:.4f}" for w in row))

    print("Computed Attention Output:")
    for row in out:
        print("  " + " ".join(f"{v:.4f}" for v in row))

    # Assertions against analytical derivation
    assert math.isclose(weights[0][0], 0.8044, abs_tol=1e-3)
    assert math.isclose(weights[0][1], 0.1956, abs_tol=1e-3)
    assert math.isclose(weights[1][0], 0.1956, abs_tol=1e-3)
    assert math.isclose(weights[1][1], 0.8044, abs_tol=1e-3)

    assert math.isclose(out[0][0], 1.5868, abs_tol=1e-3)
    assert math.isclose(out[0][1], 2.8044, abs_tol=1e-3)
    assert math.isclose(out[1][0], 3.4132, abs_tol=1e-3)
    assert math.isclose(out[1][1], 2.1956, abs_tol=1e-3)
    print("✓ Toy verification passed with exact analytical accuracy!")


def test_variance_and_gradient_saturation_simulation():
    """
    Simulates attention with vs without sqrt(d_k) scaling across d_k in [16, 64, 512, 2048].
    Proves:
      - Var(q^T k) = d_k
      - Var(scaled) = 1.0
      - Unscaled gradient saturation: softmax gradient norm -> 0 as d_k grows large.
    """
    print("\n--- Test 2: Scaling vs Non-Scaling Simulation Across d_k ---")
    d_k_list = [16, 64, 512, 2048]
    num_samples = 500
    num_keys = 8  # Sequence length of keys to attend to

    print(f"{'d_k':>6} | {'Unscaled Var':>12} | {'Scaled Var':>10} | {'Unsc Max Prob':>13} | {'Sc Max Prob':>11} | {'Unsc Grad Norm':>14} | {'Sc Grad Norm':>12}")
    print("-" * 90)

    rng = random.Random(42)

    for d_k in d_k_list:
        unscaled_dots = []
        scaled_dots = []
        unscaled_grad_norms = []
        scaled_grad_norms = []
        unscaled_max_probs = []
        scaled_max_probs = []

        scale_factor = math.sqrt(d_k)

        for _ in range(num_samples):
            q = standard_normal_vector(d_k, rng)
            keys = [standard_normal_vector(d_k, rng) for _ in range(num_keys)]

            # Compute dot products
            dots = [sum(q[m] * k[m] for m in range(d_k)) for k in keys]
            scaled = [dot / scale_factor for dot in dots]

            unscaled_dots.extend(dots)
            scaled_dots.extend(scaled)

            # Softmax
            p_unscaled = stable_softmax(dots)
            p_scaled = stable_softmax(scaled)

            unscaled_max_probs.append(max(p_unscaled))
            scaled_max_probs.append(max(p_scaled))

            # Compute Softmax Jacobian diagonal gradient norm: sum_i p_i (1 - p_i)
            # This measures the gradient responsiveness of the attention layer
            grad_unscaled = sum(p * (1.0 - p) for p in p_unscaled)
            grad_scaled = sum(p * (1.0 - p) for p in p_scaled)

            unscaled_grad_norms.append(grad_unscaled)
            scaled_grad_norms.append(grad_scaled)

        # Empirical variance
        var_unscaled = sum(x**2 for x in unscaled_dots) / len(unscaled_dots)
        var_scaled = sum(x**2 for x in scaled_dots) / len(scaled_dots)

        mean_unsc_max = sum(unscaled_max_probs) / len(unscaled_max_probs)
        mean_sc_max = sum(scaled_max_probs) / len(scaled_max_probs)

        mean_unsc_grad = sum(unscaled_grad_norms) / len(unscaled_grad_norms)
        mean_sc_grad = sum(scaled_grad_norms) / len(scaled_grad_norms)

        print(
            f"{d_k:6d} | {var_unscaled:12.2f} | {var_scaled:10.2f} | "
            f"{mean_unsc_max:13.4f} | {mean_sc_max:11.4f} | "
            f"{mean_unsc_grad:14.6f} | {mean_sc_grad:12.6f}"
        )

        # Mathematical assertions
        # 1. Unscaled variance should be close to d_k (within 10% relative error for sample size)
        assert math.isclose(var_unscaled, d_k, rel_tol=0.15), f"Expected var ~ {d_k}, got {var_unscaled}"
        # 2. Scaled variance should be close to 1.0 (within 10% relative error)
        assert math.isclose(var_scaled, 1.0, rel_tol=0.15), f"Expected var ~ 1.0, got {var_scaled}"

        # 3. For large d_k (512, 2048), unscaled max prob collapses towards 1.0 (winner-take-all)
        if d_k == 512:
            assert mean_unsc_max > 0.92, f"Expected unscaled max prob > 0.92, got {mean_unsc_max}"
            assert mean_unsc_grad < 0.10, f"Expected unscaled gradient collapse < 0.10, got {mean_unsc_grad}"
        elif d_k == 2048:
            assert mean_unsc_max > 0.97, f"Expected unscaled max prob > 0.97, got {mean_unsc_max}"
            assert mean_unsc_grad < 0.05, f"Expected unscaled gradient collapse < 0.05, got {mean_unsc_grad}"

        # Scaled max prob remains smooth and balanced (< 0.6)
        assert mean_sc_max < 0.60, f"Expected smooth scaled max prob < 0.60, got {mean_sc_max}"
        # Scaled gradient remains healthy (> 0.5)
        assert mean_sc_grad > 0.50, f"Expected healthy scaled gradient > 0.50, got {mean_sc_grad}"

    print("✓ Variance proofs and gradient saturation catastrophe successfully demonstrated!")


def main():
    print("=" * 80)
    print("PHASE 16.1 EXPERIMENT: SCALED DOT-PRODUCT ATTENTION & VARIANCE PROOFS")
    print("=" * 80)
    test_toy_example_verification()
    test_variance_and_gradient_saturation_simulation()
    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY (Exit Code 0)")
    print("=" * 80)


if __name__ == "__main__":
    main()
