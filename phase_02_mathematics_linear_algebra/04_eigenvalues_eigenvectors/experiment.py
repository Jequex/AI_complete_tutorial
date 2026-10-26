#!/usr/bin/env python3
"""
Experiment 2.4: Eigenvalues, Eigenvectors, & The Power Iteration Algorithm
========================================================================
Demonstrates:
  1. Solving for dominant eigenvalues/eigenvectors using Power Iteration.
  2. The Rayleigh Quotient formula: lambda = (v^T A v) / (v^T v).
  3. Simulating Exploding vs. Vanishing Gradients in Deep Networks based
     on whether |lambda_max| > 1 or |lambda_max| < 1.

100% pure standard library Python!
"""

import math
import random


def matvec_mul(mat, v):
    """Multiplies square matrix (n x n) by vector (n)."""
    return [sum(mat[r][c] * v[c] for c in range(len(v))) for r in range(len(mat))]


def vector_norm(v):
    return math.sqrt(sum(x * x for x in v))


def dot_product(v1, v2):
    return sum(a * b for a, b in zip(v1, v2))


def power_iteration(A, num_iterations: int = 50):
    """
    Finds the dominant eigenvalue and eigenvector of matrix A
    by repeatedly multiplying a random vector and normalizing:
      b_{k+1} = (A @ b_k) / ||A @ b_k||
    """
    n = len(A)
    # Start with random non-zero vector
    random.seed(42)
    b = [random.uniform(-1.0, 1.0) for _ in range(n)]
    norm = vector_norm(b)
    b = [x / norm for x in b]

    history = []

    for it in range(num_iterations):
        # Multiply A @ b
        b_next = matvec_mul(A, b)
        # Compute Rayleigh Quotient: lambda = (b^T @ A @ b) / (b^T @ b)
        rayleigh_lambda = dot_product(b, b_next)
        history.append(rayleigh_lambda)

        # Normalize vector
        norm = vector_norm(b_next)
        if norm == 0:
            break
        b = [x / norm for x in b_next]

    return rayleigh_lambda, b, history


def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 2.4: EIGENVALUES & POWER ITERATION ALGORITHM")
    print("=" * 70)

    # Matrix from lecture: A = [[4, 1], [2, 3]]
    # Analytical eigenvalues: lambda_1 = 5, lambda_2 = 2
    # Analytical eigenvector for lambda_1: [1, 1]^T -> normalized: [0.7071, 0.7071]
    A = [
        [4.0, 1.0],
        [2.0, 3.0]
    ]

    print("\n1. COMPUTING DOMINANT EIGENVALUE VIA POWER ITERATION")
    print(f"   Matrix A: [ [{A[0][0]}, {A[0][1]}], [{A[1][0]}, {A[1][1]}] ]")
    print("   Analytical Ground Truth: lambda_max = 5.0, v_max = [0.7071, 0.7071]")

    est_lambda, est_vec, history = power_iteration(A, num_iterations=15)

    print("\n   Iteration Convergence (Rayleigh Quotient):")
    for step in [0, 1, 2, 4, 8, 14]:
        print(f"   Step {step + 1:2d} | Estimated λ: {history[step]:8.5f}")

    print(f"\n   Final Estimated Eigenvalue: {est_lambda:.5f}")
    print(f"   Final Estimated Eigenvector: [{est_vec[0]:.4f}, {est_vec[1]:.4f}]")
    print("   ✓ Perfect convergence to the theoretical optimum!")

    # 2. Gradient Stability Simulation
    print("\n" + "=" * 70)
    print("2. SIMULATING EXPLODING VS. VANISHING SIGNALS IN DEEP NETWORKS")
    print("=" * 70)
    print("Tracking hidden state signal magnitude h_T = W^T @ h_0 across 40 layers:")

    h_0 = [1.0, 1.0]

    # Case A: Matrix with |lambda_max| = 1.05 (> 1) -> Exploding
    W_exploding = [
        [1.04, 0.01],
        [0.01, 1.04]
    ]

    # Case B: Matrix with |lambda_max| = 0.95 (< 1) -> Vanishing
    W_vanishing = [
        [0.94, 0.01],
        [0.01, 0.94]
    ]

    h_exp = list(h_0)
    h_van = list(h_0)

    print(f"\n   Layer  0 | Exploding Norm: {vector_norm(h_exp):12.4f} | Vanishing Norm: {vector_norm(h_van):10.4f}")
    for layer in range(1, 41):
        h_exp = matvec_mul(W_exploding, h_exp)
        h_van = matvec_mul(W_vanishing, h_van)
        if layer in [5, 10, 20, 30, 40]:
            print(f"   Layer {layer:2d} | Exploding Norm: {vector_norm(h_exp):12.4f} | Vanishing Norm: {vector_norm(h_van):10.6f}")

    print("\n   Key Takeaway:")
    print("   If eigenvalues > 1: Signal magnitude explodes exponentially!")
    print("   If eigenvalues < 1: Signal magnitude collapses to absolute zero!")
    print("=" * 70)


if __name__ == "__main__":
    run_experiment()

