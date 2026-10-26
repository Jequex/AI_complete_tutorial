#!/usr/bin/env python3
"""
Topic 3.4 Experiment: Jacobians, Softmax Derivatives, and Vector-Jacobian Products
Phase 03: Mathematics II — Multivariable Calculus

This script provides a pure Python implementation (zero external dependencies) of:
1. General numerical Jacobian computation via central finite differences.
2. Exact analytical Softmax Jacobian evaluation.
3. High-precision gradient verification of analytical vs numerical Jacobians.
4. Fast O(K) Vector-Jacobian Product (VJP) vs O(K^2) explicit matrix multiplication.
5. Softmax + Cross-Entropy end-to-end gradient collapse verification (grad = p - y).
"""

import math
from typing import Callable, List


# ---------------------------------------------------------------------------
# 1. Core Mathematical Helpers (Pure Python)
# ---------------------------------------------------------------------------

def softmax(logits: List[float]) -> List[float]:
    """Numerically stable Softmax using max-subtraction trick."""
    max_z = max(logits)
    exp_z = [math.exp(z - max_z) for z in logits]
    total = sum(exp_z)
    return [ez / total for ez in exp_z]


def cross_entropy_loss(probs: List[float], targets: List[float]) -> float:
    """Categorical cross-entropy loss: -sum(y_i * log(p_i + 1e-15))."""
    eps = 1e-15
    return -sum(y * math.log(p + eps) for y, p in zip(targets, probs))


# ---------------------------------------------------------------------------
# 2. General Numerical Jacobian via Central Differences
# ---------------------------------------------------------------------------

def compute_numerical_jacobian(
    func: Callable[[List[float]], List[float]],
    x: List[float],
    h: float = 1e-6
) -> List[List[float]]:
    """
    Computes numerical Jacobian matrix J in R^(m x n) using central differences:
    J[i][j] = (f_i(x + h * e_j) - f_i(x - h * e_j)) / (2 * h)
    """
    n = len(x)
    # Evaluate at base to determine output dimension m
    base_out = func(x)
    m = len(base_out)
    
    jacobian = [[0.0] * n for _ in range(m)]
    
    for j in range(n):
        x_plus = list(x)
        x_minus = list(x)
        x_plus[j] += h
        x_minus[j] -= h
        
        f_plus = func(x_plus)
        f_minus = func(x_minus)
        
        for i in range(m):
            jacobian[i][j] = (f_plus[i] - f_minus[i]) / (2.0 * h)
            
    return jacobian


# ---------------------------------------------------------------------------
# 3. Analytical Softmax Jacobian
# ---------------------------------------------------------------------------

def compute_softmax_jacobian_analytical(logits: List[float]) -> List[List[float]]:
    """
    Computes exact K x K Jacobian matrix of Softmax:
    J[i][j] = p[i] * (delta[i][j] - p[j])
    """
    p = softmax(logits)
    k = len(p)
    jacobian = [[0.0] * k for _ in range(k)]
    
    for i in range(k):
        for j in range(k):
            if i == j:
                jacobian[i][j] = p[i] * (1.0 - p[i])
            else:
                jacobian[i][j] = -p[i] * p[j]
                
    return jacobian


# ---------------------------------------------------------------------------
# 4. Vector-Jacobian Product (VJP) Implementations
# ---------------------------------------------------------------------------

def vjp_matrix(v: List[float], J: List[List[float]]) -> List[float]:
    """Computes v^T @ J via explicit O(K^2) matrix-vector product."""
    m = len(J)     # rows
    n = len(J[0])  # cols
    assert len(v) == m, "Dimension mismatch in VJP!"
    
    res = [0.0] * n
    for j in range(n):
        for i in range(m):
            res[j] += v[i] * J[i][j]
    return res


def vjp_softmax_efficient(v: List[float], p: List[float]) -> List[float]:
    """
    Computes v^T @ J_softmax in O(K) time without materializing the matrix!
    Formula: v^T J = (v * p) - (v . p) * p
    """
    # Dot product v . p
    v_dot_p = sum(vi * pi for vi, pi in zip(v, p))
    # (v_i * p_i) - (v . p) * p_i
    return [(vi * pi) - (v_dot_p * pi) for vi, pi in zip(v, p)]


# ---------------------------------------------------------------------------
# 5. Verification & Demonstration Suite
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("TOPIC 3.4: JACOBIAN MATRICES & VECTOR CALCULUS IN DEEP LEARNING")
    print("=" * 72)
    
    # -----------------------------------------------------------------------
    # Section 1: Softmax Jacobian Matrix Verification
    # -----------------------------------------------------------------------
    logits = [2.0, 1.0, 0.1]
    p = softmax(logits)
    
    print("\n--- 1. Softmax Forward & Probabilities ---")
    print(f"Logits z:       {logits}")
    print(f"Probabilities p: {[round(x, 4) for x in p]}")
    print(f"Sum of probs:   {sum(p):.6f}")
    
    print("\n--- 2. Analytical vs Numerical Softmax Jacobian ---")
    J_analytical = compute_softmax_jacobian_analytical(logits)
    J_numerical = compute_numerical_jacobian(softmax, logits, h=1e-6)
    
    print("\nAnalytical Softmax Jacobian J (diag = p_i(1 - p_i), off-diag = -p_i*p_j):")
    for row in J_analytical:
        print("  [" + ", ".join(f"{val:+8.5f}" for val in row) + "]")
        
    print("\nNumerical Finite-Difference Softmax Jacobian J:")
    for row in J_numerical:
        print("  [" + ", ".join(f"{val:+8.5f}" for val in row) + "]")
        
    max_err = 0.0
    for i in range(len(logits)):
        for j in range(len(logits)):
            err = abs(J_analytical[i][j] - J_numerical[i][j])
            if err > max_err:
                max_err = err
                
    print(f"\nMax Absolute Discrepancy: {max_err:.2e}")
    assert max_err < 1e-7, "Jacobian verification failed!"
    print(">> SUCCESS: Analytical Jacobian matches Numerical Central Differences!")

    # -----------------------------------------------------------------------
    # Section 2: Vector-Jacobian Product (VJP) Efficiency Check
    # -----------------------------------------------------------------------
    print("\n--- 3. Vector-Jacobian Product (VJP) Validation ---")
    # Simulate an incoming adjoint gradient vector from an upstream loss
    v = [0.35, -0.80, 0.45]
    print(f"Incoming Adjoint Vector v: {v}")
    
    # Method A: Explicit O(K^2) matrix multiplication v^T @ J
    vjp_a = vjp_matrix(v, J_analytical)
    
    # Method B: O(K) memory-efficient direct formula (v * p) - (v . p) * p
    vjp_b = vjp_softmax_efficient(v, p)
    
    print(f"VJP via Explicit Matrix v^T @ J: {[round(x, 6) for x in vjp_a]}")
    print(f"VJP via O(K) Efficient Formula:  {[round(x, 6) for x in vjp_b]}")
    
    vjp_diff = max(abs(a - b) for a, b in zip(vjp_a, vjp_b))
    print(f"VJP Equivalence Discrepancy:    {vjp_diff:.2e}")
    assert vjp_diff < 1e-12, "VJP formula check failed!"
    print(">> SUCCESS: O(K) VJP reproduces exact matrix multiplication!")

    # -----------------------------------------------------------------------
    # Section 3: End-to-End Softmax + Cross-Entropy Gradient Collapse
    # -----------------------------------------------------------------------
    print("\n--- 4. Softmax + Cross-Entropy Gradient Collapse (grad = p - y) ---")
    # True one-hot target: class 0 is true class
    y_true = [1.0, 0.0, 0.0]
    
    # Analytical combined gradient: p - y
    grad_analytical = [pi - yi for pi, yi in zip(p, y_true)]
    
    # Numerical gradient of combined loss L(z)
    def full_loss(z_in: List[float]) -> List[float]:
        prob = softmax(z_in)
        return [cross_entropy_loss(prob, y_true)]
        
    grad_numerical = compute_numerical_jacobian(full_loss, logits, h=1e-6)[0]
    
    print(f"Target Vector y:         {y_true}")
    print(f"Predicted Probs p:       {[round(x, 4) for x in p]}")
    print(f"Analytical Grad (p - y): {[round(x, 6) for x in grad_analytical]}")
    print(f"Numerical Grad (dL/dz):  {[round(x, 6) for x in grad_numerical]}")
    
    ce_err = max(abs(a - b) for a, b in zip(grad_analytical, grad_numerical))
    print(f"Max Gradient Discrepancy: {ce_err:.2e}")
    assert ce_err < 1e-7, "Softmax + Cross-Entropy verification failed!"
    print(">> SUCCESS: The gradient collapse formula (p - y) is completely confirmed!")
    print("=" * 72)


if __name__ == "__main__":
    main()
