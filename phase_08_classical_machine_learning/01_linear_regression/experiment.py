#!/usr/bin/env python3
"""
Topic 8.1 Experiment: Linear Regression — Normal Equation vs. Gradient Descent
Author: AI Complete Tutorial
Dependencies: Pure Python 3 Standard Library (math, random)

Demonstrates:
1. Synthetic multi-feature dataset generation with controlled Gaussian noise.
2. From-scratch Linear Algebra Engine:
   - Matrix Transpose (X^T)
   - Matrix Multiplication (A @ B)
   - Matrix-Vector Multiplication (A @ v)
   - Gauss-Jordan Matrix Inversion with partial pivoting ((X^T X)^(-1))
3. Closed-Form Analytical Solution via the Normal Equation:
   w* = (X^T X)^(-1) X^T y
4. Iterative Solution via Gradient Descent with Mean Squared Error (MSE).
5. Comprehensive assertions verifying convergence, weight agreement, and residual orthogonality.
"""

import math
import random
import sys

# =====================================================================
# 1. Linear Algebra Engine from Scratch
# =====================================================================

def matrix_transpose(A):
    """Computes transpose A^T where A is an (m x n) list of lists."""
    m = len(A)
    n = len(A[0])
    return [[A[i][j] for i in range(m)] for j in range(n)]

def matrix_multiply(A, B):
    """Computes matrix product C = A @ B."""
    m = len(A)
    k = len(A[0])
    assert k == len(B), f"Dimension mismatch: {k} != {len(B)}"
    n = len(B[0])
    C = [[0.0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for p in range(k):
            a_ip = A[i][p]
            for j in range(n):
                C[i][j] += a_ip * B[p][j]
    return C

def matrix_vector_multiply(A, v):
    """Computes matrix-vector product w = A @ v."""
    m = len(A)
    k = len(A[0])
    assert k == len(v), f"Dimension mismatch: {k} != {len(v)}"
    res = [0.0 for _ in range(m)]
    for i in range(m):
        row = A[i]
        res[i] = sum(row[j] * v[j] for j in range(k))
    return res

def invert_matrix_gauss_jordan(A):
    """
    Computes inverse of square matrix A using Gauss-Jordan elimination
    with partial pivoting.
    """
    n = len(A)
    assert all(len(row) == n for row in A), "Matrix must be square."

    # Create augmented matrix [A | I]
    aug = []
    for i in range(n):
        aug_row = [float(val) for val in A[i]] + [1.0 if i == j else 0.0 for j in range(n)]
        aug.append(aug_row)

    # Forward elimination with partial pivoting
    for col in range(n):
        # Find pivot
        max_row = col
        max_val = abs(aug[col][col])
        for r in range(col + 1, n):
            if abs(aug[r][col]) > max_val:
                max_val = abs(aug[r][col])
                max_row = r
        
        if max_val < 1e-12:
            raise ValueError(f"Matrix is singular or near-singular at column {col}.")

        # Swap rows
        if max_row != col:
            aug[col], aug[max_row] = aug[max_row], aug[col]

        # Normalize pivot row
        pivot = aug[col][col]
        for c in range(2 * n):
            aug[col][c] /= pivot

        # Eliminate other rows
        for r in range(n):
            if r != col:
                factor = aug[r][col]
                if abs(factor) > 1e-14:
                    for c in range(col, 2 * n):
                        aug[r][c] -= factor * aug[col][c]

    # Extract right-hand side (inverse)
    inv = [[aug[i][j + n] for j in range(n)] for i in range(n)]
    return inv

# =====================================================================
# 2. Synthetic Dataset Generation
# =====================================================================

def generate_synthetic_data(num_samples=200, seed=42):
    """
    Generates synthetic regression data:
    y = 3.5 + 1.8 * x1 - 2.4 * x2 + epsilon, where epsilon ~ N(0, 0.1^2)
    """
    random.seed(seed)
    true_w = [3.5, 1.8, -2.4]  # [bias, w1, w2]
    
    X = []
    y = []
    for _ in range(num_samples):
        x1 = random.uniform(-2.0, 2.0)
        x2 = random.uniform(-2.0, 2.0)
        noise = random.gauss(0.0, 0.1)
        target = true_w[0] + true_w[1] * x1 + true_w[2] * x2 + noise
        
        # Augmented row with leading 1.0 for bias
        X.append([1.0, x1, x2])
        y.append(target)
        
    return X, y, true_w

# =====================================================================
# 3. Model Solvers
# =====================================================================

def solve_normal_equation(X, y):
    """
    Solves w* = (X^T X)^(-1) X^T y using scratch linear algebra.
    """
    XT = matrix_transpose(X)
    XTX = matrix_multiply(XT, X)
    XTX_inv = invert_matrix_gauss_jordan(XTX)
    XTy = matrix_vector_multiply(XT, y)
    w_star = matrix_vector_multiply(XTX_inv, XTy)
    return w_star

def solve_gradient_descent(X, y, lr=0.05, max_epochs=2000, tol=1e-8):
    """
    Fits weights using batch gradient descent:
    w_{t+1} = w_t - (lr / N) * X^T (X w_t - y)
    """
    N = len(X)
    d = len(X[0])
    w = [0.0] * d
    XT = matrix_transpose(X)
    
    for epoch in range(max_epochs):
        # Predict: y_hat = X @ w
        y_hat = matrix_vector_multiply(X, w)
        
        # Residuals: error = y_hat - y
        residuals = [y_hat[i] - y[i] for i in range(N)]
        
        # Gradient: (1 / N) * X^T @ residuals
        grad = matrix_vector_multiply(XT, residuals)
        grad = [g / N for g in grad]
        
        # Check gradient norm for convergence
        grad_norm = math.sqrt(sum(g * g for g in grad))
        if grad_norm < tol:
            break
            
        # Update weights
        for j in range(d):
            w[j] -= lr * grad[j]
            
    return w

def compute_mse(X, y, w):
    """Computes MSE: (1 / 2N) * sum((y_hat - y)^2)."""
    N = len(X)
    y_hat = matrix_vector_multiply(X, w)
    total_sq_err = sum((y_hat[i] - y[i]) ** 2 for i in range(N))
    return total_sq_err / (2.0 * N)

# =====================================================================
# 4. Main Experiment & Benchmark
# =====================================================================

def run_experiment():
    print("=" * 70)
    print("Topic 8.1 Experiment: Linear Regression (Normal Equation vs. GD)")
    print("=" * 70)

    # 1. Generate data
    num_samples = 250
    X, y, true_w = generate_synthetic_data(num_samples=num_samples, seed=42)
    print(f"Generated {num_samples} samples with true parameters:")
    print(f"  True Bias (w0) : {true_w[0]:+.4f}")
    print(f"  True Weight w1 : {true_w[1]:+.4f}")
    print(f"  True Weight w2 : {true_w[2]:+.4f}\n")

    # 2. Solve via Normal Equation
    w_normal = solve_normal_equation(X, y)
    mse_normal = compute_mse(X, y, w_normal)
    print("1. Closed-Form Normal Equation Solution:")
    print(f"   Weights: w0 = {w_normal[0]:.6f}, w1 = {w_normal[1]:.6f}, w2 = {w_normal[2]:.6f}")
    print(f"   Loss J(w*): {mse_normal:.6f}\n")

    # 3. Solve via Gradient Descent
    w_gd = solve_gradient_descent(X, y, lr=0.1, max_epochs=2500, tol=1e-7)
    mse_gd = compute_mse(X, y, w_gd)
    print("2. Iterative Gradient Descent Solution:")
    print(f"   Weights: w0 = {w_gd[0]:.6f}, w1 = {w_gd[1]:.6f}, w2 = {w_gd[2]:.6f}")
    print(f"   Loss J(w_gd): {mse_gd:.6f}\n")

    # 4. Orthogonality Verification: X^T (y - Xw*) == 0
    XT = matrix_transpose(X)
    y_hat_normal = matrix_vector_multiply(X, w_normal)
    residuals_normal = [y[i] - y_hat_normal[i] for i in range(num_samples)]
    residual_projection = matrix_vector_multiply(XT, residuals_normal)
    
    print("3. Orthogonality Verification of Residuals X^T (y - Xw*):")
    for j, val in enumerate(residual_projection):
        print(f"   Column {j} dot product: {val:+.2e}")
        assert abs(val) < 1e-8, f"Residual not orthogonal on dimension {j}: {val}"
    print("   -> Residual vector is strictly orthogonal to feature subspace!\n")

    # 5. Parameter Discrepancy & Equivalence Assertion
    print("4. Solution Equivalence Check:")
    max_weight_diff = max(abs(w_normal[j] - w_gd[j]) for j in range(len(true_w)))
    loss_diff = abs(mse_normal - mse_gd)
    print(f"   Max absolute weight difference : {max_weight_diff:.2e}")
    print(f"   Absolute MSE loss difference   : {loss_diff:.2e}")

    assert max_weight_diff < 1e-4, f"Weights diverge between Normal Eq and GD: {max_weight_diff}"
    assert loss_diff < 1e-6, f"Loss values diverge: {loss_diff}"
    print("   -> Normal Equation and Gradient Descent match within numerical tolerance!")

    print("\n" + "=" * 70)
    print("All assertions PASSED successfully! [Exit Code 0]")
    print("=" * 70)

if __name__ == "__main__":
    run_experiment()
    sys.exit(0)
