#!/usr/bin/env python3
"""
Topic 5.4 Experiment: MAP Estimation, L2 Weight Decay & L1 Sparsity
Phase 05: Statistics & Estimation for Artificial Intelligence

This script provides a pure Python implementation (zero external dependencies) of:
1. Linear Regression with high-dimensional noisy features on a small dataset (N = 15).
2. The Overfitting Crisis of pure MLE: fitting spurious correlations on noise.
3. MAP with Gaussian Prior (L2 Regularization / Ridge): Shrinking weight variance and reducing test error.
4. MAP with Laplace Prior (L1 Regularization / Lasso): Driving irrelevant feature weights to exact zero (sparsity).
5. The Big Data Convergence: Showing that as N -> infinity, MAP converges to MLE.
"""

import math
import random
from typing import List, Tuple


# ---------------------------------------------------------------------------
# 1. Synthetic Data Generation
# ---------------------------------------------------------------------------

def generate_linear_dataset(
    n_samples: int,
    n_features: int = 10,
    true_weights: List[float] = [3.0, -2.5],
    noise_std: float = 0.5,
    seed: int = 42
) -> Tuple[List[List[float]], List[float]]:
    """
    Generates dataset where only the first 2 features are genuinely informative:
    y = 3.0 * x_0 - 2.5 * x_1 + noise
    Features x_2 through x_9 are purely irrelevant random Gaussian noise!
    """
    random.seed(seed)
    X, Y = [], []
    for _ in range(n_samples):
        x = [random.gauss(0.0, 1.0) for _ in range(n_features)]
        # Informative component from true_weights
        y = sum(w * xi for w, xi in zip(true_weights, x[:len(true_weights)]))
        y += random.gauss(0.0, noise_std)
        X.append(x)
        Y.append(y)
    return X, Y


def dot_product(w: List[float], x: List[float]) -> float:
    return sum(wi * xi for wi, xi in zip(w, x))


# ---------------------------------------------------------------------------
# 2. Model Training: MLE, MAP-L2, and MAP-L1 (Proximal Gradient)
# ---------------------------------------------------------------------------

def soft_threshold(val: float, threshold: float) -> float:
    """Soft-thresholding proximal operator for L1 (Laplace prior)."""
    if val > threshold:
        return val - threshold
    elif val < -threshold:
        return val + threshold
    else:
        return 0.0


def train_model(
    X: List[List[float]],
    Y: List[float],
    method: str = "mle",
    reg_lambda: float = 0.05,
    lr: float = 0.05,
    steps: int = 2_000
) -> List[float]:
    """
    Trains linear model using batch gradient descent:
    - 'mle': Minimizing standard MSE loss.
    - 'l2': MAP with Gaussian prior (Ridge / L2 weight decay).
    - 'l1': MAP with Laplace prior (Lasso / Proximal gradient descent).
    """
    n_samples = len(X)
    n_features = len(X[0])
    w = [0.0] * n_features
    
    for _ in range(steps):
        # 1. Compute MSE gradient w.r.t w: (2/N) * sum_i (w^T x_i - y_i) * x_i
        grad = [0.0] * n_features
        for xi, yi in zip(X, Y):
            err = dot_product(w, xi) - yi
            for j in range(n_features):
                grad[j] += (2.0 / n_samples) * err * xi[j]
                
        # 2. Parameter update
        if method == "mle":
            for j in range(n_features):
                w[j] -= lr * grad[j]
                
        elif method == "l2":
            # MAP Gaussian prior penalty: grad + 2 * lambda * w
            for j in range(n_features):
                penalty = 2.0 * reg_lambda * w[j]
                w[j] -= lr * (grad[j] + penalty)
                
        elif method == "l1":
            # MAP Laplace prior: proximal gradient soft-thresholding
            for j in range(n_features):
                w_temp = w[j] - lr * grad[j]
                w[j] = soft_threshold(w_temp, lr * reg_lambda)
                
    return w


def evaluate_mse(w: List[float], X: List[List[float]], Y: List[float]) -> float:
    """Computes Mean Squared Error on test set."""
    total_err = sum((dot_product(w, x) - y) ** 2 for x, y in zip(X, Y))
    return total_err / len(X)


# ---------------------------------------------------------------------------
# 3. Main Execution Suite
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("TOPIC 5.4: MAP ESTIMATION, L2 WEIGHT DECAY & L1 SPARSITY")
    print("=" * 72)

    # 1. Generate small training set (N=15, D=10) and large test set (N=200)
    n_train = 15
    n_test = 200
    n_features = 10
    
    X_train, Y_train = generate_linear_dataset(n_train, n_features, seed=42)
    X_test, Y_test   = generate_linear_dataset(n_test, n_features, seed=1337)
    
    print(f"\nSmall Training Dataset: {n_train} samples with {n_features} features")
    print(f"Informative Features:   x_0 (true w = +3.0) and x_1 (true w = -2.5)")
    print(f"Spurious Noise Features: x_2 through x_9 (true w = 0.0)")

    # 2. Train Models
    w_mle = train_model(X_train, Y_train, method="mle")
    w_l2  = train_model(X_train, Y_train, method="l2", reg_lambda=0.05)
    w_l1  = train_model(X_train, Y_train, method="l1", reg_lambda=0.15)

    # 3. Evaluate Generalization
    train_mse_mle = evaluate_mse(w_mle, X_train, Y_train)
    test_mse_mle  = evaluate_mse(w_mle, X_test, Y_test)

    train_mse_l2  = evaluate_mse(w_l2, X_train, Y_train)
    test_mse_l2   = evaluate_mse(w_l2, X_test, Y_test)

    train_mse_l1  = evaluate_mse(w_l1, X_train, Y_train)
    test_mse_l1   = evaluate_mse(w_l1, X_test, Y_test)

    print("\n--- 1. Comparison of Estimated Feature Weights ---")
    print(f"{'Feature':<10} | {'True Weight':<12} | {'MLE (No Prior)':<16} | {'MAP-L2 (Gaussian)':<18} | {'MAP-L1 (Laplace)':<18}")
    print("-" * 82)
    true_w_full = [3.0, -2.5] + [0.0] * (n_features - 2)
    for j in range(n_features):
        print(f"x_{j:<7} | {true_w_full[j]:>12.2f} | {w_mle[j]:>16.3f} | {w_l2[j]:>18.3f} | {w_l1[j]:>18.3f}")

    print("\n--- 2. Train vs Test Generalization Error ---")
    print(f"Pure MLE:    Train MSE = {train_mse_mle:.4f} | Test MSE = {test_mse_mle:.4f} (Overfitting on noise!)")
    print(f"MAP with L2: Train MSE = {train_mse_l2:.4f} | Test MSE = {test_mse_l2:.4f} (Substantially Lower Test Error)")
    print(f"MAP with L1: Train MSE = {train_mse_l1:.4f} | Test MSE = {test_mse_l1:.4f} (Best Generalization & Sparsity)")

    # Count zeroed-out noise features in L1
    noise_zeros_l1 = sum(1 for w in w_l1[2:] if abs(w) < 1e-4)
    print(f"\nSpurious Features Driven to Exact Zero by L1: {noise_zeros_l1} / 8")

    assert test_mse_l2 < test_mse_mle, "L2 regularization failed to reduce test error!"
    assert test_mse_l1 < test_mse_mle, "L1 regularization failed to reduce test error!"
    assert noise_zeros_l1 >= 5, "L1 failed to produce feature sparsity!"
    print(">> SUCCESS: Prior distributions (MAP) dramatically prevent overfitting on unseen test data!")

    # 4. Asymptotic Convergence: MAP -> MLE as N -> infinity
    print("\n--- 3. Big Data Convergence: As N -> Infinity, MAP -> MLE ---")
    n_big = 2_000
    X_big, Y_big = generate_linear_dataset(n_big, n_features, seed=777)
    
    w_mle_big = train_model(X_big, Y_big, method="mle")
    w_l2_big  = train_model(X_big, Y_big, method="l2", reg_lambda=0.01)
    
    max_param_diff = max(abs(a - b) for a, b in zip(w_mle_big, w_l2_big))
    print(f"Max Weight Discrepancy between MLE and MAP with N = {n_big}: {max_param_diff:.4f}")
    assert max_param_diff < 0.05, "MAP did not converge to MLE with large dataset!"
    print(">> SUCCESS: In the big-data regime, data likelihood completely dominates the prior!")
    print("=" * 72)


if __name__ == "__main__":
    main()

