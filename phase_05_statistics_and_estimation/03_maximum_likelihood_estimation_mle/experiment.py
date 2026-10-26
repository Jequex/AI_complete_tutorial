#!/usr/bin/env python3
"""
Topic 5.3 Experiment: Maximum Likelihood Estimation (MLE) in Machine Learning
Phase 05: Statistics & Estimation for Artificial Intelligence

This script provides a pure Python implementation (zero external dependencies) of:
1. Analytical vs Numerical MLE optimization for Gaussian parameters (mu, sigma).
2. Equivalence test: Proving Linear Regression MSE minimization is mathematically
   identical to Gaussian Negative Log-Likelihood (NLL) optimization.
3. Equivalence test: Proving Logistic Regression Cross-Entropy loss is mathematically
   identical to Bernoulli Negative Log-Likelihood (NLL) optimization.
"""

import math
import random
from typing import List, Tuple


# ---------------------------------------------------------------------------
# 1. Section 1: Gaussian Analytical vs Numerical MLE
# ---------------------------------------------------------------------------

def gaussian_log_likelihood(data: List[float], mu: float, sigma: float) -> float:
    """Computes total log-likelihood ell(mu, sigma) = sum ln N(x_i; mu, sigma^2)."""
    n = len(data)
    if sigma <= 1e-9:
        return -float('inf')
    var = sigma ** 2
    log_norm = -0.5 * math.log(2.0 * math.pi * var)
    sq_err_sum = sum((x - mu) ** 2 for x in data)
    return n * log_norm - (0.5 / var) * sq_err_sum


def numerical_mle_gaussian(
    data: List[float],
    lr: float = 0.01,
    steps: int = 1_000
) -> Tuple[float, float]:
    """Finds MLE parameters (mu, sigma) via gradient ascent on log-likelihood."""
    # Initialize guesses far from truth
    mu = 0.0
    sigma = 5.0
    n = len(data)
    
    for _ in range(steps):
        var = sigma ** 2
        # d(ell)/d(mu) = (1 / sigma^2) * sum(x_i - mu)
        grad_mu = sum(x - mu for x in data) / var
        # d(ell)/d(sigma) = -n / sigma + (1 / sigma^3) * sum((x_i - mu)^2)
        sq_err = sum((x - mu) ** 2 for x in data)
        grad_sigma = (-n / sigma) + (sq_err / (sigma ** 3))
        
        # Gradient ascent step (maximizing likelihood)
        mu += lr * grad_mu
        sigma += lr * grad_sigma
        sigma = max(sigma, 0.01)  # keep positive
        
    return mu, sigma


# ---------------------------------------------------------------------------
# 2. Section 2: Equivalence of MSE and Gaussian NLL in Linear Regression
# ---------------------------------------------------------------------------

def simulate_linear_mle_vs_mse(
    true_w: float = 2.5,
    true_b: float = 1.0,
    n_samples: int = 200,
    lr: float = 0.02,
    steps: int = 500
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Trains Model 1 with standard MSE Loss.
    Trains Model 2 with Gaussian Negative Log-Likelihood (NLL).
    Proves weight trajectories converge to the exact same parameter values.
    """
    random.seed(42)
    X = [random.uniform(-2.0, 2.0) for _ in range(n_samples)]
    # y = w * x + b + Gaussian noise
    Y = [true_w * x + true_b + random.gauss(0.0, 0.5) for x in X]
    
    # Model 1: Trained by minimizing MSE = (1/n) * sum (w*x + b - y)^2
    w_mse, b_mse = 0.0, 0.0
    for _ in range(steps):
        grad_w = (2.0 / n_samples) * sum(x * (w_mse * x + b_mse - y) for x, y in zip(X, Y))
        grad_b = (2.0 / n_samples) * sum((w_mse * x + b_mse - y) for x, y in zip(X, Y))
        w_mse -= lr * grad_w
        b_mse -= lr * grad_b
        
    # Model 2: Trained by minimizing Gaussian NLL = (1 / (2*sigma^2)) * sum (w*x + b - y)^2
    # grad_w_nll = (1 / sigma^2) * sum(x * (w*x + b - y))
    # Notice: grad_w_nll = (n_samples / (2 * sigma^2)) * grad_w_mse
    # To take the exact identical parameter step: lr_nll = lr * (2 * sigma^2 / n_samples)
    w_nll, b_nll = 0.0, 0.0
    sigma_fixed = 0.5
    scale_factor = (2.0 * (sigma_fixed ** 2)) / n_samples  # converts NLL grad to MSE grad
    for _ in range(steps):
        grad_w_nll = (1.0 / (sigma_fixed ** 2)) * sum(x * (w_nll * x + b_nll - y) for x, y in zip(X, Y))
        grad_b_nll = (1.0 / (sigma_fixed ** 2)) * sum((w_nll * x + b_nll - y) for x, y in zip(X, Y))
        w_nll -= lr * (scale_factor * grad_w_nll)
        b_nll -= lr * (scale_factor * grad_b_nll)
        
    return (w_mse, b_mse), (w_nll, b_nll)


# ---------------------------------------------------------------------------
# 3. Section 3: Bernoulli NLL and Binary Cross-Entropy
# ---------------------------------------------------------------------------

def sigmoid(z: float) -> float:
    """Numerically stable sigmoid."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)


def binary_cross_entropy_mle(
    X: List[float],
    Y: List[int],
    w: float,
    b: float
) -> Tuple[float, float]:
    """Computes BCE loss and verifies it equals Negative Log-Likelihood."""
    eps = 1e-15
    bce = 0.0
    nll = 0.0
    
    for x, y in zip(X, Y):
        p = sigmoid(w * x + b)
        p = max(eps, min(1.0 - eps, p))
        # BCE formula
        bce -= y * math.log(p) + (1 - y) * math.log(1.0 - p)
        # Bernoulli likelihood P(y|x) = p^y * (1-p)^(1-y) => ln P = y*ln(p) + (1-y)*ln(1-p)
        log_prob = y * math.log(p) + (1 - y) * math.log(1.0 - p)
        nll -= log_prob
        
    n = len(X)
    return bce / n, nll / n


# ---------------------------------------------------------------------------
# 4. Main Execution Suite
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("TOPIC 5.3: MAXIMUM LIKELIHOOD ESTIMATION & AI LOSS FOUNDATIONS")
    print("=" * 72)

    # 1. Analytical vs Numerical MLE
    print("\n--- 1. Gaussian Distribution MLE: Analytical vs Numerical ---")
    true_mu = 14.2
    true_sigma = 3.6
    n_data = 1_000
    
    random.seed(42)
    data = [random.gauss(true_mu, true_sigma) for _ in range(n_data)]
    
    # Analytical formulas
    mu_analytic = sum(data) / n_data
    sigma_analytic = math.sqrt(sum((x - mu_analytic)**2 for x in data) / n_data)
    
    # Numerical gradient ascent
    mu_num, sigma_num = numerical_mle_gaussian(data, lr=0.0005, steps=2_000)
    
    print(f"Ground Truth Parameters:      mu = {true_mu:.4f}, sigma = {true_sigma:.4f}")
    print(f"Analytical Closed-Form MLE:   mu = {mu_analytic:.4f}, sigma = {sigma_analytic:.4f}")
    print(f"Numerical Gradient Ascent:    mu = {mu_num:.4f}, sigma = {sigma_num:.4f}")
    
    diff_mu = abs(mu_analytic - mu_num)
    diff_sigma = abs(sigma_analytic - sigma_num)
    print(f"Discrepancy (mu, sigma):      ({diff_mu:.2e}, {diff_sigma:.2e})")
    assert diff_mu < 0.01 and diff_sigma < 0.01, "Gaussian MLE solver mismatch!"
    print(">> SUCCESS: Numerical gradient ascent perfectly converged to the analytical MLE!")

    # 2. Linear Regression MSE vs Gaussian NLL
    print("\n--- 2. Equivalence Proof: Linear Regression MSE vs Gaussian NLL ---")
    (w_mse, b_mse), (w_nll, b_nll) = simulate_linear_mle_vs_mse()
    
    print(f"Trained via MSE Loss:         w = {w_mse:.5f}, b = {b_mse:.5f}")
    print(f"Trained via Gaussian NLL:     w = {w_nll:.5f}, b = {b_nll:.5f}")
    
    w_diff = abs(w_mse - w_nll)
    b_diff = abs(b_mse - b_nll)
    print(f"Parameter Discrepancy:        dw = {w_diff:.2e}, db = {b_diff:.2e}")
    assert w_diff < 1e-9 and b_diff < 1e-9, "MSE and NLL parameters do not match!"
    print(">> SUCCESS: Minimizing MSE is mathematically identical to Maximizing Gaussian Likelihood!")

    # 3. Binary Cross-Entropy vs Bernoulli Likelihood
    print("\n--- 3. Equivalence Proof: Cross-Entropy vs Bernoulli NLL ---")
    X_cls = [-2.0, -1.0, 0.0, 1.0, 2.0]
    Y_cls = [0, 0, 1, 1, 1]
    w_test, b_test = 1.5, -0.2
    
    bce_loss, nll_loss = binary_cross_entropy_mle(X_cls, Y_cls, w_test, b_test)
    print(f"Binary Cross-Entropy Loss:   {bce_loss:.8f}")
    print(f"Bernoulli Neg Log-Likelihood: {nll_loss:.8f}")
    
    loss_diff = abs(bce_loss - nll_loss)
    print(f"Loss Discrepancy:            {loss_diff:.2e}")
    assert loss_diff < 1e-12, "BCE and Bernoulli NLL do not match!"
    print(">> SUCCESS: Cross-Entropy is mathematically identical to Bernoulli Negative Log-Likelihood!")
    print("=" * 72)


if __name__ == "__main__":
    main()
