#!/usr/bin/env python3
"""
Topic 5.1 Experiment: Populations, Samples, Estimators & Bessel's Correction
Phase 05: Statistics & Estimation for Artificial Intelligence

This script provides a pure Python implementation (zero external dependencies) of:
1. Monte Carlo empirical verification of Bessel's Correction:
   Comparing naive sample variance (1/n) vs Bessel-corrected variance (1/(n-1)).
2. Empirical proof that E[S_n^2] = (n - 1)/n * sigma^2.
3. Standard Error of the mean scaling test: verifying SE(X_bar) = sigma / sqrt(n).
4. Mini-batch gradient estimator in machine learning: verifying unbiasedness E[g_B] = g.
"""

import math
import random
from typing import List, Tuple


# ---------------------------------------------------------------------------
# 1. Statistical Functions (Pure Python)
# ---------------------------------------------------------------------------

def compute_mean(data: List[float]) -> float:
    """Computes the sample mean X_bar = (1/n) * sum(x_i)."""
    return sum(data) / len(data)


def compute_naive_variance(data: List[float]) -> float:
    """Computes biased sample variance with 1/n divisor."""
    n = len(data)
    mean = compute_mean(data)
    return sum((x - mean) ** 2 for x in data) / n


def compute_unbiased_variance(data: List[float]) -> float:
    """Computes Bessel-corrected sample variance with 1/(n - 1) divisor."""
    n = len(data)
    if n < 2:
        return 0.0
    mean = compute_mean(data)
    return sum((x - mean) ** 2 for x in data) / (n - 1)


# ---------------------------------------------------------------------------
# 2. Section 1: Empirical Verification of Bessel's Correction
# ---------------------------------------------------------------------------

def simulate_bessel_correction(
    true_mu: float = 10.0,
    true_sigma: float = 2.0,
    sample_size: int = 5,
    num_trials: int = 50_000
) -> Tuple[float, float, float]:
    """
    Draws num_trials samples of size sample_size from N(true_mu, true_sigma^2).
    Averages both naive (1/n) and unbiased (1/(n-1)) variance estimates.
    """
    true_var = true_sigma ** 2
    naive_vars = []
    unbiased_vars = []
    
    random.seed(42)
    for _ in range(num_trials):
        # Draw sample of size n from Gaussian population
        sample = [random.gauss(true_mu, true_sigma) for _ in range(sample_size)]
        naive_vars.append(compute_naive_variance(sample))
        unbiased_vars.append(compute_unbiased_variance(sample))
        
    avg_naive = sum(naive_vars) / num_trials
    avg_unbiased = sum(unbiased_vars) / num_trials
    expected_biased = ((sample_size - 1) / sample_size) * true_var
    
    return avg_naive, avg_unbiased, expected_biased


# ---------------------------------------------------------------------------
# 3. Section 2: Standard Error Scaling Test (SE = sigma / sqrt(n))
# ---------------------------------------------------------------------------

def simulate_standard_error_scaling(
    true_sigma: float = 2.0,
    sample_sizes: List[int] = [4, 16, 64, 256],
    num_trials: int = 20_000
) -> List[Tuple[int, float, float]]:
    """
    Tests how the empirical standard deviation of sample means
    scales with sample size n, verifying SE = sigma / sqrt(n).
    """
    results = []
    for n in sample_sizes:
        sample_means = []
        for _ in range(num_trials):
            sample = [random.gauss(0.0, true_sigma) for _ in range(n)]
            sample_means.append(compute_mean(sample))
            
        empirical_se = math.sqrt(compute_unbiased_variance(sample_means))
        theoretical_se = true_sigma / math.sqrt(n)
        results.append((n, empirical_se, theoretical_se))
    return results


# ---------------------------------------------------------------------------
# 4. Section 3: Mini-Batch Gradient Estimator in Machine Learning
# ---------------------------------------------------------------------------

def simulate_minibatch_gradient(
    dataset_size: int = 10_000,
    batch_size: int = 32,
    num_batches: int = 5_000
) -> Tuple[float, float, float]:
    """
    Simulates a 1D loss function L(w) = (1/N) * sum_i (w * x_i - y_i)^2.
    Computes true full-dataset gradient, and averages mini-batch gradients.
    """
    random.seed(1337)
    # Generate synthetic dataset: y = 3*x + noise
    X = [random.uniform(-2.0, 2.0) for _ in range(dataset_size)]
    Y = [3.0 * x + random.gauss(0.0, 0.5) for x in X]
    
    # Evaluate at current weight guess w = 1.0 (true w = 3.0)
    w = 1.0
    
    # True gradient on full dataset: (2/N) * sum(x_i * (w * x_i - y_i))
    individual_grads = [2.0 * x * (w * x - y) for x, y in zip(X, Y)]
    true_gradient = sum(individual_grads) / dataset_size
    
    # Sample random mini-batches
    batch_grads = []
    for _ in range(num_batches):
        batch_indices = [random.randint(0, dataset_size - 1) for _ in range(batch_size)]
        batch_g = sum(individual_grads[idx] for idx in batch_indices) / batch_size
        batch_grads.append(batch_g)
        
    avg_batch_gradient = sum(batch_grads) / num_batches
    batch_grad_variance = compute_unbiased_variance(batch_grads)
    
    return true_gradient, avg_batch_gradient, batch_grad_variance


# ---------------------------------------------------------------------------
# 5. Main Execution Suite
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("TOPIC 5.1: POPULATIONS, SAMPLES, ESTIMATORS & BESSEL'S CORRECTION")
    print("=" * 72)
    
    # Part 1: Bessel's Correction
    true_mu = 10.0
    true_sigma = 2.0
    true_var = true_sigma ** 2  # 4.0
    sample_size = 5
    num_trials = 50_000
    
    print("\n--- 1. Empirical Proof of Bessel's Correction ---")
    print(f"Population Parameters:   mu = {true_mu}, sigma^2 = {true_var:.4f}")
    print(f"Small Sample Size:       n = {sample_size}")
    print(f"Monte Carlo Simulations: {num_trials:,} trials")
    
    avg_naive, avg_unbiased, expected_biased = simulate_bessel_correction(
        true_mu, true_sigma, sample_size, num_trials
    )
    
    print(f"\nEmpirical Naive Variance E[S_n^2]:       {avg_naive:.4f}")
    print(f"Theoretical Biased Value ((n-1)/n)*var:  {expected_biased:.4f}")
    print(f"Empirical Bessel Variance E[S_(n-1)^2]:  {avg_unbiased:.4f}")
    print(f"True Population Variance sigma^2:        {true_var:.4f}")
    
    bias_ratio = avg_naive / true_var
    print(f"\nObserved Ratio (Naive / True): {bias_ratio:.4f} (Expected: {4/5:.4f} = 0.8000)")
    assert abs(avg_naive - expected_biased) < 0.02, "Biased variance formula mismatch!"
    assert abs(avg_unbiased - true_var) < 0.02, "Bessel's correction failed to restore unbiasedness!"
    print(">> SUCCESS: Bessel's correction completely eliminates estimator bias!")

    # Part 2: Standard Error Scaling
    print("\n--- 2. Standard Error of the Mean: SE = sigma / sqrt(n) ---")
    print(f"Population Sigma: {true_sigma}")
    print(f"{'Sample Size n':>13} | {'Empirical SE':>14} | {'Theoretical SE':>16} | {'Discrepancy':>12}")
    print("-" * 62)
    
    se_results = simulate_standard_error_scaling(true_sigma)
    for n, emp_se, theo_se in se_results:
        diff = abs(emp_se - theo_se)
        print(f"{n:>13} | {emp_se:>14.5f} | {theo_se:>16.5f} | {diff:>12.2e}")
        assert diff < 0.03, f"Standard error scaling failed at n={n}!"
        
    print("\n>> SUCCESS: Uncertainty shrinks at exact rate 1 / sqrt(n)!")

    # Part 3: Mini-Batch Gradient Estimator
    print("\n--- 3. Mini-Batch Gradient Estimator in Deep Learning ---")
    dataset_size = 10_000
    batch_size = 32
    
    true_g, avg_batch_g, g_var = simulate_minibatch_gradient(dataset_size, batch_size)
    print(f"Full Dataset Gradient (N = {dataset_size}): {true_g:.6f}")
    print(f"Mean Mini-Batch Gradient (B = {batch_size}):    {avg_batch_g:.6f}")
    print(f"Batch Gradient Variance:              {g_var:.6f}")
    
    grad_bias = abs(avg_batch_g - true_g)
    print(f"Observed Gradient Bias:               {grad_bias:.2e}")
    assert grad_bias < 0.05, "Mini-batch gradient is not unbiased!"
    print(">> SUCCESS: Mini-batch gradient is an unconditionally unbiased estimator of the full loss gradient!")
    print("=" * 72)


if __name__ == "__main__":
    main()

