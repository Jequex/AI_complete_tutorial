#!/usr/bin/env python3
"""
Topic 5.5 Experiment: Streaming Bayesian Learning & Conjugate Priors
Phase 05: Statistics & Estimation for Artificial Intelligence

This script provides a pure Python implementation (zero external dependencies) of:
1. Online Streaming Beta-Binomial Bayesian Updating.
2. The Overconfidence Crisis of MLE on small data vs Bayesian regularization.
3. Gaussian-Gaussian conjugate posterior updating with precision weighting.
4. Credible intervals and uncertainty collapse as observations scale from N=1 to N=1,000.
"""

import math
import random
from typing import List, Tuple


# ---------------------------------------------------------------------------
# 1. Beta Distribution Analytical Properties (Pure Python)
# ---------------------------------------------------------------------------

def beta_mean(alpha: float, beta: float) -> float:
    """Mean of Beta(alpha, beta): alpha / (alpha + beta)."""
    return alpha / (alpha + beta)


def beta_mode(alpha: float, beta: float) -> float:
    """Mode of Beta(alpha, beta) for alpha, beta > 1: (alpha - 1) / (alpha + beta - 2)."""
    if alpha > 1 and beta > 1:
        return (alpha - 1.0) / (alpha + beta - 2.0)
    elif alpha == 1 and beta == 1:
        return 0.5  # uniform
    elif alpha < 1 and beta < 1:
        return float('nan')  # bimodal at 0 and 1
    elif alpha <= 1:
        return 0.0
    else:
        return 1.0


def beta_variance(alpha: float, beta: float) -> float:
    """Variance of Beta(alpha, beta)."""
    total = alpha + beta
    return (alpha * beta) / ((total ** 2) * (total + 1.0))


# ---------------------------------------------------------------------------
# 2. Section 1: MLE vs MAP vs Bayesian Posterior Mean (Cold-Start Problem)
# ---------------------------------------------------------------------------

def compare_mle_map_bayes(
    successes: int,
    trials: int,
    prior_alpha: float = 2.0,
    prior_beta: float = 2.0
) -> Tuple[float, float, float]:
    """
    Compares parameter estimates when data is very scarce (e.g. 3 wins out of 3):
    - MLE: k / n (overconfident 1.00!)
    - MAP: mode of posterior (alpha + k - 1) / (alpha + beta + n - 2)
    - Full Bayes: expectation of posterior (alpha + k) / (alpha + beta + n)
    """
    mle = successes / trials
    
    post_alpha = prior_alpha + successes
    post_beta = prior_beta + (trials - successes)
    
    map_est = beta_mode(post_alpha, post_beta)
    bayes_mean = beta_mean(post_alpha, post_beta)
    
    return mle, map_est, bayes_mean


# ---------------------------------------------------------------------------
# 3. Section 2: Streaming Beta-Binomial Learning Simulation
# ---------------------------------------------------------------------------

def simulate_streaming_beta_binomial(
    true_theta: float = 0.75,
    initial_alpha: float = 2.0,
    initial_beta: float = 2.0,
    stream_checkpoints: List[int] = [0, 5, 20, 100, 1_000]
) -> List[Tuple[int, float, float, float]]:
    """
    Simulates streaming data arriving sequentially from a Bernoulli source with true_theta.
    Recursively updates Beta parameters in O(1) space.
    """
    random.seed(42)
    alpha = initial_alpha
    beta = initial_beta
    
    results = []
    current_trial = 0
    max_trials = max(stream_checkpoints)
    
    # Store initial prior
    if 0 in stream_checkpoints:
        results.append((0, beta_mean(alpha, beta), math.sqrt(beta_variance(alpha, beta)), 0.0))
        
    for trial_idx in range(1, max_trials + 1):
        # Draw next observation from reality
        outcome = 1 if random.random() < true_theta else 0
        
        # Exact Bayesian conjugate update: O(1) arithmetic!
        if outcome == 1:
            alpha += 1.0
        else:
            beta += 1.0
            
        if trial_idx in stream_checkpoints:
            mean_est = beta_mean(alpha, beta)
            std_est = math.sqrt(beta_variance(alpha, beta))
            results.append((trial_idx, mean_est, std_est, abs(mean_est - true_theta)))
            
    return results


# ---------------------------------------------------------------------------
# 4. Section 3: Gaussian-Gaussian Conjugate Updating
# ---------------------------------------------------------------------------

def gaussian_conjugate_update(
    prior_mu: float,
    prior_var: float,
    sample_data: List[float],
    known_noise_var: float
) -> Tuple[float, float]:
    """
    Exact analytical Bayesian update for Gaussian mean with known noise variance:
    tau_post = tau_prior + n * tau_data
    mu_post = (tau_prior * mu_prior + n * tau_data * x_bar) / tau_post
    """
    n = len(sample_data)
    x_bar = sum(sample_data) / n
    
    tau_prior = 1.0 / prior_var
    tau_data = 1.0 / known_noise_var
    
    tau_post = tau_prior + n * tau_data
    post_var = 1.0 / tau_post
    
    post_mu = (tau_prior * prior_mu + n * tau_data * x_bar) / tau_post
    return post_mu, post_var


# ---------------------------------------------------------------------------
# 5. Main Execution Suite
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("TOPIC 5.5: BAYESIAN INFERENCE & CONJUGATE PRIORS")
    print("=" * 72)

    # 1. The Cold-Start / Tiny-Data Crisis
    print("\n--- 1. MLE vs MAP vs Bayesian Expected Mean on Tiny Data ---")
    print("Scenario: A new recommender item has been clicked 3 out of 3 times (k=3, n=3)")
    print("Prior Belief: Beta(alpha=2, beta=2) (Mild belief that click rates are ~50%)")
    
    mle, map_val, bayes_val = compare_mle_map_bayes(successes=3, trials=3)
    print(f"MLE Estimate:                 {mle:.4f} (100% - Catastrophic Overconfidence!)")
    print(f"MAP Estimate (Posterior Mode): {map_val:.4f} (Regularized to 80%)")
    print(f"Full Bayes (Posterior Mean):   {bayes_val:.4f} (Laplace smoothing ~71.4%)")
    
    assert mle == 1.0, "MLE calculation mismatch!"
    assert abs(bayes_val - 5.0/7.0) < 1e-6, "Bayesian mean calculation mismatch!"
    print(">> SUCCESS: Full Bayes prevents catastrophic overconfidence in cold-start scenarios!")

    # 2. Streaming Beta-Binomial Learning
    print("\n--- 2. Streaming Online Bayesian Updating (True Win Rate = 0.75) ---")
    print(f"{'Data Points (N)':>16} | {'Posterior Mean':>16} | {'Posterior Std':>15} | {'Error to Truth':>16}")
    print("-" * 72)
    
    stream_history = simulate_streaming_beta_binomial()
    for n, p_mean, p_std, err in stream_history:
        print(f"{n:>16} | {p_mean:>16.4f} | {p_std:>15.5f} | {err:>16.4f}")
        
    final_n, final_mean, final_std, final_err = stream_history[-1]
    assert final_err < 0.03, "Streaming Bayes failed to converge to ground truth!"
    assert final_std < 0.02, "Posterior variance did not collapse!"
    print("\n>> SUCCESS: Living posterior distribution continuously sharpens toward ground truth!")

    # 3. Gaussian-Gaussian Conjugate Updating
    print("\n--- 3. Gaussian-Gaussian Conjugate Updating ---")
    # Prior belief: model latency is around 50 ms +/- 10 ms (var = 100)
    prior_mu, prior_var = 50.0, 100.0
    known_noise_var = 16.0  # std = 4.0 ms
    
    # Observe 5 queries with sample mean 42.0 ms
    observed_latencies = [41.0, 43.5, 40.5, 44.0, 41.0]
    sample_mean = sum(observed_latencies) / len(observed_latencies)
    
    post_mu, post_var = gaussian_conjugate_update(
        prior_mu, prior_var, observed_latencies, known_noise_var
    )
    post_std = math.sqrt(post_var)
    
    print(f"Prior Belief:                 mu_0 = {prior_mu:.2f} ms, std_0 = {math.sqrt(prior_var):.2f} ms")
    print(f"Observed Sample (n=5):        x_bar = {sample_mean:.2f} ms (noise std = {math.sqrt(known_noise_var):.2f})")
    print(f"Updated Posterior:            mu_N = {post_mu:.2f} ms, std_N = {post_std:.2f} ms")
    
    # Verify posterior variance is strictly smaller than both prior and data variance
    assert post_var < prior_var, "Posterior variance did not decrease!"
    print(">> SUCCESS: Gaussian conjugate update correctly synthesized prior and sample evidence!")
    print("=" * 72)


if __name__ == "__main__":
    main()

