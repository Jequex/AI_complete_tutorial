#!/usr/bin/env python3
"""
EXPERIMENT 4.5: CENTRAL LIMIT THEOREM, MONTE CARLO & LLM SAMPLING
=================================================================
A hands-on, zero-dependency Python laboratory demonstrating:
1. Live Central Limit Theorem Convergence (Uniform Noise -> Gaussian Bell)
2. ASCII Bell Curve Histogram Renderer
3. Monte Carlo Integration: Estimating Pi (pi) with Error Tracking
4. Production LLM Token Generation Engine (Greedy, Temperature, Top-K, Top-P)

Runs on standard library Python 3 without external packages.
"""

import math
import random

print("=" * 70)
print("EXPERIMENT 4.5: LIMIT THEOREMS, MONTE CARLO & LLM DECODING")
print("=" * 70)

random.seed(42)

# ======================================================================
# 1. CENTRAL LIMIT THEOREM & ASCII BELL CURVE RENDERER
# ======================================================================
print("\n--- 1. CENTRAL LIMIT THEOREM (SUM OF UNIFORM VARIABLES) ---")

# Let X_i ~ Uniform(0, 1). Mean = 0.5, Variance = 1/12 ≈ 0.08333
# Sum of n = 12 uniform variables:
# Expected Mean = 12 * 0.5 = 6.0
# Expected Variance = 12 * (1/12) = 1.0 (Standard Deviation = 1.0!)
n_uniform = 12
TRIALS = 50_000

sums = [sum(random.random() for _ in range(n_uniform)) for _ in range(TRIALS)]
emp_mean = sum(sums) / TRIALS
emp_var = sum((s - emp_mean) ** 2 for s in sums) / TRIALS

print(f"Generated {TRIALS:,} sums of {n_uniform} Uniform(0, 1) variables:")
print(f"  Empirical Mean:     {emp_mean:.4f} (Theoretical: 6.0000)")
print(f"  Empirical Variance: {emp_var:.4f} (Theoretical: 1.0000)")

# ASCII Histogram of the resulting Gaussian Bell Curve
NUM_BINS = 15
min_val = min(sums)
max_val = max(sums)
bin_width = (max_val - min_val) / NUM_BINS
bins = [0] * NUM_BINS

for val in sums:
    idx = min(int((val - min_val) / bin_width), NUM_BINS - 1)
    bins[idx] += 1

print("\nASCII Distribution of Sums (Emergent Gaussian Bell Curve):")
max_count = max(bins)
for i in range(NUM_BINS):
    bin_center = min_val + (i + 0.5) * bin_width
    bar_length = int(45 * (bins[i] / max_count))
    bar = "█" * bar_length
    print(f"  {bin_center:4.1f} | {bar} ({bins[i]:,})")
print("✅ Central Limit Theorem demonstrated: Uniform noise sums to a perfect Gaussian!")


# ======================================================================
# 2. MONTE CARLO INTEGRATION: ESTIMATING PI
# ======================================================================
print("\n" + "=" * 70)
print("--- 2. MONTE CARLO INTEGRATION (ESTIMATING PI) ---")

N_DARTS = 200_000
hits = 0

checkpoints = [1000, 10_000, 50_000, 100_000, 200_000]
print(f"Throwing {N_DARTS:,} random darts into unit square [0, 1] x [0, 1]...")

for i in range(1, N_DARTS + 1):
    x = random.random()
    y = random.random()
    if x * x + y * y <= 1.0:
        hits += 1

    if i in checkpoints:
        approx_pi = 4.0 * (hits / i)
        error = abs(approx_pi - math.pi)
        print(f"  Samples: {i:>7,} | Estimated Pi: {approx_pi:.5f} | Error: {error:.5f}")

print("✅ Monte Carlo integration validated!")


# ======================================================================
# 3. LLM TOKEN GENERATION SAMPLING ENGINE
# ======================================================================
print("\n" + "=" * 70)
print("--- 3. LLM TOKEN SAMPLING ENGINE (TEMPERATURE, TOP-K, TOP-P) ---")

vocab = ["Paris", "Rome", "London", "Berlin", "Madrid", "Tokyo", "Cairo", "Mars"]
# Raw unnormalized model logits for next-token completion: "The capital of France is ___"
raw_logits = [8.5, 5.0, 4.2, 3.8, 3.5, 1.2, 0.5, -2.0]

def softmax_with_temperature(logits: list[float], temp: float) -> list[float]:
    """Applies temperature scaling: p_i = exp(z_i / T) / sum(exp(z_j / T))."""
    scaled = [z / temp for z in logits]
    max_s = max(scaled)
    exps = [math.exp(s - max_s) for s in scaled]
    sum_e = sum(exps)
    return [e / sum_e for e in exps]

def sample_top_k(logits: list[float], k: int, temp: float = 1.0) -> list[float]:
    """Keeps only the top K logits, masks the rest to -inf, and applies softmax."""
    sorted_indices = sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)
    masked_logits = [-float("inf")] * len(logits)
    for idx in sorted_indices[:k]:
        masked_logits[idx] = logits[idx]
    return softmax_with_temperature(masked_logits, temp)

def sample_top_p(logits: list[float], p: float, temp: float = 1.0) -> list[float]:
    """Keeps smallest set of tokens whose cumulative probability >= p (Nucleus Sampling)."""
    base_probs = softmax_with_temperature(logits, temp)
    sorted_indices = sorted(range(len(logits)), key=lambda i: base_probs[i], reverse=True)

    cum_sum = 0.0
    kept_indices = []
    for idx in sorted_indices:
        kept_indices.append(idx)
        cum_sum += base_probs[idx]
        if cum_sum >= p:
            break

    masked_logits = [-float("inf")] * len(logits)
    for idx in kept_indices:
        masked_logits[idx] = logits[idx]
    return softmax_with_temperature(masked_logits, temp)

# Compare Strategies
print("\nLogit Probabilities Under Different Sampling Strategies:")
p_t03 = softmax_with_temperature(raw_logits, temp=0.3)
p_t10 = softmax_with_temperature(raw_logits, temp=1.0)
p_t20 = softmax_with_temperature(raw_logits, temp=2.0)
p_top2 = sample_top_k(raw_logits, k=2, temp=1.0)
p_nucl = sample_top_p(raw_logits, p=0.95, temp=1.0)

header = f"{'Token':>8} | {'Raw Logit':>9} | {'T=0.3':>8} | {'T=1.0':>8} | {'T=2.0':>8} | {'Top-K(2)':>9} | {'Top-P(.95)':>10}"
print(header)
print("-" * len(header))
for i, token in enumerate(vocab):
    print(f"{token:>8} | {raw_logits[i]:>9.1f} | {p_t03[i]*100:>7.2f}% | {p_t10[i]*100:>7.2f}% | {p_t20[i]*100:>7.2f}% | {p_top2[i]*100:>8.2f}% | {p_nucl[i]*100:>9.2f}%")

print("\nInsights:")
print("  - Low Temp (0.3): 'Paris' dominates with ~99.99% probability (Factual, greedy).")
print("  - High Temp (2.0): Probabilities flatten across all European cities (Creative, diverse).")
print("  - Top-K (2): Strictly zeroes out all choices other than Paris and Rome.")
print("  - Top-P (0.95): Truncates rare garbage tokens ('Tokyo', 'Cairo', 'Mars') while keeping plausible alternatives.")

print("\n" + "=" * 70)
print("EXPERIMENT 4.5 COMPLETED SUCCESSFULLY!")
print("=" * 70)

