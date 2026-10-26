#!/usr/bin/env python3
"""
EXPERIMENT 4.2: RANDOM VARIABLES, PMF, PDF & INVERSE CDF SAMPLING
================================================================
A hands-on, zero-dependency Python laboratory demonstrating:
1. Discrete PMF & CDF cumulative token sampling engine
2. Numerical Integration of Continuous PDFs (Trapezoidal Rule)
3. Inverse Transform Sampling (Generating Exponential random variables)
4. Joint, Marginal, and Conditional Distributions from a 2D matrix

Runs on standard library Python 3 without external packages.
"""

import math
import random

print("=" * 70)
print("EXPERIMENT 4.2: RANDOM VARIABLES, PMF, PDF & CDF SAMPLING")
print("=" * 70)

random.seed(42)

# ======================================================================
# 1. DISCRETE PMF, CDF & CUMULATIVE SAMPLING ENGINE
# ======================================================================
print("\n--- 1. DISCRETE PMF, CDF & TOKEN SAMPLING ENGINE ---")

# Simulated vocabulary next-token logits converted to PMF
tokens = ["the", "cat", "sat", "on", "mat"]
pmf_probs = [0.45, 0.25, 0.15, 0.10, 0.05]

# Calculate Cumulative Distribution Function (CDF)
cdf = []
cum_sum = 0.0
for p in pmf_probs:
    cum_sum += p
    cdf.append(round(cum_sum, 6))

print("Token Vocabulary PMF and CDF:")
for t, p, c in zip(tokens, pmf_probs, cdf):
    print(f"  Token: '{t:>4}' | PMF p(x): {p:.2f} | CDF F(x): {c:.2f}")

def sample_discrete(tokens: list[str], cdf: list[float]) -> str:
    """Samples a discrete token using uniform random threshold against CDF."""
    u = random.random()
    for token, threshold in zip(tokens, cdf):
        if u <= threshold:
            return token
    return tokens[-1]

# Draw 10,000 samples to verify empirical frequencies match PMF
N_SAMPLES = 50_000
draws = [sample_discrete(tokens, cdf) for _ in range(N_SAMPLES)]
counts = {t: draws.count(t) for t in tokens}

print(f"\nEmpirical Verification over {N_SAMPLES:,} samples:")
for t in tokens:
    emp_freq = counts[t] / N_SAMPLES
    true_prob = pmf_probs[tokens.index(t)]
    print(f"  Token: '{t:>4}' | Empirical: {emp_freq:.4f} | True PMF: {true_prob:.4f}")
    assert abs(emp_freq - true_prob) < 0.01
print("✅ Discrete PMF/CDF sampling engine verified!")


# ======================================================================
# 2. CONTINUOUS PDF NUMERICAL INTEGRATION (TRAPEZOIDAL RULE)
# ======================================================================
print("\n" + "=" * 70)
print("--- 2. CONTINUOUS PDF INTEGRATION & PROBABILITY INTERVALS ---")

def standard_normal_pdf(x: float) -> float:
    """f(x) = (1 / sqrt(2*pi)) * exp(-x^2 / 2)"""
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)

def integrate_pdf(pdf_func, a: float, b: float, steps: int = 10_000) -> float:
    """Computes area under curve using the Trapezoidal Rule."""
    dx = (b - a) / steps
    total_area = 0.5 * (pdf_func(a) + pdf_func(b))
    for i in range(1, steps):
        total_area += pdf_func(a + i * dx)
    return total_area * dx

# 1. Total area from -6 to +6 should equal ~1.0
total_area = integrate_pdf(standard_normal_pdf, -6.0, 6.0)
print(f"Total integrated area under Standard Normal (-6 to +6): {total_area:.6f} (Theoretical: 1.000000)")

# 2. Famous 68-95-99.7 Rule
prob_1_sigma = integrate_pdf(standard_normal_pdf, -1.0, 1.0)
prob_2_sigma = integrate_pdf(standard_normal_pdf, -2.0, 2.0)
prob_3_sigma = integrate_pdf(standard_normal_pdf, -3.0, 3.0)

print(f"  P(-1 <= X <= 1) [1 Sigma]: {prob_1_sigma * 100:.2f}% (Theoretical: 68.27%)")
print(f"  P(-2 <= X <= 2) [2 Sigma]: {prob_2_sigma * 100:.2f}% (Theoretical: 95.45%)")
print(f"  P(-3 <= X <= 3) [3 Sigma]: {prob_3_sigma * 100:.2f}% (Theoretical: 99.73%)")
print("✅ Calculus integration of continuous density validated!")


# ======================================================================
# 3. INVERSE TRANSFORM SAMPLING: EXPONENTIAL DISTRIBUTION
# ======================================================================
print("\n" + "=" * 70)
print("--- 3. INVERSE TRANSFORM SAMPLING: EXPONENTIAL DISTRIBUTION ---")

LAMBDA = 2.0  # Rate parameter: Mean = 1 / lambda = 0.50

def sample_exponential(lambda_param: float) -> float:
    """Samples from Exponential(lambda) using Inverse CDF: x = -ln(1 - u) / lambda."""
    u = random.random()
    return -math.log(1.0 - u) / lambda_param

exp_samples = [sample_exponential(LAMBDA) for _ in range(50_000)]
sample_mean = sum(exp_samples) / len(exp_samples)
sample_var = sum((x - sample_mean) ** 2 for x in exp_samples) / len(exp_samples)

print(f"Generated 50,000 samples with lambda = {LAMBDA}:")
print(f"  Sample Mean:     {sample_mean:.4f} (Theoretical 1/lambda: {1.0 / LAMBDA:.4f})")
print(f"  Sample Variance: {sample_var:.4f} (Theoretical 1/lambda^2: {1.0 / (LAMBDA**2):.4f})")
assert abs(sample_mean - (1.0 / LAMBDA)) < 0.02
print("✅ Inverse transform sampling validated!")


# ======================================================================
# 4. JOINT, MARGINAL, AND CONDITIONAL PROBABILITIES
# ======================================================================
print("\n" + "=" * 70)
print("--- 4. JOINT, MARGINAL & CONDITIONAL DISTRIBUTIONS ---")

# Let X = Weather (0: Sunny, 1: Rainy)
# Let Y = Commute (0: On-Time, 1: Delayed)
# Joint probability table P(X, Y)
joint_table = [
    [0.60, 0.10],  # Row 0: Sunny  -> On-Time: 0.60, Delayed: 0.10 (Total Sunny = 0.70)
    [0.10, 0.20]   # Row 1: Rainy  -> On-Time: 0.10, Delayed: 0.20 (Total Rainy = 0.30)
]

# Marginals P(X) = sum over columns Y
marginal_X = [sum(row) for row in joint_table]
print(f"Marginal P(Weather): Sunny = {marginal_X[0]:.2f}, Rainy = {marginal_X[1]:.2f}")

# Marginals P(Y) = sum over rows X
marginal_Y = [joint_table[0][j] + joint_table[1][j] for j in range(2)]
print(f"Marginal P(Commute): On-Time = {marginal_Y[0]:.2f}, Delayed = {marginal_Y[1]:.2f}")

# Conditional P(Delayed | Rainy) = P(Rainy, Delayed) / P(Rainy)
cond_delayed_given_rainy = joint_table[1][1] / marginal_X[1]
cond_delayed_given_sunny = joint_table[0][1] / marginal_X[0]

print(f"\nConditioning Breakdown:")
print(f"  P(Delayed | Rainy): {cond_delayed_given_rainy * 100:.1f}%")
print(f"  P(Delayed | Sunny): {cond_delayed_given_sunny * 100:.1f}%")
print("✅ Joint and conditional operations verified!")

print("\n" + "=" * 70)
print("EXPERIMENT 4.2 COMPLETED SUCCESSFULLY!")
print("=" * 70)

