#!/usr/bin/env python3
"""
EXPERIMENT 4.3: CORE PROBABILITY DISTRIBUTIONS IN AI
===================================================
A hands-on, zero-dependency Python laboratory demonstrating:
1. Bernoulli & Categorical Token Sampling with Cross-Entropy Loss
2. Exact Gaussian Generation via the Box-Muller Transform
3. Multivariate Gaussian Generator via Cholesky Decomposition (Sigma = L @ L^T)
4. Empirical Verification of Sample Covariance Matrix Recovery
5. Simulation of VAE Reparameterization & DDPM Diffusion Noise Injection

Runs on standard library Python 3 without external packages.
"""

import math
import random

print("=" * 70)
print("EXPERIMENT 4.3: CORE DISTRIBUTIONS & MULTIVARIATE GAUSSIAN ENGINE")
print("=" * 70)

random.seed(42)

# ======================================================================
# 1. BERNOULLI & CATEGORICAL SAMPLING
# ======================================================================
print("\n--- 1. BERNOULLI & CATEGORICAL TOKEN SAMPLING ---")

def sample_bernoulli(p: float) -> int:
    return 1 if random.random() < p else 0

p_head = 0.70
bern_draws = [sample_bernoulli(p_head) for _ in range(10_000)]
emp_p = sum(bern_draws) / len(bern_draws)
print(f"Bernoulli(p=0.70) over 10,000 draws -> Empirical mean: {emp_p:.4f}")

# Categorical Token Sampling via Softmax Logits
def softmax(logits: list[float]) -> list[float]:
    max_l = max(logits)
    exps = [math.exp(l - max_l) for l in logits]
    sum_e = sum(exps)
    return [e / sum_e for e in exps]

logits = [2.0, 1.0, 0.1, -1.0]
probs = softmax(logits)
print(f"\nModel Output Logits: {logits}")
print(f"Softmax Categorical Probabilities: {[round(p, 4) for p in probs]}")


# ======================================================================
# 2. BOX-MULLER TRANSFORM: GAUSSIAN SAMPLES FROM UNIFORM NOISE
# ======================================================================
print("\n" + "=" * 70)
print("--- 2. EXACT GAUSSIAN GENERATION (BOX-MULLER TRANSFORM) ---")

def sample_standard_normal() -> float:
    """Generates exact N(0, 1) using the Box-Muller polar transform."""
    u1 = random.random()
    u2 = random.random()
    # Avoid log(0)
    while u1 == 0.0:
        u1 = random.random()
    r = math.sqrt(-2.0 * math.log(u1))
    theta = 2.0 * math.pi * u2
    return r * math.cos(theta)

normal_samples = [sample_standard_normal() for _ in range(50_000)]
mu_emp = sum(normal_samples) / len(normal_samples)
var_emp = sum((x - mu_emp) ** 2 for x in normal_samples) / len(normal_samples)

print(f"Generated 50,000 Box-Muller Gaussian draws:")
print(f"  Empirical Mean:     {mu_emp:.4f} (Theoretical: 0.0000)")
print(f"  Empirical Variance: {var_emp:.4f} (Theoretical: 1.0000)")
assert abs(mu_emp) < 0.02 and abs(var_emp - 1.0) < 0.02
print("✅ Standard Gaussian N(0, 1) generator validated!")


# ======================================================================
# 3. MULTIVARIATE GAUSSIAN GENERATOR VIA CHOLESKY FACTORIZATION
# ======================================================================
print("\n" + "=" * 70)
print("--- 3. MULTIVARIATE GAUSSIAN GENERATOR VIA CHOLESKY DECOMPOSITION ---")

# Target Mean Vector mu and Covariance Matrix Sigma:
# Sigma = [ [4.0, 1.8], [1.8, 1.0] ]
# Target Variances: Var(X) = 4.0 (std=2.0), Var(Y) = 1.0 (std=1.0)
# Target Covariance: Cov(X, Y) = 1.8 -> Correlation rho = 1.8 / (2.0 * 1.0) = 0.90 (High positive correlation!)

mu = [10.0, 5.0]
target_Sigma = [
    [4.0, 1.8],
    [1.8, 1.0]
]

# Cholesky Decomposition for 2x2: Sigma = L @ L^T where L is lower-triangular:
# [ [L11, 0], [L21, L22] ]
# L11 = sqrt(Sigma[0][0]) = sqrt(4.0) = 2.0
# L21 = Sigma[1][0] / L11 = 1.8 / 2.0 = 0.90
# L22 = sqrt(Sigma[1][1] - L21^2) = sqrt(1.0 - 0.81) = sqrt(0.19) ≈ 0.4358899

L11 = math.sqrt(target_Sigma[0][0])
L21 = target_Sigma[1][0] / L11
L22 = math.sqrt(target_Sigma[1][1] - (L21 ** 2))

L = [
    [L11, 0.0],
    [L21, L22]
]

print(f"Target Covariance Matrix Σ:\n  [ [{target_Sigma[0][0]:.1f}, {target_Sigma[0][1]:.1f}], [{target_Sigma[1][0]:.1f}, {target_Sigma[1][1]:.1f}] ]")
print(f"Computed Cholesky Factor L:\n  [ [{L[0][0]:.4f}, {L[0][1]:.4f}], [{L[1][0]:.4f}, {L[1][1]:.4f}] ]")

def sample_multivariate_gaussian(mu: list[float], L: list[list[float]]) -> list[float]:
    """Generates x = mu + L @ z where z ~ N(0, I)."""
    z0 = sample_standard_normal()
    z1 = sample_standard_normal()
    x0 = mu[0] + L[0][0] * z0 + L[0][1] * z1
    x1 = mu[1] + L[1][0] * z0 + L[1][1] * z1
    return [x0, x1]

# Draw 50,000 2D samples
N_2D = 50_000
samples_2D = [sample_multivariate_gaussian(mu, L) for _ in range(N_2D)]

# Compute Empirical Mean and Covariance
emp_mu0 = sum(s[0] for s in samples_2D) / N_2D
emp_mu1 = sum(s[1] for s in samples_2D) / N_2D

cov00 = sum((s[0] - emp_mu0) ** 2 for s in samples_2D) / (N_2D - 1)
cov11 = sum((s[1] - emp_mu1) ** 2 for s in samples_2D) / (N_2D - 1)
cov01 = sum((s[0] - emp_mu0) * (s[1] - emp_mu1) for s in samples_2D) / (N_2D - 1)

print(f"\nEmpirical Estimates from {N_2D:,} Multivariate Draws:")
print(f"  Recovered Mean:       [{emp_mu0:.4f}, {emp_mu1:.4f}] (Target: [10.0, 5.0])")
print(f"  Recovered Var(X):     {cov00:.4f} (Target: 4.0000)")
print(f"  Recovered Var(Y):     {cov11:.4f} (Target: 1.0000)")
print(f"  Recovered Cov(X, Y):  {cov01:.4f} (Target: 1.8000)")
assert abs(cov00 - 4.0) < 0.1 and abs(cov11 - 1.0) < 0.05 and abs(cov01 - 1.8) < 0.08
print("✅ Multivariate Gaussian generation perfectly verified!")


# ======================================================================
# 4. VAE REPARAMETERIZATION TRICK & DIFFUSION NOISE INJECTION
# ======================================================================
print("\n" + "=" * 70)
print("--- 4. VAE REPARAMETERIZATION & DIFFUSION NOISE SIMULATION ---")

# VAE: z = mu + sigma * epsilon
encoder_mu = [0.5, -1.2]
encoder_log_var = [-0.2, 0.4]  # log(sigma^2)
encoder_std = [math.exp(0.5 * lv) for lv in encoder_log_var]

epsilon = [sample_standard_normal(), sample_standard_normal()]
latent_z = [m + s * eps for m, s, eps in zip(encoder_mu, encoder_std, epsilon)]

print(f"VAE Latent Mean:   {encoder_mu}")
print(f"VAE Latent Std:    {[round(s, 4) for s in encoder_std]}")
print(f"Sampled Latent z:  {[round(val, 4) for val in latent_z]}")

# Diffusion Model Forward Step: x_t = sqrt(1 - beta) * x_{t-1} + sqrt(beta) * epsilon
beta_t = 0.02
x_prev = [1.0, 0.5, -0.8]  # Clean pixel values
noise = [sample_standard_normal() for _ in range(3)]
x_noisy = [math.sqrt(1.0 - beta_t) * x + math.sqrt(beta_t) * n for x, n in zip(x_prev, noise)]

print(f"\nDiffusion Step:")
print(f"  Input Clean Signal:  {x_prev}")
print(f"  Noised Output Signal:{[round(val, 4) for val in x_noisy]}")

print("\n" + "=" * 70)
print("EXPERIMENT 4.3 COMPLETED SUCCESSFULLY!")
print("=" * 70)

