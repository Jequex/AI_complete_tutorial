#!/usr/bin/env python3
"""
EXPERIMENT 4.4: EXPECTATION, VARIANCE, COVARIANCE & BATCH NORMALIZATION
======================================================================
A hands-on, zero-dependency Python laboratory demonstrating:
1. Empirical Proof of Linearity of Expectation under Correlated Variables
2. Verification of Variance of Sums: Var(X + Y) = Var(X) + Var(Y) + 2*Cov(X, Y)
3. Pure Python Sample Covariance Matrix & Pearson Correlation Matrix Calculator
4. Full Batch Normalization Forward Pass from First Principles

Runs on standard library Python 3 without external packages.
"""

import math
import random

print("=" * 70)
print("EXPERIMENT 4.4: EXPECTATION, COVARIANCE & BATCH NORMALIZATION")
print("=" * 70)

random.seed(42)

# ======================================================================
# 1. LINEARITY OF EXPECTATION UNDER CORRELATED VARIABLES
# ======================================================================
print("\n--- 1. LINEARITY OF EXPECTATION UNDER DEPENDENT VARIABLES ---")

N = 100_000
# Generate strongly correlated variables X and Y
X_samples = [random.uniform(0.0, 10.0) for _ in range(N)]
# Y depends directly on X: Y = 2*X + noise
Y_samples = [2.0 * x + random.uniform(-1.0, 1.0) for x in X_samples]

# Linear combination: Z = 3*X + 4*Y - 5
Z_samples = [3.0 * x + 4.0 * y - 5.0 for x, y in zip(X_samples, Y_samples)]

emp_E_X = sum(X_samples) / N
emp_E_Y = sum(Y_samples) / N
emp_E_Z = sum(Z_samples) / N
theoretical_E_Z = 3.0 * emp_E_X + 4.0 * emp_E_Y - 5.0

print(f"Sample size: {N:,}")
print(f"  Empirical E[X]:                  {emp_E_X:.4f}")
print(f"  Empirical E[Y]:                  {emp_E_Y:.4f}")
print(f"  Direct Measured E[3X + 4Y - 5]:   {emp_E_Z:.4f}")
print(f"  Linearity Formula 3E[X]+4E[Y]-5: {theoretical_E_Z:.4f}")
assert abs(emp_E_Z - theoretical_E_Z) < 1e-4
print("✅ Linearity of expectation confirmed even under strong correlation!")


# ======================================================================
# 2. VARIANCE OF SUMS AND COVARIANCE EXPANSION
# ======================================================================
print("\n" + "=" * 70)
print("--- 2. VARIANCE OF SUMS & COVARIANCE VERIFICATION ---")

# Let S = X + Y
S_samples = [x + y for x, y in zip(X_samples, Y_samples)]

var_X = sum((x - emp_E_X) ** 2 for x in X_samples) / (N - 1)
var_Y = sum((y - emp_E_Y) ** 2 for y in Y_samples) / (N - 1)
var_S = sum((s - (emp_E_X + emp_E_Y)) ** 2 for s in S_samples) / (N - 1)

# Covariance Cov(X, Y)
cov_XY = sum((x - emp_E_X) * (y - emp_E_Y) for x, y in zip(X_samples, Y_samples)) / (N - 1)
var_S_formula = var_X + var_Y + 2.0 * cov_XY

# Pearson Correlation rho
std_X = math.sqrt(var_X)
std_Y = math.sqrt(var_Y)
rho_XY = cov_XY / (std_X * std_Y)

print(f"  Sample Variance Var(X):       {var_X:.4f}")
print(f"  Sample Variance Var(Y):       {var_Y:.4f}")
print(f"  Sample Covariance Cov(X, Y):  {cov_XY:.4f}")
print(f"  Pearson Correlation rho(X, Y):{rho_XY:.4f} (Very high linear correlation!)")
print(f"\n  Direct Measured Var(X + Y):   {var_S:.4f}")
print(f"  Formula Var(X)+Var(Y)+2Cov:   {var_S_formula:.4f}")
assert abs(var_S - var_S_formula) < 1e-3
print("✅ Variance of sum formula verified!")


# ======================================================================
# 3. HIGH-DIMENSIONAL COVARIANCE MATRIX ENGINE
# ======================================================================
print("\n" + "=" * 70)
print("--- 3. 3D SAMPLE COVARIANCE & CORRELATION MATRIX ENGINE ---")

# 3D dataset: 5 samples, 3 features
data_3D = [
    [1.0, 2.0, 5.0],
    [2.0, 4.0, 4.0],
    [3.0, 6.0, 3.0],
    [4.0, 8.0, 2.0],
    [5.0, 10.0, 1.0]
]
M = len(data_3D)     # 5 samples
D = len(data_3D[0])  # 3 dimensions

# Means per feature
means = [sum(data_3D[i][j] for i in range(M)) / M for j in range(D)]
print(f"Feature Means: {[round(m, 2) for m in means]}")

# Compute Covariance Matrix
cov_matrix = [[0.0] * D for _ in range(D)]
for j1 in range(D):
    for j2 in range(D):
        cov = sum((data_3D[i][j1] - means[j1]) * (data_3D[i][j2] - means[j2]) for i in range(M)) / (M - 1)
        cov_matrix[j1][j2] = cov

print("\nComputed Sample Covariance Matrix Σ:")
for row in cov_matrix:
    print("  [" + ", ".join(f"{val:>7.2f}" for val in row) + "]")

# Compute Pearson Correlation Matrix
corr_matrix = [[0.0] * D for _ in range(D)]
for j1 in range(D):
    for j2 in range(D):
        s1 = math.sqrt(cov_matrix[j1][j1])
        s2 = math.sqrt(cov_matrix[j2][j2])
        corr_matrix[j1][j2] = cov_matrix[j1][j2] / (s1 * s2)

print("\nComputed Pearson Correlation Matrix P:")
for row in corr_matrix:
    print("  [" + ", ".join(f"{val:>7.2f}" for val in row) + "]")
print("  Note: Feature 1 & 2 have perfect +1.00 correlation; Feature 1 & 3 have perfect -1.00 correlation!")


# ======================================================================
# 4. BATCH NORMALIZATION FROM FIRST PRINCIPLES
# ======================================================================
print("\n" + "=" * 70)
print("--- 4. BATCH NORMALIZATION FORWARD PASS ---")

# Batch of 4 samples, 2 hidden neurons
batch_X = [
    [10.0, 100.0],
    [12.0, 105.0],
    [8.0,   95.0],
    [14.0, 120.0]
]
B = len(batch_X)
features = len(batch_X[0])
EPSILON = 1e-5

# 1. Batch Mean per feature
batch_mu = [sum(batch_X[i][j] for i in range(B)) / B for j in range(features)]

# 2. Batch Variance per feature
batch_var = [sum((batch_X[i][j] - batch_mu[j]) ** 2 for i in range(B)) / B for j in range(features)]

# 3. Normalize: x_hat = (x - mu) / sqrt(var + eps)
# Learnable scale (gamma=1.0) and shift (beta=0.0)
normalized_batch = []
for i in range(B):
    row_norm = []
    for j in range(features):
        x_hat = (batch_X[i][j] - batch_mu[j]) / math.sqrt(batch_var[j] + EPSILON)
        row_norm.append(round(x_hat, 4))
    normalized_batch.append(row_norm)

print(f"Batch Means:     {[round(m, 2) for m in batch_mu]}")
print(f"Batch Variances: {[round(v, 2) for v in batch_var]}")
print("\nNormalized Batch Activations (Mean ≈ 0, Std ≈ 1):")
for row in normalized_batch:
    print(f"  {row}")

# Check that normalized batch has mean 0
norm_mu = [sum(normalized_batch[i][j] for i in range(B)) / B for j in range(features)]
print(f"Post-Normalization Means: {[round(m, 4) for m in norm_mu]}")
assert all(abs(m) < 1e-3 for m in norm_mu)
print("✅ Batch Normalization verified!")

print("\n" + "=" * 70)
print("EXPERIMENT 4.4 COMPLETED SUCCESSFULLY!")
print("=" * 70)

