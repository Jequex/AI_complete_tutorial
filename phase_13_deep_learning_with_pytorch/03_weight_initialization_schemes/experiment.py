#!/usr/bin/env python3
"""
Topic 13.3 Experiment: Weight Initialization Schemes & Deep Variance Propagation
Author: AI Complete Tutorial
Dependencies: Pure Python 3 Standard Library (math, random, sys)

This experiment demonstrates the mathematics of variance propagation through deep networks:
1. Mathematical Proof of Variance Propagation:
   Var(y) = n_in * Var(w) * Var(x) for linear layers.
   Under ReLU activation, half the distribution is zeroed, reducing variance by 1/2.
2. 50-Layer Deep Network Simulation comparing:
   - Naive Gaussian Initialization (std = 0.05): rapid activation collapse to 0.0.
   - Xavier / Glorot Initialization (std = sqrt(2 / (n_in + n_out))): fails under ReLU, shrinking exponentially.
   - He / Kaiming Initialization (std = sqrt(2 / n_in)): perfectly stabilizes activation variance ~ 1.0 through 50 layers.
3. Tanh Activation Simulation:
   - Xavier stabilizes variance under symmetric zero-centered activations.
"""

import math
import random
import sys

def set_seed(seed=42):
    random.seed(seed)


def compute_mean_and_var(matrix):
    """Computes empirical mean and variance over 2D matrix (B x D)."""
    flat = [val for row in matrix for val in row]
    n = len(flat)
    mean = sum(flat) / n
    var = sum((x - mean) ** 2 for x in flat) / n
    return mean, var


def matmul(A, B):
    """Matrix multiplication A (B x K) @ B (K x M) -> (B x M)."""
    rows_A = len(A)
    cols_A = len(A[0])
    cols_B = len(B[0])
    out = [[0.0 for _ in range(cols_B)] for _ in range(rows_A)]
    for i in range(rows_A):
        for k in range(cols_A):
            a_ik = A[i][k]
            if a_ik != 0.0:
                for j in range(cols_B):
                    out[i][j] += a_ik * B[k][j]
    return out


def relu(matrix):
    return [[max(0.0, val) for val in row] for row in matrix]


def tanh(matrix):
    return [[math.tanh(val) for val in row] for row in matrix]


# ==============================================================================
# Initialization Generators
# ==============================================================================

def init_naive(in_dim: int, out_dim: int, std: float = 0.05):
    """Naive Gaussian initialization with fixed standard deviation."""
    return [[random.gauss(0.0, std) for _ in range(out_dim)] for _ in range(in_dim)]


def init_xavier(in_dim: int, out_dim: int):
    """Xavier / Glorot normal: std = sqrt(2.0 / (in_dim + out_dim))."""
    std = math.sqrt(2.0 / (in_dim + out_dim))
    return [[random.gauss(0.0, std) for _ in range(out_dim)] for _ in range(in_dim)]


def init_kaiming(in_dim: int, out_dim: int):
    """He / Kaiming normal for ReLU: std = sqrt(2.0 / in_dim)."""
    std = math.sqrt(2.0 / in_dim)
    return [[random.gauss(0.0, std) for _ in range(out_dim)] for _ in range(in_dim)]


# ==============================================================================
# Forward Propagation across 50 Layers
# ==============================================================================

def run_deep_network_simulation(init_func, activation="relu", num_layers: int = 50, dim: int = 256, batch_size: int = 64):
    # Initial input: zero-mean, unit-variance Gaussian
    x = [[random.gauss(0.0, 1.0) for _ in range(dim)] for _ in range(batch_size)]
    layer_variances = []

    m0, v0 = compute_mean_and_var(x)
    layer_variances.append(v0)

    for layer_idx in range(1, num_layers + 1):
        w = init_func(dim, dim)
        z = matmul(x, w)
        if activation == "relu":
            x = relu(z)
        elif activation == "tanh":
            x = tanh(z)
        elif activation == "linear":
            x = z
        else:
            raise ValueError(f"Unknown activation: {activation}")

        _, var = compute_mean_and_var(x)
        layer_variances.append(var)

    return layer_variances


def main():
    set_seed(42)
    print("\n" + "#" * 78)
    print("RUNNING TOPIC 13.3 EXPERIMENT: 50-LAYER DEEP NETWORK INITIALIZATION")
    print("#" * 78 + "\n")

    num_layers = 50
    dim = 128
    batch_size = 32

    print(f"Network Topology: 50 Hidden Layers | Dimension: {dim} -> {dim} | Batch Size: {batch_size}")
    print("=" * 78)
    print("EXPERIMENT 1: ReLU Network Variance Propagation Across 50 Layers")
    print("=" * 78)

    # 1. Naive Small Init (std = 0.05)
    naive_vars = run_deep_network_simulation(lambda din, dout: init_naive(din, dout, std=0.05),
                                            activation="relu", num_layers=num_layers, dim=dim, batch_size=batch_size)

    # 2. Xavier Init (std = sqrt(2 / (2*dim)) = sqrt(1 / dim))
    xavier_vars = run_deep_network_simulation(init_xavier,
                                              activation="relu", num_layers=num_layers, dim=dim, batch_size=batch_size)

    # 3. He / Kaiming Init (std = sqrt(2 / dim))
    kaiming_vars = run_deep_network_simulation(init_kaiming,
                                               activation="relu", num_layers=num_layers, dim=dim, batch_size=batch_size)

    header = f"{'Layer':<8} | {'Naive (std=0.05)':<18} | {'Xavier (Glorot)':<18} | {'He / Kaiming (ReLU)':<18}"
    print(header)
    print("─" * len(header))

    checkpoints = [0, 1, 5, 10, 20, 30, 40, 50]
    for l in checkpoints:
        print(f"Layer {l:<2d} | {naive_vars[l]:<18.6e} | {xavier_vars[l]:<18.6e} | {kaiming_vars[l]:<18.4f}")

    print("=" * 78)
    print("ANALYSIS OF RELU RESULTS:")
    print(f"1. Naive (0.05) Variance at Layer 50: {naive_vars[50]:.4e} (Vanished completely to zero)")
    print(f"2. Xavier Variance at Layer 50:      {xavier_vars[50]:.4e} (Collapsed by ~0.5^50 factor)")
    print(f"3. He/Kaiming Variance at Layer 50:   {kaiming_vars[50]:.4f} (Maintained near unit variance ~ 1.0)")

    # Assertions for ReLU
    assert naive_vars[50] < 1e-15, f"Naive did not vanish: {naive_vars[50]}"
    assert xavier_vars[50] < 1e-10, f"Xavier should have collapsed under ReLU: {xavier_vars[50]}"
    assert 0.05 <= kaiming_vars[50] <= 5.0, f"Kaiming variance exploded or collapsed: {kaiming_vars[50]}"
    print("✓ RELU INITIALIZATION ASSERTIONS PASSED.")

    print("\n" + "=" * 78)
    print("EXPERIMENT 2: Tanh Network — Xavier vs Kaiming Stabilization")
    print("=" * 78)

    # Xavier with Tanh
    xavier_tanh_vars = run_deep_network_simulation(init_xavier, activation="tanh", num_layers=num_layers, dim=dim, batch_size=batch_size)
    # Kaiming with Tanh (should slightly saturate or expand)
    kaiming_tanh_vars = run_deep_network_simulation(init_kaiming, activation="tanh", num_layers=num_layers, dim=dim, batch_size=batch_size)

    header_tanh = f"{'Layer':<8} | {'Xavier with Tanh':<22} | {'Kaiming with Tanh':<22}"
    print(header_tanh)
    print("─" * len(header_tanh))
    for l in checkpoints:
        print(f"Layer {l:<2d} | {xavier_tanh_vars[l]:<22.4f} | {kaiming_tanh_vars[l]:<22.4f}")

    print("=" * 78)
    print("ANALYSIS OF TANH RESULTS:")
    print(f"Xavier with Tanh maintains smooth controlled variance: Layer 50 = {xavier_tanh_vars[50]:.4f}")
    assert 0.005 <= xavier_tanh_vars[50] <= 1.0, f"Xavier unstable under Tanh: {xavier_tanh_vars[50]}"
    print("✓ TANH INITIALIZATION ASSERTIONS PASSED.")

    print("\n" + "=" * 78)
    print("ALL INITIALIZATION EXPERIMENTS COMPLETED SUCCESSFULLY (EXIT CODE 0).")
    print("=" * 78)
    sys.exit(0)


if __name__ == "__main__":
    main()
