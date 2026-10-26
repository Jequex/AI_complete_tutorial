#!/usr/bin/env python3
"""
Phase 18 — Topic 03: Distributed Training & Parallelism Strategies
Experiment: Pure Python 3 Simulation of ZeRO Memory Sharding & Megatron-LM Tensor Parallelism.

This script implements:
1. Exact Memory Wall accounting for LLMs from 7B to 70B parameters:
   - Weights (FP16/BF16: 2N bytes)
   - Gradients (FP16/BF16: 2N bytes)
   - AdamW Optimizer states (FP32 master weights 4N + first momentum 4N + second momentum 4N = 12N bytes)
   - Total static state: 16N bytes
2. Mathematical simulation of ZeRO stages (Rajbhandari et al.):
   - ZeRO-1: Optimizer sharding (4N + 12N / Nd)
   - ZeRO-2: Optimizer + Gradient sharding (2N + 14N / Nd)
   - ZeRO-3: Full parameter sharding (16N / Nd)
3. Communication volume analysis: DDP (2N words) vs ZeRO-3 (3N words).
4. Pure Python simulation of Megatron-LM Tensor Parallelism:
   - Column-parallel linear layer (W_1 column split, no forward communication)
   - Row-parallel linear layer (W_2 row split, All-Reduce SUM across ranks)
   - Strict numerical equivalence verification against monolithic matrix multiplication (< 1e-12 error).
5. Pipeline parallelism bubble overhead & 1F1B activation memory bounds.

Zero external dependencies (pure standard library: math, random, sys).
Exits with code 0 on successful verification.
"""

import math
import random
import sys


def compute_static_memory_gb(n_params: float, zero_stage: int, num_devices: int) -> float:
    """
    Computes static memory footprint per device in Gigabytes (1 GB = 10^9 bytes).
    Total static bytes = 16 * N:
      - Weights: 2N bytes (FP16/BF16)
      - Gradients: 2N bytes (FP16/BF16)
      - Optimizer: 12N bytes (FP32 master weights + m_t + v_t)
    """
    if zero_stage == 0:  # Standard DDP: Replicate everything
        total_bytes = 16.0 * n_params
    elif zero_stage == 1:  # ZeRO-1: Shard optimizer states
        total_bytes = 4.0 * n_params + (12.0 * n_params / num_devices)
    elif zero_stage == 2:  # ZeRO-2: Shard optimizer states + gradients
        total_bytes = 2.0 * n_params + (14.0 * n_params / num_devices)
    elif zero_stage == 3:  # ZeRO-3: Shard weights + optimizer states + gradients
        total_bytes = 16.0 * n_params / num_devices
    else:
        raise ValueError(f"Unknown ZeRO stage: {zero_stage}")
    return total_bytes / 1e9


def pipeline_bubble_fraction(num_stages: int, num_microbatches: int) -> float:
    """Computes pipeline bubble fraction: F_bubble = (p - 1) / (m + p - 1)."""
    return (num_stages - 1.0) / (num_microbatches + num_stages - 1.0)


# ---------------------------------------------------------------------------
# Tensor Parallelism Matrix Simulator
# ---------------------------------------------------------------------------
def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """Monolithic 2D matrix multiplication: C = A @ B."""
    rows_a = len(a)
    cols_a = len(a[0])
    cols_b = len(b[0])
    c = [[0.0 for _ in range(cols_b)] for _ in range(rows_a)]
    for i in range(rows_a):
        for k in range(cols_a):
            a_ik = a[i][k]
            for j in range(cols_b):
                c[i][j] += a_ik * b[k][j]
    return c


def relu_matrix(x: list[list[float]]) -> list[list[float]]:
    """Element-wise ReLU activation."""
    return [[max(0.0, val) for val in row] for row in x]


def tensor_parallel_mlp_forward(x: list[list[float]],
                                w1: list[list[float]],
                                w2: list[list[float]],
                                tp_world_size: int = 2) -> list[list[float]]:
    """
    Simulates Megatron-LM Tensor Parallelism for an MLP block:
      Monolithic: Y = ReLU(X @ W1) @ W2
      where X in R^{B x d_in}, W1 in R^{d_in x d_mid}, W2 in R^{d_mid x d_out}.

    TP Strategy:
      1. Column-Parallel Linear Layer:
         W1 is partitioned column-wise into tp_world_size slices: W1 = [W1_0 | W1_1 | ...].
         Each rank computes: H_rank = ReLU(X @ W1_rank).
         COMMUNICATION: ZERO. No communication required here!

      2. Row-Parallel Linear Layer:
         W2 is partitioned row-wise into tp_world_size slices: W2 = [W2_0; W2_1; ...].
         Each rank computes: Z_rank = H_rank @ W2_rank.
         COMMUNICATION: All-Reduce (SUM) across all ranks: Z = sum_{rank} Z_rank.
    """
    b = len(x)
    d_in = len(w1)
    d_mid = len(w1[0])
    d_out = len(w2[0])
    assert d_mid % tp_world_size == 0, "d_mid must be divisible by tp_world_size"
    slice_size = d_mid // tp_world_size

    # Simulate individual GPU ranks
    rank_outputs: list[list[list[float]]] = []

    for rank in range(tp_world_size):
        # 1. Column slice of W1 on this rank: shape (d_in, slice_size)
        col_start = rank * slice_size
        col_end = col_start + slice_size
        w1_rank = [[w1[r][c] for c in range(col_start, col_end)] for r in range(d_in)]

        # Local computation: H_rank = ReLU(X @ W1_rank)
        linear1_rank = matmul(x, w1_rank)
        h_rank = relu_matrix(linear1_rank)

        # 2. Row slice of W2 on this rank: shape (slice_size, d_out)
        row_start = rank * slice_size
        row_end = row_start + slice_size
        w2_rank = [w2[r] for r in range(row_start, row_end)]

        # Local computation: Z_rank = H_rank @ W2_rank (shape B x d_out)
        z_rank = matmul(h_rank, w2_rank)
        rank_outputs.append(z_rank)

    # 3. All-Reduce (SUM) simulation across all ranks
    final_output = [[0.0 for _ in range(d_out)] for _ in range(b)]
    for z_rank in rank_outputs:
        for i in range(b):
            for j in range(d_out):
                final_output[i][j] += z_rank[i][j]

    return final_output


def run_experiment() -> None:
    print("=" * 80)
    print("PHASE 18 — TOPIC 03: DISTRIBUTED TRAINING & PARALLELISM STRATEGIES")
    print("Memory Wall Accounting, ZeRO Sharding, & Megatron-LM Tensor Parallelism")
    print("=" * 80)

    # 1. The Memory Wall: Why 70B Cannot Fit on a Single 80GB GPU
    models = {
        "7B Model": 7e9,
        "13B Model": 13e9,
        "34B Model": 34e9,
        "70B Model": 70e9
    }

    print("\n[1] THE MEMORY WALL: STATIC TRAINING MEMORY FOOTPRINT (16N BYTES):")
    print("-" * 80)
    print(f"{'Model Scale':<12} | {'Weights (2N)':<14} | {'Gradients (2N)':<16} | {'Optimizer (12N)':<16} | {'Total Static':<14}")
    print("-" * 80)
    for name, n in models.items():
        w_gb = (2.0 * n) / 1e9
        g_gb = (2.0 * n) / 1e9
        opt_gb = (12.0 * n) / 1e9
        tot_gb = (16.0 * n) / 1e9
        print(f"{name:<12} | {w_gb:>11.1f} GB | {g_gb:>13.1f} GB | {opt_gb:>13.1f} GB | {tot_gb:>11.1f} GB")

    print("\nKey Takeaway: A 70B parameter model requires 1,120 GB of raw static memory!")
    print("An NVIDIA A100/H100 has 80 GB of HBM. Fitting 70B requires at least 14 GPUs for static weights alone.")

    # 2. ZeRO Sharding Analysis across GPU Cluster Configurations
    print("\n[2] ZeRO-1, ZeRO-2, and ZeRO-3 MEMORY PER GPU (70B MODEL, 80GB GPUS):")
    print("-" * 80)
    print(f"{'Cluster Size (N_d)':<20} | {'DDP (ZeRO-0)':<14} | {'ZeRO-1':<12} | {'ZeRO-2':<12} | {'ZeRO-3 (FSDP)':<14}")
    print("-" * 80)
    
    n_70b = 70e9
    for n_devices in [1, 2, 4, 8, 16, 32, 64]:
        ddp_mem = compute_static_memory_gb(n_70b, 0, n_devices)
        z1_mem = compute_static_memory_gb(n_70b, 1, n_devices)
        z2_mem = compute_static_memory_gb(n_70b, 2, n_devices)
        z3_mem = compute_static_memory_gb(n_70b, 3, n_devices)
        
        status = "FITS in 80GB" if z3_mem <= 80.0 else "OOM"
        print(f"{n_devices:<20} | {ddp_mem:>11.1f} GB | {z1_mem:>9.1f} GB | {z2_mem:>9.1f} GB | {z3_mem:>11.1f} GB ({status})")

    # Assert that ZeRO-3 with 16 GPUs fits under 80GB (70 GB < 80 GB)
    assert compute_static_memory_gb(n_70b, 3, 16) <= 80.0, "ZeRO-3 on 16 GPUs should fit inside 80GB HBM!"

    # 3. Communication Volume Accounting: DDP vs ZeRO-3
    print("\n[3] DISTRIBUTED COMMUNICATION VOLUME PER STEP:")
    print("-" * 80)
    print("  • Standard DDP (All-Reduce Gradients): Communication = 2 * N words")
    print("  • ZeRO-3 / FSDP (All-Gather Weights Forward + All-Gather Weights Backward + Reduce-Scatter Grads):")
    print("    Communication = 1 * N + 1 * N + 1 * N = 3 * N words")
    print("  • Communication Overhead of ZeRO-3 over DDP: Exactly 3N / 2N = 1.5x (50% increase in network traffic).")

    # 4. Pure Python Megatron-LM Tensor Parallelism Verification
    print("\n[4] VERIFYING MEGATRON-LM TENSOR PARALLELISM vs MONOLITHIC MATMUL:")
    print("-" * 80)
    
    rng = random.Random(42)
    b, d_in, d_mid, d_out = 4, 8, 16, 8
    
    # Initialize random inputs and weights
    x = [[rng.gauss(0.0, 1.0) for _ in range(d_in)] for _ in range(b)]
    w1 = [[rng.gauss(0.0, 1.0) for _ in range(d_mid)] for _ in range(d_in)]
    w2 = [[rng.gauss(0.0, 1.0) for _ in range(d_out)] for _ in range(d_mid)]

    # Monolithic Forward Pass
    h_mono = relu_matrix(matmul(x, w1))
    y_mono = matmul(h_mono, w2)

    # Tensor-Parallel Forward Pass (Simulating 2 GPUs and 4 GPUs)
    for num_ranks in [2, 4]:
        y_tp = tensor_parallel_mlp_forward(x, w1, w2, tp_world_size=num_ranks)
        
        # Compute maximum absolute error between monolithic and TP
        max_abs_diff = 0.0
        for i in range(b):
            for j in range(d_out):
                diff = abs(y_mono[i][j] - y_tp[i][j])
                if diff > max_abs_diff:
                    max_abs_diff = diff
                    
        print(f"  TP World Size = {num_ranks} GPUs: Max Absolute Discrepancy = {max_abs_diff:.2e}")
        assert max_abs_diff < 1e-12, f"TP output deviated from monolithic matmul by {max_abs_diff}!"

    print("--> PASSED: Column-parallel and row-parallel partition reproduces exact mathematical output (< 1e-12).")

    # 5. Pipeline Parallelism Bubble Overhead Simulation
    print("\n[5] PIPELINE PARALLELISM BUBBLE FRACTION SIMULATION:")
    print("-" * 80)
    print(f"{'Pipeline Stages (p)':<20} | {'Microbatches (m)':<18} | {'Bubble Fraction F_bubble':<24} | {'Utilization'}")
    print("-" * 80)
    for p in [4, 8]:
        for m in [p, 4 * p, 8 * p]:
            bubble = pipeline_bubble_fraction(p, m)
            util = (1.0 - bubble) * 100.0
            print(f"{p:<20} | {m:<18} | {bubble:<24.4f} | {util:.1f}%")

    print("\n" + "=" * 80)
    print("ALL VERIFICATIONS COMPLETED SUCCESSFULLY — EXITING CODE 0")
    print("=" * 80)


if __name__ == "__main__":
    run_experiment()
    sys.exit(0)
