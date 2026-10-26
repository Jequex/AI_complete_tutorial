#!/usr/bin/env python3
"""
Topic 01: GPU Architecture, CUDA & Memory Hierarchy
=====================================================
Experiment: Roofline Model & Warp Divergence Simulator
Pure Python 3 standard library implementation (no numpy, no torch).

This script models:
1. Operational (Arithmetic) Intensity I = FLOPs / Memory_Bytes across neural network kernels.
2. The Roofline Model: Attainable TFLOPs = min(Compute_Peak, Bandwidth_Peak * I).
3. Bottleneck classification: Memory-bound vs Compute-bound regimes with ridge-point analysis.
4. SIMT Warp Divergence Simulator: 32 threads per warp executing conditional branches,
   computing active mask utilization, serialization overhead, and instruction throughput penalty.
"""

import math
import random
import sys


class HardwareProfile:
    """Hardware characteristics of modern AI accelerators."""
    def __init__(self, name: str, peak_tflops_fp16: float, peak_mem_bw_gb_s: float,
                 sm_count: int, warp_size: int = 32):
        self.name = name
        self.peak_tflops = peak_tflops_fp16          # TFLOP/s (10^12 FLOPs/sec)
        self.peak_mem_bw = peak_mem_bw_gb_s          # GB/s (10^9 Bytes/sec)
        self.sm_count = sm_count
        self.warp_size = warp_size
        # Ridge Point: FLOPs per Byte where compute ceiling meets memory bandwidth ceiling
        # (TFLOPs * 10^12) / (GB/s * 10^9) = (TFLOPs / GB/s) * 1000 = FLOP / Byte
        self.ridge_point = (self.peak_tflops * 1e12) / (self.peak_mem_bw * 1e9)

    def attainable_performance(self, arithmetic_intensity: float) -> float:
        """
        Attainable Performance P = min(P_peak, B_peak * I) in TFLOP/s.
        """
        # Memory-limited performance in TFLOP/s:
        # (B_peak * 10^9 bytes/s * I FLOP/byte) / 10^12 = B_peak * I / 1000
        mem_limited_tflops = (self.peak_mem_bw * arithmetic_intensity) / 1000.0
        return min(self.peak_tflops, mem_limited_tflops)

    def execution_time_ms(self, total_flops: float, total_bytes: float) -> float:
        """
        Calculates theoretical execution time in milliseconds under Roofline bound:
        T = max(FLOPs / Compute_Peak, Bytes / Bandwidth_Peak)
        """
        time_compute_sec = total_flops / (self.peak_tflops * 1e12)
        time_memory_sec = total_bytes / (self.peak_mem_bw * 1e9)
        time_sec = max(time_compute_sec, time_memory_sec)
        return time_sec * 1000.0


class KernelProfile:
    """Characteristics of neural network operations."""
    def __init__(self, name: str, flops: float, bytes_transferred: float, description: str):
        self.name = name
        self.flops = float(flops)
        self.bytes_transferred = float(bytes_transferred)
        self.description = description
        self.arithmetic_intensity = self.flops / self.bytes_transferred if self.bytes_transferred > 0 else 0.0

    @classmethod
    def gemm(cls, M: int, N: int, K: int, dtype_bytes: int = 2) -> "KernelProfile":
        """
        Matrix Multiply: C = A @ B.
        A: (M, K), B: (K, N), C: (M, N).
        FLOPs: 2 * M * N * K (multiply-add)
        Memory Bytes (naive load A, B and store C): (M*K + K*N + M*N) * dtype_bytes
        """
        flops = 2 * M * N * K
        bytes_transferred = (M * K + K * N + M * N) * dtype_bytes
        return cls(f"GEMM (M={M}, N={N}, K={K})", flops, bytes_transferred,
                   "Dense Matrix Multiplication (Attention Projection / FFN)")

    @classmethod
    def layernorm(cls, batch: int, seq_len: int, d_model: int, dtype_bytes: int = 2) -> "KernelProfile":
        """
        Layer Normalization across last dimension d_model.
        Per vector of length D:
        1. Mean: D adds
        2. Variance: D subtracts, D multiplies, D adds
        3. Normalize + Affine: D subtracts, D divs, D multiplies (gamma), D adds (beta)
        Total FLOPs ~ 8 * D per token.
        Memory Bytes: Read input (D), read gamma/beta (2D once per layer, neglected for large batch),
                      write output (D) -> 2 * D * dtype_bytes.
        """
        tokens = batch * seq_len
        flops = tokens * 8 * d_model
        bytes_transferred = tokens * 2 * d_model * dtype_bytes
        return cls(f"LayerNorm (Tokens={tokens}, D={d_model})", flops, bytes_transferred,
                   "Elementwise Reduction & Normalization")

    @classmethod
    def gelu(cls, num_elements: int, dtype_bytes: int = 2) -> "KernelProfile":
        """
        GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        ~ 10 FLOPs per element.
        Memory: Read x (1 elem), Write out (1 elem) -> 2 * dtype_bytes per elem.
        """
        flops = num_elements * 10
        bytes_transferred = num_elements * 2 * dtype_bytes
        return cls(f"GELU Activation (N={num_elements:,})", flops, bytes_transferred,
                   "Elementwise Non-linearity")

    @classmethod
    def self_attention_score(cls, batch: int, heads: int, seq_len: int, d_k: int, dtype_bytes: int = 2) -> "KernelProfile":
        """
        Standard Q @ K^T: (B, H, S, d_k) x (B, H, d_k, S) -> (B, H, S, S).
        FLOPs: 2 * B * H * S * S * d_k
        Memory: Read Q (B*H*S*d_k), Read K (B*H*S*d_k), Write Scores (B*H*S*S)
        """
        flops = 2 * batch * heads * (seq_len ** 2) * d_k
        bytes_transferred = (2 * batch * heads * seq_len * d_k + batch * heads * (seq_len ** 2)) * dtype_bytes
        return cls(f"Self-Attention Scores S={seq_len}", flops, bytes_transferred,
                   "BMM: Q @ K^T across attention heads")


class WarpDivergenceSimulator:
    """
    Simulates SIMT warp execution of 32 threads encountering conditional branches.
    Demonstrates warp divergence penalty when threads within a single warp branch to different paths.
    """
    def __init__(self, warp_size: int = 32, num_warps: int = 1000):
        self.warp_size = warp_size
        self.num_warps = num_warps

    def simulate_branch(self, branch_condition_prob: float,
                        cycles_then: int = 40, cycles_else: int = 40) -> dict:
        """
        Simulates execution of:
            if (condition):
                # cycles_then instructions
            else:
                # cycles_else instructions

        In SIMT execution:
        - If all threads take THEN: takes cycles_then, efficiency = 100%
        - If all threads take ELSE: takes cycles_else, efficiency = 100%
        - If threads diverge: warp executes BOTH paths serially with active masks:
          total cycles = cycles_then + cycles_else!
        """
        random.seed(42)
        total_diverged_warps = 0
        total_cycles_simt = 0
        total_cycles_ideal = 0  # Perfect MIMD / branch prediction without serialization
        active_thread_cycles = 0
        possible_thread_cycles = 0

        for _ in range(self.num_warps):
            # Evaluate condition for all 32 threads in the warp
            thread_decisions = [random.random() < branch_condition_prob for _ in range(self.warp_size)]
            then_count = sum(thread_decisions)
            else_count = self.warp_size - then_count

            if then_count == self.warp_size:
                # Uniform THEN path
                cycles = cycles_then
                diverged = False
                active_threads = self.warp_size * cycles_then
            elif else_count == self.warp_size:
                # Uniform ELSE path
                cycles = cycles_else
                diverged = False
                active_threads = self.warp_size * cycles_else
            else:
                # Diverged! The warp must execute THEN (masking off ELSE threads),
                # then execute ELSE (masking off THEN threads).
                diverged = True
                total_diverged_warps += 1
                cycles = cycles_then + cycles_else
                active_threads = (then_count * cycles_then) + (else_count * cycles_else)

            ideal_cycles = (cycles_then if then_count >= else_count else cycles_else)
            total_cycles_simt += cycles
            total_cycles_ideal += ideal_cycles
            active_thread_cycles += active_threads
            possible_thread_cycles += (self.warp_size * cycles)

        simt_efficiency = active_thread_cycles / possible_thread_cycles if possible_thread_cycles > 0 else 1.0
        slowdown_factor = total_cycles_simt / total_cycles_ideal if total_cycles_ideal > 0 else 1.0

        return {
            "branch_prob": branch_condition_prob,
            "diverged_warps_fraction": total_diverged_warps / self.num_warps,
            "simt_efficiency": simt_efficiency,
            "slowdown_factor": slowdown_factor,
            "total_cycles_simt": total_cycles_simt,
            "total_cycles_ideal": total_cycles_ideal,
            "wasted_slot_percentage": (1.0 - simt_efficiency) * 100.0
        }


def print_roofline_ascii(gpu: HardwareProfile, kernels: list):
    """Prints a clear ASCII representation of the Roofline model."""
    print("=" * 80)
    print(f"ROOFLINE ANALYSIS: {gpu.name}")
    print(f"Peak Compute: {gpu.peak_tflops:.1f} TFLOP/s (FP16) | Peak HBM Bandwidth: {gpu.peak_mem_bw:.1f} GB/s")
    print(f"Ridge Point (I*): {gpu.ridge_point:.2f} FLOPs/Byte")
    print("=" * 80)
    print(f"{'Kernel':<32} | {'Intensity (FLOP/B)':<18} | {'Regime':<14} | {'Attainable (TFLOP/s)':<20} | {'Time (ms)':<10}")
    print("-" * 105)

    for k in kernels:
        attainable = gpu.attainable_performance(k.arithmetic_intensity)
        regime = "COMPUTE-BOUND" if k.arithmetic_intensity >= gpu.ridge_point else "MEMORY-BOUND"
        t_ms = gpu.execution_time_ms(k.flops, k.bytes_transferred)
        pct_peak = (attainable / gpu.peak_tflops) * 100.0
        print(f"{k.name:<32} | {k.arithmetic_intensity:>18.2f} | {regime:<14} | {attainable:>10.2f} ({pct_peak:>4.1f}%) | {t_ms:>9.4f}")

    print("=" * 105)


def run_experiment():
    print("Initializing Topic 01: GPU Architecture & Memory Hierarchy Simulator...\n")

    # 1. Define Standard GPUs: NVIDIA A100 (SXM4 80GB) and H100 (SXM5 80GB)
    a100 = HardwareProfile(name="NVIDIA A100-SXM4-80GB", peak_tflops_fp16=312.0, peak_mem_bw_gb_s=2039.0, sm_count=108)
    h100 = HardwareProfile(name="NVIDIA H100-SXM5-80GB", peak_tflops_fp16=989.0, peak_mem_bw_gb_s=3350.0, sm_count=132)

    # 2. Define Workload Kernels
    kernels = [
        # Large GEMM: Batch=32, Seq=2048 (M=65536), Hidden=4096, Intermediate=14336 (LLaMA-70B FFN)
        KernelProfile.gemm(M=2048, N=4096, K=4096),
        KernelProfile.gemm(M=65536, N=4096, K=14336),
        # Elementwise Kernels
        KernelProfile.layernorm(batch=32, seq_len=2048, d_model=4096),
        KernelProfile.gelu(num_elements=32 * 2048 * 4096),
        # Self-Attention Scores: B=32, H=32, S=2048, d=128
        KernelProfile.self_attention_score(batch=32, heads=32, seq_len=512, d_k=128),
        KernelProfile.self_attention_score(batch=32, heads=32, seq_len=4096, d_k=128),
    ]

    print_roofline_ascii(a100, kernels)
    print("\n")
    print_roofline_ascii(h100, kernels)

    # 3. Simulate Warp Divergence
    print("\n" + "=" * 80)
    print("SIMT WARP DIVERGENCE SIMULATION (32 threads/warp, 1,000 Warps)")
    print("=" * 80)
    print(f"{'Branch Prob p':<15} | {'Diverged Warps %':<18} | {'SIMT Efficiency %':<18} | {'Wasted Slots %':<16} | {'Slowdown':<10}")
    print("-" * 85)

    divergence_sim = WarpDivergenceSimulator(warp_size=32, num_warps=1000)
    test_probs = [0.0, 0.05, 0.20, 0.50, 0.80, 0.95, 1.0]

    for p in test_probs:
        stats = divergence_sim.simulate_branch(branch_condition_prob=p, cycles_then=40, cycles_else=40)
        print(f"{stats['branch_prob']:<15.2f} | {stats['diverged_warps_fraction']*100:>16.1f}% | "
              f"{stats['simt_efficiency']*100:>16.2f}% | {stats['wasted_slot_percentage']:>14.2f}% | "
              f"{stats['slowdown_factor']:>9.2f}x")

    print("=" * 85)

    # Assertions and sanity tests
    assert a100.ridge_point > 100.0, f"Expected A100 ridge point > 100, got {a100.ridge_point}"
    assert h100.ridge_point > a100.ridge_point, "H100 should have higher arithmetic intensity threshold than A100"
    
    # Verify that GEMM is compute-bound while LayerNorm is strictly memory-bound
    ln_kernel = kernels[2]
    gemm_large = kernels[1]
    assert ln_kernel.arithmetic_intensity < a100.ridge_point, "LayerNorm must be memory bound"
    assert gemm_large.arithmetic_intensity > a100.ridge_point, "Large GEMM must be compute bound"

    # Verify warp divergence characteristics
    uniform_stats = divergence_sim.simulate_branch(0.0)
    assert uniform_stats["diverged_warps_fraction"] == 0.0, "Uniform branch should have 0 divergence"
    assert uniform_stats["simt_efficiency"] == 1.0, "Uniform branch should have 100% efficiency"

    split_stats = divergence_sim.simulate_branch(0.5)
    assert split_stats["diverged_warps_fraction"] > 0.99, "At p=0.5, nearly 100% of 32-thread warps diverge (1 - 2*(0.5^32))"
    assert split_stats["slowdown_factor"] > 1.8, "At p=0.5, serialization should nearly double cycles"

    print("\n[SUCCESS] All Topic 01 Roofline & Warp Divergence tests passed with code 0.")


if __name__ == "__main__":
    run_experiment()
