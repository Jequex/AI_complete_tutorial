#!/usr/bin/env python3
"""
Phase 19: Fine-Tuning & PEFT / LoRA
Topic 19.3: QLoRA & Quantization Mechanics

Pure Python 3 implementation (zero external dependencies).
Demonstrates:
  1. Symmetric & Asymmetric Uniform Quantization (INT8 and INT4) from scratch.
  2. NormalFloat4 (NF4) Non-Linear Information-Theoretic Quantile Quantization (Dettmers et al. 2023).
  3. Error Analysis: Proving NF4 achieves superior Signal-to-Noise Ratio (SNR) and lower MSE
     than uniform 4-bit on bell-shaped normally distributed neural network weights.
  4. Double Quantization (DQ): Quantizing quantization constants (scales) to slash scale overhead
     from 0.5 bits/parameter to 0.127 bits/parameter.
  5. Simulation of Paged Optimizers (dynamic CPU/GPU paging simulation).
"""

import math
import random
from typing import List, Tuple, Dict


def set_seed(seed: int = 42):
    random.seed(seed)


# -------------------------------------------------------------------------
# Exact 16-Quantile NormalFloat4 (NF4) Codebook (Dettmers et al. 2023)
# -------------------------------------------------------------------------
# These 16 values represent the quantiles of a standardized zero-mean unit-variance
# normal distribution N(0, 1) normalized such that values are bounded in [-1, 1],
# ensuring equal probability mass (1/16 = 6.25%) per bin.
NF4_CODEBOOK = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0
]


# -------------------------------------------------------------------------
# Part 1: Quantization Implementations
# -------------------------------------------------------------------------

def quantize_uniform_int8(weights: List[float]) -> Tuple[List[int], float]:
    """
    Symmetric uniform INT8 quantization:
      Scale S = max(|weights|) / 127
      q = clamp(round(w / S), -127, 127)
    """
    max_abs = max(abs(w) for w in weights)
    scale = max_abs / 127.0 if max_abs > 0 else 1.0
    q_weights = [max(-127, min(127, round(w / scale))) for w in weights]
    return q_weights, scale


def dequantize_uniform_int8(q_weights: List[int], scale: float) -> List[float]:
    return [q * scale for q in q_weights]


def quantize_uniform_int4(weights: List[float]) -> Tuple[List[int], float]:
    """
    Symmetric uniform INT4 quantization:
      q in [-8, 7] -> 16 levels
      Scale S = max(|weights|) / 7.0
    """
    max_abs = max(abs(w) for w in weights)
    scale = max_abs / 7.0 if max_abs > 0 else 1.0
    q_weights = [max(-8, min(7, round(w / scale))) for w in weights]
    return q_weights, scale


def dequantize_uniform_int4(q_weights: List[int], scale: float) -> List[float]:
    return [q * scale for q in q_weights]


def quantize_nf4_block(weights: List[float]) -> Tuple[List[int], float]:
    """
    NormalFloat4 (NF4) block quantization:
      1. Normalize block by its absolute maximum: w_norm = w / max(|w|) in [-1, 1]
      2. Find closest index k in NF4_CODEBOOK for each normalized weight.
    """
    max_abs = max(abs(w) for w in weights)
    scale = max_abs if max_abs > 0 else 1.0
    indices = []
    for w in weights:
        w_norm = w / scale
        # Find index in NF4_CODEBOOK minimizing |w_norm - codebook[idx]|
        best_idx = 0
        min_dist = float('inf')
        for idx, code_val in enumerate(NF4_CODEBOOK):
            dist = abs(w_norm - code_val)
            if dist < min_dist:
                min_dist = dist
                best_idx = idx
        indices.append(best_idx)
    return indices, scale


def dequantize_nf4_block(indices: List[int], scale: float) -> List[float]:
    """Dequantizes NF4 indices using codebook lookup: w_dequant = NF4_CODEBOOK[idx] * scale."""
    return [NF4_CODEBOOK[idx] * scale for idx in indices]


# -------------------------------------------------------------------------
# Part 2: Double Quantization (DQ)
# -------------------------------------------------------------------------

def double_quantize_scales(scales: List[float], block_size2: int = 256) -> Dict:
    """
    Double Quantization (DQ):
    Quantizes the quantization constants (scales) of Block Size 1 (e.g. 64)
    into 8-bit integers using a second-stage Block Size 2 (e.g. 256).
    
    Standard single scale overhead: 32 bits / 64 params = 0.5 bits/param.
    Double Quantization overhead:
      8 bits / 64 + 32 bits / (64 * 256) = 0.125 + 0.00195 = 0.12695 bits/param!
    """
    num_scales = len(scales)
    # Pad scales if needed
    pad_len = (block_size2 - (num_scales % block_size2)) % block_size2
    padded_scales = scales + [0.0] * pad_len
    
    quantized_scale_blocks = []
    second_stage_scales = []
    
    for start in range(0, len(padded_scales), block_size2):
        chunk = padded_scales[start:start + block_size2]
        q_chunk, c2_scale = quantize_uniform_int8(chunk)
        quantized_scale_blocks.extend(q_chunk)
        second_stage_scales.append(c2_scale)
        
    return {
        "quantized_scales_int8": quantized_scale_blocks[:num_scales],
        "second_stage_scales_fp32": second_stage_scales,
        "single_quant_bits_per_param": 32.0 / 64.0,
        "double_quant_bits_per_param": (8.0 / 64.0) + (32.0 / (64.0 * block_size2))
    }


# -------------------------------------------------------------------------
# Part 3: Metrics & Evaluation Helpers
# -------------------------------------------------------------------------

def compute_mse(orig: List[float], recon: List[float]) -> float:
    return sum((o - r) ** 2 for o, r in zip(orig, recon)) / len(orig)


def compute_snr_db(orig: List[float], recon: List[float]) -> float:
    """Signal-to-Noise Ratio in decibels (dB): 10 * log10(Power_signal / Power_noise)."""
    p_signal = sum(o ** 2 for o in orig) / len(orig)
    p_noise = sum((o - r) ** 2 for o, r in zip(orig, recon)) / len(orig)
    if p_noise == 0.0:
        return float('inf')
    return 10.0 * math.log10(p_signal / p_noise)


# -------------------------------------------------------------------------
# Execution & Experiment Script
# -------------------------------------------------------------------------

def run_experiment():
    print("================================================================================")
    print("  PHASE 19: FINE-TUNING & PEFT / LoRA")
    print("  Topic 19.3: QLoRA & Quantization Mechanics")
    print("================================================================================\n")
    set_seed(42)
    
    # 1. Synthesize 10,000 weights drawn from a pre-trained Gaussian weight distribution
    N_WEIGHTS = 10_240
    BLOCK_SIZE = 64
    sigma = 0.02
    weights = [random.gauss(0.0, sigma) for _ in range(N_WEIGHTS)]
    
    print(f"--- 1. GENERATED {N_WEIGHTS:,} WEIGHTS FROM N(0, {sigma}^2) ---")
    print(f"  Sample min: {min(weights):.6f}, max: {max(weights):.6f}, mean: {sum(weights)/len(weights):.6f}\n")
    
    # 2. Block-wise Quantization: Uniform INT4 vs. NormalFloat4 (NF4)
    print(f"--- 2. QUANTIZATION BENCHMARK (BLOCK SIZE = {BLOCK_SIZE}) ---")
    
    uniform_int4_recon = []
    nf4_recon = []
    int8_recon = []
    nf4_scales = []
    
    for start in range(0, N_WEIGHTS, BLOCK_SIZE):
        block = weights[start:start + BLOCK_SIZE]
        
        # Uniform INT8 (baseline standard)
        q8, s8 = quantize_uniform_int8(block)
        int8_recon.extend(dequantize_uniform_int8(q8, s8))
        
        # Uniform INT4 (linear equal-width bins)
        q4_uni, s4_uni = quantize_uniform_int4(block)
        uniform_int4_recon.extend(dequantize_uniform_int4(q4_uni, s4_uni))
        
        # NormalFloat4 (NF4: information-theoretic quantiles)
        q4_nf4, s_nf4 = quantize_nf4_block(block)
        nf4_scales.append(s_nf4)
        nf4_recon.extend(dequantize_nf4_block(q4_nf4, s_nf4))
        
    mse_int8 = compute_mse(weights, int8_recon)
    snr_int8 = compute_snr_db(weights, int8_recon)
    
    mse_uni4 = compute_mse(weights, uniform_int4_recon)
    snr_uni4 = compute_snr_db(weights, uniform_int4_recon)
    
    mse_nf4 = compute_mse(weights, nf4_recon)
    snr_nf4 = compute_snr_db(weights, nf4_recon)
    
    print(f"{'Format':<20} | {'Bits/Weight':<12} | {'MSE Error':<16} | {'SNR (dB)':<12}")
    print("-" * 68)
    print(f"{'FP16 (Reference)':<20} | {'16 bits':<12} | {'0.00000000e+00':<16} | {'+inf dB':<12}")
    print(f"{'Uniform INT8':<20} | {'8 bits':<12} | {mse_int8:<16.8e} | {snr_int8:>7.2f} dB")
    print(f"{'Uniform INT4':<20} | {'4 bits':<12} | {mse_uni4:<16.8e} | {snr_uni4:>7.2f} dB")
    print(f"{'NormalFloat4 (NF4)':<20} | {'4 bits':<12} | {mse_nf4:<16.8e} | {snr_nf4:>7.2f} dB")
    
    print("\nObservation:")
    print(f"  NF4 achieves an MSE reduction of {((mse_uni4 - mse_nf4) / mse_uni4) * 100:.2f}% compared to Uniform INT4!")
    print(f"  NF4 delivers a +{snr_nf4 - snr_uni4:.2f} dB higher Signal-to-Noise Ratio at the exact same 4-bit budget.")
    assert mse_nf4 < mse_uni4, "NF4 must yield lower MSE than Uniform INT4 on normal distributions!"
    assert snr_nf4 > snr_uni4, "NF4 must yield higher SNR than Uniform INT4 on normal distributions!"
    print("  [PASSED] NF4 information-theoretic superiority verified.\n")
    
    # 3. Double Quantization (DQ) Simulation
    print("--- 3. DOUBLE QUANTIZATION (DQ) MEMORY COMPRESSION ---")
    dq_results = double_quantize_scales(nf4_scales, block_size2=256)
    
    single_bits = dq_results["single_quant_bits_per_param"]
    double_bits = dq_results["double_quant_bits_per_param"]
    saved_bits = single_bits - double_bits
    
    print(f"  Total Stage 1 Scales:       {len(nf4_scales)} (Block Size 64)")
    print(f"  Single Quantization Cost:   {single_bits:.4f} bits/parameter (FP32 scale per 64 weights)")
    print(f"  Double Quantization Cost:   {double_bits:.4f} bits/parameter (INT8 scale + FP32 scale per 256*64)")
    print(f"  Memory Savings on Scales:   {saved_bits:.4f} bits/parameter ({saved_bits / single_bits * 100:.1f}% reduction)")
    
    # Model scale extrapolation (LLaMA-65B)
    params_65b = 65_000_000_000
    single_overhead_gb = (params_65b * single_bits / 8) / (1024**3)
    double_overhead_gb = (params_65b * double_bits / 8) / (1024**3)
    gb_saved = single_overhead_gb - double_overhead_gb
    print(f"  Extrapolation to LLaMA-65B: Saves {gb_saved:.2f} GB of GPU VRAM exclusively from quantization constants!")
    assert double_bits < 0.13, "Double quantization should achieve ~0.127 bits/param"
    print("  [PASSED] Double Quantization theoretical compression verified.\n")
    
    # 4. Paged Optimizers Logic Simulation
    print("--- 4. PAGED OPTIMIZERS SIMULATION ---")
    print("  Simulating CUDA Unified Memory eviction during activation spike...")
    gpu_vram_limit_mb = 1000
    allocated_vram_mb = 850
    activation_spike_mb = 300
    
    print(f"  Current VRAM Allocated:    {allocated_vram_mb} MB / {gpu_vram_limit_mb} MB")
    print(f"  Incoming Activation Spike: {activation_spike_mb} MB")
    
    if (allocated_vram_mb + activation_spike_mb) > gpu_vram_limit_mb:
        print("  [ALERT] VRAM Threshold Exceeded! Standard training triggers CUDA Out-Of-Memory (OOM).")
        paged_out_mb = (allocated_vram_mb + activation_spike_mb) - gpu_vram_limit_mb + 50
        print(f"  [PAGED OPTIMIZER] Swapping {paged_out_mb} MB of AdamW page tables to CPU RAM...")
        allocated_vram_mb -= paged_out_mb
        allocated_vram_mb += activation_spike_mb
        print(f"  [SUCCESS] Activation spike absorbed safely! Active VRAM: {allocated_vram_mb} MB. Zero Crash.")
    
    print("\n================================================================================")
    print("All assertions passed successfully. Topic 19.3 experiment complete.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(run_experiment())
