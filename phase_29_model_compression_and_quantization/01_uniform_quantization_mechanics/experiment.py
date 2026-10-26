#!/usr/bin/env python3
"""
Phase 29 - Topic 01: Uniform Quantization Mechanics
====================================================
Pure Python 3 implementation of uniform affine (asymmetric) and symmetric
quantization/dequantization across INT8 and INT4 precision levels.

Verifies:
1. Exact mapping between real continuous domain and integer domain.
2. Quantization error distribution and variance comparison with theoretical sigma^2 = Delta^2 / 12.
3. Empirical Signal-to-Quantization-Noise Ratio (SQNR) vs theoretical 6.02b + C dB.
4. Dynamic range and precision comparison between INT8 and INT4 on Uniform and Gaussian distributions.

Zero external dependencies: standard library only (math, random).
"""

import math
import random

def set_seed(seed=42):
    random.seed(seed)

# -----------------------------------------------------------------------------
# 1. Quantizer Implementations (Pure Python)
# -----------------------------------------------------------------------------

def asymmetric_quantize(tensor, num_bits=8):
    """
    Uniform Affine (Asymmetric) Quantization.
    Maps real range [x_min, x_max] to integer range [q_min, q_max].
    For unsigned: [0, 2^b - 1]
    Formula:
        s = (x_max - x_min) / (q_max - q_min)
        z = round(-x_min / s) + q_min
        q = clip(round(x / s) + z, q_min, q_max)
    """
    q_min = 0
    q_max = (1 << num_bits) - 1
    
    x_min = min(tensor)
    x_max = max(tensor)
    
    # Guard against zero dynamic range
    if math.isclose(x_min, x_max, abs_tol=1e-8):
        scale = 1.0
        zero_point = 0
        q_tensor = [q_min for _ in tensor]
        return q_tensor, scale, zero_point, q_min, q_max
    
    scale = (x_max - x_min) / float(q_max - q_min)
    # Zero-point calculation
    initial_zero_point = -x_min / scale + q_min
    zero_point = round(initial_zero_point)
    zero_point = max(q_min, min(q_max, zero_point))
    
    q_tensor = []
    for x in tensor:
        q = round(x / scale) + zero_point
        q_clamped = max(q_min, min(q_max, q))
        q_tensor.append(int(q_clamped))
        
    return q_tensor, scale, zero_point, q_min, q_max

def asymmetric_dequantize(q_tensor, scale, zero_point):
    """
    Dequantization for Uniform Affine Quantization:
        x_hat = s * (q - z)
    """
    return [scale * (q - zero_point) for q in q_tensor]

def symmetric_quantize(tensor, num_bits=8):
    """
    Symmetric Quantization (Zero-point = 0).
    Maps [-max(|x|), max(|x|)] to signed range [-2^(b-1), 2^(b-1) - 1].
    Formula:
        s = max(|x|) / (2^(b-1) - 1)
        q = clip(round(x / s), -2^(b-1), 2^(b-1) - 1)
    """
    q_max = (1 << (num_bits - 1)) - 1
    q_min = -(1 << (num_bits - 1))
    
    abs_max = max(abs(x) for x in tensor)
    if math.isclose(abs_max, 0.0, abs_tol=1e-8):
        scale = 1.0
        q_tensor = [0 for _ in tensor]
        return q_tensor, scale, 0, q_min, q_max
    
    scale = abs_max / float(q_max)
    
    q_tensor = []
    for x in tensor:
        q = round(x / scale)
        q_clamped = max(q_min, min(q_max, q))
        q_tensor.append(int(q_clamped))
        
    return q_tensor, scale, 0, q_min, q_max

def symmetric_dequantize(q_tensor, scale):
    """
    Dequantization for Symmetric Quantization:
        x_hat = s * q
    """
    return [scale * q for q in q_tensor]

# -----------------------------------------------------------------------------
# 2. Statistical & SQNR Metrics
# -----------------------------------------------------------------------------

def compute_sqnr(original, reconstructed):
    """
    Signal-to-Quantization-Noise Ratio:
        SQNR = 10 * log10( P_signal / P_noise )
        P_signal = (1/N) * sum( x_i^2 )
        P_noise  = (1/N) * sum( (x_i - x_hat_i)^2 )
    """
    p_signal = sum(x * x for x in original) / len(original)
    p_noise = sum((x - x_hat) ** 2 for x, x_hat in zip(original, reconstructed)) / len(original)
    
    if p_noise <= 1e-15:
        return float('inf')
    return 10.0 * math.log10(p_signal / p_noise)

def compute_noise_stats(original, reconstructed):
    """Compute empirical mean and variance of quantization noise e = x - x_hat."""
    errors = [x - x_hat for x, x_hat in zip(original, reconstructed)]
    mean_err = sum(errors) / len(errors)
    var_err = sum((e - mean_err) ** 2 for e in errors) / len(errors)
    return mean_err, var_err

# -----------------------------------------------------------------------------
# 3. Main Verification Experiments
# -----------------------------------------------------------------------------

def run_quantization_experiments():
    set_seed(42)
    n_samples = 100000
    
    print("=" * 78)
    print("PHASE 29: TOPIC 01 - UNIFORM QUANTIZATION MECHANICS & SQNR VERIFICATION")
    print("=" * 78)
    
    # -------------------------------------------------------------------------
    # Experiment A: Uniform Distribution & Theoretical SQNR Rule
    # -------------------------------------------------------------------------
    print("\n--- EXPERIMENT A: UNIFORM DISTRIBUTION [-5.0, 5.0] ---")
    uniform_data = [random.uniform(-5.0, 5.0) for _ in range(n_samples)]
    
    for bits in [8, 4]:
        # Asymmetric
        q_asym, s_asym, z_asym, _, _ = asymmetric_quantize(uniform_data, num_bits=bits)
        x_hat_asym = asymmetric_dequantize(q_asym, s_asym, z_asym)
        sqnr_asym = compute_sqnr(uniform_data, x_hat_asym)
        mean_err_asym, var_err_asym = compute_noise_stats(uniform_data, x_hat_asym)
        theory_var_asym = (s_asym ** 2) / 12.0
        
        # Symmetric
        q_sym, s_sym, z_sym, _, _ = symmetric_quantize(uniform_data, num_bits=bits)
        x_hat_sym = symmetric_dequantize(q_sym, s_sym)
        sqnr_sym = compute_sqnr(uniform_data, x_hat_sym)
        mean_err_sym, var_err_sym = compute_noise_stats(uniform_data, x_hat_sym)
        theory_var_sym = (s_sym ** 2) / 12.0
        
        # Theoretical SQNR for uniform distribution [-A, A]:
        # P_signal = A^2 / 3, P_noise = Delta^2 / 12 = (2A / (2^b - 1))^2 / 12
        # SQNR_theory = 10 * log10( (A^2 / 3) / ((4 A^2) / (12 * (2^b - 1)^2)) )
        #             = 10 * log10( (2^b - 1)^2 ) approx 20 * log10(2^b) = 6.02 * b dB
        theory_sqnr_uniform = 10.0 * math.log10(((1 << bits) - 1) ** 2)
        
        print(f"\n[Precision: INT{bits} (b={bits})]")
        print(f"  Asymmetric (Scale s={s_asym:.6f}, Zero-point z={z_asym}):")
        print(f"    Empirical SQNR:       {sqnr_asym:.2f} dB")
        print(f"    Theoretical SQNR:     {theory_sqnr_uniform:.2f} dB  (Rule: ~6.02b)")
        print(f"    Empirical Noise Var:  {var_err_asym:.8f}")
        print(f"    Theoretical Delta^2/12: {theory_var_asym:.8f} (Ratio: {var_err_asym/theory_var_asym:.4f})")
        
        print(f"  Symmetric (Scale s={s_sym:.6f}, Zero-point z=0):")
        print(f"    Empirical SQNR:       {sqnr_sym:.2f} dB")
        print(f"    Empirical Noise Var:  {var_err_sym:.8f}")
        print(f"    Theoretical Delta^2/12: {theory_var_sym:.8f} (Ratio: {var_err_sym/theory_var_sym:.4f})")
        
        # Assertions to mathematically validate the experiment
        assert math.isclose(var_err_asym, theory_var_asym, rel_tol=0.05), \
            f"Noise variance deviates from Delta^2 / 12 for INT{bits} Asymmetric"
        assert abs(sqnr_asym - theory_sqnr_uniform) < 0.8, \
            f"SQNR deviates from 6.02b rule for INT{bits} Asymmetric"

    # -------------------------------------------------------------------------
    # Experiment B: Full-Scale Sinusoidal Signal (6.02b + 1.76 dB proof)
    # -------------------------------------------------------------------------
    print("\n--- EXPERIMENT B: FULL-SCALE SINUSOIDAL SIGNAL (6.02b + 1.76 dB) ---")
    amplitude = 10.0
    sinusoid_data = [amplitude * math.sin(2.0 * math.pi * i / 1000.0) for i in range(n_samples)]
    
    for bits in [8, 4]:
        q_sym, s_sym, _, _, _ = symmetric_quantize(sinusoid_data, num_bits=bits)
        x_hat = symmetric_dequantize(q_sym, s_sym)
        sqnr = compute_sqnr(sinusoid_data, x_hat)
        theory_sqnr_sin = 6.0206 * bits + 1.7609
        
        print(f"[INT{bits}] Empirical SQNR: {sqnr:.2f} dB | Theoretical 6.02b + 1.76: {theory_sqnr_sin:.2f} dB | Error: {abs(sqnr - theory_sqnr_sin):.2f} dB")
        # For small bit-widths (b=4), discrete binning causes small deviation (< 1.0 dB); for b=8 it is exact (< 0.1 dB).
        assert abs(sqnr - theory_sqnr_sin) < 1.0, f"Sinusoidal SQNR failed benchmark for INT{bits}"

    # -------------------------------------------------------------------------
    # Experiment C: Gaussian Distributed Weights (Typical Neural Network Layer)
    # -------------------------------------------------------------------------
    print("\n--- EXPERIMENT C: GAUSSIAN WEIGHT TENSOR N(0, 1) ---")
    gaussian_data = [random.gauss(0.0, 1.0) for _ in range(n_samples)]
    
    for bits in [8, 4]:
        q_asym, s_asym, z_asym, _, _ = asymmetric_quantize(gaussian_data, num_bits=bits)
        x_hat_asym = asymmetric_dequantize(q_asym, s_asym, z_asym)
        sqnr_asym = compute_sqnr(gaussian_data, x_hat_asym)
        
        q_sym, s_sym, _, _, _ = symmetric_quantize(gaussian_data, num_bits=bits)
        x_hat_sym = symmetric_dequantize(q_sym, s_sym)
        sqnr_sym = compute_sqnr(gaussian_data, x_hat_sym)
        
        print(f"[INT{bits}] Gaussian Weights -> Asymmetric SQNR: {sqnr_asym:.2f} dB | Symmetric SQNR: {sqnr_sym:.2f} dB")
        # In INT8, SQNR is high (~38-40 dB). In INT4, it drops to ~16-18 dB.
        assert sqnr_sym > (3.5 * bits), f"Gaussian SQNR too low for INT{bits}"

    print("\n" + "=" * 78)
    print("ALL VERIFICATIONS COMPLETED SUCCESSFULLY (EXIT CODE 0)")
    print("=" * 78)

if __name__ == "__main__":
    run_quantization_experiments()
