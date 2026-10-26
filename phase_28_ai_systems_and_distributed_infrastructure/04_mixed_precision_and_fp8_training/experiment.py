#!/usr/bin/env python3
"""
Topic 04: Mixed Precision & FP8 Training
=====================================================
Experiment: Floating-Point Format Emulator & Dynamic Loss Scaling
Pure Python 3 standard library implementation (no numpy, no torch).

This script models:
1. Exact Bit-Level Emulation for:
   - IEEE 754 FP32 (1 sign, 8 exp, 23 mantissa)
   - IEEE 754 FP16 (1 sign, 5 exp, 10 mantissa, max=65,504, min_norm=6.1e-5, min_subnorm=5.96e-8)
   - BF16 (Brain Float 16: 1 sign, 8 exp, 7 mantissa, matches FP32 dynamic range)
   - FP8 E4M3 (1 sign, 4 exp, 3 mantissa, max=448, higher precision)
   - FP8 E5M2 (1 sign, 5 exp, 2 mantissa, max=57,344, higher dynamic range)
2. Gradient Underflow Catastrophe Demonstration in Raw FP16:
   - Models small backward gradients underflowing to exact 0.0.
3. Automatic Mixed Precision (AMP) Dynamic Loss Scaling Controller:
   - Simulates scaled backward pass L_scaled = S * L, unscaling in FP32,
     and dynamic scale factor adjustment (halving on Inf/NaN, doubling on consecutive stable steps).
"""

import math
import random
import sys
from typing import List, Tuple, Optional


class FloatFormat:
    """Specification of an arbitrary floating-point format."""
    def __init__(self, name: str, exp_bits: int, mantissa_bits: int, bias: Optional[int] = None,
                 custom_max: Optional[float] = None):
        self.name = name
        self.exp_bits = exp_bits
        self.mantissa_bits = mantissa_bits
        self.bias = bias if bias is not None else (2 ** (exp_bits - 1) - 1)
        self.max_exp_val = 2 ** exp_bits - 1
        
        # Format limits
        if custom_max is not None:
            self.max_val = custom_max
        else:
            max_unscaled_mantissa = 2.0 - (2.0 ** (-mantissa_bits))
            max_exponent = (self.max_exp_val - 1) - self.bias
            self.max_val = max_unscaled_mantissa * (2.0 ** max_exponent)
        
        # Smallest positive normal
        min_exponent = 1 - self.bias
        self.min_normal = 2.0 ** min_exponent
        
        # Smallest positive subnormal
        self.min_subnormal = 2.0 ** (min_exponent - mantissa_bits)

    def quantize(self, x: float) -> float:
        """Quantizes a Python 64-bit float to the target format."""
        if math.isnan(x):
            return float('nan')
        if math.isinf(x):
            return float('inf') if x > 0 else float('-inf')
        if x == 0.0:
            return 0.0

        sign = -1.0 if x < 0 else 1.0
        ax = abs(x)

        # Check overflow
        if ax > self.max_val:
            return sign * float('inf')

        # Check underflow below smallest subnormal
        if ax < self.min_subnormal / 2.0:
            return 0.0  # Underflow flush to zero

        # Subnormal representation
        if ax < self.min_normal:
            step = self.min_subnormal
            subnormal_val = round(ax / step) * step
            return sign * subnormal_val

        # Normalized representation: ax = (1 + f) * 2^e
        e = math.floor(math.log2(ax))
        # Clamp exponent
        e = max(1 - self.bias, min((self.max_exp_val - 1) - self.bias, e))
        fraction = (ax / (2.0 ** e)) - 1.0

        # Round fraction to mantissa_bits
        steps = 2 ** self.mantissa_bits
        rounded_frac = round(fraction * steps) / steps
        quantized_abs = (1.0 + rounded_frac) * (2.0 ** e)

        if quantized_abs > self.max_val:
            return sign * float('inf')

        return sign * quantized_abs


# Standard AI Floating-Point Formats
FORMAT_FP32 = FloatFormat("FP32", exp_bits=8, mantissa_bits=23)
FORMAT_FP16 = FloatFormat("FP16", exp_bits=5, mantissa_bits=10, bias=15)
FORMAT_BF16 = FloatFormat("BF16", exp_bits=8, mantissa_bits=7, bias=127)
# OCP FP8 standards
FORMAT_FP8_E4M3 = FloatFormat("FP8-E4M3", exp_bits=4, mantissa_bits=3, bias=7, custom_max=448.0)
FORMAT_FP8_E5M2 = FloatFormat("FP8-E5M2", exp_bits=5, mantissa_bits=2, bias=15, custom_max=57344.0)


class DynamicLossScaler:
    """
    Simulates Automatic Mixed Precision (AMP) Dynamic Loss Scaling.
    Maintains a scale factor S.
    Scales loss by S in backward pass to protect small gradients from FP16 underflow.
    Unscales gradients before optimizer step, inspecting for Inf/NaN.
    """
    def __init__(self, initial_scale: float = 65536.0, growth_factor: float = 2.0,
                 backoff_factor: float = 0.5, growth_interval: int = 5):
        self.scale = initial_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.consecutive_good_steps = 0
        self.total_overflows = 0

    def step(self, raw_fp32_gradients: List[float], format_type: FloatFormat) -> Tuple[List[float], bool]:
        """
        Executes one gradient backward scaling step:
        1. Scale gradients: g_scaled = g * S
        2. Cast to target format (e.g. FP16)
        3. Check for Inf or NaN
        4. If overflow: skip step, reduce scale S = S * backoff
        5. If clean: unscale g = g_scaled / S, update scale if growth_interval reached
        """
        scaled_quantized = []
        has_overflow = False

        for g in raw_fp32_gradients:
            g_scaled = g * self.scale
            q = format_type.quantize(g_scaled)
            if math.isinf(q) or math.isnan(q):
                has_overflow = True
                break
            scaled_quantized.append(q)

        if has_overflow:
            # Overflow detected! Skip optimizer step and shrink scale
            self.total_overflows += 1
            self.consecutive_good_steps = 0
            self.scale = max(1.0, self.scale * self.backoff_factor)
            return [], False

        # Clean step: Unscale gradients to FP32
        unscaled_fp32 = [q / self.scale for q in scaled_quantized]
        self.consecutive_good_steps += 1

        if self.consecutive_good_steps >= self.growth_interval:
            self.scale *= self.growth_factor
            self.consecutive_good_steps = 0

        return unscaled_fp32, True


def run_experiment():
    print("Initializing Topic 04: Mixed Precision & FP8 Format Simulator...\n")

    # 1. Format Characteristics Table
    formats = [FORMAT_FP32, FORMAT_FP16, FORMAT_BF16, FORMAT_FP8_E4M3, FORMAT_FP8_E5M2]
    print("=" * 95)
    print("FLOATING-POINT FORMAT SPECIFICATIONS")
    print("=" * 95)
    print(f"{'Format':<12} | {'Bits (Sign/Exp/Mant)':<22} | {'Min Positive Subnormal':<24} | {'Max Value':<16} | {'Primary DL Role':<16}")
    print("-" * 95)
    roles = {
        "FP32": "Master Weights, Adam",
        "FP16": "Weights, Activations",
        "BF16": "Modern LLM Training",
        "FP8-E4M3": "Forward Weights/Act",
        "FP8-E5M2": "Backward Gradients"
    }
    for fmt in formats:
        bits_str = f"1 / {fmt.exp_bits} / {fmt.mantissa_bits}"
        print(f"{fmt.name:<12} | {bits_str:<22} | {fmt.min_subnormal:>24.3e} | {fmt.max_val:>16.1f} | {roles[fmt.name]:<16}")
    print("=" * 95 + "\n")

    # 2. Underflow Catastrophe Demonstration
    # Generate realistic deep Transformer backward gradient distribution centered around 1e-8
    random.seed(42)
    sample_size = 5000
    raw_gradients = [math.exp(random.gauss(math.log(1e-8), 1.5)) for _ in range(sample_size)]

    # Quantize directly to FP16 without scaling
    fp16_direct = [FORMAT_FP16.quantize(g) for g in raw_gradients]
    direct_underflows = sum(1 for q in fp16_direct if q == 0.0)
    underflow_pct = (direct_underflows / sample_size) * 100.0

    print("=" * 85)
    print(f"GRADIENT UNDERFLOW EXPERIMENT (Sample Size: {sample_size:,} gradients, Median ~ 1e-8)")
    print("=" * 85)
    print(f"Direct FP16 (No Loss Scaling):")
    print(f"  Total underflowed to 0.0: {direct_underflows:,} / {sample_size:,} ({underflow_pct:.2f}%)")
    print(f"  Consequence: {underflow_pct:.1f}% of network weights receive EXACT ZERO update!")

    # BF16 direct comparison (BF16 has 8 exponent bits, identical dynamic range to FP32)
    bf16_direct = [FORMAT_BF16.quantize(g) for g in raw_gradients]
    bf16_underflows = sum(1 for q in bf16_direct if q == 0.0)
    print(f"Direct BF16 (No Loss Scaling):")
    print(f"  Total underflowed to 0.0: {bf16_underflows} / {sample_size:,} (0.00%)")
    print(f"  Reason: BF16 min subnormal is ~9.2e-41, completely immune to gradient underflow.")
    print("=" * 85 + "\n")

    # 3. Dynamic Loss Scaling Simulation
    print("=" * 85)
    print("AUTOMATIC MIXED PRECISION (AMP) DYNAMIC LOSS SCALING SIMULATION")
    print("=" * 85)
    scaler = DynamicLossScaler(initial_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=4)

    # Simulate 15 training steps, introducing a gradient explosion spike at Step 8
    print(f"{'Step':<6} | {'Scale Factor':<14} | {'Max Gradient':<16} | {'Scaled Max':<14} | {'Status':<16} | {'Active Non-Zeros':<16}")
    print("-" * 90)

    for step in range(1, 16):
        # Normal step or spike
        if step == 8:
            # Gradient spike: produces an overflow in FP16 (exceeds 65,504)
            step_grads = [g * 50_000_000.0 for g in raw_gradients]
        else:
            step_grads = list(raw_gradients)

        curr_scale = scaler.scale
        max_g = max(step_grads)
        scaled_max = max_g * curr_scale

        unscaled, success = scaler.step(step_grads, FORMAT_FP16)

        if success:
            non_zeros = sum(1 for g in unscaled if g > 0.0)
            status = "UPDATED"
            print(f"{step:<6} | {curr_scale:>14,.0f} | {max_g:>16.3e} | {scaled_max:>14.2f} | {status:<16} | {non_zeros:>16,}")
        else:
            status = "OVERFLOW (SKIP)"
            print(f"{step:<6} | {curr_scale:>14,.0f} | {max_g:>16.3e} | {scaled_max:>14.2f} | {status:<16} | {'0 (Skipped)':>16}")

    print("=" * 90)

    # Assertions
    assert underflow_pct > 70.0, f"Expected >70% underflow in unscaled FP16, got {underflow_pct}%"
    assert bf16_underflows == 0, "BF16 must not suffer underflow on gradients ~ 1e-8"
    assert scaler.total_overflows == 1, "Expected exactly 1 overflow at spike step 8"

    # Scaled FP16 should recover virtually all gradients
    scaled_step_grads, ok = scaler.step(raw_gradients, FORMAT_FP16)
    assert ok, "Standard step should succeed"
    recovered_nonzeros = sum(1 for g in scaled_step_grads if g > 0.0)
    assert recovered_nonzeros / sample_size > 0.99, "Scaled FP16 should recover >99% of gradients"

    print("\n[SUCCESS] All Topic 04 Mixed Precision & FP8 tests passed with code 0.")


if __name__ == "__main__":
    run_experiment()
