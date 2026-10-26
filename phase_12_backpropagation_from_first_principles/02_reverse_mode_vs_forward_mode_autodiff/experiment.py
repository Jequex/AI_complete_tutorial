#!/usr/bin/env python3
"""
Phase 12 - Topic 02: Reverse-Mode vs Forward-Mode Automatic Differentiation
Experiment: Dual Numbers (Forward-Mode JVP) vs Tape-Based Reverse-Mode (VJP)
Benchmarking Computational Complexity and Numerical Equivalence.

Pure Python 3 standard library implementation. Zero external dependencies.
"""

import math
import sys
from typing import Dict, List, Tuple


# =====================================================================
# 1. Forward-Mode Autodiff via Dual Numbers
# =====================================================================

class DualNumber:
    """
    Dual number representation: x + dot{x} * epsilon, where epsilon^2 = 0.
    - val: Real part f(x)
    - der: Dual part f'(x) * v (Tangent / Directional derivative)
    """
    def __init__(self, val: float, der: float = 0.0):
        self.val = float(val)
        self.der = float(der)

    def __add__(self, other):
        if not isinstance(other, DualNumber):
            other = DualNumber(other, 0.0)
        return DualNumber(self.val + other.val, self.der + other.der)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if not isinstance(other, DualNumber):
            other = DualNumber(other, 0.0)
        return DualNumber(self.val - other.val, self.der - other.der)

    def __rsub__(self, other):
        if not isinstance(other, DualNumber):
            other = DualNumber(other, 0.0)
        return other.__sub__(self)

    def __mul__(self, other):
        if not isinstance(other, DualNumber):
            other = DualNumber(other, 0.0)
        # (a + a_dot*eps) * (b + b_dot*eps) = ab + (a*b_dot + a_dot*b)*eps
        return DualNumber(self.val * other.val, self.val * other.der + self.der * other.val)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if not isinstance(other, DualNumber):
            other = DualNumber(other, 0.0)
        # Quotient rule: (a / b)' = (a'b - ab') / b^2
        val = self.val / other.val
        der = (self.der * other.val - self.val * other.der) / (other.val ** 2)
        return DualNumber(val, der)

    def __pow__(self, p: float):
        # f(x) = x^p => f'(x) = p * x^(p-1) * x'
        val = self.val ** p
        der = p * (self.val ** (p - 1)) * self.der
        return DualNumber(val, der)

    def exp(self):
        val = math.exp(self.val)
        return DualNumber(val, val * self.der)

    def sin(self):
        val = math.sin(self.val)
        return DualNumber(val, math.cos(self.val) * self.der)

    def cos(self):
        val = math.cos(self.val)
        return DualNumber(val, -math.sin(self.val) * self.der)

    def __repr__(self):
        return f"DualNumber(val={self.val:.6f}, der={self.der:.6f})"


# =====================================================================
# 2. Reverse-Mode Autodiff via Computation Tape (Wengert List)
# =====================================================================

class TapeEntry:
    def __init__(self, op: str, result_id: int, inputs: List[int], local_grads: List[float]):
        self.op = op
        self.result_id = result_id
        self.inputs = inputs
        self.local_grads = local_grads  # d(result)/d(input_i)


class ReverseTapeEngine:
    def __init__(self):
        self.tape: List[TapeEntry] = []
        self.values: Dict[int, float] = {}
        self.counter = 0

    def variable(self, val: float) -> int:
        node_id = self.counter
        self.counter += 1
        self.values[node_id] = float(val)
        return node_id

    def add(self, a_id: int, b_id: int) -> int:
        res_id = self.counter
        self.counter += 1
        res_val = self.values[a_id] + self.values[b_id]
        self.values[res_id] = res_val
        self.tape.append(TapeEntry("+", res_id, [a_id, b_id], [1.0, 1.0]))
        return res_id

    def sub(self, a_id: int, b_id: int) -> int:
        res_id = self.counter
        self.counter += 1
        res_val = self.values[a_id] - self.values[b_id]
        self.values[res_id] = res_val
        self.tape.append(TapeEntry("-", res_id, [a_id, b_id], [1.0, -1.0]))
        return res_id

    def mul(self, a_id: int, b_id: int) -> int:
        res_id = self.counter
        self.counter += 1
        a_val = self.values[a_id]
        b_val = self.values[b_id]
        res_val = a_val * b_val
        self.values[res_id] = res_val
        self.tape.append(TapeEntry("*", res_id, [a_id, b_id], [b_val, a_val]))
        return res_id

    def pow_const(self, a_id: int, p: float) -> int:
        res_id = self.counter
        self.counter += 1
        a_val = self.values[a_id]
        res_val = a_val ** p
        self.values[res_id] = res_val
        local_grad = p * (a_val ** (p - 1.0))
        self.tape.append(TapeEntry(f"**{p}", res_id, [a_id], [local_grad]))
        return res_id

    def exp(self, a_id: int) -> int:
        res_id = self.counter
        self.counter += 1
        a_val = self.values[a_id]
        res_val = math.exp(a_val)
        self.values[res_id] = res_val
        self.tape.append(TapeEntry("exp", res_id, [a_id], [res_val]))
        return res_id

    def sin(self, a_id: int) -> int:
        res_id = self.counter
        self.counter += 1
        a_val = self.values[a_id]
        res_val = math.sin(a_val)
        self.values[res_id] = res_val
        self.tape.append(TapeEntry("sin", res_id, [a_id], [math.cos(a_val)]))
        return res_id

    def backward(self, target_id: int) -> Dict[int, float]:
        """
        Reverse pass over recorded tape.
        Accumulates adjoints: dL/d(inp) += dL/d(res) * local_grad
        """
        adjoints: Dict[int, float] = {node_id: 0.0 for node_id in range(self.counter)}
        adjoints[target_id] = 1.0  # Seed

        # Process tape in reverse chronological order
        for entry in reversed(self.tape):
            res_grad = adjoints[entry.result_id]
            for inp_id, local_g in zip(entry.inputs, entry.local_grads):
                adjoints[inp_id] += res_grad * local_g

        return adjoints


# =====================================================================
# Target Mathematical Function: f: R^3 -> R
# f(x, y, z) = sin(x * y) + exp(y * z) - (x + z)^2
# =====================================================================

def test_func_forward_mode(x: float, y: float, z: float, seed_v: Tuple[float, float, float]) -> DualNumber:
    """Computes JVP: J * v using Dual Numbers."""
    dx = DualNumber(x, seed_v[0])
    dy = DualNumber(y, seed_v[1])
    dz = DualNumber(z, seed_v[2])

    t1 = (dx * dy).sin()        # sin(x * y)
    t2 = (dy * dz).exp()        # exp(y * z)
    t3 = (dx + dz) ** 2         # (x + z)^2
    res = (t1 + t2) - t3
    return res


def test_func_reverse_mode(x: float, y: float, z: float) -> Tuple[float, Dict[str, float], ReverseTapeEngine]:
    tape = ReverseTapeEngine()
    x_id = tape.variable(x)
    y_id = tape.variable(y)
    z_id = tape.variable(z)

    # t1 = sin(x * y)
    xy_id = tape.mul(x_id, y_id)
    t1_id = tape.sin(xy_id)

    # t2 = exp(y * z)
    yz_id = tape.mul(y_id, z_id)
    t2_id = tape.exp(yz_id)

    # t3 = (x + z)^2
    xz_id = tape.add(x_id, z_id)
    t3_id = tape.pow_const(xz_id, 2.0)

    # res = (t1 + t2) - t3
    sum12_id = tape.add(t1_id, t2_id)
    res_id = tape.sub(sum12_id, t3_id)

    # Execute single backward pass
    adjoints = tape.backward(res_id)

    grads = {
        "df_dx": adjoints[x_id],
        "df_dy": adjoints[y_id],
        "df_dz": adjoints[z_id],
    }
    return tape.values[res_id], grads, tape


def run_experiment() -> None:
    print("=" * 78)
    print("TOPIC 02 EXPERIMENT: FORWARD-MODE (DUAL NUMBERS) VS REVERSE-MODE (TAPE)")
    print("Function f: R^3 -> R: f(x, y, z) = sin(x*y) + exp(y*z) - (x + z)^2")
    print("=" * 78)

    test_points = [
        (1.2, 0.8, -0.5),
        (0.5, 1.5, 2.0),
        (-1.0, 0.4, 0.7),
    ]

    for idx, (x, y, z) in enumerate(test_points, 1):
        print(f"\n--- Evaluation at Point {idx}: (x={x}, y={y}, z={z}) ---")

        # -------------------------------------------------------------
        # FORWARD-MODE: Requires 3 separate forward sweeps for 3 inputs!
        # -------------------------------------------------------------
        print("1. Forward-Mode Evaluation (Jacobian-Vector Products):")
        # Sweep 1: v = [1, 0, 0] -> computes df/dx
        res_x = test_func_forward_mode(x, y, z, (1.0, 0.0, 0.0))
        # Sweep 2: v = [0, 1, 0] -> computes df/dy
        res_y = test_func_forward_mode(x, y, z, (0.0, 1.0, 0.0))
        # Sweep 3: v = [0, 0, 1] -> computes df/dz
        res_z = test_func_forward_mode(x, y, z, (0.0, 0.0, 1.0))

        forward_val = res_x.val
        forward_grads = {
            "df_dx": res_x.der,
            "df_dy": res_y.der,
            "df_dz": res_z.der,
        }
        print(f"   Forward Sweeps Needed: 3 (equal to d_in = 3)")
        print(f"   Value: {forward_val:.8f}")
        print(f"   Grads: df/dx={forward_grads['df_dx']:.8f}, df/dy={forward_grads['df_dy']:.8f}, df/dz={forward_grads['df_dz']:.8f}")

        # -------------------------------------------------------------
        # REVERSE-MODE: Requires 1 forward sweep + 1 backward sweep!
        # -------------------------------------------------------------
        print("\n2. Reverse-Mode Evaluation (Vector-Jacobian Product):")
        rev_val, rev_grads, tape = test_func_reverse_mode(x, y, z)
        print(f"   Forward Sweeps: 1 | Backward Sweeps: 1 (equal to d_out = 1)")
        print(f"   Tape Operations Recorded: {len(tape.tape)}")
        print(f"   Value: {rev_val:.8f}")
        print(f"   Grads: df/dx={rev_grads['df_dx']:.8f}, df/dy={rev_grads['df_dy']:.8f}, df/dz={rev_grads['df_dz']:.8f}")

        # -------------------------------------------------------------
        # Verification & Comparison
        # -------------------------------------------------------------
        val_diff = abs(forward_val - rev_val)
        err_x = abs(forward_grads["df_dx"] - rev_grads["df_dx"])
        err_y = abs(forward_grads["df_dy"] - rev_grads["df_dy"])
        err_z = abs(forward_grads["df_dz"] - rev_grads["df_dz"])

        print(f"\n3. Discrepancy Comparison:")
        print(f"   Value Diff: {val_diff:.2e}")
        print(f"   Gradient Abs Errors: dx={err_x:.2e}, dy={err_y:.2e}, dz={err_z:.2e}")

        assert val_diff < 1e-12, f"Function value mismatch: {val_diff}"
        assert err_x < 1e-12, f"dx mismatch: {err_x}"
        assert err_y < 1e-12, f"dy mismatch: {err_y}"
        assert err_z < 1e-12, f"dz mismatch: {err_z}"
        print("   => EXACT EQUIVALENCE VERIFIED (Tolerance < 1e-12)")

    # -----------------------------------------------------------------
    # Complexity Scaling Demonstration
    # -----------------------------------------------------------------
    print("\n" + "=" * 78)
    print("COMPLEXITY SCALING: WHY DEEP LEARNING USES REVERSE-MODE")
    print("=" * 78)
    d_in_scenarios = [3, 100, 10_000, 1_000_000, 175_000_000_000]
    print(f"{'Input Dim (d_in)':<22} | {'Forward-Mode Sweeps':<22} | {'Reverse-Mode Sweeps':<20}")
    print("-" * 72)
    for d in d_in_scenarios:
        print(f"{d:<22,d} | {d:<22,d} | {'1 backward pass':<20}")
    print("-" * 72)
    print("For GPT-3 (175 Billion parameters):")
    print(" - Forward Mode: 175,000,000,000 full forward passes per training step (IMPOSSIBLE).")
    print(" - Reverse Mode: 1 forward pass + 1 backward pass per training step (TRAINABLE).")
    print("=" * 78)


if __name__ == "__main__":
    run_experiment()
    sys.exit(0)
