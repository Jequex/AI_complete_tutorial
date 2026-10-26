#!/usr/bin/env python3
"""
Phase 12 - Topic 01: Computational Graphs and the Chain Rule
Experiment: Manual Multi-Branch Computational Graph Forward and Backward Pass
with Finite Difference Numerical Gradient Verification.

Pure Python 3 standard library implementation. Zero external dependencies.
"""

import math
import sys
from typing import Dict, List, Tuple


def evaluate_graph_forward(x: float, y: float) -> Dict[str, float]:
    """
    Evaluates a multi-branch computational graph forward:
    
    Inputs:
        x, y
    Intermediate Nodes:
        a = x * y           (Branch 1 from x, y)
        b = x + y           (Branch 2 from x, y)
        c = a * b           (Consumer of a and b)
        d = b ** 2          (Consumer of b, creating a diamond dependency on b)
        e = math.exp(x)     (Branch 3 directly from x)
    Output:
        L = c + d - e       (Final scalar objective)
    """
    a = x * y
    b = x + y
    c = a * b
    d = b ** 2
    e = math.exp(x)
    L = c + d - e
    
    return {
        "x": x,
        "y": y,
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "e": e,
        "L": L,
    }


def evaluate_graph_backward(vals: Dict[str, float]) -> Dict[str, float]:
    """
    Evaluates the backward pass on the computational graph using the
    multivariable chain rule:
        dL/du = sum_{v in Children(u)} (dL/dv) * (dv/du)

    Reverse topological ordering:
    1. Seed output adjoint: dL/dL = 1.0
    2. dL/dc = dL/dL * (dL/dc) = 1.0 * 1.0 = 1.0
    3. dL/dd = dL/dL * (dL/dd) = 1.0 * 1.0 = 1.0
    4. dL/de = dL/dL * (dL/de) = 1.0 * (-1.0) = -1.0
    5. dL/da = (dL/dc) * (dc/da) = dL/dc * b
    6. dL/db = (dL/dc) * (dc/db) + (dL/dd) * (dd/db)   [Accumulation from 2 consumers: c and d!]
             = (dL/dc) * a + (dL/dd) * (2 * b)
    7. dL/dx = (dL/da) * (da/dx) + (dL/db) * (db/dx) + (dL/de) * (de/dx) [Accumulation from a, b, e!]
             = (dL/da) * y + (dL/db) * 1.0 + (dL/de) * exp(x)
    8. dL/dy = (dL/da) * (da/dy) + (dL/db) * (db/dy)   [Accumulation from a and b!]
             = (dL/da) * x + (dL/db) * 1.0
    """
    x = vals["x"]
    y = vals["y"]
    a = vals["a"]
    b = vals["b"]
    e = vals["e"]

    # Adjoints (dL/d[var])
    grad: Dict[str, float] = {}

    # Seed
    grad["L"] = 1.0

    # From L = c + d - e
    grad["c"] = grad["L"] * 1.0
    grad["d"] = grad["L"] * 1.0
    grad["e"] = grad["L"] * (-1.0)

    # From c = a * b
    # a has one consumer: c
    grad["a"] = grad["c"] * b

    # b has TWO consumers: c (via c = a * b) and d (via d = b^2)
    # Summing branches:
    grad_b_from_c = grad["c"] * a
    grad_b_from_d = grad["d"] * (2.0 * b)
    grad["b"] = grad_b_from_c + grad_b_from_d

    # x has THREE consumers: a (a = x*y), b (b = x+y), e (e = exp(x))
    grad_x_from_a = grad["a"] * y
    grad_x_from_b = grad["b"] * 1.0
    grad_x_from_e = grad["e"] * math.exp(x)
    grad["x"] = grad_x_from_a + grad_x_from_b + grad_x_from_e

    # y has TWO consumers: a (a = x*y) and b (b = x+y)
    grad_y_from_a = grad["a"] * x
    grad_y_from_b = grad["b"] * 1.0
    grad["y"] = grad_y_from_a + grad_y_from_b

    return grad


def finite_difference_gradient(x: float, y: float, eps: float = 1e-7) -> Tuple[float, float]:
    """
    Approximates dL/dx and dL/dy using two-sided central finite differences:
        f'(theta) ~ (f(theta + eps) - f(theta - eps)) / (2 * eps)
    """
    # Perturb x
    L_x_plus = evaluate_graph_forward(x + eps, y)["L"]
    L_x_minus = evaluate_graph_forward(x - eps, y)["L"]
    num_grad_x = (L_x_plus - L_x_minus) / (2.0 * eps)

    # Perturb y
    L_y_plus = evaluate_graph_forward(x, y + eps)["L"]
    L_y_minus = evaluate_graph_forward(x, y - eps)["L"]
    num_grad_y = (L_y_plus - L_y_minus) / (2.0 * eps)

    return num_grad_x, num_grad_y


def run_experiment() -> None:
    print("=" * 78)
    print("TOPIC 01 EXPERIMENT: COMPUTATIONAL GRAPH & MULTIVARIABLE CHAIN RULE")
    print("=" * 78)

    test_points = [
        (2.0, 3.0),
        (0.5, 1.5),
        (-1.2, 2.4),
        (3.0, -1.0),
    ]

    for idx, (x_val, y_val) in enumerate(test_points, 1):
        print(f"\n--- Test Point {idx}: x = {x_val:.4f}, y = {y_val:.4f} ---")
        
        # Forward pass
        vals = evaluate_graph_forward(x_val, y_val)
        print("Forward Pass Node Activations:")
        for node in ["x", "y", "a", "b", "c", "d", "e", "L"]:
            print(f"  {node:>2} = {vals[node]:12.6f}")

        # Backward pass
        grads = evaluate_graph_backward(vals)
        print("\nBackward Pass Node Adjoints (dL/dNode):")
        for node in ["L", "e", "d", "c", "b", "a", "y", "x"]:
            print(f"  dL/d{node:<2} = {grads[node]:12.6f}")

        # Finite difference comparison
        num_gx, num_gy = finite_difference_gradient(x_val, y_val)
        err_x = abs(grads["x"] - num_gx)
        err_y = abs(grads["y"] - num_gy)

        print("\nNumerical Verification (Central Differences eps=1e-7):")
        print(f"  dL/dx: Analytical = {grads['x']:12.6f} | Numerical = {num_gx:12.6f} | Abs Error = {err_x:.2e}")
        print(f"  dL/dy: Analytical = {grads['y']:12.6f} | Numerical = {num_gy:12.6f} | Abs Error = {err_y:.2e}")

        # Verify precision within 1e-6 (finite difference truncation and roundoff)
        assert err_x < 1e-6, f"dL/dx verification failed! Error {err_x} >= 1e-6"
        assert err_y < 1e-6, f"dL/dy verification failed! Error {err_y} >= 1e-6"
        print("  => VERIFICATION PASSED! Exact match between Chain Rule and Finite Differences.")

    print("\n" + "=" * 78)
    print("ALL TESTS PASSED SUCCESSFULLY! Computational graph chain rule verified.")
    print("=" * 78)


if __name__ == "__main__":
    run_experiment()
    sys.exit(0)
