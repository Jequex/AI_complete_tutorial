#!/usr/bin/env python3
"""
Phase 11: Neural Networks From Scratch
Topic 11.2: Activation Functions & Vanishing Gradients

Pure Python 3 implementation (zero external dependencies).
Demonstrates:
  1. Forward pass and exact analytical derivatives for:
     - Sigmoid
     - Tanh
     - ReLU
     - LeakyReLU (alpha = 0.01)
     - GELU (Gaussian Error Linear Unit - tanh approximation)
     - SiLU / Swish (z * sigmoid(z))
  2. Finite-difference numerical gradient verification for all activations.
  3. Deep 10-layer gradient flow comparison:
     - Vanishing gradient collapse in Sigmoid (decaying exponentially by 0.25^10).
     - Healthy gradient preservation in ReLU, LeakyReLU, GELU, and SiLU.
  4. Dead ReLU simulation: neuron trapped in negative saturation.
"""

import math
from typing import Callable, Tuple, List, Dict


def sigmoid(z: float) -> float:
    # Stable sigmoid to prevent overflow
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)


def sigmoid_grad(z: float) -> float:
    s = sigmoid(z)
    return s * (1.0 - s)


def tanh_func(z: float) -> float:
    return math.tanh(z)


def tanh_grad(z: float) -> float:
    t = math.tanh(z)
    return 1.0 - t * t


def relu(z: float) -> float:
    return max(0.0, z)


def relu_grad(z: float) -> float:
    return 1.0 if z > 0.0 else 0.0


def leaky_relu(z: float, alpha: float = 0.01) -> float:
    return z if z > 0.0 else alpha * z


def leaky_relu_grad(z: float, alpha: float = 0.01) -> float:
    return 1.0 if z > 0.0 else alpha


def gelu(z: float) -> float:
    # Standard Hendrycks & Gimpel (2016) approximation used in GPT / BERT
    # 0.5 * z * (1 + tanh(sqrt(2/pi) * (z + 0.044715 * z^3)))
    sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
    inner = sqrt_2_over_pi * (z + 0.044715 * (z ** 3))
    return 0.5 * z * (1.0 + math.tanh(inner))


def gelu_grad(z: float) -> float:
    # Analytical derivative of the tanh approximation of GELU
    # Let u(z) = sqrt(2/pi) * (z + 0.044715 * z^3)
    # du/dz = sqrt(2/pi) * (1 + 3 * 0.044715 * z^2)
    # GELU(z) = 0.5 * z * (1 + tanh(u))
    # dGELU/dz = 0.5 * (1 + tanh(u)) + 0.5 * z * (1 - tanh(u)^2) * du/dz
    sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
    inner = sqrt_2_over_pi * (z + 0.044715 * (z ** 3))
    tanh_u = math.tanh(inner)
    d_inner = sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * (z ** 2))
    return 0.5 * (1.0 + tanh_u) + 0.5 * z * (1.0 - tanh_u ** 2) * d_inner


def silu(z: float) -> float:
    # SiLU / Swish: z * sigmoid(z)
    return z * sigmoid(z)


def silu_grad(z: float) -> float:
    # d/dz [z * sigma(z)] = sigma(z) + z * sigma(z) * (1 - sigma(z))
    #                     = sigma(z) * [1 + z * (1 - sigma(z))]
    s = sigmoid(z)
    return s + z * s * (1.0 - s)


def numerical_gradient(f: Callable[[float], float], z: float, eps: float = 1e-6) -> float:
    """Centered finite-difference approximation: (f(z+eps) - f(z-eps)) / (2*eps)"""
    return (f(z + eps) - f(z - eps)) / (2.0 * eps)


def test_analytical_derivatives():
    print("=" * 70)
    print("Step 1: Verification of Analytical Activation Derivatives via Finite Differences")
    print("=" * 70)

    test_points = [-3.0, -1.5, -0.5, 0.2, 1.0, 2.5]
    activations = [
        ("Sigmoid", sigmoid, sigmoid_grad),
        ("Tanh", tanh_func, tanh_grad),
        ("ReLU", relu, relu_grad),
        ("LeakyReLU", leaky_relu, leaky_relu_grad),
        ("GELU", gelu, gelu_grad),
        ("SiLU/Swish", silu, silu_grad),
    ]

    for name, f_forward, f_grad in activations:
        print(f"\nTesting {name}:")
        for z in test_points:
            # Skip exact z=0 for ReLU where derivative is discontinuous
            analytical = f_grad(z)
            numerical = numerical_gradient(f_forward, z)
            diff = abs(analytical - numerical)
            assert diff < 1e-4, f"Mismatch in {name} at z={z}: analytical={analytical}, numerical={numerical}"
            print(f"  z={z:5.2f} -> f(z)={f_forward(z):8.5f}, f'(z)={analytical:8.5f}, num_grad={numerical:8.5f} (diff={diff:1.1e})")

    print("\n✓ All activation forward passes and analytical derivatives verified numerically.")


def test_linear_collapse():
    print("\n" + "=" * 70)
    print("Step 2: Mathematical Proof of Linear Collapse in Multi-Layer Networks")
    print("=" * 70)

    # Consider 3 linear layers without activation:
    # a1 = W1 x + b1
    # a2 = W2 a1 + b2 = W2(W1 x + b1) + b2 = (W2 W1) x + (W2 b1 + b2)
    # y = W3 a2 + b3 = (W3 W2 W1) x + (W3 (W2 b1 + b2) + b3)
    #
    # Let x in R^2, layer 1: 2->3, layer 2: 3->3, layer 3: 3->1
    W1 = [[1.5, -0.5], [0.2, 1.0], [-1.0, 0.5]]  # 3x2
    b1 = [0.1, -0.2, 0.3]                         # 3
    W2 = [[0.5, 0.0, -1.0], [1.0, -0.5, 0.2], [0.0, 2.0, 1.0]]  # 3x3
    b2 = [-0.1, 0.4, 0.0]                         # 3
    W3 = [[1.2, -0.8, 0.5]]                       # 1x3
    b3 = [0.25]                                   # 1

    # Multi-layer forward pass
    def feedforward_linear(x: List[float]) -> float:
        # Layer 1
        a1 = [sum(W1[i][j] * x[j] for j in range(2)) + b1[i] for i in range(3)]
        # Layer 2
        a2 = [sum(W2[i][j] * a1[j] for j in range(3)) + b2[i] for i in range(3)]
        # Layer 3
        y = sum(W3[0][j] * a2[j] for j in range(3)) + b3[0]
        return y

    # Compute collapsed equivalent weights W_eff (1x2) and b_eff (1)
    # First: W21 = W2 @ W1 (3x2), b21 = W2 @ b1 + b2 (3)
    W21 = [[sum(W2[i][k] * W1[k][j] for k in range(3)) for j in range(2)] for i in range(3)]
    b21 = [sum(W2[i][k] * b1[k] for k in range(3)) + b2[i] for i in range(3)]

    # Next: W_eff = W3 @ W21 (1x2), b_eff = W3 @ b21 + b3 (1)
    W_eff = [sum(W3[0][k] * W21[k][j] for k in range(3)) for j in range(2)]
    b_eff = sum(W3[0][k] * b21[k] for k in range(3)) + b3[0]

    def collapsed_linear(x: List[float]) -> float:
        return W_eff[0] * x[0] + W_eff[1] * x[1] + b_eff

    test_inputs = [[1.0, 2.0], [-0.5, 3.5], [4.0, -2.0], [0.0, 0.0]]
    for x in test_inputs:
        y_multi = feedforward_linear(x)
        y_collapsed = collapsed_linear(x)
        print(f"Input x={x}: multi-layer={y_multi:.8f}, collapsed single layer={y_collapsed:.8f}")
        assert math.isclose(y_multi, y_collapsed, rel_tol=1e-12), "Linear collapse check failed!"

    print("✓ Confirmed: 3 cascaded linear layers collapse exactly to W_eff * x + b_eff.")


def test_vanishing_gradients_10_layers():
    print("\n" + "=" * 70)
    print("Step 3: Vanishing Gradient Propagation through 10 Layers")
    print("=" * 70)

    # In backpropagation, by the chain rule:
    # dL/da0 = (dL/daL) * prod_{l=1}^L [W^{[l]} * g'(z^{[l]})]
    # If weights are initialized near 1.0 and z is centered near 0:
    # For Sigmoid: g'(0) = 0.25. Over 10 layers: (0.25)^10 = (1/4)^10 = 1 / 1,048,576 ~ 9.54e-7!
    # For ReLU: g'(z) = 1.0 (for z > 0). Over 10 layers: (1.0)^10 = 1.0!
    # For GELU: g'(z) ~ 0.5 to 1.0 in active regime.

    num_layers = 10
    incoming_gradient = 1.0  # dL/da_L

    # Consider pre-activation centered at typical small magnitude (z = 0.5)
    z_val = 0.5

    activations_to_compare = [
        ("Sigmoid", sigmoid_grad(z_val)),
        ("Tanh", tanh_grad(z_val)),
        ("ReLU", relu_grad(z_val)),
        ("LeakyReLU", leaky_relu_grad(z_val)),
        ("GELU", gelu_grad(z_val)),
        ("SiLU", silu_grad(z_val)),
    ]

    print(f"Simulating gradient flow from Layer {num_layers} back to Layer 1 (incoming grad = 1.0, weight = 1.0):")
    print(f"{'Activation':<12} | {'Local g_prime':<14} | {'Grad at Layer 5':<16} | {'Grad at Layer 1 (End)':<22}")
    print("-" * 72)

    for name, local_grad in activations_to_compare:
        grad_layer_5 = incoming_gradient * (local_grad ** 5)
        grad_layer_1 = incoming_gradient * (local_grad ** 10)
        print(f"{name:<12} | {local_grad:<14.6f} | {grad_layer_5:<16.6e} | {grad_layer_1:<22.6e}")

    # Specific assertion: Sigmoid gradient decays exponentially below 1e-5
    sigmoid_g10 = sigmoid_grad(0.0) ** 10  # best case at z=0 is 0.25^10
    assert sigmoid_g10 < 1e-6, f"Sigmoid gradient {sigmoid_g10} expected < 1e-6"
    # ReLU gradient does not vanish (remains 1.0)
    assert relu_grad(z_val) ** 10 == 1.0, "ReLU gradient should remain 1.0 for positive activations"

    print("\n✓ Vanishing gradient phenomenon demonstrated: Sigmoid gradient collapses by factor > 1,000,000.")


def test_dead_relu_simulation():
    print("\n" + "=" * 70)
    print("Step 4: Dead ReLU Simulation & LeakyReLU Recovery")
    print("=" * 70)

    # A neuron with a large negative bias or shifted weights receives inputs producing z < 0.
    negative_zs = [-5.0, -2.5, -0.8, -0.01]

    for z in negative_zs:
        r_out = relu(z)
        r_g = relu_grad(z)
        lr_out = leaky_relu(z, alpha=0.01)
        lr_g = leaky_relu_grad(z, alpha=0.01)

        assert r_g == 0.0, f"Dead ReLU must have exactly 0 gradient, got {r_g}"
        assert lr_g == 0.01, f"LeakyReLU must retain alpha gradient, got {lr_g}"
        print(f"z = {z:6.2f} | ReLU: out={r_out:4.2f}, grad={r_g:4.2f} (DEAD) | LeakyReLU: out={lr_out:7.4f}, grad={lr_g:6.4f} (ALIVE)")

    print("✓ Dead ReLU frozen state and LeakyReLU gradient recovery verified.")


def main():
    print("=" * 70)
    print("PHASE 11.2 EXPERIMENT: ACTIVATION FUNCTIONS & VANISHING GRADIENTS")
    print("=" * 70)
    test_analytical_derivatives()
    test_linear_collapse()
    test_vanishing_gradients_10_layers()
    test_dead_relu_simulation()
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS PASSED SUCCESSFULLY (Exit Code 0)")
    print("=" * 70)


if __name__ == "__main__":
    main()
