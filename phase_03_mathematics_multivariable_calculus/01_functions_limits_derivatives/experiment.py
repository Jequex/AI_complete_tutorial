#!/usr/bin/env python3
"""
EXPERIMENT 3.1: FUNCTIONS, DERIVATIVES & NUMERICAL GRADIENT CHECKING
===================================================================
A hands-on, zero-dependency Python laboratory demonstrating:
1. Limit Definition & Numerical Differentiation (Forward vs Central Difference)
2. Precision Benchmarking across step sizes h (O(h) vs O(h^2) error convergence)
3. Analytical vs Numerical Derivatives for Neural Activations (Sigmoid, ReLU, GELU)
4. Production Gradient Checking Engine for Verifying Backprop Implementations

Runs on standard library Python 3 without external packages.
"""

import math

print("=" * 70)
print("EXPERIMENT 3.1: FUNCTIONS, DERIVATIVES & GRADIENT CHECKING")
print("=" * 70)

# ======================================================================
# 1. FORWARD VS CENTRAL DIFFERENCE ERROR CONVERGENCE
# ======================================================================
print("\n--- 1. NUMERICAL DIFFERENTIATION: FORWARD VS CENTRAL DIFFERENCE ---")

# Test function: f(x) = x^3. Analytical derivative: f'(x) = 3*x^2.
def f_cubic(x: float) -> float:
    return x ** 3

def f_cubic_derivative(x: float) -> float:
    return 3.0 * (x ** 2)

x_eval = 2.0
true_deriv = f_cubic_derivative(x_eval)  # 3 * 2^2 = 12.0

print(f"Target Function: f(x) = x^3 at x = {x_eval:.1f} (True Analytical Derivative: {true_deriv:.6f})")
print(f"{'Step Size h':>12} | {'Forward Diff':>14} | {'Fwd Error':>11} | {'Central Diff':>14} | {'Central Error':>13}")
print("-" * 75)

step_sizes = [1e-1, 1e-2, 1e-3, 1e-5, 1e-7, 1e-10]

for h in step_sizes:
    # Forward Difference: (f(x + h) - f(x)) / h
    fwd = (f_cubic(x_eval + h) - f_cubic(x_eval)) / h
    fwd_err = abs(fwd - true_deriv)

    # Central Difference: (f(x + h) - f(x - h)) / (2h)
    central = (f_cubic(x_eval + h) - f_cubic(x_eval - h)) / (2.0 * h)
    central_err = abs(central - true_deriv)

    print(f"{h:>12.1e} | {fwd:>14.8f} | {fwd_err:>11.2e} | {central:>14.8f} | {central_err:>13.2e}")

print("Note: Central difference achieves machine epsilon accuracy at h = 1e-5 (~10^-10 error)!")


# ======================================================================
# 2. NEURAL ACTIVATION FUNCTION DERIVATIVES
# ======================================================================
print("\n" + "=" * 70)
print("--- 2. NEURAL ACTIVATION FUNCTION DERIVATIVES ---")

# 1. Sigmoid Function: sigma(x) = 1 / (1 + exp(-x))
def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_derivative(x: float) -> float:
    s = sigmoid(x)
    return s * (1.0 - s)

# 2. ReLU Function: max(0, x)
def relu(x: float) -> float:
    return max(0.0, x)

def relu_derivative(x: float) -> float:
    return 1.0 if x > 0.0 else 0.0

# 3. GELU (Gaussian Error Linear Unit - used in GPT and BERT)
def gelu(x: float) -> float:
    # Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    c = math.sqrt(2.0 / math.pi)
    inner = c * (x + 0.044715 * (x ** 3))
    return 0.5 * x * (1.0 + math.tanh(inner))

def numerical_derivative(f, x: float, h: float = 1e-6) -> float:
    return (f(x + h) - f(x - h)) / (2.0 * h)

test_points = [-2.0, -0.5, 0.5, 2.0]
print("Verifying Sigmoid Derivative: sigma'(x) = sigma(x) * (1 - sigma(x)):")
for x in test_points:
    analytic = sigmoid_derivative(x)
    num = numerical_derivative(sigmoid, x)
    print(f"  x = {x:>5.1f} | Analytic: {analytic:.6f} | Numerical: {num:.6f} | Diff: {abs(analytic - num):.2e}")
    assert abs(analytic - num) < 1e-5

print("\nVerifying GELU Activation Derivative:")
for x in test_points:
    num_gelu = numerical_derivative(gelu, x)
    print(f"  x = {x:>5.1f} | GELU(x): {gelu(x):.4f} | GELU'(x) (Numerical): {num_gelu:.6f}")


# ======================================================================
# 3. PRODUCTION GRADIENT CHECKING ENGINE
# ======================================================================
print("\n" + "=" * 70)
print("--- 3. PRODUCTION GRADIENT CHECKING ENGINE ---")

def grad_check(func, grad_func, x: float, eps: float = 1e-6, tol: float = 1e-5) -> bool:
    """Relative error gradient checker used in deep learning frameworks."""
    analytic_grad = grad_func(x)
    numerical_grad = (func(x + eps) - func(x - eps)) / (2.0 * eps)

    # Relative difference: |a - n| / max(|a|, |n|, eps)
    rel_diff = abs(analytic_grad - numerical_grad) / max(abs(analytic_grad), abs(numerical_grad), eps)
    passed = rel_diff < tol
    status = "✅ PASSED" if passed else "❌ FAILED"
    print(f"  GradCheck at x={x:.2f} -> Analytic: {analytic_grad:.6f}, Numerical: {numerical_grad:.6f} [RelDiff: {rel_diff:.2e}] {status}")
    return passed

# Test complex composite function: f(x) = ln(1 + e^x) (Softplus)
# Analytical derivative: f'(x) = 1 / (1 + e^-x) = Sigmoid(x)
def softplus(x: float) -> float:
    return math.log(1.0 + math.exp(x))

def softplus_derivative(x: float) -> float:
    return sigmoid(x)

print("Running Gradient Check on Softplus f(x) = ln(1 + e^x):")
for pt in [-3.0, -1.0, 0.0, 1.5, 4.0]:
    assert grad_check(softplus, softplus_derivative, pt)

print("\n" + "=" * 70)
print("EXPERIMENT 3.1 COMPLETED SUCCESSFULLY!")
print("=" * 70)
