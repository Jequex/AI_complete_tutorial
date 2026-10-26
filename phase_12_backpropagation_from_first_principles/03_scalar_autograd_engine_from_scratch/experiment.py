#!/usr/bin/env python3
"""
Phase 12 - Topic 03: Scalar Autograd Engine From Scratch
Experiment: Micrograd-Style Scalar Autograd Engine with Topological Sorting,
Building a 2-Layer Neural Network, and Validating Against Finite Differences (< 1e-7).

Pure Python 3 standard library implementation. Zero external dependencies.
"""

import math
import random
import sys
from typing import List, Set, Tuple


class Value:
    """
    Scalar computational node supporting automatic reverse-mode differentiation.
    Tracks numerical data, accumulated gradient, previous nodes in DAG,
    and backward differentiation closure.
    """
    __slots__ = ("data", "grad", "_prev", "_op", "_backward")

    def __init__(self, data: float, _children: Tuple["Value", ...] = (), _op: str = ""):
        self.data: float = float(data)
        self.grad: float = 0.0
        self._prev: Set["Value"] = set(_children)
        self._op: str = _op
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f"Value(data={self.data:.6f}, grad={self.grad:.6f})"

    # --- Elementary Arithmetic Operations ---

    def __add__(self, other) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # d(x + y)/dx = 1, d(x + y)/dy = 1
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other) -> "Value":
        return self.__add__(other)

    def __neg__(self) -> "Value":
        return self * -1.0

    def __sub__(self, other) -> "Value":
        return self + (-other)

    def __rsub__(self, other) -> "Value":
        return Value(other) + (-self)

    def __mul__(self, other) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # Product rule: d(a * b)/da = b, d(a * b)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other) -> "Value":
        return self.__mul__(other)

    def __truediv__(self, other) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        return self * (other ** -1.0)

    def __rtruediv__(self, other) -> "Value":
        return Value(other) * (self ** -1.0)

    def __pow__(self, p: float) -> "Value":
        assert isinstance(p, (int, float)), "Power must be scalar float or int"
        out = Value(self.data ** p, (self,), f"**{p}")

        def _backward():
            # Power rule: d(x^p)/dx = p * x^(p-1)
            self.grad += (p * (self.data ** (p - 1.0))) * out.grad

        out._backward = _backward
        return out

    # --- Elementary Nonlinear Transcendental Functions ---

    def exp(self) -> "Value":
        x = self.data
        out = Value(math.exp(x), (self,), "exp")

        def _backward():
            # d(exp(x))/dx = exp(x) = out.data
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self) -> "Value":
        assert self.data > 0, f"Cannot take logarithm of non-positive value: {self.data}"
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            # d(ln(x))/dx = 1 / x
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> "Value":
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward():
            # Subgradient of ReLU: 1.0 if x > 0 else 0.0
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> "Value":
        x = self.data
        t = math.tanh(x)
        out = Value(t, (self,), "tanh")

        def _backward():
            # d(tanh(x))/dx = 1 - tanh^2(x)
            self.grad += (1.0 - t ** 2) * out.grad

        out._backward = _backward
        return out

    # --- Reverse Topological Traversal ---

    def backward(self) -> None:
        """
        Executes backpropagation on the computational DAG using
        Depth-First Search (DFS) topological ordering.
        """
        topo: List["Value"] = []
        visited: Set["Value"] = set()

        def build_topo(v: "Value") -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Seed terminal loss gradient
        self.grad = 1.0

        # Traverse graph in reverse topological order
        for node in reversed(topo):
            node._backward()


# =====================================================================
# Neural Network Architecture: 2-Layer MLP
# =====================================================================

class Neuron:
    def __init__(self, nin: int, nonlin: bool = True):
        # Deterministic seed initialization for repeatability
        self.w = [Value(random.uniform(-0.8, 0.8)) for _ in range(nin)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x: List[Value]) -> Value:
        # act = sum(w_i * x_i) + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh() if self.nonlin else act

    def parameters(self) -> List[Value]:
        return self.w + [self.b]


class Layer:
    def __init__(self, nin: int, nout: int, nonlin: bool = True):
        self.neurons = [Neuron(nin, nonlin) for _ in range(nout)]

    def __call__(self, x: List[Value]) -> List[Value]:
        return [n(x) for n in self.neurons]

    def parameters(self) -> List[Value]:
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self, nin: int, nouts: List[int]):
        sz = [nin] + nouts
        self.layers = [
            Layer(sz[i], sz[i+1], nonlin=(i != len(nouts)-1))
            for i in range(len(nouts))
        ]

    def __call__(self, x: List[Value]) -> Value:
        out = x
        for layer in self.layers:
            out = layer(out)
        return out[0]  # Return single scalar output for binary/regression

    def parameters(self) -> List[Value]:
        return [p for layer in self.layers for p in layer.parameters()]


def run_experiment() -> None:
    print("=" * 78)
    print("TOPIC 03 EXPERIMENT: SCALAR AUTOGRAD ENGINE FROM SCRATCH")
    print("=" * 78)

    # 1. Test basic graph differentiation with multi-branch fan-out
    print("\n--- 1. Testing Value Autograd on Multi-Branch Graph with Reused Variables ---")
    x = Value(2.5)
    # y = x * x + x^3 + tanh(x) (x is used across three branches)
    y = (x * x) + (x ** 3) + x.tanh()
    y.backward()

    # Analytical derivative:
    # dy/dx = 2x + 3x^2 + (1 - tanh^2(x))
    analytical_grad = 2.0 * 2.5 + 3.0 * (2.5 ** 2) + (1.0 - math.tanh(2.5) ** 2)
    err = abs(x.grad - analytical_grad)

    print(f"Computed x.grad:     {x.grad:.8f}")
    print(f"Analytical Formula:  {analytical_grad:.8f}")
    print(f"Discrepancy Error:   {err:.2e}")
    assert err < 1e-12, f"Gradient calculation failed: {err}"
    print("=> Multi-branch variable gradient accumulation verified!")

    # 2. Build 2-Layer Micro MLP (Architecture: 2 -> 4 -> 1)
    print("\n--- 2. Constructing 2-Layer MLP (Inputs: 2, Hidden: 4, Output: 1) ---")
    random.seed(42)
    model = MLP(nin=2, nouts=[4, 1])
    params = model.parameters()
    print(f"Total Model Parameters: {len(params)} (Layer 1: 4*2 + 4 = 12; Layer 2: 1*4 + 1 = 5; Total = 17)")

    # Training dataset: 4 samples (XOR-like problem)
    xs = [
        [Value(1.0), Value(1.0)],
        [Value(-1.0), Value(1.0)],
        [Value(1.0), Value(-1.0)],
        [Value(-1.0), Value(-1.0)],
    ]
    ys = [Value(0.5), Value(-0.5), Value(-0.5), Value(0.5)]

    def compute_loss() -> Value:
        total_loss = Value(0.0)
        for x_in, y_target in zip(xs, ys):
            pred = model(x_in)
            diff = pred - y_target
            total_loss = total_loss + (diff * diff)  # Squared error
        return total_loss

    # Forward pass
    initial_loss = compute_loss()
    print(f"Initial Loss before backprop: {initial_loss.data:.6f}")

    # Backward pass
    for p in params:
        p.grad = 0.0
    initial_loss.backward()

    # 3. Finite Difference Gradient Checking on ALL 17 Parameters!
    print("\n--- 3. Numerical Gradient Checking (Central Differences eps=1e-7) ---")
    eps = 1e-7
    max_relative_error = 0.0

    print(f"{'Param #':<8} | {'Analytical Grad':<18} | {'Numerical Grad':<18} | {'Rel Error':<12}")
    print("-" * 62)

    for i, p in enumerate(params):
        analytical_g = p.grad

        # Perturb +eps
        p.data += eps
        loss_plus = compute_loss().data

        # Perturb -eps
        p.data -= 2.0 * eps
        loss_minus = compute_loss().data

        # Restore original parameter value
        p.data += eps

        # Central difference
        numerical_g = (loss_plus - loss_minus) / (2.0 * eps)

        # Numerical gradient checking metric:
        # Standard relative error: |g_a - g_n| / (|g_a| + |g_n|)
        # When both gradients are near machine zero (< 1e-7), absolute error is checked
        if abs(analytical_g) < 1e-7 and abs(numerical_g) < 1e-7:
            rel_err = abs(analytical_g - numerical_g)
        else:
            rel_err = abs(analytical_g - numerical_g) / (abs(analytical_g) + abs(numerical_g))

        if rel_err > max_relative_error:
            max_relative_error = rel_err

        if i < 6 or i >= len(params) - 2:  # Print first few and last few
            print(f"P[{i:<2}]     | {analytical_g:18.8f} | {numerical_g:18.8f} | {rel_err:.2e}")
        elif i == 6:
            print("  ... (remaining hidden layer parameters verified) ...")

    print(f"\nMaximum Relative Error across ALL {len(params)} parameters: {max_relative_error:.2e}")
    assert max_relative_error < 1e-7, f"GradCheck failed! Max relative error {max_relative_error} >= 1e-7"
    print("=> GRADIENT CHECK PASSED: Every parameter derivative matches finite difference within 1e-7!")

    # 4. Perform One Gradient Descent Step
    print("\n--- 4. Executing One Gradient Descent Step ---")
    learning_rate = 0.05
    for p in params:
        p.data -= learning_rate * p.grad

    new_loss = compute_loss()
    print(f"Loss after 1 gradient step: {new_loss.data:.6f}")
    loss_reduction = initial_loss.data - new_loss.data
    print(f"Loss Reduction:             {loss_reduction:.6f}")
    assert new_loss.data < initial_loss.data, "Gradient descent failed to reduce loss!"
    print("=> OPTIMIZATION VERIFIED: Loss decreased strictly in direction of negative gradient.")

    print("\n" + "=" * 78)
    print("ALL TESTS PASSED SUCCESSFULLY! Scalar autograd engine fully verified.")
    print("=" * 78)


if __name__ == "__main__":
    run_experiment()
    sys.exit(0)
