#!/usr/bin/env python3
"""
EXPERIMENT 3.3: MICRO-AUTOGRAD ENGINE & REVERSE-MODE BACKPROPAGATION
===================================================================
A hands-on, zero-dependency Python laboratory demonstrating:
1. Micro-Autograd Engine built from scratch (like PyTorch / micrograd)
2. Dynamic Computational Graph Construction (Directed Acyclic Graph)
3. Reverse-Mode Topological Traversal & Adjoint Backpropagation
4. Neural Network Forward-Backward Training Step with Loss Minimization
5. High-Precision Numerical Gradient Checking of All Leaf Weights

Runs on standard library Python 3 without external packages.
"""

import math

print("=" * 70)
print("EXPERIMENT 3.3: MICRO-AUTOGRAD ENGINE & BACKPROPAGATION")
print("=" * 70)


class Value:
    """Stores a scalar value and its accumulated adjoint gradient."""

    def __init__(self, data: float, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # d(x + y)/dx = 1, d(x + y)/dy = 1
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # d(x * y)/dx = y, d(x * y)/dy = x
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __sub__(self, other) -> "Value":
        return self + (-other)

    def __neg__(self) -> "Value":
        return self * -1.0

    def relu(self) -> "Value":
        out = Value(max(0.0, self.data), (self,), "ReLU")

        def _backward():
            self.grad += (1.0 if self.data > 0.0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> "Value":
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            # d(tanh(x))/dx = 1 - tanh(x)^2
            self.grad += (1.0 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        """Topological sort and reverse-mode gradient propagation."""
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Root loss gradient dL/dL = 1.0
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __repr__(self) -> str:
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"


# ======================================================================
# 1. VERIFYING CHAIN RULE ON MULTI-PATH GRAPH
# ======================================================================
print("\n--- 1. MULTI-PATH COMPUTATIONAL GRAPH VERIFICATION ---")

# Compute f(a, b) = (a * b) + (a + b)
# Notice 'a' branches into two separate paths!
# df/da = b + 1
# df/db = a + 1
a = Value(3.0)
b = Value(4.0)

term1 = a * b       # 12.0
term2 = a + b       # 7.0
loss = term1 + term2 # 19.0

loss.backward()

print(f"Forward Result: f(3.0, 4.0) = {loss.data:.1f} (Expected: 19.0)")
print(f"Autograd Gradient df/da: {a.grad:.4f} (Theoretical: b + 1 = 5.0000)")
print(f"Autograd Gradient df/db: {b.grad:.4f} (Theoretical: a + 1 = 4.0000)")
assert a.grad == 5.0 and b.grad == 4.0
print("✅ Multi-path gradient accumulation verified!")


# ======================================================================
# 2. NEURAL NETWORK FORWARD-BACKWARD PASS & NUMERICAL GRADCHECK
# ======================================================================
print("\n" + "=" * 70)
print("--- 2. NEURAL NETWORK TRAINING STEP & GRADIENT CHECKING ---")

# Inputs x1, x2
x1 = Value(2.0)
x2 = Value(-1.0)
target = 1.0

# Trainable Weights w1, w2 and Bias b
w1 = Value(0.5)
w2 = Value(-1.5)
b = Value(0.1)

def forward_network(w1_v, w2_v, b_v):
    # Linear projection: z = w1*x1 + w2*x2 + b
    z = (w1_v * x1) + (w2_v * x2) + b_v
    # Activation: a = tanh(z)
    pred = z.tanh()
    # Loss: MSE = (pred - target)^2
    diff = pred - target
    loss_val = diff * diff
    return loss_val, pred

loss, pred = forward_network(w1, w2, b)
loss.backward()

print(f"Forward Pass -> Prediction: {pred.data:.4f} | Target: {target:.1f} | Loss: {loss.data:.4f}")
print("Autograd Computed Gradients:")
print(f"  dL/dw1: {w1.grad:.6f}")
print(f"  dL/dw2: {w2.grad:.6f}")
print(f"  dL/db:  {b.grad:.6f}")

# Numerical Gradient Check (Finite Differences)
print("\nRunning Finite Difference Gradient Checks on Leaf Weights:")
eps = 1e-6

for name, val_obj, get_loss in [
    ("w1", w1, lambda delta: forward_network(Value(w1.data + delta), w2, b)[0].data),
    ("w2", w2, lambda delta: forward_network(w1, Value(w2.data + delta), b)[0].data),
    ("b",  b,  lambda delta: forward_network(w1, w2, Value(b.data + delta))[0].data)
]:
    loss_plus = get_loss(+eps)
    loss_minus = get_loss(-eps)
    num_grad = (loss_plus - loss_minus) / (2.0 * eps)
    analytic_grad = val_obj.grad
    err = abs(analytic_grad - num_grad)
    print(f"  {name:>2} | Autograd: {analytic_grad:.6f} | Numerical: {num_grad:.6f} | Error: {err:.2e} ✅")
    assert err < 1e-5

print("\n" + "=" * 70)
print("EXPERIMENT 3.3 COMPLETED SUCCESSFULLY!")
print("=" * 70)
