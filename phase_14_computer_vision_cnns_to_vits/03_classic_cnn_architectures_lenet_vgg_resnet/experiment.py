#!/usr/bin/env python3
"""
Phase 14: Computer Vision: CNNs to Vision Transformers
Topic 14.3: Classic CNN Architectures: LeNet, VGG, and ResNet

Pure Python 3 implementation (zero external dependencies).
Demonstrates:
  1. The Degradation Problem: Vanishing gradients in deep vanilla networks.
  2. ResNet Residual Block with skip connection: y = F(x) + x.
  3. The Gradient Highway Proof: dL/dx = dL/dy * (dF/dx + I).
  4. Simulation of gradient flow through a 20-block network (Vanilla vs. ResNet).
  5. Quantitative verification proving the elimination of vanishing gradients.
"""

import math
from typing import List, Tuple


# =====================================================================
# 1. Linear Algebra Vector & Matrix Utilities (Pure Python)
# =====================================================================

def dot(v1: List[float], v2: List[float]) -> float:
    return sum(a * b for a, b in zip(v1, v2))


def matvec(W: List[List[float]], x: List[float]) -> List[float]:
    """W @ x where W is [d_out, d_in] and x is [d_in]."""
    return [dot(row, x) for row in W]


def matvec_transposed(W: List[List[float]], grad: List[float]) -> List[float]:
    """W^T @ grad where W is [d_out, d_in] and grad is [d_out] -> [d_in]."""
    d_out = len(W)
    d_in = len(W[0])
    result = [0.0 for _ in range(d_in)]
    for j in range(d_in):
        s = 0.0
        for i in range(d_out):
            s += W[i][j] * grad[i]
        result[j] = s
    return result


def vec_add(v1: List[float], v2: List[float]) -> List[float]:
    return [a + b for a, b in zip(v1, v2)]


def vec_norm(v: List[float]) -> float:
    return math.sqrt(sum(a * a for a in v))


def relu(x: List[float]) -> List[float]:
    return [max(0.0, v) for v in x]


def relu_backward(grad: List[float], pre_act: List[float]) -> List[float]:
    return [g if z > 0 else 0.0 for g, z in zip(grad, pre_act)]


# =====================================================================
# 2. Vanilla Block vs. Residual Block
# =====================================================================

class VanillaBlock:
    """
    A single block of a plain / vanilla deep network:
      z = W x + b
      y = ReLU(z)
    """
    def __init__(self, dim: int, weight_scale: float = 0.7):
        self.dim = dim
        # Initialize a fixed deterministic orthogonal/scaled weight matrix
        # Using a slight contraction factor to simulate deep vanilla attenuation
        self.W = [[(weight_scale / math.sqrt(dim)) * (1.0 if (i + j) % 2 == 0 else -0.8) 
                   for j in range(dim)] for i in range(dim)]
        self.b = [0.01 for _ in range(dim)]
        self.cache = {}

    def forward(self, x: List[float]) -> List[float]:
        z = vec_add(matvec(self.W, x), self.b)
        y = relu(z)
        self.cache = {"x": x, "z": z}
        return y

    def backward(self, grad_y: List[float]) -> List[float]:
        z = self.cache["z"]
        grad_z = relu_backward(grad_y, z)
        grad_x = matvec_transposed(self.W, grad_z)
        return grad_x


class ResidualBlock:
    """
    A ResNet Residual Block with identity shortcut:
      F(x) = ReLU(W2 * ReLU(W1 * x + b1) + b2)
      y = F(x) + x
    """
    def __init__(self, dim: int, weight_scale: float = 0.7):
        self.dim = dim
        self.W1 = [[(weight_scale / math.sqrt(dim)) * (1.0 if (i + j) % 2 == 0 else -0.8) 
                    for j in range(dim)] for i in range(dim)]
        self.b1 = [0.01 for _ in range(dim)]
        self.W2 = [[(weight_scale / math.sqrt(dim)) * (0.8 if (i + j) % 2 == 1 else -0.6) 
                    for j in range(dim)] for i in range(dim)]
        self.b2 = [0.0 for _ in range(dim)]
        self.cache = {}

    def forward(self, x: List[float]) -> List[float]:
        # Branch F(x)
        z1 = vec_add(matvec(self.W1, x), self.b1)
        a1 = relu(z1)
        z2 = vec_add(matvec(self.W2, a1), self.b2)
        fx = relu(z2)
        # Skip connection: y = F(x) + x
        y = vec_add(fx, x)

        self.cache = {"x": x, "z1": z1, "a1": a1, "z2": z2}
        return y

    def backward(self, grad_y: List[float]) -> List[float]:
        """
        Gradient Highway:
          dL/dx = dL/dy * (dF/dx + I) = dF_grad + dL/dy
        """
        z2 = self.cache["z2"]
        a1 = self.cache["a1"]
        z1 = self.cache["z1"]

        # 1. Backprop through F(x)
        grad_z2 = relu_backward(grad_y, z2)
        grad_a1 = matvec_transposed(self.W2, grad_z2)
        grad_z1 = relu_backward(grad_a1, z1)
        grad_fx = matvec_transposed(self.W1, grad_z1)

        # 2. Gradient Highway addition: grad_x = grad_fx + grad_y (unhindered identity flow!)
        grad_x = vec_add(grad_fx, grad_y)
        return grad_x


# =====================================================================
# 3. 20-Layer Gradient Flow Simulation
# =====================================================================

def simulate_20_block_networks():
    dim = 16
    depth = 20
    print(f"\nSimulating forward and backward propagation through {depth} sequential blocks (dim={dim}):")

    # Initial input vector x_0
    x_init = [1.0 / math.sqrt(dim) for _ in range(dim)]
    upstream_grad = [1.0 for _ in range(dim)]
    init_grad_norm = vec_norm(upstream_grad)

    # -------------------------------------------------------------
    # 1. Plain / Vanilla Network of 20 Blocks
    # -------------------------------------------------------------
    vanilla_blocks = [VanillaBlock(dim, weight_scale=0.7) for _ in range(depth)]
    
    # Forward pass
    curr_x = x_init[:]
    for block in vanilla_blocks:
        curr_x = block.forward(curr_x)
    
    # Backward pass & gradient tracking
    curr_grad = upstream_grad[:]
    vanilla_grad_norms = [vec_norm(curr_grad)]
    for i in reversed(range(depth)):
        curr_grad = vanilla_blocks[i].backward(curr_grad)
        vanilla_grad_norms.append(vec_norm(curr_grad))

    vanilla_input_grad_norm = vec_norm(curr_grad)

    # -------------------------------------------------------------
    # 2. Residual Network of 20 Blocks
    # -------------------------------------------------------------
    resnet_blocks = [ResidualBlock(dim, weight_scale=0.7) for _ in range(depth)]
    
    # Forward pass
    curr_x = x_init[:]
    for block in resnet_blocks:
        curr_x = block.forward(curr_x)
    
    # Backward pass & gradient tracking
    curr_grad = upstream_grad[:]
    resnet_grad_norms = [vec_norm(curr_grad)]
    for i in reversed(range(depth)):
        curr_grad = resnet_blocks[i].backward(curr_grad)
        resnet_grad_norms.append(vec_norm(curr_grad))

    resnet_input_grad_norm = vec_norm(curr_grad)

    # -------------------------------------------------------------
    # Step 3: Print and Assert Comparison
    # -------------------------------------------------------------
    print("\nBlock-by-Block Gradient Norm Flow (from Output Block 20 back to Input Layer 0):")
    print(f"  {'Step (Backprop)':<18} | {'Vanilla Grad Norm':<20} | {'ResNet Grad Norm':<20}")
    print("  " + "-" * 64)
    for step in range(depth + 1):
        if step % 4 == 0 or step == depth:
            print(f"  Block {20 - step:>2} (Backprop {step:>2}) | {vanilla_grad_norms[step]:<20.6e} | {resnet_grad_norms[step]:<20.6f}")

    print("\nSummary of Final Gradient Reaching Input Layer (x_0):")
    print(f"  Original Output Loss Gradient Norm: {init_grad_norm:.6f}")
    print(f"  Vanilla Network Input Gradient Norm: {vanilla_input_grad_norm:.6e}")
    print(f"  ResNet Input Gradient Norm:          {resnet_input_grad_norm:.6f}")

    # Assertions:
    # 1. Vanilla network gradient has collapsed severely (vanishing gradient)
    assert vanilla_input_grad_norm < 1e-6, f"Expected vanilla gradient to vanish, got {vanilla_input_grad_norm}"
    # 2. ResNet gradient remains healthy and unhindered (order of magnitude ~ 1.0 or higher)
    assert resnet_input_grad_norm > 0.5, f"Expected healthy ResNet gradient, got {resnet_input_grad_norm}"

    print("\n[VERIFIED] ResNet completely eliminates vanishing gradients through its identity shortcut highway!")


# =====================================================================
# 4. VGG-Style Stacking Mathematics Check
# =====================================================================

def verify_vgg_stacking_math():
    print("\n--- Verifying VGG 3x3 Stacking Factorization ---")
    C = 64
    # Two 3x3 convs:
    weights_two_3x3 = 2 * (C * C * 3 * 3)
    # One 5x5 conv:
    weights_one_5x5 = 1 * (C * C * 5 * 5)

    reduction = (weights_one_5x5 - weights_two_3x3) / weights_one_5x5
    print(f"  Weights for two 3x3 conv layers: {weights_two_3x3:,}")
    print(f"  Weights for one 5x5 conv layer:  {weights_one_5x5:,}")
    print(f"  Parameter reduction: {reduction * 100:.2f}% (Expected: 28.00%)")
    assert math.isclose(reduction, 0.28, abs_tol=1e-5)

    # Three 3x3 convs vs one 7x7 conv:
    weights_three_3x3 = 3 * (C * C * 3 * 3)
    weights_one_7x7 = 1 * (C * C * 7 * 7)
    reduction_7x7 = (weights_one_7x7 - weights_three_3x3) / weights_one_7x7
    print(f"  Weights for three 3x3 conv layers: {weights_three_3x3:,}")
    print(f"  Weights for one 7x7 conv layer:    {weights_one_7x7:,}")
    print(f"  Parameter reduction: {reduction_7x7 * 100:.2f}% (Expected: 44.90%)")
    assert math.isclose(reduction_7x7, 0.448979, abs_tol=1e-4)
    print("  VGG factorized convolution savings verified!")


def main():
    print("=" * 70)
    print("Phase 14 - Topic 14.3: Classic CNNs & ResNet Experiment")
    print("=" * 70)

    verify_vgg_stacking_math()
    simulate_20_block_networks()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED SUCCESSFULLY! (Exit Code 0)")
    print("=" * 70)


if __name__ == "__main__":
    main()
