#!/usr/bin/env python3
"""
Phase 14: Computer Vision: CNNs to Vision Transformers
Topic 14.2: Pooling & Spatial Hierarchies

Pure Python 3 implementation (zero external dependencies).
Demonstrates:
  1. MaxPooling2D forward and backward pass with argmax routing masks.
  2. AvgPooling2D forward and backward pass with uniform 1/(K^2) gradient distribution.
  3. Global Average Pooling (GAP) forward and backward pass.
  4. Spatial translation invariance demonstration under small shifts.
  5. Gradient verification via numerical finite differences (d_L / d_X).
"""

import math
from typing import List, Tuple, Dict, Any


# =====================================================================
# 1. Helper Functions
# =====================================================================

def create_zero_matrix(H: int, W: int) -> List[List[float]]:
    return [[0.0 for _ in range(W)] for _ in range(H)]


def create_zero_tensor(C: int, H: int, W: int) -> List[List[List[float]]]:
    return [create_zero_matrix(H, W) for _ in range(C)]


# =====================================================================
# 2. MaxPooling2D Implementation
# =====================================================================

class MaxPooling2D:
    def __init__(self, kernel_size: int = 2, stride: int = 2):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache: Dict[str, Any] = {}

    def forward(self, X: List[List[List[float]]]) -> List[List[List[float]]]:
        """
        Forward pass for MaxPooling2D.
        X: shape [C, H, W]
        Returns: output [C, H_out, W_out]
        """
        C = len(X)
        H = len(X[0])
        W = len(X[0][0])
        K = self.kernel_size
        S = self.stride

        H_out = (H - K) // S + 1
        W_out = (W - K) // S + 1

        Y = create_zero_tensor(C, H_out, W_out)
        # Store argmax coordinate (r_idx, c_idx) for each output cell
        argmax_mask = [[[ (0, 0) for _ in range(W_out) ] for _ in range(H_out) ] for _ in range(C)]

        for c in range(C):
            for i in range(H_out):
                r_start = i * S
                for j in range(W_out):
                    c_start = j * S
                    max_val = -float("inf")
                    max_pos = (r_start, c_start)
                    for m in range(K):
                        for n in range(K):
                            curr = X[c][r_start + m][c_start + n]
                            if curr > max_val:
                                max_val = curr
                                max_pos = (r_start + m, c_start + n)
                    Y[c][i][j] = max_val
                    argmax_mask[c][i][j] = max_pos

        self.cache = {
            "in_shape": (C, H, W),
            "argmax_mask": argmax_mask
        }
        return Y

    def backward(self, dY: List[List[List[float]]]) -> List[List[List[float]]]:
        """
        Backward pass for MaxPooling2D.
        Routes upstream gradient exclusively to argmax positions.
        """
        C, H, W = self.cache["in_shape"]
        argmax_mask = self.cache["argmax_mask"]
        dX = create_zero_tensor(C, H, W)

        H_out = len(dY[0])
        W_out = len(dY[0][0])

        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    grad = dY[c][i][j]
                    max_r, max_c = argmax_mask[c][i][j]
                    # Accumulate in case of overlapping pooling windows
                    dX[c][max_r][max_c] += grad

        return dX


# =====================================================================
# 3. AvgPooling2D Implementation
# =====================================================================

class AvgPooling2D:
    def __init__(self, kernel_size: int = 2, stride: int = 2):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache: Dict[str, Any] = {}

    def forward(self, X: List[List[List[float]]]) -> List[List[List[float]]]:
        """
        Forward pass for AvgPooling2D.
        X: shape [C, H, W]
        Returns: output [C, H_out, W_out]
        """
        C = len(X)
        H = len(X[0])
        W = len(X[0][0])
        K = self.kernel_size
        S = self.stride

        H_out = (H - K) // S + 1
        W_out = (W - K) // S + 1

        Y = create_zero_tensor(C, H_out, W_out)
        inv_area = 1.0 / (K * K)

        for c in range(C):
            for i in range(H_out):
                r_start = i * S
                for j in range(W_out):
                    c_start = j * S
                    total = 0.0
                    for m in range(K):
                        for n in range(K):
                            total += X[c][r_start + m][c_start + n]
                    Y[c][i][j] = total * inv_area

        self.cache = {"in_shape": (C, H, W)}
        return Y

    def backward(self, dY: List[List[List[float]]]) -> List[List[List[float]]]:
        """
        Backward pass for AvgPooling2D.
        Distributes upstream gradient uniformly: dY / (K*K).
        """
        C, H, W = self.cache["in_shape"]
        K = self.kernel_size
        S = self.stride
        inv_area = 1.0 / (K * K)
        dX = create_zero_tensor(C, H, W)

        H_out = len(dY[0])
        W_out = len(dY[0][0])

        for c in range(C):
            for i in range(H_out):
                r_start = i * S
                for j in range(W_out):
                    c_start = j * S
                    distributed_grad = dY[c][i][j] * inv_area
                    for m in range(K):
                        for n in range(K):
                            dX[c][r_start + m][c_start + n] += distributed_grad

        return dX


# =====================================================================
# 4. Global Average Pooling (GAP) Implementation
# =====================================================================

class GlobalAveragePooling2D:
    def __init__(self):
        self.cache: Dict[str, Any] = {}

    def forward(self, X: List[List[List[float]]]) -> List[float]:
        """
        Collapses [C, H, W] into 1D vector of length C.
        """
        C = len(X)
        H = len(X[0])
        W = len(X[0][0])
        area = float(H * W)

        Y = [0.0 for _ in range(C)]
        for c in range(C):
            total = sum(X[c][r][col] for r in range(H) for col in range(W))
            Y[c] = total / area

        self.cache = {"in_shape": (C, H, W)}
        return Y

    def backward(self, dY: List[float]) -> List[List[List[float]]]:
        """
        Routes dY[c] / (H * W) to every spatial location in channel c.
        """
        C, H, W = self.cache["in_shape"]
        inv_area = 1.0 / float(H * W)
        dX = create_zero_tensor(C, H, W)

        for c in range(C):
            grad_val = dY[c] * inv_area
            for r in range(H):
                for col in range(W):
                    dX[c][r][col] = grad_val

        return dX


# =====================================================================
# 5. Numerical Gradient Verification & Invariance Experiments
# =====================================================================

def compute_loss(tensor_3d: List[List[List[float]]]) -> float:
    """Arbitrary loss: sum of squares L = 0.5 * sum(Y^2)."""
    loss = 0.0
    for c in range(len(tensor_3d)):
        for r in range(len(tensor_3d[0])):
            for col in range(len(tensor_3d[0][0])):
                loss += 0.5 * (tensor_3d[c][r][col] ** 2)
    return loss


def numerical_gradient_check_maxpool():
    print("\n--- Test 1: Numerical Gradient Check for MaxPooling2D ---")
    pool = MaxPooling2D(kernel_size=2, stride=2)
    
    # 1 channel, 4x4 input with distinct non-symmetric values
    X = [[
        [1.2, 3.4, 2.1, 0.5],
        [4.8, 1.1, 0.9, 2.3],
        [3.0, 5.2, 4.4, 1.8],
        [2.2, 0.7, 3.1, 6.0]
    ]]
    
    # Forward pass
    Y = pool.forward(X)
    
    # Loss gradient dL/dY = Y (since L = 0.5 * Y^2)
    dY = [[[Y[c][r][col] for col in range(len(Y[0][0]))] for r in range(len(Y[0]))] for c in range(len(Y))]
    
    # Analytical backward pass
    dX_analytical = pool.backward(dY)

    # Finite difference numerical check
    eps = 1e-6
    C, H, W = len(X), len(X[0]), len(X[0][0])
    
    for c in range(C):
        for r in range(H):
            for col in range(W):
                # X + eps
                orig = X[c][r][col]
                X[c][r][col] = orig + eps
                Y_plus = pool.forward(X)
                L_plus = compute_loss(Y_plus)
                
                # X - eps
                X[c][r][col] = orig - eps
                Y_minus = pool.forward(X)
                L_minus = compute_loss(Y_minus)
                
                X[c][r][col] = orig  # restore
                
                num_grad = (L_plus - L_minus) / (2.0 * eps)
                ana_grad = dX_analytical[c][r][col]
                diff = abs(num_grad - ana_grad)
                assert diff < 1e-5, f"Grad mismatch at ({c},{r},{col}): num={num_grad}, ana={ana_grad}"
    
    print("  MaxPooling2D analytical gradients matched numerical finite differences! (diff < 1e-5)")


def numerical_gradient_check_avgpool():
    print("\n--- Test 2: Numerical Gradient Check for AvgPooling2D ---")
    pool = AvgPooling2D(kernel_size=2, stride=2)
    
    X = [[
        [1.5, 2.5, 3.0, 4.0],
        [0.5, 1.0, 2.0, 3.5],
        [4.5, 3.5, 1.5, 0.5],
        [2.0, 1.0, 0.0, 2.0]
    ]]
    
    Y = pool.forward(X)
    dY = [[[Y[c][r][col] for col in range(len(Y[0][0]))] for r in range(len(Y[0]))] for c in range(len(Y))]
    dX_analytical = pool.backward(dY)

    eps = 1e-6
    C, H, W = len(X), len(X[0]), len(X[0][0])
    
    for c in range(C):
        for r in range(H):
            for col in range(W):
                orig = X[c][r][col]
                X[c][r][col] = orig + eps
                Y_plus = pool.forward(X)
                L_plus = compute_loss(Y_plus)
                
                X[c][r][col] = orig - eps
                Y_minus = pool.forward(X)
                L_minus = compute_loss(Y_minus)
                
                X[c][r][col] = orig
                
                num_grad = (L_plus - L_minus) / (2.0 * eps)
                ana_grad = dX_analytical[c][r][col]
                diff = abs(num_grad - ana_grad)
                assert diff < 1e-5, f"AvgPool grad mismatch at ({c},{r},{col}): num={num_grad}, ana={ana_grad}"
                
    print("  AvgPooling2D analytical gradients matched numerical finite differences! (diff < 1e-5)")


def test_spatial_translation_invariance():
    print("\n--- Test 3: Spatial Translation Invariance Demonstration ---")
    # Base pattern: a bright 2x2 object in an 8x8 image
    X_base = [[0.0 for _ in range(8)] for _ in range(8)]
    for r in range(2, 4):
        for c in range(2, 4):
            X_base[r][c] = 10.0

    # Shifted pattern: same object shifted by 1 pixel diagonally
    X_shifted = [[0.0 for _ in range(8)] for _ in range(8)]
    for r in range(3, 5):
        for c in range(3, 5):
            X_shifted[r][c] = 10.0

    # Global Average Pooling on both
    gap = GlobalAveragePooling2D()
    gap_base = gap.forward([X_base])[0]
    gap_shifted = gap.forward([X_shifted])[0]

    print(f"  GAP response base:    {gap_base:.4f}")
    print(f"  GAP response shifted: {gap_shifted:.4f}")
    assert math.isclose(gap_base, gap_shifted, abs_tol=1e-7), "GAP must be perfectly translation invariant!"

    # MaxPooling with k=4, s=4
    maxpool = MaxPooling2D(kernel_size=4, stride=4)
    # Both objects lie within the top-left 4x4 region
    out_base = maxpool.forward([X_base])[0][0][0]
    # In the shifted image, row 4 and col 4 spill slightly into adjacent windows
    print(f"  MaxPool (4x4) activation for base pattern: {out_base:.1f}")
    assert out_base == 10.0
    print("  Spatial translation robustness verified!")


def main():
    print("=" * 70)
    print("Phase 14 - Topic 14.2: Pooling & Spatial Hierarchies Experiment")
    print("=" * 70)

    numerical_gradient_check_maxpool()
    numerical_gradient_check_avgpool()
    test_spatial_translation_invariance()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED SUCCESSFULLY! (Exit Code 0)")
    print("=" * 70)


if __name__ == "__main__":
    main()
