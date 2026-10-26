#!/usr/bin/env python3
"""
Phase 12 - Topic 04: Tensor Backpropagation and Gradient Checking
Experiment: 2D Tensor Linear Layer (Y = XW + b) with Analytical Backprop
and Automated Numerical GradCheck via Two-Sided Central Differences (< 1e-7).

Pure Python 3 standard library implementation. Zero external dependencies.
"""

import math
import random
import sys
from typing import Callable, List, Tuple


# Matrix type alias: List of lists of floats
Matrix = List[List[float]]


# =====================================================================
# Matrix Utility Functions (Pure Python)
# =====================================================================

def mat_zeros(rows: int, cols: int) -> Matrix:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def mat_random(rows: int, cols: int, low: float = -1.0, high: float = 1.0) -> Matrix:
    return [[random.uniform(low, high) for _ in range(cols)] for _ in range(rows)]


def mat_mul(A: Matrix, B: Matrix) -> Matrix:
    rA, cA = len(A), len(A[0])
    rB, cB = len(B), len(B[0])
    assert cA == rB, f"Matrix dimension mismatch: ({rA}x{cA}) and ({rB}x{cB})"
    out = mat_zeros(rA, cB)
    for i in range(rA):
        for k in range(cA):
            a_ik = A[i][k]
            for j in range(cB):
                out[i][j] += a_ik * B[k][j]
    return out


def mat_transpose(A: Matrix) -> Matrix:
    rows, cols = len(A), len(A[0])
    out = mat_zeros(cols, rows)
    for i in range(rows):
        for j in range(cols):
            out[j][i] = A[i][j]
    return out


def mat_add_broadcast_row(A: Matrix, b: Matrix) -> Matrix:
    """Adds row vector b (1 x C) to each row of matrix A (R x C)."""
    rows, cols = len(A), len(A[0])
    assert len(b) == 1 and len(b[0]) == cols, f"Bias dimension mismatch: ({len(b)}x{len(b[0])}) vs (1x{cols})"
    out = mat_zeros(rows, cols)
    for i in range(rows):
        for j in range(cols):
            out[i][j] = A[i][j] + b[0][j]
    return out


def frobenius_norm(A: Matrix) -> float:
    total = sum(val ** 2 for row in A for val in row)
    return math.sqrt(total)


def mat_sub(A: Matrix, B: Matrix) -> Matrix:
    rows, cols = len(A), len(A[0])
    return [[A[i][j] - B[i][j] for j in range(cols)] for i in range(rows)]


# =====================================================================
# Linear Layer with Tensor Forward and Backward Passes
# =====================================================================

class LinearLayer:
    """
    Affine transformation layer: Y = X * W + b
    Dimensions:
        X: (B, Din)
        W: (Din, Dout)
        b: (1, Dout)
        Y: (B, Dout)
    """
    def __init__(self, din: int, dout: int):
        self.din = din
        self.dout = dout
        self.W: Matrix = mat_random(din, dout, -0.5, 0.5)
        self.b: Matrix = mat_random(1, dout, -0.1, 0.1)

        # Activations cached during forward pass
        self.X_cache: Matrix = []

        # Gradients
        self.grad_W: Matrix = mat_zeros(din, dout)
        self.grad_b: Matrix = mat_zeros(1, dout)
        self.grad_X: Matrix = []

    def forward(self, X: Matrix) -> Matrix:
        self.X_cache = [row[:] for row in X]
        # Y = XW + b
        XW = mat_mul(X, self.W)
        Y = mat_add_broadcast_row(XW, self.b)
        return Y

    def backward(self, grad_Y: Matrix) -> Matrix:
        """
        Matrix Calculus Derivations:
            dL/dX = grad_Y * W^T       shape: (B, Dout) * (Dout, Din) = (B, Din)
            dL/dW = X^T * grad_Y       shape: (Din, B) * (B, Dout) = (Din, Dout)
            dL/db = sum_{batch} grad_Y shape: (1, Dout)
        """
        B = len(grad_Y)
        Din = self.din
        Dout = self.dout

        # 1. dL/dW = X^T * grad_Y
        X_T = mat_transpose(self.X_cache)
        self.grad_W = mat_mul(X_T, grad_Y)

        # 2. dL/db = sum across batch rows
        self.grad_b = mat_zeros(1, Dout)
        for i in range(B):
            for j in range(Dout):
                self.grad_b[0][j] += grad_Y[i][j]

        # 3. dL/dX = grad_Y * W^T
        W_T = mat_transpose(self.W)
        self.grad_X = mat_mul(grad_Y, W_T)

        return self.grad_X


# =====================================================================
# Scalar Loss Objective for Testing: Frobenius Mean Squared Error
# L = 0.5 * sum((Y - Y_target)^2)
# dL/dY = Y - Y_target
# =====================================================================

def compute_loss(Y: Matrix, Y_target: Matrix) -> Tuple[float, Matrix]:
    B, Dout = len(Y), len(Y[0])
    loss = 0.0
    grad_Y = mat_zeros(B, Dout)
    for i in range(B):
        for j in range(Dout):
            diff = Y[i][j] - Y_target[i][j]
            loss += 0.5 * (diff ** 2)
            grad_Y[i][j] = diff
    return loss, grad_Y


# =====================================================================
# Automated GradCheck Engine (Central Differences)
# =====================================================================

def relative_error(analytical: Matrix, numerical: Matrix, eps_denom: float = 1e-12) -> float:
    """
    Computes standard relative error metric:
        RelErr = ||g_num - g_analytical||_2 / (||g_num||_2 + ||g_analytical||_2 + eps)
    """
    diff_matrix = mat_sub(numerical, analytical)
    diff_norm = frobenius_norm(diff_matrix)
    num_norm = frobenius_norm(numerical)
    ana_norm = frobenius_norm(analytical)

    denom = num_norm + ana_norm + eps_denom
    return diff_norm / denom


def gradcheck_matrix(
    loss_fn: Callable[[Matrix], float],
    param: Matrix,
    analytical_grad: Matrix,
    eps: float = 1e-7,
) -> Tuple[float, Matrix]:
    """
    Computes two-sided central difference numerical gradient for every entry:
        g_num[i][j] = (L(theta + eps) - L(theta - eps)) / (2 * eps)
    """
    rows, cols = len(param), len(param[0])
    numerical_grad = mat_zeros(rows, cols)

    for i in range(rows):
        for j in range(cols):
            orig_val = param[i][j]

            # Perturb +eps
            param[i][j] = orig_val + eps
            loss_plus = loss_fn(param)

            # Perturb -eps
            param[i][j] = orig_val - eps
            loss_minus = loss_fn(param)

            # Restore original
            param[i][j] = orig_val

            # Central difference
            numerical_grad[i][j] = (loss_plus - loss_minus) / (2.0 * eps)

    rel_err = relative_error(analytical_grad, numerical_grad)
    return rel_err, numerical_grad


def run_experiment() -> None:
    print("=" * 78)
    print("TOPIC 04 EXPERIMENT: TENSOR BACKPROPAGATION & GRADIENT CHECKING")
    print("=" * 78)

    random.seed(42)

    # Dimensional configuration
    B = 4      # Batch size
    Din = 3    # Input features
    Dout = 2   # Output features

    print(f"\nConfiguration:")
    print(f"  Batch size (B):      {B}")
    print(f"  Input Dim (Din):     {Din}")
    print(f"  Output Dim (Dout):   {Dout}")
    print(f"  Linear Transform:    Y = XW + b")
    print(f"  X shape: ({B}, {Din}) | W shape: ({Din}, {Dout}) | b shape: (1, {Dout})")

    # Instantiate layer and test tensors
    layer = LinearLayer(Din, Dout)
    X = mat_random(B, Din, -1.0, 1.0)
    Y_target = mat_random(B, Dout, -1.0, 1.0)

    # 1. Forward Pass
    Y = layer.forward(X)
    loss, grad_Y = compute_loss(Y, Y_target)
    print(f"\nForward Pass Loss: {loss:.8f}")

    # 2. Backward Pass (Analytical Matrix Calculus)
    grad_X = layer.backward(grad_Y)
    grad_W = layer.grad_W
    grad_b = layer.grad_b

    print("\nAnalytical Matrix Gradients Computed:")
    print(f"  dL/dW shape: ({len(grad_W)}, {len(grad_W[0])}) | Frobenius norm: {frobenius_norm(grad_W):.6f}")
    print(f"  dL/db shape: ({len(grad_b)}, {len(grad_b[0])}) | Frobenius norm: {frobenius_norm(grad_b):.6f}")
    print(f"  dL/dX shape: ({len(grad_X)}, {len(grad_X[0])}) | Frobenius norm: {frobenius_norm(grad_X):.6f}")

    # 3. Automated GradCheck on W
    print("\n" + "-" * 78)
    print("Running GradCheck on Weight Matrix W:")
    def loss_w(W_pert: Matrix) -> float:
        orig_W = layer.W
        layer.W = W_pert
        Y_pert = mat_add_broadcast_row(mat_mul(X, layer.W), layer.b)
        l, _ = compute_loss(Y_pert, Y_target)
        layer.W = orig_W
        return l

    rel_err_W, num_W = gradcheck_matrix(loss_w, layer.W, grad_W)
    print(f"  Frobenius ||g_num - g_ana||: {frobenius_norm(mat_sub(num_W, grad_W)):.2e}")
    print(f"  Relative Error for W:       {rel_err_W:.2e}")
    assert rel_err_W < 1e-7, f"GradCheck for W failed! {rel_err_W} >= 1e-7"
    print("  => Weight Matrix W GRADCHECK PASSED (< 1e-7)!")

    # 4. Automated GradCheck on Bias b
    print("\n" + "-" * 78)
    print("Running GradCheck on Bias Vector b:")
    def loss_b(b_pert: Matrix) -> float:
        orig_b = layer.b
        layer.b = b_pert
        Y_pert = mat_add_broadcast_row(mat_mul(X, layer.W), layer.b)
        l, _ = compute_loss(Y_pert, Y_target)
        layer.b = orig_b
        return l

    rel_err_b, num_b = gradcheck_matrix(loss_b, layer.b, grad_b)
    print(f"  Frobenius ||g_num - g_ana||: {frobenius_norm(mat_sub(num_b, grad_b)):.2e}")
    print(f"  Relative Error for b:       {rel_err_b:.2e}")
    assert rel_err_b < 1e-7, f"GradCheck for b failed! {rel_err_b} >= 1e-7"
    print("  => Bias Vector b GRADCHECK PASSED (< 1e-7)!")

    # 5. Automated GradCheck on Input X
    print("\n" + "-" * 78)
    print("Running GradCheck on Input Tensor X:")
    def loss_X(X_pert: Matrix) -> float:
        Y_pert = mat_add_broadcast_row(mat_mul(X_pert, layer.W), layer.b)
        l, _ = compute_loss(Y_pert, Y_target)
        return l

    rel_err_X, num_X = gradcheck_matrix(loss_X, X, grad_X)
    print(f"  Frobenius ||g_num - g_ana||: {frobenius_norm(mat_sub(num_X, grad_X)):.2e}")
    print(f"  Relative Error for X:       {rel_err_X:.2e}")
    assert rel_err_X < 1e-7, f"GradCheck for X failed! {rel_err_X} >= 1e-7"
    print("  => Input Tensor X GRADCHECK PASSED (< 1e-7)!")

    # Print summary tolerance guide
    print("\n" + "=" * 78)
    print("GRADCHECK TOLERANCE BAND SUMMARY:")
    print("  RelErr < 1e-7 : PERFECT (Analytical and numerical derivatives match)")
    print("  RelErr < 1e-5 : ACCEPTABLE (Minor floating-point / kink noise)")
    print("  RelErr > 1e-4 : WARNING / DEFECTIVE (Probable algebraic or index bug)")
    print("=" * 78)
    print("ALL TESTS PASSED SUCCESSFULLY! Exit code 0.")


if __name__ == "__main__":
    run_experiment()
    sys.exit(0)
