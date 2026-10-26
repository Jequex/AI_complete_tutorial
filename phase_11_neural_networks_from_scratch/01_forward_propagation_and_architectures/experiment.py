#!/usr/bin/env python3
"""
Phase 11: Neural Networks From Scratch
Topic 11.1: Forward Propagation & Architectures

Pure Python 3 implementation (zero external dependencies).
Demonstrates:
  1. The Artificial Neuron: affine transformation z = w^T x + b.
  2. Pure Python 2D Matrix Multiplication and Tensor Bookkeeping.
  3. Dense / Linear Layer: Z = A W^T + b with shape [B, d_in] x [d_in, d_out] -> [B, d_out].
  4. Multi-Layer Feedforward Sequential Network (Input -> Hidden -> Output).
  5. Exact analytical verification vs hand-calculated forward pass.
"""

import math
from typing import List, Tuple


def matmul_2d(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """
    Pure Python matrix multiplication: C = A @ B.
    A: shape [rows_A, cols_A]
    B: shape [rows_B, cols_B]
    Constraint: cols_A == rows_B
    Returns C: shape [rows_A, cols_B]
    """
    rows_A = len(A)
    cols_A = len(A[0]) if rows_A > 0 else 0
    rows_B = len(B)
    cols_B = len(B[0]) if rows_B > 0 else 0

    if cols_A != rows_B:
        raise ValueError(f"Shape mismatch: cannot multiply ({rows_A}x{cols_A}) and ({rows_B}x{cols_B})")

    # Initialize output matrix with zeros
    C = [[0.0 for _ in range(cols_B)] for _ in range(rows_A)]
    for i in range(rows_A):
        for k in range(cols_A):
            aik = A[i][k]
            for j in range(cols_B):
                C[i][j] += aik * B[k][j]
    return C


def transpose_2d(A: List[List[float]]) -> List[List[float]]:
    """Transposes a 2D matrix A [rows, cols] -> [cols, rows]."""
    rows = len(A)
    cols = len(A[0]) if rows > 0 else 0
    return [[A[r][c] for r in range(rows)] for c in range(cols)]


def add_bias(Z: List[List[float]], b: List[float]) -> List[List[float]]:
    """
    Broadcasting addition of 1D bias vector b [d_out] to 2D pre-activation matrix Z [B, d_out].
    """
    B = len(Z)
    d_out = len(Z[0]) if B > 0 else 0
    if len(b) != d_out:
        raise ValueError(f"Bias dimension mismatch: Z has {d_out} cols but b has {len(b)} elements")

    out = [[0.0 for _ in range(d_out)] for _ in range(B)]
    for i in range(B):
        for j in range(d_out):
            out[i][j] = Z[i][j] + b[j]
    return out


class LinearLayer:
    """
    Fully-connected (Dense / Affine) Layer:
      Z = A W^T + b
    Weights W: shape [d_out, d_in] (standard PyTorch convention)
    Biases b: shape [d_out]
    Input A: shape [B, d_in]
    Output Z: shape [B, d_out]
    """
    def __init__(self, in_features: int, out_features: int, weights: List[List[float]] = None, bias: List[float] = None):
        self.in_features = in_features
        self.out_features = out_features

        if weights is not None:
            # Validate shape [out_features, in_features]
            assert len(weights) == out_features, f"Weight rows {len(weights)} != out_features {out_features}"
            for row in weights:
                assert len(row) == in_features, f"Weight cols {len(row)} != in_features {in_features}"
            self.W = weights
        else:
            # Deterministic initialization for reproducible experiment
            self.W = [[0.1 * ((i + 1) * (j + 1) % 7 - 3) for j in range(in_features)] for i in range(out_features)]

        if bias is not None:
            assert len(bias) == out_features, f"Bias len {len(bias)} != out_features {out_features}"
            self.b = bias
        else:
            self.b = [0.05 * (i + 1) for i in range(out_features)]

    def forward(self, A: List[List[float]]) -> List[List[float]]:
        """
        Forward pass for a batch:
        Z = A @ W^T + b
        """
        B = len(A)
        d_in = len(A[0]) if B > 0 else 0
        if d_in != self.in_features:
            raise ValueError(f"Input feature dimension {d_in} != layer in_features {self.in_features}")

        W_T = transpose_2d(self.W)  # [d_in, d_out]
        linear_term = matmul_2d(A, W_T)  # [B, d_in] x [d_in, d_out] -> [B, d_out]
        Z = add_bias(linear_term, self.b)
        return Z


class SequentialNetwork:
    """
    Multi-Layer Feedforward Sequential Network composed of LinearLayers.
    """
    def __init__(self, layers: List[LinearLayer]):
        self.layers = layers
        # Verify layer dimension chain
        for idx in range(len(layers) - 1):
            assert layers[idx].out_features == layers[idx + 1].in_features, (
                f"Dimension mismatch between layer {idx} (out={layers[idx].out_features}) "
                f"and layer {idx+1} (in={layers[idx+1].in_features})"
            )

    def forward(self, X: List[List[float]]) -> List[List[float]]:
        current_activation = X
        for l_idx, layer in enumerate(self.layers):
            current_activation = layer.forward(current_activation)
        return current_activation


def test_artificial_neuron():
    print("=" * 70)
    print("Step 1: The Artificial Neuron: Analytical Hand Verification")
    print("=" * 70)

    # 1. Biological vs Mathematical single neuron
    # Input vector x = [2.0, -1.0, 3.0]
    # Weights w = [0.5, -0.2, 0.4]
    # Bias b = 0.35
    x = [2.0, -1.0, 3.0]
    w = [0.5, -0.2, 0.4]
    b = 0.35

    # Hand calculation:
    # z = (2.0 * 0.5) + (-1.0 * -0.2) + (3.0 * 0.4) + 0.35
    #   = 1.0 + 0.2 + 1.2 + 0.35 = 2.75
    expected_z = 2.75
    computed_z = sum(xi * wi for xi, wi in zip(x, w)) + b

    print(f"Input x: {x}")
    print(f"Weights w: {w}, Bias b: {b}")
    print(f"Computed z: {computed_z:.6f}, Expected z: {expected_z:.6f}")
    assert math.isclose(computed_z, expected_z, rel_tol=1e-12), "Neuron scalar forward pass failed!"
    print("✓ Single neuron dot-product affine map matched analytical calculation exactly.")


def test_linear_layer_and_batch_vectorization():
    print("\n" + "=" * 70)
    print("Step 2: Linear Layer & Batch Vectorization ([B, d_in] -> [B, d_out])")
    print("=" * 70)

    # Batch of 3 samples, 2 input features:
    # X = [[1.0, 2.0],
    #      [-0.5, 0.5],
    #      [0.0, 4.0]] (shape 3 x 2)
    X = [
        [1.0, 2.0],
        [-0.5, 0.5],
        [0.0, 4.0]
    ]

    # Layer mapping 2 inputs -> 3 outputs:
    # W shape: [3, 2]
    # W = [[0.2, -0.1],
    #      [1.5,  0.5],
    #      [-1.0, 2.0]]
    W = [
        [0.2, -0.1],
        [1.5,  0.5],
        [-1.0, 2.0]
    ]
    # Bias b shape: [3]
    # b = [0.1, -0.2, 0.5]
    b = [0.1, -0.2, 0.5]

    layer = LinearLayer(in_features=2, out_features=3, weights=W, bias=b)
    Z = layer.forward(X)

    # Hand calculations:
    # Sample 0: x = [1.0, 2.0]
    # z0_0 = 1.0*0.2 + 2.0*(-0.1) + 0.1 = 0.2 - 0.2 + 0.1 = 0.1
    # z0_1 = 1.0*1.5 + 2.0*0.5 - 0.2 = 1.5 + 1.0 - 0.2 = 2.3
    # z0_2 = 1.0*(-1.0) + 2.0*2.0 + 0.5 = -1.0 + 4.0 + 0.5 = 3.5
    #
    # Sample 1: x = [-0.5, 0.5]
    # z1_0 = -0.5*0.2 + 0.5*(-0.1) + 0.1 = -0.1 - 0.05 + 0.1 = -0.05
    # z1_1 = -0.5*1.5 + 0.5*0.5 - 0.2 = -0.75 + 0.25 - 0.2 = -0.70
    # z1_2 = -0.5*(-1.0) + 0.5*2.0 + 0.5 = 0.5 + 1.0 + 0.5 = 2.0
    #
    # Sample 2: x = [0.0, 4.0]
    # z2_0 = 0.0*0.2 + 4.0*(-0.1) + 0.1 = -0.4 + 0.1 = -0.3
    # z2_1 = 0.0*1.5 + 4.0*0.5 - 0.2 = 2.0 - 0.2 = 1.8
    # z2_2 = 0.0*(-1.0) + 4.0*2.0 + 0.5 = 8.0 + 0.5 = 8.5

    expected_Z = [
        [0.10,  2.30, 3.50],
        [-0.05, -0.70, 2.00],
        [-0.30,  1.80, 8.50]
    ]

    print(f"Batch input shape: [{len(X)}, {len(X[0])}]")
    print(f"Weight shape: [{len(W)}, {len(W[0])}], Bias shape: [{len(b)}]")
    print(f"Output Z shape: [{len(Z)}, {len(Z[0])}]")

    assert len(Z) == 3 and len(Z[0]) == 3, "Output shape mismatch!"

    for r in range(3):
        for c in range(3):
            val = Z[r][c]
            exp = expected_Z[r][c]
            assert math.isclose(val, exp, rel_tol=1e-12), f"Mismatch at ({r},{c}): computed {val} vs expected {exp}"

    print("Computed Batch Output Z:")
    for row in Z:
        print("  " + " ".join(f"{v:8.4f}" for v in row))
    print("✓ Batch vectorization and shape propagation [3, 2] -> [3, 3] verified.")


def test_multi_layer_deep_network():
    print("\n" + "=" * 70)
    print("Step 3: Multi-Layer Deep Feedforward Architecture Verification")
    print("=" * 70)

    # 3-Layer Architecture:
    # Layer 1: 4 in -> 5 hidden
    # Layer 2: 5 hidden -> 3 hidden
    # Layer 3: 3 hidden -> 2 output
    layer1 = LinearLayer(in_features=4, out_features=5)
    layer2 = LinearLayer(in_features=5, out_features=3)
    layer3 = LinearLayer(in_features=3, out_features=2)

    net = SequentialNetwork([layer1, layer2, layer3])

    # Synthetic batch of 6 samples with 4 features each
    B = 6
    X_batch = [[0.1 * i + 0.05 * j for j in range(4)] for i in range(B)]

    # Forward pass through all layers
    Y_hat = net.forward(X_batch)

    assert len(Y_hat) == B, f"Expected {B} samples, got {len(Y_hat)}"
    assert len(Y_hat[0]) == 2, f"Expected 2 output features, got {len(Y_hat[0])}"

    # Also step through layer-by-layer manually to verify sequential consistency
    A1 = layer1.forward(X_batch)
    assert len(A1) == B and len(A1[0]) == 5
    A2 = layer2.forward(A1)
    assert len(A2) == B and len(A2[0]) == 3
    A3 = layer3.forward(A2)
    assert len(A3) == B and len(A3[0]) == 2

    for r in range(B):
        for c in range(2):
            assert math.isclose(Y_hat[r][c], A3[r][c], rel_tol=1e-12)

    print(f"Sequential Network Architecture: [4] -> [5] -> [3] -> [2]")
    print(f"Batch size: {B}")
    print(f"Output shape verified: [{len(Y_hat)}, {len(Y_hat[0])}]")
    print("Final output Y_hat:")
    for row in Y_hat:
        print("  " + " ".join(f"{v:10.5f}" for v in row))
    print("✓ Deep feedforward architecture forward pass validated across all layers.")


def main():
    print("=" * 70)
    print("PHASE 11.1 EXPERIMENT: FORWARD PROPAGATION & ARCHITECTURES")
    print("=" * 70)
    test_artificial_neuron()
    test_linear_layer_and_batch_vectorization()
    test_multi_layer_deep_network()
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS PASSED SUCCESSFULLY (Exit Code 0)")
    print("=" * 70)


if __name__ == "__main__":
    main()
