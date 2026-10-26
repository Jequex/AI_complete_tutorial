#!/usr/bin/env python3
"""
Phase 14: Computer Vision: CNNs to Vision Transformers
Topic 14.1: Convolutions & Feature Maps

Pure Python 3 implementation (zero external dependencies).
Demonstrates:
  1. 2D Discrete Cross-Correlation / Convolution with padding and stride.
  2. Spatial output dimension arithmetic: O = floor((W - K + 2P)/S) + 1.
  3. Multi-channel convolution: [C_out, C_in, K_h, K_w] tensor operations.
  4. Sobel edge detection (horizontal and vertical edge filters) on synthetic image.
  5. Analytical Receptive Field (RF) propagation across multi-layer pipelines.
  6. Computational complexity (FLOPs and MACs) tracking.
"""

import math
from typing import List, Tuple, Union


# =====================================================================
# 1. Tensor Shapes and Helper Operations (Pure Python Lists)
# =====================================================================

def pad_2d(
    matrix: List[List[float]],
    padding: int,
    pad_val: float = 0.0
) -> List[List[float]]:
    """
    Apply symmetric zero-padding of width P to a 2D matrix [H, W].
    Returns a new matrix of shape [H + 2*P, W + 2*P].
    """
    if padding == 0:
        return [row[:] for row in matrix]
    
    H = len(matrix)
    W = len(matrix[0]) if H > 0 else 0
    new_H = H + 2 * padding
    new_W = W + 2 * padding

    padded = [[pad_val for _ in range(new_W)] for _ in range(new_H)]
    for r in range(H):
        for c in range(W):
            padded[r + padding][c + padding] = matrix[r][c]
    return padded


def compute_output_dim(dim_in: int, kernel_size: int, padding: int, stride: int) -> int:
    """
    Standard CNN spatial output dimension arithmetic:
    O = floor((dim_in - kernel_size + 2 * padding) / stride) + 1
    """
    if dim_in + 2 * padding < kernel_size:
        raise ValueError(
            f"Input size ({dim_in}) + 2*pad ({2*padding}) is smaller than kernel size ({kernel_size})."
        )
    return ((dim_in - kernel_size + 2 * padding) // stride) + 1


# =====================================================================
# 2. 2D Single-Channel Convolution / Cross-Correlation
# =====================================================================

def conv2d_single_channel(
    image: List[List[float]],
    kernel: List[List[float]],
    bias: float = 0.0,
    stride: int = 1,
    padding: int = 0
) -> List[List[float]]:
    """
    Computes 2D Cross-Correlation of single-channel image [H, W] with kernel [K_h, K_w].
    Output shape: [H_out, W_out].
    """
    H = len(image)
    W = len(image[0]) if H > 0 else 0
    K_h = len(kernel)
    K_w = len(kernel[0]) if K_h > 0 else 0

    H_out = compute_output_dim(H, K_h, padding, stride)
    W_out = compute_output_dim(W, K_w, padding, stride)

    padded_img = pad_2d(image, padding)

    out = [[0.0 for _ in range(W_out)] for _ in range(H_out)]

    for i in range(H_out):
        r_start = i * stride
        for j in range(W_out):
            c_start = j * stride
            acc = bias
            for m in range(K_h):
                for n in range(K_w):
                    acc += padded_img[r_start + m][c_start + n] * kernel[m][n]
            out[i][j] = acc

    return out


# =====================================================================
# 3. Multi-Channel 2D Convolution
# =====================================================================

def conv2d_multi_channel(
    input_tensor: List[List[List[float]]],  # Shape: [C_in, H, W]
    weight_tensor: List[List[List[List[float]]]],  # Shape: [C_out, C_in, K_h, K_w]
    bias_vector: List[float],  # Shape: [C_out]
    stride: int = 1,
    padding: int = 0
) -> Tuple[List[List[List[float]]], int]:
    """
    Multi-channel 2D Convolution:
      Input: [C_in, H, W]
      Weights: [C_out, C_in, K_h, K_w]
      Bias: [C_out]
    Returns:
      Output tensor [C_out, H_out, W_out] and total MAC operations performed.
    """
    C_in = len(input_tensor)
    H = len(input_tensor[0])
    W = len(input_tensor[0][0])

    C_out = len(weight_tensor)
    if len(weight_tensor[0]) != C_in:
        raise ValueError(f"Weight kernel C_in ({len(weight_tensor[0])}) does not match input C_in ({C_in})")
    
    K_h = len(weight_tensor[0][0])
    K_w = len(weight_tensor[0][0][0])

    H_out = compute_output_dim(H, K_h, padding, stride)
    W_out = compute_output_dim(W, K_w, padding, stride)

    # Pad each input channel independently
    padded_input = [pad_2d(input_tensor[c], padding) for c in range(C_in)]

    output = [[[0.0 for _ in range(W_out)] for _ in range(H_out)] for _ in range(C_out)]

    macs = 0
    for k in range(C_out):
        b = bias_vector[k] if bias_vector else 0.0
        for i in range(H_out):
            r_start = i * stride
            for j in range(W_out):
                c_start = j * stride
                acc = b
                for c in range(C_in):
                    for m in range(K_h):
                        for n in range(K_w):
                            acc += padded_input[c][r_start + m][c_start + n] * weight_tensor[k][c][m][n]
                            macs += 1
                output[k][i][j] = acc

    return output, macs


# =====================================================================
# 4. Receptive Field Tracker
# =====================================================================

def compute_receptive_field(layers: List[Tuple[int, int]]) -> List[dict]:
    """
    Computes receptive field expansion across sequential layers.
    layers: list of (kernel_size, stride) tuples.
    Formulas:
      RF_0 = 1, J_0 = 1
      RF_l = RF_{l-1} + (k_l - 1) * J_{l-1}
      J_l = J_{l-1} * s_l  (cumulative stride / jump)
    """
    rf = 1
    jump = 1
    history = [{"layer": 0, "kernel": 1, "stride": 1, "jump": jump, "rf": rf}]

    for idx, (k, s) in enumerate(layers, 1):
        rf = rf + (k - 1) * jump
        jump = jump * s
        history.append({"layer": idx, "kernel": k, "stride": s, "jump": jump, "rf": rf})

    return history


# =====================================================================
# 5. Sobel Filter Edge Detection Demonstration
# =====================================================================

def create_synthetic_image(H: int = 8, W: int = 8) -> List[List[float]]:
    """
    Creates an 8x8 synthetic image with a bright 4x4 centered square.
    Background = 0.0, Square = 10.0.
    Produces crisp horizontal and vertical edges.
    """
    img = [[0.0 for _ in range(W)] for _ in range(H)]
    for r in range(2, 6):
        for c in range(2, 6):
            img[r][c] = 10.0
    return img


def run_convolution_experiment():
    print("=" * 70)
    print("Phase 14 - Topic 14.1: Convolutions & Feature Maps Experiment")
    print("=" * 70)

    # -------------------------------------------------------------
    # Step 1: Spatial Arithmetic Verification
    # -------------------------------------------------------------
    print("\n--- Test 1: Spatial Output Arithmetic ---")
    test_cases = [
        # (W_in, K, P, S, expected_out)
        (32, 3, 1, 1, 32),   # Standard "Same" padding
        (32, 3, 0, 1, 30),   # "Valid" padding
        (32, 3, 1, 2, 16),   # Strided conv (downsampling)
        (224, 7, 3, 2, 112), # ResNet conv1
        (7, 3, 0, 1, 5),     # Tiny conv
    ]
    for W, K, P, S, expected in test_cases:
        calculated = compute_output_dim(W, K, P, S)
        assert calculated == expected, f"Failed: W={W}, K={K}, P={P}, S={S} -> got {calculated}, expected {expected}"
        print(f"  Input: {W}x{W}, K: {K}x{K}, Pad: {P}, Stride: {S} -> Output: {calculated}x{calculated} (Passed)")

    # -------------------------------------------------------------
    # Step 2: Sobel Edge Detection on Synthetic Image
    # -------------------------------------------------------------
    print("\n--- Test 2: Edge Detection with Sobel Filters ---")
    image = create_synthetic_image(8, 8)
    
    # Sobel Vertical Edge Detector (detects horizontal gradient dI/dx)
    sobel_v = [
        [-1.0, 0.0, 1.0],
        [-2.0, 0.0, 2.0],
        [-1.0, 0.0, 1.0]
    ]

    # Sobel Horizontal Edge Detector (detects vertical gradient dI/dy)
    sobel_h = [
        [-1.0, -2.0, -1.0],
        [ 0.0,  0.0,  0.0],
        [ 1.0,  2.0,  1.0]
    ]

    # Convolve with Sobel filters (P=0, S=1) -> 8x8 input -> 6x6 output
    edge_v = conv2d_single_channel(image, sobel_v, padding=0, stride=1)
    edge_h = conv2d_single_channel(image, sobel_h, padding=0, stride=1)

    assert len(edge_v) == 6 and len(edge_v[0]) == 6
    assert len(edge_h) == 6 and len(edge_h[0]) == 6

    # Verify edge detector characteristics:
    # Top edge of square is between row 1 and row 2.
    # In edge_h (6x6), rows correspond to image rows 1 to 6.
    # Positive edge when transitioning from dark (0) to bright (10) downwards:
    # edge_h[0][2] examines rows 0..2 of image -> row 2 is 10, rows 0,1 are 0:
    #   sum = 1*10 + 2*10 + 1*10 = 40.0
    print(f"  Horizontal Sobel response at top boundary: {edge_h[0][2]:.1f} (Expected: 40.0)")
    assert math.isclose(edge_h[0][2], 40.0, abs_tol=1e-5), "Sobel top edge mismatch!"

    # Bottom boundary transition from bright (10) to dark (0):
    # edge_h[4][2] examines rows 4..6 -> row 4 is 10, rows 5,6 are 0:
    #   sum = -1*10 + -2*10 + -1*10 = -40.0
    print(f"  Horizontal Sobel response at bottom boundary: {edge_h[4][2]:.1f} (Expected: -40.0)")
    assert math.isclose(edge_h[4][2], -40.0, abs_tol=1e-5), "Sobel bottom edge mismatch!"

    # Vertical edge detector at left boundary:
    # edge_v[2][0] examines columns 0..2 -> col 2 is 10, cols 0,1 are 0:
    #   sum = 1*10 + 2*10 + 1*10 = 40.0
    print(f"  Vertical Sobel response at left boundary: {edge_v[2][0]:.1f} (Expected: 40.0)")
    assert math.isclose(edge_v[2][0], 40.0, abs_tol=1e-5), "Sobel left edge mismatch!"

    # Center of square should have zero gradient (flat region):
    print(f"  Gradient at interior center: {edge_h[2][2]:.1f}, {edge_v[2][2]:.1f} (Expected: 0.0)")
    assert math.isclose(edge_h[2][2], 0.0, abs_tol=1e-5)
    assert math.isclose(edge_v[2][2], 0.0, abs_tol=1e-5)

    # -------------------------------------------------------------
    # Step 3: Multi-Channel Convolution & FLOPs Bookkeeping
    # -------------------------------------------------------------
    print("\n--- Test 3: Multi-Channel Convolution & FLOPs ---")
    # Synthetic RGB input: 3 channels, 4x4 spatial
    C_in, H, W = 3, 4, 4
    input_rgb = [[[float(c + 1) for _ in range(W)] for _ in range(H)] for c in range(C_in)]

    # Two filters: C_out = 2, kernel = 2x2
    C_out = 2
    weights = [
        # Filter 0: All weights = 1.0
        [[[1.0, 1.0], [1.0, 1.0]] for _ in range(C_in)],
        # Filter 1: All weights = 0.5
        [[[0.5, 0.5], [0.5, 0.5]] for _ in range(C_in)]
    ]
    biases = [1.0, -2.0]

    # With padding=0, stride=1, output spatial size is (4 - 2)/1 + 1 = 3x3
    out_tensor, macs = conv2d_multi_channel(input_rgb, weights, biases, stride=1, padding=0)

    assert len(out_tensor) == 2
    assert len(out_tensor[0]) == 3
    assert len(out_tensor[0][0]) == 3

    # Analytical calculation for Filter 0:
    # Channel 0 (val 1.0) * 4 pixels = 4.0
    # Channel 1 (val 2.0) * 4 pixels = 8.0
    # Channel 2 (val 3.0) * 4 pixels = 12.0
    # Sum across channels = 24.0 + bias (1.0) = 25.0
    print(f"  Multi-channel Filter 0 output value: {out_tensor[0][0][0]} (Expected: 25.0)")
    assert math.isclose(out_tensor[0][0][0], 25.0, abs_tol=1e-5)

    # Filter 1:
    # 24.0 * 0.5 + bias (-2.0) = 12.0 - 2.0 = 10.0
    print(f"  Multi-channel Filter 1 output value: {out_tensor[1][0][0]} (Expected: 10.0)")
    assert math.isclose(out_tensor[1][0][0], 10.0, abs_tol=1e-5)

    # Expected MACs: H_out * W_out * C_out * (C_in * K_h * K_w)
    # = 3 * 3 * 2 * (3 * 2 * 2) = 18 * 12 = 216 MACs
    print(f"  Total MACs computed: {macs} (Expected: 216)")
    assert macs == 216

    # -------------------------------------------------------------
    # Step 4: Receptive Field Propagation
    # -------------------------------------------------------------
    print("\n--- Test 4: Receptive Field (RF) Expansion ---")
    # Simulate a classic 4-layer conv stack:
    # Layer 1: k=3, s=1
    # Layer 2: k=3, s=2
    # Layer 3: k=3, s=1
    # Layer 4: k=3, s=2
    network_spec = [(3, 1), (3, 2), (3, 1), (3, 2)]
    rf_history = compute_receptive_field(network_spec)

    print("  Layer | Kernel | Stride | Jump (Stride Product) | Receptive Field")
    print("  " + "-" * 58)
    for entry in rf_history:
        print(f"   L{entry['layer']}   |   {entry['kernel']}    |   {entry['stride']}    |          {entry['jump']:2d}           |       {entry['rf']:2d}")

    # RF calculations:
    # L0: RF=1
    # L1: k=3, s=1 -> RF = 1 + (3-1)*1 = 3, J = 1*1 = 1
    # L2: k=3, s=2 -> RF = 3 + (3-1)*1 = 5, J = 1*2 = 2
    # L3: k=3, s=1 -> RF = 5 + (3-1)*2 = 9, J = 2*1 = 2
    # L4: k=3, s=2 -> RF = 9 + (3-1)*2 = 13, J = 2*2 = 4
    assert rf_history[1]["rf"] == 3
    assert rf_history[2]["rf"] == 5
    assert rf_history[3]["rf"] == 9
    assert rf_history[4]["rf"] == 13

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED SUCCESSFULLY! (Exit Code 0)")
    print("=" * 70)


if __name__ == "__main__":
    run_convolution_experiment()
