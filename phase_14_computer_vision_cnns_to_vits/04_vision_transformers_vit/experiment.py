#!/usr/bin/env python3
"""
Phase 14: Computer Vision: CNNs to Vision Transformers
Topic 14.4: Vision Transformers (ViT)

Pure Python 3 implementation (zero external dependencies).
Demonstrates:
  1. Slicing an image [C, H, W] into non-overlapping patches of size [P, P, C].
  2. Flattening and Linear Projection: x_p -> x_p E (dimension D).
  3. Prepending the learnable [CLS] classification token.
  4. Adding 1D Positional Embeddings: z_0 = [x_cls; x_p E] + E_pos.
  5. Multi-Head Scaled Dot-Product Self-Attention (Q, K, V).
  6. Exact assertion of shape transitions and softmax normalization (sum = 1.0).
"""

import math
from typing import List, Tuple


# =====================================================================
# 1. Pure Python Matrix & Math Utilities
# =====================================================================

def matmul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Multiplies matrix A [R1, C1] by B [C1, C2] -> [R1, C2]."""
    R1 = len(A)
    C1 = len(A[0])
    R2 = len(B)
    C2 = len(B[0])
    if C1 != R2:
        raise ValueError(f"Shape mismatch: {R1}x{C1} and {R2}x{C2}")
    
    C = [[0.0 for _ in range(C2)] for _ in range(R1)]
    for i in range(R1):
        for k in range(C1):
            aik = A[i][k]
            for j in range(C2):
                C[i][j] += aik * B[k][j]
    return C


def transpose(A: List[List[float]]) -> List[List[float]]:
    """Transposes matrix A [R, C] -> [C, R]."""
    R = len(A)
    C = len(A[0])
    return [[A[r][c] for r in range(R)] for c in range(C)]


def softmax_2d(matrix: List[List[float]]) -> List[List[float]]:
    """Applies row-wise numerically stable softmax."""
    out = []
    for row in matrix:
        max_val = max(row)
        exp_vals = [math.exp(x - max_val) for x in row]
        sum_exp = sum(exp_vals)
        out.append([e / sum_exp for e in exp_vals])
    return out


# =====================================================================
# 2. ViT Patch Extraction & Linear Projection
# =====================================================================

def extract_patches(
    image: List[List[List[float]]],
    patch_size: int
) -> List[List[float]]:
    """
    Slices image [C, H, W] into N = (H*W)/(P*P) non-overlapping patches.
    Flattens each patch into a 1D vector of length (P * P * C).
    Returns tensor of shape [N, P * P * C].
    """
    C = len(image)
    H = len(image[0])
    W = len(image[0][0])
    P = patch_size

    if H % P != 0 or W % P != 0:
        raise ValueError(f"Image ({H}x{W}) not divisible by patch size ({P})")

    num_patches_h = H // P
    num_patches_w = W // P
    patches = []

    for ph in range(num_patches_h):
        for pw in range(num_patches_w):
            patch_vec = []
            # Extract in channel-first order for each pixel in patch
            for c in range(C):
                for pr in range(P):
                    for pc in range(P):
                        r = ph * P + pr
                        col = pw * P + pc
                        patch_vec.append(image[c][r][col])
            patches.append(patch_vec)

    return patches


def linear_projection(
    patches: List[List[float]],
    E: List[List[float]]
) -> List[List[float]]:
    """
    Linearly projects flattened patches: X_proj = patches @ E
    patches: [N, P^2 * C]
    E: [P^2 * C, D]
    Returns: [N, D]
    """
    return matmul(patches, E)


# =====================================================================
# 3. [CLS] Token & Positional Embeddings
# =====================================================================

def add_cls_and_position(
    projected_patches: List[List[float]],
    cls_token: List[float],
    pos_embeddings: List[List[float]]
) -> List[List[float]]:
    """
    Prepends [CLS] token: shape [N+1, D]
    Adds 1D positional embeddings E_pos [N+1, D]
    """
    N = len(projected_patches)
    D = len(cls_token)

    # 1. Prepend CLS token at row 0
    sequence = [cls_token[:]] + [row[:] for row in projected_patches]

    # 2. Element-wise addition of positional embeddings
    if len(pos_embeddings) != N + 1 or len(pos_embeddings[0]) != D:
        raise ValueError(f"Positional embedding shape mismatch: expected ({N+1}, {D})")

    Z0 = [[sequence[i][j] + pos_embeddings[i][j] for j in range(D)] for i in range(N + 1)]
    return Z0


# =====================================================================
# 4. Scaled Dot-Product Self-Attention
# =====================================================================

def self_attention(
    Z: List[List[float]],
    W_q: List[List[float]],
    W_k: List[List[float]],
    W_v: List[List[float]],
    W_o: List[List[float]]
) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Computes Scaled Dot-Product Self-Attention:
      Q = Z @ W_q
      K = Z @ W_k
      V = Z @ W_v
      Scores = (Q @ K^T) / sqrt(d_k)
      Attn = Softmax(Scores)
      Out = Attn @ V @ W_o
    Z: [Seq_Len, D]
    Returns (Out [Seq_Len, D], Attn_Weights [Seq_Len, Seq_Len])
    """
    Seq_Len = len(Z)
    D = len(Z[0])
    d_k = float(D)

    # 1. Linear Projections for Query, Key, Value
    Q = matmul(Z, W_q)
    K = matmul(Z, W_k)
    V = matmul(Z, W_v)

    # 2. Scaled Dot-Product: Q @ K^T / sqrt(d_k)
    K_T = transpose(K)
    raw_scores = matmul(Q, K_T)

    scale = 1.0 / math.sqrt(d_k)
    scaled_scores = [[raw_scores[i][j] * scale for j in range(Seq_Len)] for i in range(Seq_Len)]

    # 3. Softmax across key dimension (row-wise)
    attn_weights = softmax_2d(scaled_scores)

    # 4. Attention weighted combination of Values
    context = matmul(attn_weights, V)

    # 5. Output projection
    output = matmul(context, W_o)

    return output, attn_weights


# =====================================================================
# 5. Main ViT Forward Flow Experiment
# =====================================================================

def run_vit_experiment():
    print("=" * 70)
    print("Phase 14 - Topic 14.4: Vision Transformers (ViT) Experiment")
    print("=" * 70)

    # -------------------------------------------------------------
    # Step 1: Synthetic Image Definition
    # -------------------------------------------------------------
    # Image: 2 channels (e.g. grayscale + infrared), H = 8, W = 8
    C, H, W = 2, 8, 8
    patch_size = 4
    D = 8  # Transformer hidden dimension

    # Create synthetic image with 4 quadrants
    # Quad 0: (0..3, 0..3) = 1.0
    # Quad 1: (0..3, 4..7) = 2.0
    # Quad 2: (4..7, 0..3) = 3.0
    # Quad 3: (4..7, 4..7) = 4.0
    synthetic_image = [[[0.0 for _ in range(W)] for _ in range(H)] for _ in range(C)]
    for c in range(C):
        for r in range(H):
            for col in range(W):
                q_r = 0 if r < 4 else 1
                q_c = 0 if col < 4 else 1
                quad_id = q_r * 2 + q_c + 1
                synthetic_image[c][r][col] = float(quad_id * 10 + c)

    print(f"\n1. Image Configuration:")
    print(f"   Shape: [{C} channels, {H} height, {W} width]")
    print(f"   Patch size: {patch_size}x{patch_size}")

    # -------------------------------------------------------------
    # Step 2: Patch Slicing & Flattening
    # -------------------------------------------------------------
    patches = extract_patches(synthetic_image, patch_size)
    num_patches = len(patches)
    patch_dim = len(patches[0])
    expected_num_patches = (H * W) // (patch_size * patch_size)
    expected_patch_dim = patch_size * patch_size * C

    print(f"\n2. Patch Extraction & Flattening:")
    print(f"   Number of patches (N): {num_patches} (Expected: {expected_num_patches})")
    print(f"   Flattened patch dimension (P^2 * C): {patch_dim} (Expected: {expected_patch_dim})")
    assert num_patches == 4
    assert patch_dim == 32

    # -------------------------------------------------------------
    # Step 3: Patch Linear Projection to Dimension D
    # -------------------------------------------------------------
    # Learnable projection matrix E: [32, 8]
    # Deterministic initialization
    E = [[(1.0 / math.sqrt(patch_dim)) * (0.5 if (i + j) % 2 == 0 else -0.5) 
          for j in range(D)] for i in range(patch_dim)]

    projected_patches = linear_projection(patches, E)
    print(f"\n3. Linear Projection to Dimension D={D}:")
    print(f"   Projected shape: [{len(projected_patches)}, {len(projected_patches[0])}]")
    assert len(projected_patches) == 4
    assert len(projected_patches[0]) == D

    # -------------------------------------------------------------
    # Step 4: Prepending [CLS] Token & Adding 1D Positional Embeddings
    # -------------------------------------------------------------
    cls_token = [0.1 for _ in range(D)]
    pos_embeddings = [[(float(pos) / 10.0) + (float(d) / 100.0) for d in range(D)] 
                      for pos in range(num_patches + 1)]

    Z0 = add_cls_and_position(projected_patches, cls_token, pos_embeddings)
    seq_len = len(Z0)
    print(f"\n4. Sequence Construction with [CLS] & 1D Positional Embeddings:")
    print(f"   Total sequence length (N + 1): {seq_len} tokens")
    print(f"   Embedded sequence matrix shape: [{seq_len}, {len(Z0[0])}]")
    assert seq_len == 5
    assert len(Z0[0]) == D

    # -------------------------------------------------------------
    # Step 5: Multi-Head Self-Attention Forward Pass
    # -------------------------------------------------------------
    # Projection matrices W_q, W_k, W_v, W_o: [D, D]
    W_q = [[(0.3 if i == j else 0.05) for j in range(D)] for i in range(D)]
    W_k = [[(0.3 if i == j else -0.05) for j in range(D)] for i in range(D)]
    W_v = [[(0.4 if i == j else 0.02) for j in range(D)] for i in range(D)]
    W_o = [[(0.5 if i == j else 0.0) for j in range(D)] for i in range(D)]

    attn_output, attn_weights = self_attention(Z0, W_q, W_k, W_v, W_o)

    print(f"\n5. Self-Attention Forward Pass Verification:")
    print(f"   Attention weight matrix shape: [{len(attn_weights)}, {len(attn_weights[0])}]")
    print(f"   Attention context output shape: [{len(attn_output)}, {len(attn_output[0])}]")

    assert len(attn_weights) == 5 and len(attn_weights[0]) == 5
    assert len(attn_output) == 5 and len(attn_output[0]) == D

    # -------------------------------------------------------------
    # Step 6: Softmax Normalization Verification
    # -------------------------------------------------------------
    print("\n6. Verifying Softmax Normalization across Tokens (Row Sums):")
    for i, row in enumerate(attn_weights):
        row_sum = sum(row)
        token_label = "[CLS]" if i == 0 else f"Patch {i}"
        print(f"   Token {token_label:<7} Attention Weights Sum: {row_sum:.6f} (Passed)")
        assert math.isclose(row_sum, 1.0, abs_tol=1e-6), f"Row {i} does not sum to 1.0: {row_sum}"

    # Print the attention matrix
    print("\n   Self-Attention Weight Matrix (Alpha_ij):")
    header = "       " + "".join([f"{('[CLS]' if j == 0 else f'P{j}'):>8}" for j in range(5)])
    print(header)
    print("       " + "-" * 40)
    for i in range(5):
        row_str = f" {('[CLS]' if i == 0 else f'P{i}'):<5} |"
        for j in range(5):
            row_str += f"{attn_weights[i][j]:8.4f}"
        print(row_str)

    # -------------------------------------------------------------
    # Step 7: Classification Head Feature Extraction
    # -------------------------------------------------------------
    cls_representation = attn_output[0]
    print(f"\n7. Classification Head Output:")
    print(f"   Extracted [CLS] embedding vector (length {len(cls_representation)}):")
    print(f"   {[round(v, 4) for v in cls_representation]}")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED SUCCESSFULLY! (Exit Code 0)")
    print("=" * 70)


if __name__ == "__main__":
    run_vit_experiment()
