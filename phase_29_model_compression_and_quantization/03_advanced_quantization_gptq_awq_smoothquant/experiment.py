#!/usr/bin/env python3
"""
Phase 29 - Topic 03: Advanced Quantization: GPTQ, AWQ & SmoothQuant
==================================================================
Pure Python 3 implementation of:
1. SmoothQuant: mathematically shifting outlier activation dynamic range into weights
   via per-channel scaling matrix S = diag(s_j), achieving >2x MSE reduction in W8A8.
2. Mini-GPTQ / Optimal Brain Quantizer (OBQ): second-order greedy weight quantization
   using inverse Hessian updates Delta w = - ((w_q - hat_w_q) / [H^-1]_qq) * H^-1_{:, q},
   verifying reconstruction error reduction over naive Round-To-Nearest (RTN).

Zero external dependencies: Python standard library only (math, random).
"""

import math
import random

def set_seed(seed=42):
    random.seed(seed)

# -----------------------------------------------------------------------------
# 1. Matrix & Linear Algebra Utilities (Pure Python Standard Library)
# -----------------------------------------------------------------------------

def matmul(A, B):
    """Matrix multiplication C = A * B."""
    rows_A = len(A)
    cols_A = len(A[0])
    cols_B = len(B[0])
    C = [[0.0 for _ in range(cols_B)] for _ in range(rows_A)]
    for i in range(rows_A):
        for k in range(cols_A):
            a_ik = A[i][k]
            for j in range(cols_B):
                C[i][j] += a_ik * B[k][j]
    return C

def transpose(A):
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]

def invert_symmetric_matrix(A, damping=1e-3):
    """Invert square matrix with diagonal damping using Gauss-Jordan elimination with pivoting."""
    n = len(A)
    # Add damping lambda * I
    M = [[A[i][j] + (damping if i == j else 0.0) for j in range(n)] + 
         [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    
    for i in range(n):
        # Partial pivoting
        pivot_row = i
        max_val = abs(M[i][i])
        for r in range(i + 1, n):
            if abs(M[r][i]) > max_val:
                max_val = abs(M[r][i])
                pivot_row = r
        if pivot_row != i:
            M[i], M[pivot_row] = M[pivot_row], M[i]
            
        pivot = M[i][i]
        if abs(pivot) < 1e-12:
            pivot = 1e-12
            
        for c in range(2 * n):
            M[i][c] /= pivot
            
        for r in range(n):
            if r != i:
                factor = M[r][i]
                for c in range(2 * n):
                    M[r][c] -= factor * M[i][c]
                    
    return [[M[i][j + n] for j in range(n)] for i in range(n)]

def quantize_symmetric(val, scale, q_min, q_max):
    """Quantize and dequantize a scalar symmetrically."""
    if abs(scale) < 1e-12:
        return 0.0
    scaled = val / scale
    rounded = round(scaled)
    clamped = max(q_min, min(q_max, rounded))
    return clamped * scale

# -----------------------------------------------------------------------------
# 2. SmoothQuant Implementation
# -----------------------------------------------------------------------------

def run_smoothquant_experiment():
    print("\n--- PART 1: SMOOTHQUANT OUTLIER MIGRATION ---")
    set_seed(42)
    
    # Setup dimensions: M tokens, K input channels, N output channels
    M = 120
    K = 16
    N = 8
    
    # Generate activations X with two severe persistent outlier channels (channels 3 and 11)
    X = [[random.gauss(0.0, 1.0) for _ in range(K)] for _ in range(M)]
    for row in X:
        row[3] *= 35.0  # Outlier channel 1 (35x magnitude)
        row[11] *= 25.0 # Outlier channel 2 (25x magnitude)
        
    # Generate weights W
    W = [[random.gauss(0.0, 1.0 / math.sqrt(K)) for _ in range(N)] for _ in range(K)]
    
    # Ground truth reference output Y = X * W
    Y = matmul(X, W)
    
    # Case A: Naive INT8 Quantization (Per-tensor activations & Per-tensor weights)
    max_X = max(abs(x) for row in X for x in row)
    scale_X = max_X / 127.0
    X_naive_q = [[quantize_symmetric(x, scale_X, -128, 127) for x in row] for row in X]
    
    max_W = max(abs(w) for row in W for w in row)
    scale_W = max_W / 127.0
    W_naive_q = [[quantize_symmetric(w, scale_W, -128, 127) for w in row] for row in W]
    
    Y_naive = matmul(X_naive_q, W_naive_q)
    mse_naive = sum((Y[i][j] - Y_naive[i][j]) ** 2 for i in range(M) for j in range(N)) / (M * N)
    
    # Case B: SmoothQuant per-channel migration
    # s_j = (max(|X_j|)^alpha) / (max(|W_j|)^(1 - alpha))
    alpha = 0.5
    s = []
    for j in range(K):
        act_max = max(abs(X[i][j]) for i in range(M))
        weight_max = max(abs(W[j][k]) for k in range(N))
        act_max = max(1e-5, act_max)
        weight_max = max(1e-5, weight_max)
        scale_j = (act_max ** alpha) / (weight_max ** (1.0 - alpha))
        s.append(scale_j)
        
    # Migrate: X_smooth = X * diag(s)^-1, W_smooth = diag(s) * W
    X_smooth = [[X[i][j] / s[j] for j in range(K)] for i in range(M)]
    W_smooth = [[W[j][k] * s[j] for k in range(N)] for j in range(K)]
    
    # Verify exact mathematical equivalence in FP32: Y_smooth == Y
    Y_fp32_smooth = matmul(X_smooth, W_smooth)
    diff_fp32 = max(abs(Y[i][j] - Y_fp32_smooth[i][j]) for i in range(M) for j in range(N))
    assert diff_fp32 < 1e-6, "SmoothQuant FP32 equivalence failed!"
    
    # Now quantize smoothed representations to INT8
    max_X_smooth = max(abs(x) for row in X_smooth for x in row)
    scale_X_smooth = max_X_smooth / 127.0
    X_smooth_q = [[quantize_symmetric(x, scale_X_smooth, -128, 127) for x in row] for row in X_smooth]
    
    max_W_smooth = max(abs(w) for row in W_smooth for w in row)
    scale_W_smooth = max_W_smooth / 127.0
    W_smooth_q = [[quantize_symmetric(w, scale_W_smooth, -128, 127) for w in row] for row in W_smooth]
    
    Y_smooth = matmul(X_smooth_q, W_smooth_q)
    mse_smooth = sum((Y[i][j] - Y_smooth[i][j]) ** 2 for i in range(M) for j in range(N)) / (M * N)
    
    print(f"  Activation Dynamic Range Before SmoothQuant: {max_X:.2f}")
    print(f"  Activation Dynamic Range After SmoothQuant:  {max_X_smooth:.2f} (Smoothed by {max_X/max_X_smooth:.1f}x)")
    print(f"  Naive INT8 Quantization MSE:                 {mse_naive:.6f}")
    print(f"  SmoothQuant INT8 Quantization MSE:           {mse_smooth:.6f}")
    print(f"  Error Reduction Ratio:                       {mse_naive / mse_smooth:.2f}x lower MSE")
    
    assert mse_smooth < mse_naive * 0.5, "SmoothQuant failed to reduce quantization error by at least 2x"

# -----------------------------------------------------------------------------
# 3. Mini-GPTQ / Optimal Brain Quantizer (OBQ) Implementation
# -----------------------------------------------------------------------------

def run_gptq_experiment():
    print("\n--- PART 2: MINI-GPTQ SECOND-ORDER GREEDY COMPENSATION ---")
    set_seed(42)
    
    # Setup single linear projection: y = X * w
    M = 200
    K = 24
    
    # Calibration inputs X
    X = [[random.gauss(0.0, 1.0) for _ in range(K)] for _ in range(M)]
    
    # Original FP32 weight vector w
    w = [random.gauss(0.0, 1.0 / math.sqrt(K)) for _ in range(K)]
    
    # Target projection: y_ref = X * w
    y_ref = [sum(X[i][j] * w[j] for j in range(K)) for i in range(M)]
    
    # 1. Compute empirical Hessian matrix: H = 2 * (X^T * X) / M + lambda * I
    X_T = transpose(X)
    H_raw = matmul(X_T, X)
    H = [[2.0 * H_raw[i][j] / float(M) for j in range(K)] for i in range(K)]
    
    # Invert Hessian H
    H_inv = invert_symmetric_matrix(H, damping=1e-2)
    
    # Target INT4 precision (symmetric: -8 to 7)
    q_max = 7
    q_min = -8
    max_w = max(abs(x) for x in w)
    scale_w = max_w / float(q_max)
    
    # Strategy A: Naive Round-to-Nearest (RTN)
    w_rtn = [quantize_symmetric(val, scale_w, q_min, q_max) for val in w]
    y_rtn = [sum(X[i][j] * w_rtn[j] for j in range(K)) for i in range(M)]
    err_rtn = sum((y_ref[i] - y_rtn[i]) ** 2 for i in range(M)) / float(M)
    
    # Strategy B: Mini-GPTQ (OBQ Sequential Second-Order Compensation with rank-1 update)
    w_gptq = list(w)
    H_inv_active = [list(row) for row in H_inv]
    
    for q in range(K):
        w_curr = w_gptq[q]
        w_hat = quantize_symmetric(w_curr, scale_w, q_min, q_max)
        w_gptq[q] = w_hat
        
        delta_q = w_curr - w_hat
        h_qq = H_inv_active[q][q]
        if abs(h_qq) < 1e-12:
            h_qq = 1e-12
            
        # Update unquantized weights
        for j in range(q + 1, K):
            w_gptq[j] -= delta_q * (H_inv_active[j][q] / h_qq)
            
        # Rank-1 update of the remaining active inverse Hessian
        for r in range(q + 1, K):
            for c in range(q + 1, K):
                H_inv_active[r][c] -= (H_inv_active[r][q] * H_inv_active[q][c]) / h_qq
            
    y_gptq = [sum(X[i][j] * w_gptq[j] for j in range(K)) for i in range(M)]
    err_gptq = sum((y_ref[i] - y_gptq[i]) ** 2 for i in range(M)) / float(M)
    
    print(f"  Naive Round-to-Nearest (RTN) INT4 Output MSE: {err_rtn:.6f}")
    print(f"  GPTQ Second-Order Compensated INT4 MSE:       {err_gptq:.6f}")
    print(f"  GPTQ Error Reduction:                         {err_rtn / err_gptq:.2f}x lower output error")
    
    assert err_gptq < err_rtn, "GPTQ failed to improve reconstruction error over RTN"
    assert (err_rtn - err_gptq) / err_rtn > 0.05, "GPTQ should achieve at least 5% error reduction"

# -----------------------------------------------------------------------------
# 4. Main Entrypoint
# -----------------------------------------------------------------------------

def run_experiment():
    print("=" * 78)
    print("PHASE 29: TOPIC 03 - ADVANCED QUANTIZATION: GPTQ & SMOOTHQUANT")
    print("=" * 78)
    run_smoothquant_experiment()
    run_gptq_experiment()
    print("\n" + "=" * 78)
    print("ALL ADVANCED QUANTIZATION EXPERIMENTS PASSED (EXIT CODE 0)")
    print("=" * 78)

if __name__ == "__main__":
    run_experiment()
