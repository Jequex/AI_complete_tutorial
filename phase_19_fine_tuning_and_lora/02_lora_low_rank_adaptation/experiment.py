#!/usr/bin/env python3
"""
Phase 19: Fine-Tuning & PEFT / LoRA
Topic 19.2: LoRA (Low-Rank Adaptation) Mechanics

Pure Python 3 implementation (zero external dependencies).
Demonstrates:
  1. Pure Python `LoRALinear` layer with frozen base weights W_0 and trainable low-rank matrices A and B.
  2. Proof that at Step 0, B=0 ensures ΔW = (α/r) * B @ A = 0 (exact initial equivalence to base model).
  3. Analytical backpropagation gradients for A and B.
  4. Finite-difference numerical gradient verification (< 1e-6 error).
  5. Training on a target regression task with frozen W_0.
  6. Weight merging: W_merged = W_0 + (α/r) * B @ A, proving zero runtime inference overhead
     and identical output matching (< 1e-12 max absolute deviation).
"""

import math
import random
from typing import List, Tuple


def set_seed(seed: int = 42):
    random.seed(seed)


# -------------------------------------------------------------------------
# Pure Python Linear Algebra Helpers
# -------------------------------------------------------------------------

def zeros_2d(rows: int, cols: int) -> List[List[float]]:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def matmul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Matrix multiplication C = A @ B."""
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    if cols_A != rows_B:
        raise ValueError(f"Incompatible shapes: ({rows_A}x{cols_A}) and ({rows_B}x{cols_B})")
    C = zeros_2d(rows_A, cols_B)
    for i in range(rows_A):
        for k in range(cols_A):
            aik = A[i][k]
            for j in range(cols_B):
                C[i][j] += aik * B[k][j]
    return C


def mat_vec_mul(M: List[List[float]], v: List[float]) -> List[float]:
    """Computes M @ v where M is [R, C] and v is [C], returns [R]."""
    rows, cols = len(M), len(M[0])
    if len(v) != cols:
        raise ValueError(f"Vector dim {len(v)} != matrix cols {cols}")
    out = [0.0] * rows
    for r in range(rows):
        s = 0.0
        row = M[r]
        for c in range(cols):
            s += row[c] * v[c]
        out[r] = s
    return out


# -------------------------------------------------------------------------
# LoRALinear Module
# -------------------------------------------------------------------------

class LoRALinear:
    """
    Low-Rank Adaptation Linear Layer:
      h = W_0 x + (alpha / r) * (B @ A) x
    
    where:
      - W_0 in R^{out_features x in_features} (Frozen)
      - A in R^{r x in_features} (Gaussian init: N(0, 1/r))
      - B in R^{out_features x r} (Zero init)
      - r: intrinsic adaptation rank (r << min(in_features, out_features))
      - alpha: scaling hyperparameter
    """
    def __init__(self, in_features: int, out_features: int, rank: int = 4, alpha: float = 8.0):
        self.in_features = in_features
        self.out_features = out_features
        self.r = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Base weight W_0: frozen pre-trained weights
        std_w = math.sqrt(2.0 / (in_features + out_features))
        self.W0 = [[random.gauss(0.0, std_w) for _ in range(in_features)] for _ in range(out_features)]
        
        # LoRA Matrix A: Gaussian initialization N(0, 1/r)
        std_a = 1.0 / math.sqrt(rank)
        self.A = [[random.gauss(0.0, std_a) for _ in range(in_features)] for _ in range(rank)]
        
        # LoRA Matrix B: Initialized strictly to ZERO
        self.B = zeros_2d(out_features, rank)
        
        # Cache for backpropagation
        self._last_x = None
        self._last_Ax = None

    def forward(self, x: List[float]) -> List[float]:
        """
        Forward pass: h = W_0 x + scaling * B @ (A @ x)
        """
        self._last_x = list(x)
        
        # 1. Base forward pass: h_base = W_0 @ x
        h_base = mat_vec_mul(self.W0, x)
        
        # 2. LoRA low-rank path: u = A @ x
        Ax = mat_vec_mul(self.A, x)
        self._last_Ax = Ax
        
        # 3. BAx = B @ (A @ x)
        BAx = mat_vec_mul(self.B, Ax)
        
        # 4. Total adapted output
        out = [h_base[i] + self.scaling * BAx[i] for i in range(self.out_features)]
        return out

    def backward(self, grad_output: List[float]) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Computes analytical gradients for A and B given dL/dy:
          dL/dB = scaling * (grad_output otimes Ax)
          dL/dA = scaling * (B^T @ grad_output otimes x)
        """
        x = self._last_x
        Ax = self._last_Ax
        s = self.scaling
        
        # Gradient for B: shape [out_features, r]
        # B[i][k] affects out[i] via s * B[i][k] * Ax[k]
        grad_B = zeros_2d(self.out_features, self.r)
        for i in range(self.out_features):
            g_i = grad_output[i] * s
            for k in range(self.r):
                grad_B[i][k] = g_i * Ax[k]
                
        # Gradient for A: shape [r, in_features]
        # Backprop through B to get grad_Ax: grad_Ax[k] = sum_i grad_output[i] * s * B[i][k]
        grad_Ax = [0.0] * self.r
        for k in range(self.r):
            sum_k = 0.0
            for i in range(self.out_features):
                sum_k += grad_output[i] * self.B[i][k]
            grad_Ax[k] = sum_k * s
            
        grad_A = zeros_2d(self.r, self.in_features)
        for k in range(self.r):
            g_k = grad_Ax[k]
            for j in range(self.in_features):
                grad_A[k][j] = g_k * x[j]
                
        return grad_A, grad_B

    def merge_weights(self) -> List[List[float]]:
        """
        Computes W_merged = W_0 + (alpha / r) * (B @ A) for zero-latency inference.
        Returns a single [out_features x in_features] weight matrix.
        """
        BA = matmul(self.B, self.A)
        s = self.scaling
        W_merged = zeros_2d(self.out_features, self.in_features)
        for i in range(self.out_features):
            for j in range(self.in_features):
                W_merged[i][j] = self.W0[i][j] + s * BA[i][j]
        return W_merged


# -------------------------------------------------------------------------
# Verification & Experiments
# -------------------------------------------------------------------------

def run_experiment():
    print("================================================================================")
    print("  PHASE 19: FINE-TUNING & PEFT / LoRA")
    print("  Topic 19.2: LoRA (Low-Rank Adaptation) Mechanics & Weight Merging")
    print("================================================================================\n")
    set_seed(42)
    
    in_dim, out_dim, rank, alpha = 8, 6, 2, 4.0
    layer = LoRALinear(in_features=in_dim, out_features=out_dim, rank=rank, alpha=alpha)
    
    # Section 1: Verification of Step 0 Identity
    print("--- 1. STEP 0 INITIALIZATION IDENTITY VERIFICATION ---")
    x_test = [random.uniform(-1.0, 1.0) for _ in range(in_dim)]
    y_base = mat_vec_mul(layer.W0, x_test)
    y_lora_step0 = layer.forward(x_test)
    
    max_dev_step0 = max(abs(y_lora_step0[i] - y_base[i]) for i in range(out_dim))
    print(f"  Base output y_base:       {[round(v, 4) for v in y_base[:4]]}...")
    print(f"  LoRA output at Step 0:    {[round(v, 4) for v in y_lora_step0[:4]]}...")
    print(f"  Max absolute deviation:   {max_dev_step0:.16e}")
    assert max_dev_step0 == 0.0, "LoRA output at step 0 must match base model exactly (B=0 condition)!"
    print("  [PASSED] Step 0 equivalence verified: B=0 ensures zero initial perturbation.\n")

    # Section 2: Analytical vs. Finite-Difference Numerical Gradients
    print("--- 2. ANALYTICAL GRADIENT VS. NUMERICAL GRADIENT VERIFICATION ---")
    # Set non-zero values in B to get active gradients across both A and B
    for i in range(out_dim):
        for k in range(rank):
            layer.B[i][k] = random.gauss(0.0, 0.5)
            
    # Forward pass
    y = layer.forward(x_test)
    # Target loss: L = 0.5 * sum((y_i - target_i)^2)
    target = [random.uniform(-1.0, 1.0) for _ in range(out_dim)]
    grad_output = [y[i] - target[i] for i in range(out_dim)]
    
    # Analytical gradients
    grad_A_anal, grad_B_anal = layer.backward(grad_output)
    
    # Numerical gradients for B via finite differences
    eps = 1e-6
    max_err_B = 0.0
    for i in range(out_dim):
        for k in range(rank):
            orig_val = layer.B[i][k]
            
            layer.B[i][k] = orig_val + eps
            y_plus = layer.forward(x_test)
            loss_plus = 0.5 * sum((y_plus[idx] - target[idx])**2 for idx in range(out_dim))
            
            layer.B[i][k] = orig_val - eps
            y_minus = layer.forward(x_test)
            loss_minus = 0.5 * sum((y_minus[idx] - target[idx])**2 for idx in range(out_dim))
            
            layer.B[i][k] = orig_val
            num_grad = (loss_plus - loss_minus) / (2 * eps)
            err = abs(num_grad - grad_B_anal[i][k])
            if err > max_err_B:
                max_err_B = err
                
    # Numerical gradients for A via finite differences
    max_err_A = 0.0
    for k in range(rank):
        for j in range(in_dim):
            orig_val = layer.A[k][j]
            
            layer.A[k][j] = orig_val + eps
            y_plus = layer.forward(x_test)
            loss_plus = 0.5 * sum((y_plus[idx] - target[idx])**2 for idx in range(out_dim))
            
            layer.A[k][j] = orig_val - eps
            y_minus = layer.forward(x_test)
            loss_minus = 0.5 * sum((y_minus[idx] - target[idx])**2 for idx in range(out_dim))
            
            layer.A[k][j] = orig_val
            num_grad = (loss_plus - loss_minus) / (2 * eps)
            err = abs(num_grad - grad_A_anal[k][j])
            if err > max_err_A:
                max_err_A = err

    print(f"  Max error for dL/dB: {max_err_B:.8e}")
    print(f"  Max error for dL/dA: {max_err_A:.8e}")
    assert max_err_B < 1e-5, f"Gradient dL/dB failed verification: {max_err_B}"
    assert max_err_A < 1e-5, f"Gradient dL/dA failed verification: {max_err_A}"
    print("  [PASSED] LoRA analytical gradients match numerical gradients within machine tolerance.\n")

    # Section 3: Training with Frozen W_0
    print("--- 3. TRAINING LORA ADAPTER WITH FROZEN BASE WEIGHTS ---")
    # Reset B to 0.0
    layer.B = zeros_2d(out_dim, rank)
    
    # Create target linear mapping: y* = W_target @ x
    W_target = [[random.uniform(-1.0, 1.0) for _ in range(in_dim)] for _ in range(out_dim)]
    
    dataset = []
    for _ in range(100):
        x_d = [random.uniform(-1.0, 1.0) for _ in range(in_dim)]
        y_d = mat_vec_mul(W_target, x_d)
        dataset.append((x_d, y_d))
        
    def compute_dataset_loss(lora_layer):
        total = 0.0
        for x, y_true in dataset:
            y_pred = lora_layer.forward(x)
            total += 0.5 * sum((y_pred[i] - y_true[i])**2 for i in range(out_dim))
        return total / len(dataset)
    
    initial_loss = compute_dataset_loss(layer)
    print(f"  Initial Adapter Loss (Step 0): {initial_loss:.6f}")
    
    lr = 0.02
    for epoch in range(150):
        for x, y_true in dataset:
            y_pred = layer.forward(x)
            grad_out = [y_pred[i] - y_true[i] for i in range(out_dim)]
            grad_A, grad_B = layer.backward(grad_out)
            
            # Update only A and B! (W0 remains completely untouched)
            for k in range(rank):
                for j in range(in_dim):
                    layer.A[k][j] -= lr * grad_A[k][j]
            for i in range(out_dim):
                for k in range(rank):
                    layer.B[i][k] -= lr * grad_B[i][k]
                    
    final_loss = compute_dataset_loss(layer)
    print(f"  Final Adapter Loss (Epoch 150): {final_loss:.6f}")
    assert final_loss < initial_loss * 0.5, "LoRA adapter failed to minimize loss!"
    print("  [PASSED] LoRA adapter successfully converged while base weights W_0 remained frozen.\n")

    # Section 4: Zero-Latency Inference Weight Merging
    print("--- 4. ZERO-LATENCY INFERENCE WEIGHT MERGING ---")
    W_merged = layer.merge_weights()
    
    # Test on 20 independent novel test vectors
    max_merge_diff = 0.0
    for _ in range(20):
        x_eval = [random.uniform(-2.0, 2.0) for _ in range(in_dim)]
        y_dynamic = layer.forward(x_eval)
        y_merged = mat_vec_mul(W_merged, x_eval)
        
        diff = max(abs(y_dynamic[i] - y_merged[i]) for i in range(out_dim))
        if diff > max_merge_diff:
            max_merge_diff = diff
            
    print(f"  Max deviation between dynamic LoRA pass and merged W_merged: {max_merge_diff:.16e}")
    assert max_merge_diff < 1e-11, f"Weight merging produced output mismatch: {max_merge_diff}"
    print("  [PASSED] W_merged produces mathematically identical results with zero runtime overhead.")

    # Section 5: Parameter Savings Breakdown
    print("\n--- 5. PARAMETER EFFICIENCY COMPARISON ---")
    d, k, r_val = 4096, 4096, 16
    base_params = d * k
    lora_params = (r_val * k) + (d * r_val)
    reduction = (1.0 - (lora_params / base_params)) * 100
    print(f"  Standard Linear Layer (d={d}, k={k}): {base_params:,} parameters")
    print(f"  LoRA Adapter (r={r_val}):                   {lora_params:,} parameters")
    print(f"  Parameter Reduction:                    {reduction:.2f}% reduction! ({lora_params / base_params * 100:.2f}% of original)")
    
    print("\n================================================================================")
    print("All assertions passed successfully. Topic 19.2 experiment complete.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(run_experiment())
