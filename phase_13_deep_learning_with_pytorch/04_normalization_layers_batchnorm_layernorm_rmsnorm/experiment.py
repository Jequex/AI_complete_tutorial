#!/usr/bin/env python3
"""
Topic 13.4 Experiment: Normalization Layers — BatchNorm, LayerNorm & RMSNorm
Author: AI Complete Tutorial
Dependencies: Pure Python 3 Standard Library (math, random, sys)

This experiment implements and mathematically validates the 3 core normalization architectures:
1. Batch Normalization (BatchNorm1d):
   - Normalization across mini-batch dimension B.
   - Tracking running mean and variance for train vs eval modes.
   - Exact analytical backward pass with respect to inputs, gamma, and beta.
2. Layer Normalization (LayerNorm):
   - Normalization across hidden features D for each independent sample.
   - Invariance to batch size (vital for Transformers and autoregressive decoding with B=1).
3. Root Mean Square Normalization (RMSNorm - LLaMA / Mistral architecture):
   - Omits mean-centering: RMS(x) = sqrt((1/D) * sum(x_i^2) + eps).
   - Eliminates shift parameter beta, reducing memory bandwidth by ~20%.
4. Numerical Gradient Verification:
   - Compares analytical backward formulas against two-sided finite difference approximations.
5. Scale Invariance Demonstration:
   - Verifies that scaling input x by 1000x preserves normalized output magnitude.
"""

import math
import random
import sys

def set_seed(seed=42):
    random.seed(seed)

# ==============================================================================
# SECTION 1: Batch Normalization (BatchNorm1d)
# ==============================================================================

class PureBatchNorm1d:
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.training = True

        # Learnable affine parameters
        self.gamma = [1.0] * num_features
        self.beta = [0.0] * num_features

        # Persistent running statistics (buffers)
        self.running_mean = [0.0] * num_features
        self.running_var = [1.0] * num_features

        # Cached tensors for backprop
        self.x = None
        self.x_hat = None
        self.mean = None
        self.var = None

    def forward(self, x):
        """x: shape (B, D)"""
        self.x = x
        B = len(x)
        D = self.num_features

        if self.training:
            # 1. Batch mean per feature
            mean = [sum(x[b][d] for b in range(B)) / B for d in range(D)]
            # 2. Batch variance per feature
            var = [sum((x[b][d] - mean[d]) ** 2 for b in range(B)) / B for d in range(D)]

            self.mean = mean
            self.var = var

            # 3. Update running statistics
            for d in range(D):
                self.running_mean[d] = (1.0 - self.momentum) * self.running_mean[d] + self.momentum * mean[d]
                self.running_var[d] = (1.0 - self.momentum) * self.running_var[d] + self.momentum * var[d]

            curr_mean = mean
            curr_var = var
        else:
            curr_mean = self.running_mean
            curr_var = self.running_var

        # 4. Normalize and affine transform
        self.x_hat = [[0.0] * D for _ in range(B)]
        out = [[0.0] * D for _ in range(B)]
        for b in range(B):
            for d in range(D):
                inv_std = 1.0 / math.sqrt(curr_var[d] + self.eps)
                self.x_hat[b][d] = (x[b][d] - curr_mean[d]) * inv_std
                out[b][d] = self.gamma[d] * self.x_hat[b][d] + self.beta[d]

        return out

    def backward(self, grad_output):
        """
        Analytical backward pass for BatchNorm.
        grad_output: shape (B, D)
        Returns: (grad_input, grad_gamma, grad_beta)
        """
        B = len(grad_output)
        D = self.num_features

        grad_gamma = [sum(grad_output[b][d] * self.x_hat[b][d] for b in range(B)) for d in range(D)]
        grad_beta = [sum(grad_output[b][d] for b in range(B)) for d in range(D)]

        # Gradient w.r.t input x
        grad_input = [[0.0] * D for _ in range(B)]
        for d in range(D):
            inv_std = 1.0 / math.sqrt(self.var[d] + self.eps)
            sum_dy = sum(grad_output[b][d] for b in range(B))
            sum_dy_xhat = sum(grad_output[b][d] * self.x_hat[b][d] for b in range(B))

            for b in range(B):
                # Standard batchnorm backward formula:
                # dx = (gamma / (B * std)) * [B * dy - sum(dy) - x_hat * sum(dy * x_hat)]
                term = B * grad_output[b][d] - sum_dy - self.x_hat[b][d] * sum_dy_xhat
                grad_input[b][d] = (self.gamma[d] * inv_std / B) * term

        return grad_input, grad_gamma, grad_beta


# ==============================================================================
# SECTION 2: Layer Normalization (LayerNorm)
# ==============================================================================

class PureLayerNorm:
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        self.D = normalized_shape
        self.eps = eps
        self.gamma = [1.0] * self.D
        self.beta = [0.0] * self.D

        self.x = None
        self.x_hat = None
        self.mean = None
        self.var = None

    def forward(self, x):
        """x: shape (B, D)"""
        self.x = x
        B = len(x)
        D = self.D

        self.mean = [sum(x[b]) / D for b in range(B)]
        self.var = [sum((x[b][d] - self.mean[b]) ** 2 for d in range(D)) / D for b in range(B)]

        self.x_hat = [[0.0] * D for _ in range(B)]
        out = [[0.0] * D for _ in range(B)]
        for b in range(B):
            inv_std = 1.0 / math.sqrt(self.var[b] + self.eps)
            for d in range(D):
                self.x_hat[b][d] = (x[b][d] - self.mean[b]) * inv_std
                out[b][d] = self.gamma[d] * self.x_hat[b][d] + self.beta[d]
        return out

    def backward(self, grad_output):
        """grad_output: shape (B, D)"""
        B = len(grad_output)
        D = self.D

        grad_gamma = [sum(grad_output[b][d] * self.x_hat[b][d] for b in range(B)) for d in range(D)]
        grad_beta = [sum(grad_output[b][d] for b in range(B)) for d in range(D)]

        grad_input = [[0.0] * D for _ in range(B)]
        for b in range(B):
            inv_std = 1.0 / math.sqrt(self.var[b] + self.eps)
            sum_dy_gamma = sum(grad_output[b][d] * self.gamma[d] for d in range(D))
            sum_dy_gamma_xhat = sum(grad_output[b][d] * self.gamma[d] * self.x_hat[b][d] for d in range(D))

            for d in range(D):
                term = D * (grad_output[b][d] * self.gamma[d]) - sum_dy_gamma - self.x_hat[b][d] * sum_dy_gamma_xhat
                grad_input[b][d] = (inv_std / D) * term

        return grad_input, grad_gamma, grad_beta


# ==============================================================================
# SECTION 3: Root Mean Square Normalization (RMSNorm)
# ==============================================================================

class PureRMSNorm:
    """
    Modern LLM normalization (used in LLaMA 1/2/3, Mistral, Gemma).
    Omits mean centering: RMS(x) = sqrt(1/D * sum(x_i^2) + eps).
    Eliminates bias beta.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        self.D = dim
        self.eps = eps
        self.gamma = [1.0] * dim

        self.x = None
        self.rms = None
        self.x_norm = None

    def forward(self, x):
        """x: shape (B, D)"""
        self.x = x
        B = len(x)
        D = self.D

        self.rms = [math.sqrt(sum(x[b][d] ** 2 for d in range(D)) / D + self.eps) for b in range(B)]
        self.x_norm = [[0.0] * D for _ in range(B)]
        out = [[0.0] * D for _ in range(B)]

        for b in range(B):
            inv_rms = 1.0 / self.rms[b]
            for d in range(D):
                self.x_norm[b][d] = x[b][d] * inv_rms
                out[b][d] = self.gamma[d] * self.x_norm[b][d]
        return out

    def backward(self, grad_output):
        B = len(grad_output)
        D = self.D

        grad_gamma = [sum(grad_output[b][d] * self.x_norm[b][d] for b in range(B)) for d in range(D)]

        grad_input = [[0.0] * D for _ in range(B)]
        for b in range(B):
            inv_rms = 1.0 / self.rms[b]
            # d(Loss)/d(RMS) * d(RMS)/d(x_i)
            # sum_d = sum_j (dy_j * gamma_j * x_j) / (D * RMS^2)
            dot = sum(grad_output[b][d] * self.gamma[d] * self.x[b][d] for d in range(D))
            factor = dot / (D * (self.rms[b] ** 2))

            for d in range(D):
                grad_input[b][d] = inv_rms * (grad_output[b][d] * self.gamma[d] - self.x[b][d] * factor)

        return grad_input, grad_gamma


# ==============================================================================
# SECTION 4: Finite Difference Gradient Checking
# ==============================================================================

def check_gradients():
    print("=" * 78)
    print("TEST 1: Analytical vs Finite-Difference Numerical Gradient Checking")
    print("=" * 78)

    set_seed(42)
    B, D = 4, 5
    x = [[random.gauss(0.0, 1.5) for _ in range(D)] for _ in range(B)]
    grad_out = [[random.gauss(0.0, 1.0) for _ in range(D)] for _ in range(B)]
    eps = 1e-6

    # 1. Check BatchNorm
    bn = PureBatchNorm1d(num_features=D)
    _ = bn.forward(x)
    dx_ana, dg_ana, db_ana = bn.backward(grad_out)

    # Numerical grad w.r.t x[0][0]
    x_plus = [row[:] for row in x]
    x_minus = [row[:] for row in x]
    x_plus[0][0] += eps
    x_minus[0][0] -= eps

    out_p = bn.forward(x_plus)
    loss_p = sum(out_p[b][d] * grad_out[b][d] for b in range(B) for d in range(D))
    out_m = bn.forward(x_minus)
    loss_m = sum(out_m[b][d] * grad_out[b][d] for b in range(B) for d in range(D))
    dx_num = (loss_p - loss_m) / (2.0 * eps)

    diff_bn = abs(dx_ana[0][0] - dx_num)
    print(f"BatchNorm  - dx[0,0] Analytical: {dx_ana[0][0]:.6f} | Numerical: {dx_num:.6f} | Abs Diff: {diff_bn:.2e}")
    assert diff_bn < 1e-5, f"BatchNorm gradient check failed! Diff: {diff_bn}"

    # 2. Check LayerNorm
    ln = PureLayerNorm(normalized_shape=D)
    _ = ln.forward(x)
    dx_ana_ln, _, _ = ln.backward(grad_out)

    out_p = ln.forward(x_plus)
    loss_p = sum(out_p[b][d] * grad_out[b][d] for b in range(B) for d in range(D))
    out_m = ln.forward(x_minus)
    loss_m = sum(out_m[b][d] * grad_out[b][d] for b in range(B) for d in range(D))
    dx_num_ln = (loss_p - loss_m) / (2.0 * eps)

    diff_ln = abs(dx_ana_ln[0][0] - dx_num_ln)
    print(f"LayerNorm  - dx[0,0] Analytical: {dx_ana_ln[0][0]:.6f} | Numerical: {dx_num_ln:.6f} | Abs Diff: {diff_ln:.2e}")
    assert diff_ln < 1e-5, f"LayerNorm gradient check failed! Diff: {diff_ln}"

    # 3. Check RMSNorm
    rms = PureRMSNorm(dim=D)
    _ = rms.forward(x)
    dx_ana_rms, _ = rms.backward(grad_out)

    out_p = rms.forward(x_plus)
    loss_p = sum(out_p[b][d] * grad_out[b][d] for b in range(B) for d in range(D))
    out_m = rms.forward(x_minus)
    loss_m = sum(out_m[b][d] * grad_out[b][d] for b in range(B) for d in range(D))
    dx_num_rms = (loss_p - loss_m) / (2.0 * eps)

    diff_rms = abs(dx_ana_rms[0][0] - dx_num_rms)
    print(f"RMSNorm    - dx[0,0] Analytical: {dx_ana_rms[0][0]:.6f} | Numerical: {dx_num_rms:.6f} | Abs Diff: {diff_rms:.2e}")
    assert diff_rms < 1e-5, f"RMSNorm gradient check failed! Diff: {diff_rms}"

    print("✓ TEST 1 PASSED: All analytical backward passes match numerical finite differences.\n")


def test_scale_invariance_and_batch_independence():
    print("=" * 78)
    print("TEST 2: Scale Invariance and Batch Size Independence")
    print("=" * 78)

    set_seed(42)
    B, D = 4, 8
    x_base = [[random.gauss(10.0, 5.0) for _ in range(D)] for _ in range(B)]
    # Multiply inputs by 1000x
    x_scaled = [[val * 1000.0 for val in row] for row in x_base]

    ln = PureLayerNorm(normalized_shape=D)
    rms = PureRMSNorm(dim=D)

    out_base_ln = ln.forward(x_base)
    out_scaled_ln = ln.forward(x_scaled)

    out_base_rms = rms.forward(x_base)
    out_scaled_rms = rms.forward(x_scaled)

    # LayerNorm scale invariance: (k*x - k*mu) / (k*std) = (x - mu) / std
    max_ln_diff = max(abs(out_base_ln[b][d] - out_scaled_ln[b][d]) for b in range(B) for d in range(D))
    max_rms_diff = max(abs(out_base_rms[b][d] - out_scaled_rms[b][d]) for b in range(B) for d in range(D))

    print(f"LayerNorm Output on 1x Input  [0,:3]: {[round(v, 4) for v in out_base_ln[0][:3]]}")
    print(f"LayerNorm Output on 1000x Input [0,:3]: {[round(v, 4) for v in out_scaled_ln[0][:3]]}")
    print(f"Max Difference between 1x and 1000x: {max_ln_diff:.2e}")
    assert max_ln_diff < 1e-4, f"LayerNorm failed scale invariance! Diff: {max_ln_diff}"

    print(f"\nRMSNorm Output on 1x Input  [0,:3]:   {[round(v, 4) for v in out_base_rms[0][:3]]}")
    print(f"RMSNorm Output on 1000x Input [0,:3]:   {[round(v, 4) for v in out_scaled_rms[0][:3]]}")
    print(f"Max Difference between 1x and 1000x: {max_rms_diff:.2e}")
    assert max_rms_diff < 1e-4, f"RMSNorm failed scale invariance! Diff: {max_rms_diff}"

    # Batch Size 1 Test (LayerNorm and RMSNorm operate flawlessly on single token)
    single_token = [[2.0, -1.0, 3.5, 0.4]]
    ln_single = PureLayerNorm(normalized_shape=4)
    rms_single = PureRMSNorm(dim=4)
    out_ln_s = ln_single.forward(single_token)
    out_rms_s = rms_single.forward(single_token)

    mean_s = sum(out_ln_s[0]) / 4
    var_s = sum((v - mean_s) ** 2 for v in out_ln_s[0]) / 4
    print(f"\nSingle-Token LayerNorm (B=1): Mean = {mean_s:.6f}, Var = {var_s:.6f}")
    assert abs(mean_s) < 1e-5 and abs(var_s - 1.0) < 1e-4
    print("✓ TEST 2 PASSED: Scale invariance and single-token normalization verified.\n")


def main():
    print("\n" + "#" * 78)
    print("RUNNING TOPIC 13.4 EXPERIMENT: BATCHNORM, LAYERNORM & RMSNORM")
    print("#" * 78 + "\n")

    check_gradients()
    test_scale_invariance_and_batch_independence()

    print("=" * 78)
    print("ALL NORMALIZATION TESTS PASSED CLEANLY (EXIT CODE 0).")
    print("=" * 78)
    sys.exit(0)


if __name__ == "__main__":
    main()
