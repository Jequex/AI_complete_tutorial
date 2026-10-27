#!/usr/bin/env python3
"""
Topic 05: Model Steering & Weight Editing
-----------------------------------------
A pure Python 3 standard library implementation demonstrating:
1. Activation Steering (Representation Engineering, Zou et al. 2023):
   - Computing concept steering vectors from contrastive pairs:
     v_steer = mean(h_pos) - mean(h_neg)
   - Runtime intervention in residual stream: h' = h + alpha * v_steer
   - Steering model output logits across a continuous spectrum:
     from negative sentiment to neutral to positive sentiment.
2. Rank-One Model Editing (ROME, Meng et al. 2022):
   - Closed-form rank-1 MLP weight modification preserving existing knowledge covariance C = K K^T:
     W' = W + Lambda * (C^-1 * k_*)^T, where Lambda = (v_* - W * k_*) / (k_*^T * C^-1 * k_*)
   - Updating/inserting a target factual memory (e.g. "Space Needle" -> "Seattle")
     while preserving all existing factual associations (Efficacy = 100%, Specificity > 98%).

Zero external dependencies (pure Python standard library).
"""

import math
import random

random.seed(42)

# =====================================================================
# 1. Pure Python Linear Algebra Utilities
# =====================================================================

def dot(u, v):
    return sum(a * b for a, b in zip(u, v))

def norm(u):
    return math.sqrt(sum(a * a for a in u)) + 1e-12

def vec_add(u, v):
    return [a + b for a, b in zip(u, v)]

def vec_sub(u, v):
    return [a - b for a, b in zip(u, v)]

def vec_scale(u, s):
    return [a * s for a in u]

def matmul_vec(W, x):
    """W is list of rows: res = W * x"""
    return [dot(row, x) for row in W]

def outer_product(u, v):
    """u (m), v (n) -> (m x n) matrix"""
    return [[u[i] * v[j] for j in range(len(v))] for i in range(len(u))]

def mat_add(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def invert_matrix_cholesky_or_gauss(A):
    """
    Invert an (n x n) matrix using Gauss-Jordan elimination with partial pivoting.
    """
    n = len(A)
    # Augmented matrix [A | I]
    aug = [list(A[i]) + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    
    for col in range(n):
        # Pivot selection
        max_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[max_row] = aug[max_row], aug[col]
        
        pivot = aug[col][col]
        if abs(pivot) < 1e-12:
            raise ValueError("Matrix singular or near-singular.")
            
        inv_pivot = 1.0 / pivot
        for j in range(2 * n):
            aug[col][j] *= inv_pivot
            
        for r in range(n):
            if r != col:
                factor = aug[r][col]
                if abs(factor) > 1e-15:
                    for j in range(2 * n):
                        aug[r][j] -= factor * aug[col][j]
                        
    inv = [[aug[i][n + j] for j in range(n)] for i in range(n)]
    return inv

def softmax(logits):
    m = max(logits)
    exps = [math.exp(l - m) for l in logits]
    s = sum(exps)
    return [e / s for e in exps]


# =====================================================================
# 2. Part 1: Activation Steering (Representation Engineering)
# =====================================================================

class SteeringExperiment:
    """
    Demonstrates runtime activation intervention using contrastive steering vectors.
    """
    def __init__(self, dim=8):
        self.dim = dim
        # Latent sentiment direction in activation space
        self.true_sentiment_axis = [0.0] * dim
        self.true_sentiment_axis[0] = 1.0
        
        # Unembedding layer projecting to logits: 0=Negative, 1=Positive
        self.W_unembed = [
            [-2.0, 0.1, -0.2, 0.0, 0.1, -0.1, 0.0, 0.0],  # Negative logit
            [+2.0, -0.1, 0.2, 0.0, -0.1, 0.1, 0.0, 0.0]   # Positive logit
        ]

    def generate_contrastive_dataset(self, n_samples=50):
        pos_samples = []
        neg_samples = []
        for _ in range(n_samples):
            # Positives: high coordinate 0 + random noise
            pos = [random.gauss(0, 0.3) for _ in range(self.dim)]
            pos[0] += random.uniform(1.5, 2.5)
            pos_samples.append(pos)
            
            # Negatives: low coordinate 0 + random noise
            neg = [random.gauss(0, 0.3) for _ in range(self.dim)]
            neg[0] -= random.uniform(1.5, 2.5)
            neg_samples.append(neg)
            
        return pos_samples, neg_samples

    def compute_steering_vector(self, pos_samples, neg_samples):
        """v_steer = mean(pos) - mean(neg)"""
        mean_pos = [sum(s[d] for s in pos_samples) / len(pos_samples) for d in range(self.dim)]
        mean_neg = [sum(s[d] for s in neg_samples) / len(neg_samples) for d in range(self.dim)]
        v_steer = vec_sub(mean_pos, mean_neg)
        # Normalize
        v_norm = norm(v_steer)
        return [v / v_norm for v in v_steer]

    def predict(self, h):
        logits = matmul_vec(self.W_unembed, h)
        return softmax(logits), logits


# =====================================================================
# 3. Part 2: Rank-One Model Editing (ROME)
# =====================================================================

class ROMEModelEditor:
    """
    Implements closed-form Rank-One Model Editing for associative key-value recall:
      W * k_i = v_i  (e.g., Subject embedding k_i -> Target entity representation v_i)
    
    Given covariance matrix C = sum(k_i * k_i^T) + lambda * I,
    to update W' * k_* = v_* while minimizing sum_i ||W' * k_i - W * k_i||^2:
      W' = W + Lambda * (C^-1 * k_*)^T
      Lambda = (v_* - W * k_*) / (k_*^T * C^-1 * k_*)
    """
    def __init__(self, d_in=4, d_out=4):
        self.d_in = d_in
        self.d_out = d_out
        
        # Initial weight matrix W
        self.W = [
            [1.0, 0.2, -0.1, 0.0],
            [0.1, 1.2, 0.0, -0.2],
            [-0.2, 0.0, 1.1, 0.1],
            [0.0, -0.1, 0.2, 1.0]
        ]

    def compute_covariance(self, keys, l2_reg=1e-3):
        """Computes C = sum(k * k^T) + l2_reg * I"""
        C = [[0.0 for _ in range(self.d_in)] for _ in range(self.d_in)]
        for k in keys:
            outer = outer_product(k, k)
            C = mat_add(C, outer)
        # Add diagonal Tikhonov regularization
        for i in range(self.d_in):
            C[i][i] += l2_reg
        return C

    def edit(self, k_star, v_star, background_keys):
        """
        Applies closed-form ROME update:
        W' = W + Lambda * (C^-1 * k_*)^T
        """
        # 1. Background covariance C and C^-1
        C = self.compute_covariance(background_keys + [k_star])
        C_inv = invert_matrix_cholesky_or_gauss(C)
        
        # 2. Projected key vector: u = C^-1 * k_*
        u = matmul_vec(C_inv, k_star)
        
        # 3. Denominator scalar: k_*^T * C^-1 * k_* = dot(k_star, u)
        denom = dot(k_star, u)
        
        # 4. Residual error: delta = v_* - W * k_*
        cur_v = matmul_vec(self.W, k_star)
        delta = vec_sub(v_star, cur_v)
        
        # 5. Vector Lambda = delta / denom
        Lambda = vec_scale(delta, 1.0 / denom)
        
        # 6. Rank-one update matrix: dW = Lambda * u^T
        dW = outer_product(Lambda, u)
        W_prime = mat_add(self.W, dW)
        return W_prime


# =====================================================================
# 4. Pipeline Execution
# =====================================================================

def run_experiment():
    print("=" * 78)
    print("PHASE 33 — TOPIC 05: MODEL STEERING & WEIGHT EDITING")
    print("Representation Engineering and Rank-One Model Editing (ROME)")
    print("=" * 78)

    # -----------------------------------------------------------------
    # Part 1: Activation Steering Demonstration
    # -----------------------------------------------------------------
    print("\n--- PART 1: ACTIVATION STEERING (REPRESENTATION ENGINEERING) ---")
    steerer = SteeringExperiment(dim=8)
    pos_data, neg_data = steerer.generate_contrastive_dataset(n_samples=100)
    v_steer = steerer.compute_steering_vector(pos_data, neg_data)
    
    print(f"Computed Concept Steering Vector v_steer (norm={norm(v_steer):.3f}):")
    print(f"  v_steer[0] (sentiment dim) = {v_steer[0]:+.4f} (expected dominant positive)")
    
    # Neutral/Ambiguous baseline activation
    h_neutral = [0.0, 0.2, -0.1, 0.05, 0.0, -0.05, 0.1, 0.0]
    
    print("\nSteering Intervention: h' = h_neutral + alpha * v_steer")
    print(f"{'Alpha':<8} | {'P(Negative)':<14} | {'P(Positive)':<14} | {'Logit Diff':<12} | {'Output Sentiment'}")
    print("-" * 68)
    
    alphas = [-3.0, -1.5, 0.0, +1.5, +3.0]
    steering_probs = []
    
    for alpha in alphas:
        h_intervened = vec_add(h_neutral, vec_scale(v_steer, alpha))
        probs, logits = steerer.predict(h_intervened)
        diff = logits[1] - logits[0]
        sentiment = "STRONG POSITIVE" if probs[1] > 0.90 else ("STRONG NEGATIVE" if probs[0] > 0.90 else "NEUTRAL")
        print(f"{alpha:>+6.1f}   | {probs[0]*100:>10.2f}%   | {probs[1]*100:>10.2f}%   | {diff:>+10.3f} | {sentiment}")
        steering_probs.append(probs[1])

    # -----------------------------------------------------------------
    # Part 2: Rank-One Model Editing (ROME) Demonstration
    # -----------------------------------------------------------------
    print("\n--- PART 2: RANK-ONE MODEL EDITING (ROME) ---")
    editor = ROMEModelEditor(d_in=4, d_out=4)
    
    # 3 Existing Facts stored in associative memory:
    # Fact 1: [1.0, 0.0, 0.0, 0.0] -> Eiffel Tower -> Paris
    # Fact 2: [0.0, 1.0, 0.0, 0.0] -> Colosseum -> Rome
    # Fact 3: [0.0, 0.0, 1.0, 0.0] -> Big Ben -> London
    existing_facts = [
        ([1.0, 0.0, 0.0, 0.0], matmul_vec(editor.W, [1.0, 0.0, 0.0, 0.0]), "Eiffel Tower -> Paris"),
        ([0.0, 1.0, 0.0, 0.0], matmul_vec(editor.W, [0.0, 1.0, 0.0, 0.0]), "Colosseum -> Rome"),
        ([0.0, 0.0, 1.0, 0.0], matmul_vec(editor.W, [0.0, 0.0, 1.0, 0.0]), "Big Ben -> London")
    ]
    background_keys = [f[0] for f in existing_facts]
    
    # Target New Fact to insert / edit:
    # "Space Needle" key k_* -> Target city "Seattle" representation v_*
    k_star = [0.0, 0.0, 0.0, 1.0]
    v_star = [0.5, -0.5, 1.5, 4.0]  # Ground truth representation for Seattle
    
    v_orig = matmul_vec(editor.W, k_star)
    orig_err = norm(vec_sub(v_orig, v_star))
    print(f"Pre-Edit Model Output for New Key k_*:   {v_orig}")
    print(f"Target Desired Representation v_*:      {v_star} (Initial Error = {orig_err:.3f})")

    # Apply closed-form ROME update
    W_edited = editor.edit(k_star, v_star, background_keys)
    
    # 1. Evaluate Efficacy on Target Edited Key
    v_edited = matmul_vec(W_edited, k_star)
    post_err = norm(vec_sub(v_edited, v_star))
    print(f"\nPost-Edit Model Output for k_*:          {v_edited}")
    print(f"Post-Edit Error on Target Key:          {post_err:.6e} (Expected ~0.0)")

    # 2. Evaluate Specificity / Preservation on Existing Facts
    print("\nPreservation of Pre-existing Factual Associations (Specificity):")
    max_drift = 0.0
    for k_orig, v_true, fact_name in existing_facts:
        v_check = matmul_vec(W_edited, k_orig)
        drift = norm(vec_sub(v_check, v_true))
        max_drift = max(max_drift, drift)
        print(f"  {fact_name:<25}: Drift = {drift:.6e} | Status: PRESERVED")

    # Verifications & Assertions
    print("\n[Scientific Verifications]")
    # Steering verifications
    assert steering_probs[0] < 0.05, "Negative steering should suppress positive sentiment!"
    assert steering_probs[-1] > 0.95, "Positive steering should drive positive sentiment > 95%!"
    print("1. Activation Steering successfully shifted distribution from <5% to >95% probability.")
    
    # ROME verifications
    assert post_err < 1e-4, f"ROME edit efficacy failed, post error = {post_err}"
    assert max_drift < 1e-4, f"ROME edit caused catastrophic forgetting, max drift = {max_drift}"
    print(f"2. ROME Efficacy: Exact target convergence (error < 1e-4).")
    print(f"3. ROME Specificity: Zero catastrophic interference on existing memories (max drift < 1e-4).")

    print("\n>>> ALL TESTS PASSED: Model steering and weight editing verified successfully!")
    print("=" * 78)

if __name__ == "__main__":
    run_experiment()
