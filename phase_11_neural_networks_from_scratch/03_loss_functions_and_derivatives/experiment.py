#!/usr/bin/env python3
"""
Phase 11: Neural Networks From Scratch
Topic 11.3: Loss Functions & Analytical Derivatives

Pure Python 3 implementation (zero external dependencies).
Demonstrates:
  1. Mean Squared Error (MSE) loss & analytical gradient w.r.t predictions.
  2. Binary Cross-Entropy (BCE) loss & combined Sigmoid + BCE gradient (p - y).
  3. Multi-Class Categorical Cross-Entropy (CCE) with numerically stable Softmax:
     combined Softmax + CCE gradient (p - y).
  4. Focal Loss for extreme class imbalance.
  5. Rigorous numerical gradient verification via centered finite differences:
     (L(theta + eps) - L(theta - eps)) / (2 * eps).
"""

import math
from typing import List, Tuple


# =====================================================================
# 1. Mean Squared Error (MSE)
# =====================================================================
def mse_loss(y_true: List[float], y_pred: List[float]) -> float:
    """MSE = (1 / (2 * N)) * sum((y_pred_i - y_true_i)^2)"""
    N = len(y_true)
    return sum((p - t) ** 2 for t, p in zip(y_true, y_pred)) / (2.0 * N)


def mse_gradient(y_true: List[float], y_pred: List[float]) -> List[float]:
    """dL/dy_pred = (1 / N) * (y_pred - y_true)"""
    N = len(y_true)
    return [(p - t) / N for t, p in zip(y_true, y_pred)]


# =====================================================================
# 2. Binary Cross-Entropy (BCE)
# =====================================================================
def stable_sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)


def bce_loss(y_true: List[float], y_prob: List[float], eps: float = 1e-15) -> float:
    """
    BCE = - (1 / N) * sum(y * ln(p) + (1 - y) * ln(1 - p))
    Clamped to [eps, 1 - eps] to prevent log(0).
    """
    N = len(y_true)
    loss = 0.0
    for y, p in zip(y_true, y_prob):
        p_c = max(eps, min(1.0 - eps, p))
        loss -= (y * math.log(p_c) + (1.0 - y) * math.log(1.0 - p_c))
    return loss / N


def bce_loss_from_logits(y_true: List[float], logits: List[float]) -> float:
    """
    Numerically stable BCE with logits using log-sum-exp trick:
    max(z, 0) - z * y + log(1 + exp(-|z|))
    """
    N = len(y_true)
    loss = 0.0
    for y, z in zip(y_true, logits):
        loss += max(z, 0.0) - z * y + math.log(1.0 + math.exp(-abs(z)))
    return loss / N


def bce_logit_gradient(y_true: List[float], logits: List[float]) -> List[float]:
    """
    dL/dz = (1 / N) * (sigmoid(z) - y)
    The Sigmoid derivative cancels out the denominator!
    """
    N = len(y_true)
    return [(stable_sigmoid(z) - y) / N for y, z in zip(y_true, logits)]


# =====================================================================
# 3. Softmax & Categorical Cross-Entropy (CCE)
# =====================================================================
def softmax(logits: List[float]) -> List[float]:
    """Numerically stable softmax by subtracting max logit."""
    max_z = max(logits)
    exp_shifted = [math.exp(z - max_z) for z in logits]
    sum_exp = sum(exp_shifted)
    return [e / sum_exp for e in exp_shifted]


def categorical_cross_entropy(y_one_hot: List[List[float]], probs: List[List[float]], eps: float = 1e-15) -> float:
    """
    CCE = - (1 / N) * sum_{i} sum_{k} y_{ik} * ln(p_{ik})
    """
    N = len(y_one_hot)
    total_loss = 0.0
    for y_row, p_row in zip(y_one_hot, probs):
        for y, p in zip(y_row, p_row):
            if y > 0:
                p_c = max(eps, p)
                total_loss -= y * math.log(p_c)
    return total_loss / N


def cce_loss_from_logits(y_one_hot: List[List[float]], logits_batch: List[List[float]]) -> float:
    """Stable CCE from raw logits."""
    probs_batch = [softmax(logits) for logits in logits_batch]
    return categorical_cross_entropy(y_one_hot, probs_batch)


def softmax_cce_logit_gradient(y_one_hot: List[List[float]], logits_batch: List[List[float]]) -> List[List[float]]:
    """
    dL/dz_{ik} = (1 / N) * (p_{ik} - y_{ik})
    """
    N = len(y_one_hot)
    grads = []
    for y_row, logits in zip(y_one_hot, logits_batch):
        p_row = softmax(logits)
        grad_row = [(p - y) / N for y, p in zip(y_row, p_row)]
        grads.append(grad_row)
    return grads


# =====================================================================
# 4. Focal Loss for Class Imbalance
# =====================================================================
def focal_loss(y_true: List[float], y_prob: List[float], gamma: float = 2.0, alpha: float = 0.25, eps: float = 1e-15) -> float:
    """
    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    Down-weights easy examples, focuses on hard positives/negatives.
    """
    N = len(y_true)
    total_loss = 0.0
    for y, p in zip(y_true, y_prob):
        p_c = max(eps, min(1.0 - eps, p))
        if y == 1.0:
            p_t = p_c
            alpha_t = alpha
        else:
            p_t = 1.0 - p_c
            alpha_t = 1.0 - alpha
        total_loss -= alpha_t * ((1.0 - p_t) ** gamma) * math.log(p_t)
    return total_loss / N


# =====================================================================
# Numerical Finite-Difference Gradient Verifier
# =====================================================================
def test_mse_gradients():
    print("=" * 70)
    print("Step 1: MSE Loss & Finite-Difference Gradient Verification")
    print("=" * 70)
    y_true = [1.5, -0.5, 2.0, 0.0]
    y_pred = [1.2, 0.3, 1.8, -0.4]

    loss = mse_loss(y_true, y_pred)
    anal_grads = mse_gradient(y_true, y_pred)
    eps = 1e-6

    print(f"MSE Loss: {loss:.6f}")
    for i in range(len(y_pred)):
        # Perturb y_pred[i] + eps
        y_pred_plus = list(y_pred)
        y_pred_plus[i] += eps
        loss_plus = mse_loss(y_true, y_pred_plus)

        # Perturb y_pred[i] - eps
        y_pred_minus = list(y_pred)
        y_pred_minus[i] -= eps
        loss_minus = mse_loss(y_true, y_pred_minus)

        num_grad = (loss_plus - loss_minus) / (2.0 * eps)
        diff = abs(anal_grads[i] - num_grad)
        print(f"  Feature {i}: Analytical={anal_grads[i]:10.6f}, Numerical={num_grad:10.6f}, Diff={diff:1.1e}")
        assert diff < 1e-6, f"MSE gradient mismatch at {i}!"

    print("✓ MSE analytical gradients verified via finite differences.")


def test_bce_logits_gradients():
    print("\n" + "=" * 70)
    print("Step 2: Binary Cross-Entropy (BCE) Logit Gradient Verification")
    print("=" * 70)
    y_true = [1.0, 0.0, 1.0, 0.0]
    logits = [2.5, 1.2, -1.8, -0.5]

    loss = bce_loss_from_logits(y_true, logits)
    anal_grads = bce_logit_gradient(y_true, logits)
    eps = 1e-6

    print(f"BCE Loss from Logits: {loss:.6f}")
    for i in range(len(logits)):
        logits_plus = list(logits)
        logits_plus[i] += eps
        loss_plus = bce_loss_from_logits(y_true, logits_plus)

        logits_minus = list(logits)
        logits_minus[i] -= eps
        loss_minus = bce_loss_from_logits(y_true, logits_minus)

        num_grad = (loss_plus - loss_minus) / (2.0 * eps)
        diff = abs(anal_grads[i] - num_grad)
        print(f"  Sample {i} (y={y_true[i]}, z={logits[i]:4.1f}): Analytical={anal_grads[i]:10.6f}, Numerical={num_grad:10.6f}, Diff={diff:1.1e}")
        assert diff < 1e-6, f"BCE logit gradient mismatch at {i}!"

    print("✓ BCE logit gradient (p - y) verified via finite differences.")


def test_softmax_cce_gradients():
    print("\n" + "=" * 70)
    print("Step 3: Softmax + Categorical Cross-Entropy (CCE) Gradient Verification")
    print("=" * 70)
    # Batch of 2 samples, 3 classes
    y_one_hot = [
        [1.0, 0.0, 0.0],  # Sample 0 is class 0
        [0.0, 1.0, 0.0],  # Sample 1 is class 1
    ]
    logits_batch = [
        [2.0, 1.0, 0.1],  # Sample 0
        [-0.5, 3.0, 1.5], # Sample 1
    ]

    loss = cce_loss_from_logits(y_one_hot, logits_batch)
    anal_grads = softmax_cce_logit_gradient(y_one_hot, logits_batch)
    eps = 1e-6

    print(f"Softmax CCE Loss: {loss:.6f}")
    for i in range(len(logits_batch)):
        for k in range(len(logits_batch[i])):
            # Perturb logits_batch[i][k] + eps
            batch_plus = [list(row) for row in logits_batch]
            batch_plus[i][k] += eps
            loss_plus = cce_loss_from_logits(y_one_hot, batch_plus)

            batch_minus = [list(row) for row in logits_batch]
            batch_minus[i][k] -= eps
            loss_minus = cce_loss_from_logits(y_one_hot, batch_minus)

            num_grad = (loss_plus - loss_minus) / (2.0 * eps)
            diff = abs(anal_grads[i][k] - num_grad)
            print(f"  Sample {i}, Class {k}: Anal={anal_grads[i][k]:10.6f}, Num={num_grad:10.6f}, Diff={diff:1.1e}")
            assert diff < 1e-6, f"Softmax CCE gradient mismatch at ({i}, {k})!"

    print("✓ Softmax + CCE analytical gradient (p - y) verified via finite differences.")


def test_focal_loss_weighting():
    print("\n" + "=" * 70)
    print("Step 4: Focal Loss vs Standard BCE Under Extreme Imbalance")
    print("=" * 70)
    # Scenario: 98 easy negative samples (prob = 0.01), 1 hard positive sample (prob = 0.05)
    # Standard BCE treats easy negatives with significant cumulative weight.
    y_easy_neg = [0.0]
    p_easy_neg = [0.01]
    y_hard_pos = [1.0]
    p_hard_pos = [0.05]

    bce_neg = bce_loss(y_easy_neg, p_easy_neg)
    bce_pos = bce_loss(y_hard_pos, p_hard_pos)
    focal_neg = focal_loss(y_easy_neg, p_easy_neg, gamma=2.0, alpha=0.25)
    focal_pos = focal_loss(y_hard_pos, p_hard_pos, gamma=2.0, alpha=0.25)

    print(f"Easy Negative (y=0, p=0.01): BCE={bce_neg:.6f}, Focal={focal_neg:.8f} (Suppressed by {bce_neg/focal_neg:.1f}x)")
    print(f"Hard Positive (y=1, p=0.05): BCE={bce_pos:.6f}, Focal={focal_pos:.6f}")
    assert focal_neg < bce_neg / 100.0, "Focal loss did not sufficiently suppress easy negative!"
    print("✓ Focal Loss correctly suppresses easy background examples.")


def main():
    print("=" * 70)
    print("PHASE 11.3 EXPERIMENT: LOSS FUNCTIONS & DERIVATIVES")
    print("=" * 70)
    test_mse_gradients()
    test_bce_logits_gradients()
    test_softmax_cce_gradients()
    test_focal_loss_weighting()
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS PASSED SUCCESSFULLY (Exit Code 0)")
    print("=" * 70)


if __name__ == "__main__":
    main()
