#!/usr/bin/env python3
"""
Phase 31: AI Evaluation, Benchmarks & Metrics
Topic 02: Classification, Calibration & Thresholding

Implementation of:
1. Exact ROC curve generation and trapezoidal ROC-AUC calculation.
2. Verification of ROC-AUC via Mann-Whitney U-statistic.
3. Precision-Recall curve and PR-AUC (Average Precision).
4. Brier score and Murphy's 3-component decomposition (Reliability, Resolution, Uncertainty).
5. Expected Calibration Error (ECE) and Maximum Calibration Error (MCE) with M equal-width bins.
6. Post-hoc Temperature Scaling optimizer minimizing Negative Log-Likelihood (NLL).
7. Mathematical proof-verification that Temperature Scaling preserves ROC-AUC and accuracy.

Pure Python 3 standard library only. No external dependencies.
"""

import math
import random
from typing import List, Tuple, Dict, Any


def sigmoid(z: float) -> float:
    """Numerically stable sigmoid function."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    else:
        exp_z = math.exp(z)
        return exp_z / (1.0 + exp_z)


def logit(p: float) -> float:
    """Log-odds function."""
    p_clamped = max(1e-12, min(1.0 - 1e-12, p))
    return math.log(p_clamped / (1.0 - p_clamped))


def compute_roc_auc(labels: List[int], probs: List[float]) -> Tuple[float, List[Tuple[float, float, float]]]:
    """
    Computes ROC curve operating points (threshold, fpr, tpr) and exact trapezoidal AUC.
    """
    assert len(labels) == len(probs)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    assert n_pos > 0 and n_neg > 0, "Both classes must be present"

    # Pair and sort by descending probability
    paired = sorted(zip(probs, labels), key=lambda x: x[0], reverse=True)

    # Unique thresholds
    unique_scores = sorted(list(set(probs)), reverse=True)
    operating_points = [(float("inf"), 0.0, 0.0)]

    tp = 0
    fp = 0
    pair_idx = 0
    n = len(paired)

    for thresh in unique_scores:
        while pair_idx < n and paired[pair_idx][0] >= thresh:
            if paired[pair_idx][1] == 1:
                tp += 1
            else:
                fp += 1
            pair_idx += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        operating_points.append((thresh, fpr, tpr))

    # Trapezoidal integration
    auc = 0.0
    for k in range(1, len(operating_points)):
        fpr_prev = operating_points[k - 1][1]
        fpr_curr = operating_points[k][1]
        tpr_prev = operating_points[k - 1][2]
        tpr_curr = operating_points[k][2]
        auc += (tpr_curr + tpr_prev) * 0.5 * (fpr_curr - fpr_prev)

    return auc, operating_points


def compute_mann_whitney_u(labels: List[int], probs: List[float]) -> float:
    """
    Computes Mann-Whitney U-statistic divided by (N_pos * N_neg).
    """
    pos_scores = [p for p, y in zip(probs, labels) if y == 1]
    neg_scores = [p for p, y in zip(probs, labels) if y == 0]

    u_count = 0.0
    for sp in pos_scores:
        for sn in neg_scores:
            if sp > sn:
                u_count += 1.0
            elif abs(sp - sn) < 1e-12:
                u_count += 0.5

    return u_count / (len(pos_scores) * len(neg_scores))


def compute_pr_auc(labels: List[int], probs: List[float]) -> Tuple[float, List[Tuple[float, float, float]]]:
    """
    Computes PR curve operating points (threshold, recall, precision) and Average Precision (PR-AUC).
    """
    assert len(labels) == len(probs)
    n_pos = sum(labels)

    paired = sorted(zip(probs, labels), key=lambda x: x[0], reverse=True)
    unique_scores = sorted(list(set(probs)), reverse=True)

    points = [(float("inf"), 0.0, 1.0)]
    tp = 0
    fp = 0
    pair_idx = 0
    n = len(paired)

    pr_auc = 0.0
    prev_recall = 0.0

    for thresh in unique_scores:
        while pair_idx < n and paired[pair_idx][0] >= thresh:
            if paired[pair_idx][1] == 1:
                tp += 1
            else:
                fp += 1
            pair_idx += 1
        recall = tp / n_pos
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        points.append((thresh, recall, precision))

        delta_r = recall - prev_recall
        if delta_r > 0:
            pr_auc += delta_r * precision
            prev_recall = recall

    return pr_auc, points


def compute_brier_decomposition(labels: List[int], probs: List[float], num_bins: int = 10) -> Dict[str, float]:
    """
    Computes Brier score and Murphy's decomposition: BS = Reliability - Resolution + Uncertainty.
    """
    n = len(labels)
    assert n == len(probs) and n > 0
    base_rate = sum(labels) / n

    # Partition into equal-width bins
    bin_counts = [0] * num_bins
    bin_sum_probs = [0.0] * num_bins
    bin_sum_labels = [0.0] * num_bins

    for p, y in zip(probs, labels):
        b_idx = min(num_bins - 1, int(p * num_bins))
        bin_counts[b_idx] += 1
        bin_sum_probs[b_idx] += p
        bin_sum_labels[b_idx] += y

    rel = 0.0
    res = 0.0

    for count, s_p, s_y in zip(bin_counts, bin_sum_probs, bin_sum_labels):
        if count == 0:
            continue
        p_bar = s_p / count
        y_bar = s_y / count
        weight = count / n
        rel += weight * ((p_bar - y_bar) ** 2)
        res += weight * ((y_bar - base_rate) ** 2)

    unc = base_rate * (1.0 - base_rate)
    brier_decomposed = rel - res + unc
    brier_exact = sum((p - y) ** 2 for p, y in zip(probs, labels)) / n

    return {
        "brier_exact": brier_exact,
        "reliability": rel,
        "resolution": res,
        "uncertainty": unc,
        "brier_decomposed": brier_decomposed,
        "decomposition_gap": abs(brier_exact - brier_decomposed),
    }


def compute_calibration_errors(labels: List[int], probs: List[float], num_bins: int = 10) -> Dict[str, Any]:
    """
    Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    """
    n = len(labels)
    bin_counts = [0] * num_bins
    bin_sum_probs = [0.0] * num_bins
    bin_sum_labels = [0.0] * num_bins

    for p, y in zip(probs, labels):
        b_idx = min(num_bins - 1, int(p * num_bins))
        bin_counts[b_idx] += 1
        bin_sum_probs[b_idx] += p
        bin_sum_labels[b_idx] += y

    ece = 0.0
    mce = 0.0
    bins_data = []

    for idx, (cnt, s_p, s_y) in enumerate(zip(bin_counts, bin_sum_probs, bin_sum_labels)):
        if cnt == 0:
            continue
        conf = s_p / cnt
        acc = s_y / cnt
        gap = abs(acc - conf)
        weight = cnt / n
        ece += weight * gap
        if gap > mce:
            mce = gap
        bins_data.append({
            "bin": idx + 1,
            "range": (idx / num_bins, (idx + 1) / num_bins),
            "count": cnt,
            "confidence": conf,
            "accuracy": acc,
            "gap": gap,
        })

    return {
        "ece": ece,
        "mce": mce,
        "bins": bins_data,
    }


def optimize_temperature(logits: List[float], labels: List[int]) -> float:
    """
    Finds optimal temperature T > 0 minimizing Binary Cross-Entropy via 1D Golden Section search.
    """
    def nll(temp: float) -> float:
        total = 0.0
        for z, y in zip(logits, labels):
            scaled_z = z / temp
            p = sigmoid(scaled_z)
            p = max(1e-12, min(1.0 - 1e-12, p))
            total += -(y * math.log(p) + (1 - y) * math.log(1.0 - p))
        return total / len(logits)

    # Search in T in [0.1, 10.0]
    a = 0.1
    b = 10.0
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    tol = 1e-5

    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc = nll(c)
    fd = nll(d)

    while abs(b - a) > tol:
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - phi * (b - a)
            fc = nll(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + phi * (b - a)
            fd = nll(d)

    return (a + b) / 2.0


def run_experiment():
    print("=" * 80)
    print("PHASE 31 — TOPIC 02: CLASSIFICATION, CALIBRATION & THRESHOLDING")
    print("=" * 80)

    # 1. Simulate an overconfident neural network on validation and test cohorts
    rng = random.Random(42)
    n_samples = 1000
    true_labels = []
    uncal_logits = []

    for _ in range(n_samples):
        # Latent true probability
        p_true = rng.betavariate(2.0, 2.0)
        y = 1 if rng.random() < p_true else 0
        true_labels.append(y)

        # Overconfident network expands logits by ~2.5x
        true_logit = logit(p_true)
        overconf_logit = 2.4 * true_logit + rng.gauss(0, 0.4)
        uncal_logits.append(overconf_logit)

    uncal_probs = [sigmoid(z) for z in uncal_logits]

    # Split into 500 val and 500 test
    val_labels = true_labels[:500]
    val_logits = uncal_logits[:500]
    val_probs = uncal_probs[:500]

    test_labels = true_labels[500:]
    test_logits = uncal_logits[500:]
    test_probs = uncal_probs[500:]

    # 2. Compute Discrimination Metrics (ROC-AUC & PR-AUC)
    roc_auc, _ = compute_roc_auc(test_labels, test_probs)
    mann_whitney = compute_mann_whitney_u(test_labels, test_probs)
    pr_auc, _ = compute_pr_auc(test_labels, test_probs)

    print("\n[1] Model Discrimination Metrics on Test Set (500 samples):")
    print(f"  • ROC-AUC (Trapezoidal)     : {roc_auc:.5f}")
    print(f"  • ROC-AUC (Mann-Whitney U)  : {mann_whitney:.5f}")
    print(f"  • Trapezoidal vs U Gap      : {abs(roc_auc - mann_whitney):.2e}")
    print(f"  • PR-AUC (Average Precision): {pr_auc:.5f}")
    assert abs(roc_auc - mann_whitney) < 1e-9, "ROC-AUC must equal Mann-Whitney U"

    # 3. Pre-Calibration Evaluation
    pre_cal = compute_calibration_errors(test_labels, test_probs, num_bins=10)
    pre_brier = compute_brier_decomposition(test_labels, test_probs, num_bins=10)
    pre_acc = sum((p >= 0.5) == y for p, y in zip(test_probs, test_labels)) / len(test_labels)

    print("\n[2] Pre-Calibration Calibration Status (Uncalibrated Overconfident Model):")
    print(f"  • Accuracy (threshold 0.50) : {pre_acc * 100:.2f}%")
    print(f"  • Expected Calibration Error: {pre_cal['ece'] * 100:.2f}%")
    print(f"  • Maximum Calibration Error : {pre_cal['mce'] * 100:.2f}%")
    print(f"  • Total Brier Score         : {pre_brier['brier_exact']:.5f}")
    print(f"    - Reliability (REL)       : {pre_brier['reliability']:.5f}")
    print(f"    - Resolution (RES)        : {pre_brier['resolution']:.5f}")
    print(f"    - Uncertainty (UNC)       : {pre_brier['uncertainty']:.5f}")

    # 4. Fit Temperature Scaling on Validation Set
    optimal_T = optimize_temperature(val_logits, val_labels)
    print(f"\n[3] Optimal Temperature Learned on Validation Set: T = {optimal_T:.4f}")
    assert optimal_T > 1.5, "Overconfident network should yield optimal T > 1.5"

    # 5. Apply Temperature Scaling to Held-Out Test Set
    cal_logits = [z / optimal_T for z in test_logits]
    cal_probs = [sigmoid(z) for z in cal_logits]

    post_cal = compute_calibration_errors(test_labels, cal_probs, num_bins=10)
    post_brier = compute_brier_decomposition(test_labels, cal_probs, num_bins=10)
    post_roc_auc, _ = compute_roc_auc(test_labels, cal_probs)
    post_acc = sum((p >= 0.5) == y for p, y in zip(cal_probs, test_labels)) / len(test_labels)

    print("\n[4] Post-Calibration Status on Test Set (Temperature Scaled):")
    print(f"  • Accuracy (threshold 0.50) : {post_acc * 100:.2f}% (Identical to pre-calibration)")
    print(f"  • ROC-AUC                   : {post_roc_auc:.5f} (Identical to pre-calibration)")
    print(f"  • Post-Cal ECE              : {post_cal['ece'] * 100:.2f}% (Reduced from {pre_cal['ece'] * 100:.2f}%)")
    print(f"  • Post-Cal MCE              : {post_cal['mce'] * 100:.2f}% (Reduced from {pre_cal['mce'] * 100:.2f}%)")
    print(f"  • Post-Cal Brier Score      : {post_brier['brier_exact']:.5f} (Improved from {pre_brier['brier_exact']:.5f})")
    print(f"    - Reliability (REL)       : {post_brier['reliability']:.5f} (Calibration component slashed)")

    # 6. Print Reliability Bin Comparison Table
    print("\n[5] Reliability Table: Pre- vs Post-Calibration Bins:")
    print(f"{'Bin':<5} | {'Range':<14} | {'Count':<6} | {'Pre Conf':<10} | {'Post Conf':<10} | {'Actual Acc':<10} | {'Pre Gap':<10} | {'Post Gap':<10}")
    print("-" * 90)
    for pre_b, post_b in zip(pre_cal["bins"], post_cal["bins"]):
        rng_str = f"[{pre_b['range'][0]:.1f}, {pre_b['range'][1]:.1f})"
        print(f"{pre_b['bin']:<5} | {rng_str:<14} | {pre_b['count']:<6} | {pre_b['confidence']:<10.3f} | {post_b['confidence']:<10.3f} | {pre_b['accuracy']:<10.3f} | {pre_b['gap']:<10.3f} | {post_b['gap']:<10.3f}")

    # Assertions
    assert abs(post_roc_auc - roc_auc) < 1e-9, "ROC-AUC must be invariant under temperature scaling"
    assert abs(post_acc - pre_acc) < 1e-9, "Accuracy at threshold 0.5 must be invariant under temperature scaling"
    assert post_cal["ece"] < pre_cal["ece"], "ECE must improve following temperature scaling"
    assert post_brier["reliability"] < pre_brier["reliability"], "Brier reliability component must improve"

    print("\nAll classification & calibration assertions passed successfully! (Exit Code 0)")


if __name__ == "__main__":
    run_experiment()
