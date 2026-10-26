#!/usr/bin/env python3
"""
Topic 10.3 Experiment: Comprehensive Evaluation Metrics & Probability Calibration
Author: AI Complete Tutorial
Dependencies: Pure Python 3 Standard Library (math, random)

Implements completely from scratch (zero dependencies):
1. Confusion Matrix (TP, FP, FN, TN)
2. Classification Metrics: Accuracy, Precision, Recall, Specificity, F1-Score, MCC
3. Full ROC Curve generation and exact Trapezoidal ROC-AUC
4. Precision-Recall Curve generation and PR-AUC
5. Brier Score & Reliability Calibration Bins
6. Platt Scaling: Calibrating uncalibrated model scores via Logistic Regression
"""

import math
import random

# ----------------------------------------------------------------------
# 1. Confusion Matrix & Point Metrics
# ----------------------------------------------------------------------

def compute_confusion_matrix(y_true, y_pred):
    """
    Returns (TP, FP, FN, TN).
    Assumes binary labels in {0, 1}.
    """
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    return tp, fp, fn, tn

def compute_metrics(y_true, y_pred):
    tp, fp, fn, tn = compute_confusion_matrix(y_true, y_pred)
    n = tp + fp + fn + tn

    accuracy = (tp + tn) / n if n > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Harmonic mean: 2 * P * R / (P + R)
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Matthews Correlation Coefficient (MCC)
    mcc_denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / mcc_denom if mcc_denom > 0 else 0.0

    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1": f1,
        "MCC": mcc
    }

# ----------------------------------------------------------------------
# 2. ROC & PR Curves with Trapezoidal AUC
# ----------------------------------------------------------------------

def compute_roc_and_auc(y_true, y_probs):
    """
    Generates ROC curve coordinates (FPR, TPR) across all distinct thresholds
    and computes Area Under the Curve (AUC) via trapezoidal integration.
    """
    # Total actual positives and negatives
    total_pos = sum(y_true)
    total_neg = len(y_true) - total_pos

    if total_pos == 0 or total_neg == 0:
        return [(0.0, 0.0), (1.0, 1.0)], 0.5

    # Pair probabilities with true labels and sort descending
    paired = sorted(zip(y_probs, y_true), key=lambda x: x[0], reverse=True)

    roc_points = [(0.0, 0.0)]
    tp = 0
    fp = 0

    for i in range(len(paired)):
        prob, label = paired[i]
        if label == 1:
            tp += 1
        else:
            fp += 1

        # Check if next score is different or at end
        if i == len(paired) - 1 or paired[i + 1][0] != prob:
            tpr = tp / total_pos
            fpr = fp / total_neg
            roc_points.append((fpr, tpr))

    # Ensure point (1.0, 1.0) is present
    if roc_points[-1] != (1.0, 1.0):
        roc_points.append((1.0, 1.0))

    # Trapezoidal rule for AUC: sum_{i} (x_{i} - x_{i-1}) * (y_{i} + y_{i-1}) / 2
    auc = 0.0
    for i in range(1, len(roc_points)):
        dx = roc_points[i][0] - roc_points[i - 1][0]
        avg_y = (roc_points[i][1] + roc_points[i - 1][1]) / 2.0
        auc += dx * avg_y

    return roc_points, auc

def compute_pr_curve_and_auc(y_true, y_probs):
    """Computes Precision-Recall curve and trapezoidal area."""
    total_pos = sum(y_true)
    if total_pos == 0:
        return [(0.0, 0.0)], 0.0

    paired = sorted(zip(y_probs, y_true), key=lambda x: x[0], reverse=True)
    pr_points = []
    tp = 0
    fp = 0

    for i in range(len(paired)):
        prob, label = paired[i]
        if label == 1:
            tp += 1
        else:
            fp += 1

        if i == len(paired) - 1 or paired[i + 1][0] != prob:
            recall = tp / total_pos
            precision = tp / (tp + fp)
            pr_points.append((recall, precision))

    # Compute PR-AUC (Average Precision trapezoidal approximation)
    pr_auc = 0.0
    prev_r = 0.0
    prev_p = pr_points[0][1] if pr_points else 1.0
    for r, p in pr_points:
        dr = r - prev_r
        pr_auc += dr * ((p + prev_p) / 2.0)
        prev_r = r
        prev_p = p

    return pr_points, pr_auc

# ----------------------------------------------------------------------
# 3. Calibration & Brier Score
# ----------------------------------------------------------------------

def compute_brier_score(y_true, y_probs):
    """Mean Squared Error between predicted probabilities and binary outcomes."""
    return sum((p - y) ** 2 for y, p in zip(y_true, y_probs)) / len(y_true)

def reliability_bins(y_true, y_probs, n_bins=5):
    """Groups predictions into confidence bins to evaluate empirical calibration."""
    bin_size = 1.0 / n_bins
    bins = []
    for b in range(n_bins):
        low = b * bin_size
        high = (b + 1) * bin_size
        # Collect predictions falling in [low, high)
        if b == n_bins - 1:
            indices = [i for i, p in enumerate(y_probs) if low <= p <= high]
        else:
            indices = [i for i, p in enumerate(y_probs) if low <= p < high]

        count = len(indices)
        if count > 0:
            avg_pred = sum(y_probs[i] for i in indices) / count
            avg_true = sum(y_true[i] for i in indices) / count
        else:
            avg_pred = (low + high) / 2.0
            avg_true = 0.0

        bins.append({
            "bin": f"[{low:.1f}-{high:.1f}]",
            "count": count,
            "avg_predicted": avg_pred,
            "fraction_positive": avg_true,
            "calibration_gap": abs(avg_pred - avg_true)
        })
    return bins

class PlattScaler:
    """
    Fits a 1D logistic regression model P(Y=1 | s) = 1 / (1 + exp(A*s + B))
    to calibrate uncalibrated decision scores s.
    """
    def __init__(self, lr=0.1, epochs=300):
        self.lr = lr
        self.epochs = epochs
        self.A = 1.0
        self.B = 0.0

    def fit(self, scores, y_true):
        n = len(scores)
        for _ in range(self.epochs):
            grad_A = 0.0
            grad_B = 0.0
            for s, y in zip(scores, y_true):
                z = self.A * s + self.B
                z_clamped = max(-25.0, min(25.0, z))
                p = 1.0 / (1.0 + math.exp(-z_clamped))
                err = p - y
                grad_A += err * s
                grad_B += err
            self.A -= (self.lr / n) * grad_A
            self.B -= (self.lr / n) * grad_B
        return self

    def predict_proba(self, scores):
        calibrated = []
        for s in scores:
            z = self.A * s + self.B
            z_clamped = max(-25.0, min(25.0, z))
            calibrated.append(1.0 / (1.0 + math.exp(-z_clamped)))
        return calibrated

# ----------------------------------------------------------------------
# 4. Main Experiment Execution
# ----------------------------------------------------------------------

def run_experiment():
    print("=" * 78)
    print("TOPIC 10.3: EVALUATION METRICS, CONFUSION MATRIX & CALIBRATION")
    print("=" * 78)

    random.seed(42)

    # 1. Simulate an Imbalanced Medical / Fraud Dataset
    # Total samples = 1,000; Prevalence = 5% (50 positives, 950 negatives)
    n_samples = 1000
    prevalence = 0.05
    y_true = [1 if random.random() < prevalence else 0 for _ in range(n_samples)]
    actual_pos = sum(y_true)
    actual_neg = n_samples - actual_pos

    print(f"Dataset summary: N = {n_samples} samples | Positives = {actual_pos} (5%) | Negatives = {actual_neg} (95%)")

    # Generate synthetic uncalibrated margin scores for a good classifier:
    # Positives have higher average margin, negatives have lower
    uncalibrated_scores = []
    for y in y_true:
        if y == 1:
            margin = random.gauss(2.2, 1.2) # Positive class score
        else:
            margin = random.gauss(-1.8, 1.2) # Negative class score
        uncalibrated_scores.append(margin)

    # Convert margin to raw naive sigmoid probabilities:
    raw_probs = [1.0 / (1.0 + math.exp(-max(-20, min(20, s)))) for s in uncalibrated_scores]
    raw_preds = [1 if p >= 0.5 else 0 for p in raw_probs]

    # Also simulate a "Naive Majority / Zero Predictor" that predicts 0 for everyone
    majority_preds = [0] * n_samples

    print("\n" + "─" * 78)
    print("PART 1: THE ACCURACY PARADOX ON IMBALANCED DATA")
    print("─" * 78)
    m_naive = compute_metrics(y_true, majority_preds)
    m_model = compute_metrics(y_true, raw_preds)

    print(f"{'Metric':<25} | {'Naive Zero-Predictor':<22} | {'Trained Model (Thresh 0.5)'}")
    print("─" * 78)
    print(f"{'Accuracy':<25} | {m_naive['Accuracy']*100:>20.2f}% | {m_model['Accuracy']*100:>23.2f}%")
    print(f"{'Precision':<25} | {m_naive['Precision']*100:>20.2f}% | {m_model['Precision']*100:>23.2f}%")
    print(f"{'Recall (Sensitivity)':<25} | {m_naive['Recall']*100:>20.2f}% | {m_model['Recall']*100:>23.2f}%")
    print(f"{'Specificity':<25} | {m_naive['Specificity']*100:>20.2f}% | {m_model['Specificity']*100:>23.2f}%")
    print(f"{'F1-Score':<25} | {m_naive['F1']*100:>20.2f}% | {m_model['F1']*100:>23.2f}%")
    print(f"{'Matthews Corr (MCC)':<25} | {m_naive['MCC']:>21.4f}  | {m_model['MCC']:>24.4f}")
    print(f"\nTakeaway: The Naive Predictor achieves {m_naive['Accuracy']*100:.1f}% Accuracy while catching 0% of fraud!")
    print(f"          F1 and MCC reveal the truth: Naive F1 = 0.0%, Trained F1 = {m_model['F1']*100:.1f}%, MCC = {m_model['MCC']:.4f}.")

    print("\n" + "─" * 78)
    print("PART 2: ROC-AUC & PRECISION-RECALL AUC COMPUTATION")
    print("─" * 78)
    roc_points, roc_auc = compute_roc_and_auc(y_true, raw_probs)
    pr_points, pr_auc = compute_pr_curve_and_auc(y_true, raw_probs)

    print(f"• Evaluated {len(roc_points)} distinct operating points on the ROC curve.")
    print(f"• Exact Trapezoidal ROC-AUC: {roc_auc:.4f} (Probability a random positive ranks above a negative)")
    print(f"• Exact Trapezoidal PR-AUC:  {pr_auc:.4f} (Baseline random PR-AUC is equal to prevalence: {prevalence:.2f})")

    print("\n" + "─" * 78)
    print("PART 3: PROBABILITY CALIBRATION & PLATT SCALING")
    print("─" * 78)
    # Distort raw scores to simulate an overconfident, uncalibrated model (e.g. boosting or naive Bayes)
    uncalibrated_probs = [min(0.999, max(0.001, p ** 0.4 if p >= 0.5 else p ** 2.2)) for p in raw_probs]
    brier_uncal = compute_brier_score(y_true, uncalibrated_probs)
    print(f"Uncalibrated Model Brier Score: {brier_uncal:.4f}")

    # Inspect reliability bins
    print("\nReliability Diagram Bins (Uncalibrated):")
    print(f"{'Bin':<12} | {'Count':<7} | {'Mean Pred Prob':<16} | {'Observed Fraction':<18} | {'Gap'}")
    print("─" * 70)
    for b in reliability_bins(y_true, uncalibrated_probs, n_bins=5):
        print(f"{b['bin']:<12} | {b['count']:<7} | {b['avg_predicted']*100:>14.1f}% | {b['fraction_positive']*100:>16.1f}% | {b['calibration_gap']*100:>5.1f}%")

    # Fit Platt Scaler
    platt = PlattScaler(lr=0.5, epochs=400)
    # Convert distorted probs back to logits for Platt fitting
    distorted_logits = [math.log(p / (1.0 - p)) for p in uncalibrated_probs]
    platt.fit(distorted_logits, y_true)
    calibrated_probs = platt.predict_proba(distorted_logits)

    brier_cal = compute_brier_score(y_true, calibrated_probs)
    print(f"\nCalibrated Model Brier Score: {brier_cal:.4f} (Improved by {(brier_uncal - brier_cal)/brier_uncal*100:.1f}%)")

    print("\nReliability Diagram Bins (Calibrated via Platt Scaling):")
    print(f"{'Bin':<12} | {'Count':<7} | {'Mean Pred Prob':<16} | {'Observed Fraction':<18} | {'Gap'}")
    print("─" * 70)
    for b in reliability_bins(y_true, calibrated_probs, n_bins=5):
        print(f"{b['bin']:<12} | {b['count']:<7} | {b['avg_predicted']*100:>14.1f}% | {b['fraction_positive']*100:>16.1f}% | {b['calibration_gap']*100:>5.1f}%")

    print("\n" + "=" * 78)
    print("SUCCESS: Topic 10.3 experiment completed with exit code 0.")

if __name__ == "__main__":
    run_experiment()
