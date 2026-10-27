#!/usr/bin/env python3
"""
Phase 32 — Topic 03: Data Poisoning, Backdoors, and Model Extraction
Experiment: Backdoor Injection, Spectral Signature Defense (Tran et al. 2018), and Watermark Verification

This script demonstrates from first mathematical principles:
1. Synthetic classification dataset generation with representation features.
2. Backdoor trigger implantation (x + Delta -> target label y_target) with poison budget alpha.
3. Training a linear classifier on poisoned data: demonstrating high clean accuracy + high attack success rate (ASR).
4. Spectral Signature Defense (Tran, Li, Madry 2018):
   - Centered feature covariance matrix construction: C = (1/N) * sum (h_i - mu)(h_i - mu)^T.
   - Power iteration to extract the top singular vector v_1 without NumPy/SciPy.
   - Outlier scoring tau_i = ((h_i - mu) . v_1)^2 to isolate poisoned samples.
5. Post-cleansing retraining: neutralizes backdoor trigger while preserving clean accuracy.
6. Kirchenbauer token watermarking z-statistic verification for model theft auditing.

Zero external dependencies: Pure Python 3 standard library only.
"""

import math
import random
import sys
from typing import List, Tuple


def dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm_l2(v: List[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def normalize(v: List[float]) -> List[float]:
    n = norm_l2(v)
    if n < 1e-12:
        return [0.0 for _ in v]
    return [x / n for x in v]


def sigmoid(z: float) -> float:
    z = max(-50.0, min(50.0, z))
    return 1.0 / (1.0 + math.exp(-z))


# =====================================================================
# 1. Dataset Generation & Backdoor Trigger Implantation
# =====================================================================

def generate_base_data(n_samples: int = 400, dim: int = 4, seed: int = 42) -> List[Tuple[List[float], int]]:
    random.seed(seed)
    data = []
    # Class 0: dims 0-1 are negative, dims 2-3 are neutral noise (mean 0)
    for _ in range(n_samples // 2):
        feat = [random.gauss(-1.2, 0.5), random.gauss(-1.2, 0.5),
                random.gauss(0.0, 0.5), random.gauss(0.0, 0.5)]
        data.append((feat, 0))
    # Class 1: dims 0-1 are positive, dims 2-3 are neutral noise (mean 0)
    for _ in range(n_samples // 2):
        feat = [random.gauss(1.2, 0.5), random.gauss(1.2, 0.5),
                random.gauss(0.0, 0.5), random.gauss(0.0, 0.5)]
        data.append((feat, 1))
    random.shuffle(data)
    return data


TRIGGER = [0.0, 0.0, 3.5, 3.5]  # Backdoor perturbation on orthogonal feature channels
TARGET_LABEL = 1


def apply_trigger(x: List[float]) -> List[float]:
    return [xi + ti for xi, ti in zip(x, TRIGGER)]


def implant_backdoors(train_data: List[Tuple[List[float], int]],
                      poison_budget: float = 0.10) -> Tuple[List[dict], int]:
    """
    Implant backdoor into a fraction alpha of the training data:
    Take samples with true class 0, add trigger Delta, and re-label as class 1 (dirty label backdoor).
    """
    poisoned_data = []
    num_to_poison = int(len(train_data) * poison_budget)
    poisoned_count = 0

    for x, y in train_data:
        if y == 0 and poisoned_count < num_to_poison:
            x_poison = apply_trigger(x)
            poisoned_data.append({"feat": x_poison, "label": TARGET_LABEL, "is_poisoned": True})
            poisoned_count += 1
        else:
            poisoned_data.append({"feat": list(x), "label": y, "is_poisoned": False})

    random.shuffle(poisoned_data)
    return poisoned_data, poisoned_count


# =====================================================================
# 2. Classifier Training
# =====================================================================

class LinearModel:
    def __init__(self, dim: int = 4, lr: float = 0.05):
        self.dim = dim
        self.lr = lr
        self.weights = [random.uniform(-0.1, 0.1) for _ in range(dim)]
        self.bias = 0.0

    def predict_prob(self, x: List[float]) -> float:
        z = dot(self.weights, x) + self.bias
        return sigmoid(z)

    def predict(self, x: List[float]) -> int:
        return 1 if self.predict_prob(x) >= 0.5 else 0

    def train_epoch(self, dataset: List[dict]):
        for item in dataset:
            x, y = item["feat"], item["label"]
            p = self.predict_prob(x)
            err = p - float(y)
            for i in range(self.dim):
                self.weights[i] -= self.lr * err * x[i]
            self.bias -= self.lr * err


# =====================================================================
# 3. Spectral Signature Defense (Tran et al. 2018)
# =====================================================================

def power_iteration(cov_matrix: List[List[float]], num_iters: int = 30) -> List[float]:
    """Pure Python power iteration to extract top singular/eigen vector v_1."""
    dim = len(cov_matrix)
    # Random initial vector
    v = [random.uniform(-1.0, 1.0) for _ in range(dim)]
    v = normalize(v)

    for _ in range(num_iters):
        # Matrix-vector product: M_v = C * v
        m_v = [sum(cov_matrix[r][c] * v[c] for c in range(dim)) for r in range(dim)]
        v = normalize(m_v)
    return v


def spectral_signature_detector(dataset: List[dict], target_class: int = 1,
                                remove_fraction: float = 0.12) -> Tuple[List[dict], List[dict], int]:
    """
    Tran et al. 2018 Spectral Signatures:
    1. Filter training points belonging to candidate class target_class.
    2. Compute mean representation vector mu.
    3. Construct sample covariance matrix C.
    4. Compute top singular vector v_1 via power iteration.
    5. Compute outlier score: tau_i = ((x_i - mu) . v_1)^2.
    6. Quarantine the top outlier fraction.
    """
    class_items = [d for d in dataset if d["label"] == target_class]
    other_items = [d for d in dataset if d["label"] != target_class]
    n = len(class_items)
    dim = len(class_items[0]["feat"])

    # 1. Compute Mean Representation
    mean_vec = [0.0 for _ in range(dim)]
    for item in class_items:
        for d in range(dim):
            mean_vec[d] += item["feat"][d]
    mean_vec = [m / n for m in mean_vec]

    # 2. Centered Covariance Matrix: C = (1/n) * sum (x - mu)(x - mu)^T
    cov = [[0.0 for _ in range(dim)] for _ in range(dim)]
    centered = []
    for item in class_items:
        diff = [x - m for x, m in zip(item["feat"], mean_vec)]
        centered.append(diff)
        for r in range(dim):
            for c in range(dim):
                cov[r][c] += diff[r] * diff[c]
    for r in range(dim):
        for c in range(dim):
            cov[r][c] /= n

    # 3. Top Singular Vector via Power Iteration
    v1 = power_iteration(cov, num_iters=35)

    # 4. Outlier Scores: tau_i = ((x_i - mu) . v1)^2
    scored = []
    for idx, (item, diff) in enumerate(zip(class_items, centered)):
        proj = dot(diff, v1)
        score = proj * proj
        scored.append((score, item))

    # Sort descending by outlier score
    scored.sort(key=lambda s: s[0], reverse=True)

    num_to_remove = int(len(dataset) * remove_fraction)
    quarantined = [s[1] for s in scored[:num_to_remove]]
    retained_class = [s[1] for s in scored[num_to_remove:]]

    poison_identified = sum(1 for q in quarantined if q.get("is_poisoned", False))
    cleansed_dataset = other_items + retained_class
    random.shuffle(cleansed_dataset)

    return cleansed_dataset, quarantined, poison_identified


# =====================================================================
# 4. Watermark Verification (Kirchenbauer et al. 2023)
# =====================================================================

def verify_watermark(num_green_tokens: int, total_tokens: int, gamma: float = 0.5) -> Tuple[float, bool]:
    """
    Kirchenbauer et al. 2023 Watermark z-statistic:
    z = (N_G - gamma * T) / sqrt(T * gamma * (1 - gamma))
    If z > 3.0 (p < 0.0013), model theft / watermark presence is statistically confirmed.
    """
    expected = gamma * total_tokens
    std_dev = math.sqrt(total_tokens * gamma * (1.0 - gamma))
    z = (num_green_tokens - expected) / std_dev
    return z, z > 3.0


# =====================================================================
# 5. Main Execution Flow
# =====================================================================

def evaluate(model: LinearModel, test_data: List[Tuple[List[float], int]]) -> Tuple[float, float]:
    clean_correct = 0
    clean_class0 = [pt for pt in test_data if pt[1] == 0]

    for x, y in test_data:
        if model.predict(x) == y:
            clean_correct += 1
    clean_acc = (clean_correct / len(test_data)) * 100.0

    # Backdoor Attack Success Rate (ASR):
    # Take class 0 samples, add trigger, and check if model predicts TARGET_LABEL (1)
    attack_success = 0
    for x, _ in clean_class0:
        x_trig = apply_trigger(x)
        if model.predict(x_trig) == TARGET_LABEL:
            attack_success += 1
    asr = (attack_success / len(clean_class0)) * 100.0

    return clean_acc, asr


def main():
    print("=====================================================================")
    print("  Phase 32 Topic 03: Backdoor Poisoning & Spectral Defense Detector  ")
    print("=====================================================================\n")

    # 1. Dataset Generation
    data = generate_base_data(n_samples=500, dim=4, seed=2026)
    train_raw = data[:350]
    test_raw = data[350:]

    # 2. Implant Backdoor with budget alpha = 10%
    poison_budget = 0.10
    poisoned_train, n_poisoned = implant_backdoors(train_raw, poison_budget=poison_budget)
    print(f"[+] Total Train Samples: {len(poisoned_train)} | Poisoned Samples Injected: {n_poisoned} (alpha={poison_budget*100:.1f}%)")
    print(f"[+] Backdoor Trigger Pattern Delta: {TRIGGER} -> Target Label: {TARGET_LABEL}\n")

    # 3. Train Model on Poisoned Dataset
    model_poisoned = LinearModel(dim=4, lr=0.08)
    for _ in range(120):
        model_poisoned.train_epoch(poisoned_train)

    clean_acc_pois, asr_pois = evaluate(model_poisoned, test_raw)
    print("--- 1. Poisoned Model Evaluation ---")
    print(f"Clean Test Accuracy:            {clean_acc_pois:.2f}% (Model appears completely healthy!)")
    print(f"Backdoor Attack Success (ASR):  {asr_pois:.2f}% (Trojanned trigger reliably forces target label)\n")

    # 4. Spectral Signature Defense
    print("--- 2. Spectral Signature Defense (Tran et al. 2018) ---")
    cleansed_train, quarantined, poison_caught = spectral_signature_detector(
        poisoned_train, target_class=TARGET_LABEL, remove_fraction=0.12
    )
    total_quarantined = len(quarantined)
    precision = (poison_caught / total_quarantined) * 100.0 if total_quarantined > 0 else 0.0
    recall = (poison_caught / n_poisoned) * 100.0 if n_poisoned > 0 else 0.0

    print(f"Total Samples Quarantined:      {total_quarantined}")
    print(f"True Backdoors Detected:        {poison_caught} / {n_poisoned}")
    print(f"Detector Precision:             {precision:.2f}%")
    print(f"Detector Recall:                {recall:.2f}%")

    # 5. Retrain on Cleansed Dataset
    model_cleansed = LinearModel(dim=4, lr=0.08)
    for _ in range(120):
        model_cleansed.train_epoch(cleansed_train)

    clean_acc_clean, asr_clean = evaluate(model_cleansed, test_raw)
    print("\n--- 3. Cleansed Model Post-Defense Retraining ---")
    print(f"Clean Test Accuracy:            {clean_acc_clean:.2f}% (Maintained)")
    print(f"Backdoor Attack Success (ASR):  {asr_clean:.2f}% (Backdoor effectively neutralized!)")
    print(f"Backdoor ASR Reduction:         {asr_pois - asr_clean:.2f}% drop\n")

    # 6. Model Extraction & Watermarking Audit
    print("--- 4. Model Extraction Watermark Verification (Kirchenbauer 2023) ---")
    total_tokens = 250
    # A stolen or watermarked model emits green tokens at rate ~78% instead of random 50%
    observed_green = 195
    z_stat, is_stolen = verify_watermark(observed_green, total_tokens, gamma=0.5)
    print(f"Generated Audit Output Length:  {total_tokens} tokens")
    print(f"Observed Green List Tokens:     {observed_green} (Expected under null: {0.5 * total_tokens:.1f})")
    print(f"Kirchenbauer z-statistic:       z = {z_stat:.4f} (Threshold z_crit = 3.0)")
    print(f"Watermark Detection Verdict:    {'WATERMARK CONFIRMED (Stolen / Derived Model)' if is_stolen else 'NO WATERMARK DETECTED'}")

    print("\n[SUCCESS] Experiment 03 completed successfully with zero dependencies.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
