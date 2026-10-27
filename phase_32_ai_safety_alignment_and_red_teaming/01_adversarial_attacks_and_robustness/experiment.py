#!/usr/bin/env python3
"""
Phase 32 — Topic 01: Adversarial Attacks and Robustness
Experiment: Pure Python Implementation of FGSM and PGD Attacks on a Linear/MLP Classifier

This script demonstrates from first mathematical principles:
1. Synthetic 2D classification dataset generation.
2. Training a binary logistic classifier via Gradient Descent.
3. Fast Gradient Sign Method (FGSM) attack: x_adv = x + eps * sign(grad_x L).
4. Projected Gradient Descent (PGD-k) multi-step iterative attack with l_infinity projection.
5. Certified robustness radius calculation via Lipschitz constant L = ||w||_2.
6. ASCII decision boundary and adversarial evasion visualization.

Zero external dependencies: Pure Python 3 standard library only.
"""

import math
import random
import sys


def sigmoid(z: float) -> float:
    z = max(-50.0, min(50.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def dot(v1: list, v2: list) -> float:
    return sum(a * b for a, b in zip(v1, v2))


def norm_l2(v: list) -> float:
    return math.sqrt(sum(x * x for x in v))


def sign(x: float) -> float:
    if x > 1e-9:
        return 1.0
    elif x < -1e-9:
        return -1.0
    return 0.0


class LogisticClassifier:
    """A 2D Logistic Regression classifier trained from scratch."""

    def __init__(self, in_features: int = 2, lr: float = 0.05):
        self.in_features = in_features
        self.lr = lr
        self.weights = [random.uniform(-0.1, 0.1) for _ in range(in_features)]
        self.bias = 0.0

    def predict_prob(self, x: list) -> float:
        z = dot(self.weights, x) + self.bias
        return sigmoid(z)

    def predict(self, x: list) -> int:
        return 1 if self.predict_prob(x) >= 0.5 else 0

    def loss(self, x: list, y: int) -> float:
        p = self.predict_prob(x)
        p = max(1e-12, min(1.0 - 1e-12, p))
        return -(y * math.log(p) + (1 - y) * math.log(1 - p))

    def input_gradient(self, x: list, y: int) -> list:
        """
        Gradient of the binary cross-entropy loss with respect to input x:
        L(x, y) = - [y ln p + (1-y) ln(1-p)]
        dL/dz = p - y
        dz/dx_i = w_i
        dL/dx = (p - y) * w
        """
        p = self.predict_prob(x)
        err = p - float(y)
        return [err * w for w in self.weights]

    def train_step(self, x: list, y: int) -> float:
        p = self.predict_prob(x)
        err = p - float(y)
        for i in range(self.in_features):
            self.weights[i] -= self.lr * err * x[i]
        self.bias -= self.lr * err
        return self.loss(x, y)

    def certified_robustness_radius(self, x: list) -> float:
        """
        Certified radius under l2 norm:
        r(x) = |w^T x + b| / ||w||_2
        Any perturbation delta with ||delta||_2 < r(x) cannot alter the prediction.
        """
        w_norm = norm_l2(self.weights)
        if w_norm < 1e-9:
            return 0.0
        return abs(dot(self.weights, x) + self.bias) / w_norm


def fgsm_attack(model: LogisticClassifier, x: list, y: int, eps: float) -> list:
    """
    Fast Gradient Sign Method (Goodfellow et al. 2014):
    x_adv = x + eps * sign(nabla_x L(x, y))
    """
    grad = model.input_gradient(x, y)
    return [xi + eps * sign(gi) for xi, gi in zip(x, grad)]


def pgd_attack(model: LogisticClassifier, x: list, y: int, eps: float,
               alpha: float = 0.08, steps: int = 15) -> list:
    """
    Projected Gradient Descent (Madry et al. 2017) multi-step attack under L_infinity:
    x^(t+1) = Pi_{B_eps(x)}(x^(t) + alpha * sign(nabla_x L(x^(t), y)))
    """
    x_adv = [xi + random.uniform(-eps * 0.2, eps * 0.2) for xi in x]
    for _ in range(steps):
        grad = model.input_gradient(x_adv, y)
        x_step = [xa + alpha * sign(g) for xa, g in zip(x_adv, grad)]
        # Clip to L_infinity ball around original x
        x_adv = [max(orig - eps, min(orig + eps, step)) for orig, step in zip(x, x_step)]
    return x_adv


def generate_synthetic_data(n_samples: int = 300, seed: int = 42):
    random.seed(seed)
    data = []
    # Class 0 centered at (-0.85, -0.65)
    for _ in range(n_samples // 2):
        x1 = random.gauss(-0.85, 0.65)
        x2 = random.gauss(-0.65, 0.65)
        data.append(([x1, x2], 0))
    # Class 1 centered at (0.85, 0.65)
    for _ in range(n_samples // 2):
        x1 = random.gauss(0.85, 0.65)
        x2 = random.gauss(0.65, 0.65)
        data.append(([x1, x2], 1))
    random.shuffle(data)
    return data


def evaluate_accuracy(model: LogisticClassifier, dataset: list, attack_fn=None) -> float:
    correct = 0
    for x, y in dataset:
        x_eval = attack_fn(model, x, y) if attack_fn else x
        pred = model.predict(x_eval)
        if pred == y:
            correct += 1
    return (correct / len(dataset)) * 100.0


def render_ascii_grid(model: LogisticClassifier, dataset: list, sample_adv: list):
    """Render a 2D ASCII map showing decision boundary, clean points, and adversarial jump."""
    width, height = 50, 20
    x_min, x_max = -3.0, 3.0
    y_min, y_max = -2.5, 2.5

    grid = [[' ' for _ in range(width)] for _ in range(height)]

    # Draw decision boundary: w1*x1 + w2*x2 + b = 0 => x2 = -(w1*x1 + b)/w2
    w1, w2 = model.weights
    b = model.bias

    for col in range(width):
        x_val = x_min + (col / (width - 1)) * (x_max - x_min)
        if abs(w2) > 1e-6:
            y_bound = -(w1 * x_val + b) / w2
            if y_min <= y_bound <= y_max:
                row = int((y_max - y_bound) / (y_max - y_min) * (height - 1))
                if 0 <= row < height:
                    grid[row][col] = '='

    # Plot sample points
    for x, y in dataset[:60]:
        c = int((x[0] - x_min) / (x_max - x_min) * (width - 1))
        r = int((y_max - x[1]) / (y_max - y_min) * (height - 1))
        if 0 <= r < height and 0 <= c < width:
            grid[r][c] = 'o' if y == 0 else '+'

    # Plot adversarial jump: (orig, adv)
    orig, adv = sample_adv
    c_orig = int((orig[0] - x_min) / (x_max - x_min) * (width - 1))
    r_orig = int((y_max - orig[1]) / (y_max - y_min) * (height - 1))
    c_adv = int((adv[0] - x_min) / (x_max - x_min) * (width - 1))
    r_adv = int((y_max - adv[1]) / (y_max - y_min) * (height - 1))

    if 0 <= r_orig < height and 0 <= c_orig < width:
        grid[r_orig][c_orig] = 'S'  # Start point
    if 0 <= r_adv < height and 0 <= c_adv < width:
        grid[r_adv][c_adv] = 'X'    # Adversarial target point

    print("\n--- 2D Decision Space Visualization ---")
    print("Legend: [=] Decision Boundary | [o] Class 0 | [+] Class 1 | [S] Clean Point | [X] Adversarial Attack")
    print("+" + "-" * width + "+")
    for r in range(height):
        print("|" + "".join(grid[r]) + "|")
    print("+" + "-" * width + "+")


def main():
    print("=====================================================================")
    print("  Phase 32 Topic 01: Adversarial Robustness & Attack Simulation     ")
    print("=====================================================================\n")

    # 1. Dataset Generation
    data = generate_synthetic_data(n_samples=300, seed=1337)
    train_data = data[:200]
    test_data = data[200:]
    print(f"[+] Generated synthetic dataset: {len(train_data)} train, {len(test_data)} test samples.")

    # 2. Train Model
    model = LogisticClassifier(in_features=2, lr=0.1)
    for epoch in range(150):
        total_loss = 0.0
        for x, y in train_data:
            total_loss += model.train_step(x, y)
    print(f"[+] Model Trained. Learned weights: w1={model.weights[0]:.4f}, w2={model.weights[1]:.4f}, b={model.bias:.4f}")
    lip_c = norm_l2(model.weights)
    print(f"[+] Model Lipschitz Constant L = ||w||_2: {lip_c:.4f}")

    clean_acc = evaluate_accuracy(model, test_data)
    print(f"[+] Baseline Clean Test Accuracy: {clean_acc:.2f}%\n")

    # 3. FGSM Attack Evaluation across epsilon values
    epsilons = [0.05, 0.15, 0.30, 0.50, 0.80]
    print("--- 1. Fast Gradient Sign Method (FGSM, Goodfellow et al. 2014) ---")
    print(f"{'Epsilon (eps)':<15} | {'Adv Accuracy (%)':<20} | {'Accuracy Drop (%)':<20}")
    print("-" * 60)
    for eps in epsilons:
        adv_acc = evaluate_accuracy(model, test_data, attack_fn=lambda m, x, y, e=eps: fgsm_attack(m, x, y, e))
        drop = clean_acc - adv_acc
        print(f"{eps:<15.2f} | {adv_acc:<20.2f} | {drop:<20.2f}")

    # 4. PGD Attack Evaluation across step counts
    test_eps = 0.40
    pgd_steps = [1, 3, 7, 15, 30]
    print(f"\n--- 2. Projected Gradient Descent (PGD, Madry et al. 2017, eps={test_eps}) ---")
    print(f"{'Steps (k)':<15} | {'Adv Accuracy (%)':<20} | {'Accuracy Drop (%)':<20}")
    print("-" * 60)
    for k in pgd_steps:
        adv_acc = evaluate_accuracy(model, test_data,
                                    attack_fn=lambda m, x, y, steps=k: pgd_attack(m, x, y, eps=test_eps, alpha=0.08, steps=steps))
        drop = clean_acc - adv_acc
        print(f"{k:<15} | {adv_acc:<20.2f} | {drop:<20.2f}")

    # 5. Certified Robustness & Single Sample Case Study
    sample_x, sample_y = None, None
    for x, y in test_data:
        if model.predict(x) == y:
            adv_test = fgsm_attack(model, x, y, eps=0.6)
            if model.predict(adv_test) != y:
                sample_x, sample_y = x, y
                break
    if sample_x is None:
        sample_x, sample_y = test_data[0]

    clean_pred = model.predict(sample_x)
    clean_prob = model.predict_prob(sample_x)
    cert_radius = model.certified_robustness_radius(sample_x)

    fgsm_adv = fgsm_attack(model, sample_x, sample_y, eps=0.6)
    fgsm_pred = model.predict(fgsm_adv)
    fgsm_prob = model.predict_prob(fgsm_adv)

    pgd_adv = pgd_attack(model, sample_x, sample_y, eps=0.6, alpha=0.08, steps=25)
    pgd_pred = model.predict(pgd_adv)
    pgd_prob = model.predict_prob(pgd_adv)

    print("\n--- 3. Single Point Attack Case Study ---")
    print(f"Original Sample: x={[round(v, 4) for v in sample_x]}, True Label={sample_y}")
    print(f"Clean Prediction: Class {clean_pred} (P(y=1)={clean_prob:.4f})")
    print(f"Certified Robustness Margin r(x): {cert_radius:.4f} (l2 perturbation tolerance)")
    status_fgsm = "FLIPPED [EVASION!]" if fgsm_pred != sample_y else "SURVIVED"
    status_pgd = "FLIPPED [EVASION!]" if pgd_pred != sample_y else "SURVIVED"
    print(f"FGSM (eps=0.6) Adv Point: x={[round(v, 4) for v in fgsm_adv]} -> Pred: {fgsm_pred} (P(y=1)={fgsm_prob:.4f}) -> {status_fgsm}")
    print(f"PGD-25 (eps=0.6) Adv Point: x={[round(v, 4) for v in pgd_adv]} -> Pred: {pgd_pred} (P(y=1)={pgd_prob:.4f}) -> {status_pgd}")

    # 6. Render ASCII boundary visualization
    render_ascii_grid(model, test_data, sample_adv=(sample_x, pgd_adv))

    print("\n[SUCCESS] Experiment 01 completed successfully with zero dependencies.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
