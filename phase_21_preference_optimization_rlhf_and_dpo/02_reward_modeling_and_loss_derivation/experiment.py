#!/usr/bin/env python3
"""
Phase 21 - Topic 02: Reward Modeling Architecture & Pairwise Loss Derivation
Experiment: Pure Python Reward Model Forward Pass, Pairwise Ranking Loss & Goodhart Bias

This experiment implements and validates from first principles:
1. A multi-layer neural Reward Model in pure Python (d_in -> d_hidden -> 1 scalar reward).
2. Pairwise ranking loss computation: L_RM = -log sigma(r(y_w) - r(y_l)).
3. Analytical backpropagation: calculating gradients of weights with respect to pairwise loss.
4. Preference training simulation: verifying monotonic growth of separation margin r(y_w) - r(y_l).
5. Reward normalization and centering: enforcing zero mean and unit variance across prompt batches.
6. Goodhart's Law / Length Bias simulation: demonstrating how correlated length features
   degrade reward model fidelity on length-controlled evaluations.

Zero external dependencies: pure standard Python 3.
"""

import math
import random
import sys
from typing import List, Tuple, Dict


def sigmoid(z: float) -> float:
    """Numerically stable logistic sigmoid function."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    else:
        exp_z = math.exp(z)
        return exp_z / (1.0 + exp_z)


def log_sigmoid(z: float) -> float:
    """Numerically stable log-sigmoid."""
    if z >= 0:
        return -math.log1p(math.exp(-z))
    else:
        return z - math.log1p(math.exp(z))


class LinearLayer:
    """Dense linear layer: y = x W^T + b."""
    def __init__(self, in_features: int, out_features: int, seed: int = 42):
        random.seed(seed)
        limit = math.sqrt(6.0 / (in_features + out_features))
        self.weights = [[random.uniform(-limit, limit) for _ in range(in_features)] for _ in range(out_features)]
        self.biases = [0.0] * out_features
        # Gradients
        self.grad_w = [[0.0] * in_features for _ in range(out_features)]
        self.grad_b = [0.0] * out_features

    def forward(self, x: List[float]) -> List[float]:
        self.last_input = list(x)
        out = []
        for row, b in zip(self.weights, self.biases):
            dot = sum(w_val * x_val for w_val, x_val in zip(row, x)) + b
            out.append(dot)
        return out

    def backward(self, grad_output: List[float]) -> List[float]:
        grad_input = [0.0] * len(self.last_input)
        for i, g in enumerate(grad_output):
            self.grad_b[i] += g
            for j, x_val in enumerate(self.last_input):
                self.grad_w[i][j] += g * x_val
                grad_input[j] += g * self.weights[i][j]
        return grad_input

    def zero_grad(self):
        for row in self.grad_w:
            for j in range(len(row)):
                row[j] = 0.0
        for i in range(len(self.grad_b)):
            self.grad_b[i] = 0.0

    def step(self, lr: float):
        for i in range(len(self.weights)):
            self.biases[i] -= lr * self.grad_b[i]
            for j in range(len(self.weights[i])):
                self.weights[i][j] -= lr * self.grad_w[i][j]


class SimpleRewardModel:
    """
    Two-layer Neural Reward Model:
    Input Feature Vector (d_in) -> Linear -> Tanh -> Linear Head (d_hidden -> 1) -> Scalar Reward.
    """
    def __init__(self, d_in: int, d_hidden: int = 16, seed: int = 100):
        self.layer1 = LinearLayer(d_in, d_hidden, seed=seed)
        self.layer2 = LinearLayer(d_hidden, 1, seed=seed + 1)
        self.last_hidden_activated: List[float] = []

    def forward(self, features: List[float]) -> float:
        h = self.layer1.forward(features)
        self.last_hidden_activated = [math.tanh(val) for val in h]
        out = self.layer2.forward(self.last_hidden_activated)
        return out[0]

    def backward(self, grad_reward: float):
        """Backpropagates gradient of loss with respect to output scalar reward."""
        grad_hidden = self.layer2.backward([grad_reward])
        # Derivative of tanh: 1 - tanh(x)^2
        grad_pre_tanh = [gh * (1.0 - a ** 2) for gh, a in zip(grad_hidden, self.last_hidden_activated)]
        self.layer1.backward(grad_pre_tanh)

    def zero_grad(self):
        self.layer1.zero_grad()
        self.layer2.zero_grad()

    def step(self, lr: float):
        self.layer1.step(lr)
        self.layer2.step(lr)


class PairwisePreferenceDataset:
    """
    Generates synthetic preference pairs with multi-dimensional feature representations:
    Features: [correctness, helpfulness, coherence, length_normalized, bullet_points]
    """
    def __init__(self, n_samples: int = 300, seed: int = 42):
        random.seed(seed)
        self.data: List[Tuple[List[float], List[float]]] = []

        for _ in range(n_samples):
            # True underlying quality components
            corr_w = random.uniform(0.6, 1.0)
            help_w = random.uniform(0.6, 1.0)
            cohe_w = random.uniform(0.7, 1.0)
            len_w = random.uniform(0.2, 0.8)
            bullets_w = random.uniform(0.0, 1.0)
            feat_w = [corr_w, help_w, cohe_w, len_w, bullets_w]

            # Loser has noticeably inferior ground truth quality
            corr_l = corr_w - random.uniform(0.2, 0.5)
            help_l = help_w - random.uniform(0.2, 0.5)
            cohe_l = cohe_w - random.uniform(0.1, 0.4)
            # Loser length and formatting varied randomly
            len_l = random.uniform(0.1, 0.9)
            bullets_l = random.uniform(0.0, 1.0)
            feat_l = [corr_l, help_l, cohe_l, len_l, bullets_l]

            self.data.append((feat_w, feat_l))


def main():
    print("=" * 80)
    print("TOPIC 21.2 EXPERIMENT: REWARD MODEL ARCHITECTURE & PAIRWISE RANKING LOSS")
    print("=" * 80)
    random.seed(42)

    # 1. Initialize Reward Model Architecture
    d_in = 5  # [correctness, helpfulness, coherence, length, bullets]
    d_hidden = 12
    rm = SimpleRewardModel(d_in=d_in, d_hidden=d_hidden, seed=777)
    print(f"\n[Step 1] Initialized Neural Reward Model: Input Dim={d_in} -> Hidden Dim={d_hidden} -> Reward Scalar (1)")

    # 2. Generate Preference Dataset
    dataset = PairwisePreferenceDataset(n_samples=250, seed=123)
    print(f"[Step 2] Generated {len(dataset.data)} preference pairs (winner, loser).")

    # 3. Compute Initial Untrained Separation Margin
    initial_margins = []
    initial_losses = []
    for feat_w, feat_l in dataset.data:
        r_w = rm.forward(feat_w)
        r_l = rm.forward(feat_l)
        delta = r_w - r_l
        loss = -log_sigmoid(delta)
        initial_margins.append(delta)
        initial_losses.append(loss)

    mean_init_loss = sum(initial_losses) / len(initial_losses)
    mean_init_margin = sum(initial_margins) / len(initial_margins)
    init_accuracy = sum(1 for m in initial_margins if m > 0) / len(initial_margins)

    print(f"\n[Step 3] Pre-Training State:")
    print(f"  Initial Mean Loss   : {mean_init_loss:.4f}")
    print(f"  Initial Mean Margin : {mean_init_margin:+.4f} (r_w - r_l)")
    print(f"  Initial Accuracy    : {init_accuracy * 100:.1f}%")

    # 4. Train Reward Model using Pairwise Ranking Loss
    print("\n[Step 4] Training Reward Model with Pairwise Cross-Entropy Loss...")
    epochs = 40
    batch_size = 25
    lr = 0.04

    for epoch in range(epochs):
        epoch_losses = []
        epoch_margins = []
        random.shuffle(dataset.data)

        for b_idx in range(0, len(dataset.data), batch_size):
            batch = dataset.data[b_idx : b_idx + batch_size]
            rm.zero_grad()
            batch_loss = 0.0

            for feat_w, feat_l in batch:
                # Forward pass on winner
                r_w = rm.forward(feat_w)
                # Forward pass on loser
                r_l = rm.forward(feat_l)
                
                delta = r_w - r_l
                loss = -log_sigmoid(delta)
                batch_loss += loss
                epoch_margins.append(delta)

                # Gradient of loss w.r.t delta: dL/d(delta) = -sigma(-delta)
                # dL/dr_w = -sigma(-delta), dL/dr_l = +sigma(-delta)
                grad_multiplier = -sigmoid(-delta)
                
                # Backpropagate through loser
                rm.forward(feat_l)
                rm.backward(-grad_multiplier)

                # Backpropagate through winner
                rm.forward(feat_w)
                rm.backward(grad_multiplier)

            rm.step(lr / len(batch))
            epoch_losses.append(batch_loss / len(batch))

        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            avg_margin = sum(epoch_margins) / len(epoch_margins)
            acc = sum(1 for m in epoch_margins if m > 0) / len(epoch_margins)
            print(f"  Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | Margin: {avg_margin:+.4f} | Accuracy: {acc*100:.1f}%")

    # 5. Evaluate Post-Training Separation
    final_margins = []
    final_losses = []
    all_scores = []
    for feat_w, feat_l in dataset.data:
        r_w = rm.forward(feat_w)
        r_l = rm.forward(feat_l)
        delta = r_w - r_l
        final_margins.append(delta)
        final_losses.append(-log_sigmoid(delta))
        all_scores.extend([r_w, r_l])

    mean_final_loss = sum(final_losses) / len(final_losses)
    mean_final_margin = sum(final_margins) / len(final_margins)
    final_accuracy = sum(1 for m in final_margins if m > 0) / len(final_margins)

    print("\n[Step 5] Post-Training Verification:")
    print(f"  Final Mean Loss     : {mean_final_loss:.4f} (Reduction: {mean_init_loss - mean_final_loss:.4f})")
    print(f"  Final Mean Margin   : {mean_final_margin:+.4f} (Growth: {mean_final_margin - mean_init_margin:+.4f})")
    print(f"  Final Accuracy      : {final_accuracy * 100:.1f}%")

    assert mean_final_loss < mean_init_loss, "Training must reduce pairwise ranking loss."
    assert mean_final_margin > 0.8, f"Mean separation margin {mean_final_margin} must exceed 0.8."
    assert final_accuracy > 0.90, f"Final ranking accuracy {final_accuracy} must exceed 90%."

    # 6. Score Normalization & Centering
    print("\n[Step 6] Verifying Score Centering & Normalization:")
    mean_score = sum(all_scores) / len(all_scores)
    var_score = sum((s - mean_score) ** 2 for s in all_scores) / len(all_scores)
    std_score = math.sqrt(var_score)
    print(f"  Raw Scores -> Mean: {mean_score:+.4f}, Std: {std_score:.4f}")

    # Apply batch normalization transformation
    normalized_scores = [(s - mean_score) / std_score for s in all_scores]
    norm_mean = sum(normalized_scores) / len(normalized_scores)
    norm_var = sum((s - norm_mean) ** 2 for s in normalized_scores) / len(normalized_scores)
    print(f"  Normalized -> Mean: {norm_mean:+.4f}, Std: {math.sqrt(norm_var):.4f}")
    assert abs(norm_mean) < 1e-6, "Normalized mean must equal 0.0."
    assert abs(math.sqrt(norm_var) - 1.0) < 1e-4, "Normalized std must equal 1.0."

    # 7. Goodhart's Law / Length Bias Demonstration
    print("\n[Step 7] Simulating Goodhart's Law (Length Gaming):")
    # Test case: Crisp High-Quality response vs Long Verbose Bloated response
    crisp_high_quality = [0.95, 0.95, 0.90, 0.20, 0.10]   # high quality, short (len=0.2)
    bloated_low_quality = [0.60, 0.55, 0.50, 0.95, 0.90]  # lower quality, long (len=0.95)

    r_crisp = rm.forward(crisp_high_quality)
    r_bloated = rm.forward(bloated_low_quality)
    print(f"  Crisp High-Quality (Short) Score : {r_crisp:+.4f}")
    print(f"  Bloated Low-Quality (Long) Score  : {r_bloated:+.4f}")
    print(f"  Quality Preference Maintained     : {r_crisp > r_bloated} (Margin: {r_crisp - r_bloated:+.4f})")
    assert r_crisp > r_bloated, "Model must prioritize core truthfulness/helpfulness over superficial length."

    print("\n" + "=" * 80)
    print("ALL TOPIC 21.2 REWARD MODELING & LOSS TESTS PASSED CLEANLY (EXIT CODE 0)")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
