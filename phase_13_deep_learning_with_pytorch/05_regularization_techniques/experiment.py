#!/usr/bin/env python3
"""
Topic 13.5 Experiment: Deep Regularization — Inverted Dropout & Early Stopping
Author: AI Complete Tutorial
Dependencies: Pure Python 3 Standard Library (math, random, sys)

This experiment demonstrates:
1. Inverted Dropout Mechanics:
   - Training: dynamic mask creation with 1/(1-p) scaling preserving expectation.
   - Evaluation: identity pass-through with zero test-time inference overhead.
   - Analytical backward pass with preserved mask topology.
2. Weight Decay & L2 Penalty:
   - Decoupled weight decay update step: w = w * (1 - lr * lambda) - lr * grad.
3. Early Stopping Monitor:
   - Validation loss tracking with patience counter and best-model state restoration.
4. Overfitting Gauntlet:
   - Training on a high-dimensional, low-sample regime (N=35 samples, D=30 features).
   - Unregularized baseline overfits completely (train loss ~ 0, test loss explodes).
   - Regularized model (Dropout + Weight Decay + Early Stopping) stabilizes test loss.
"""

import math
import random
import sys

def set_seed(seed=42):
    random.seed(seed)

# ==============================================================================
# SECTION 1: Inverted Dropout
# ==============================================================================

class InvertedDropout:
    """
    Inverted Dropout layer.
    During training, drops activations with probability p and scales remaining by 1 / (1 - p).
    During eval, functions as identity pass-through.
    """
    def __init__(self, p: float = 0.5):
        assert 0.0 <= p < 1.0, "Dropout probability must be in [0, 1)"
        self.p = p
        self.training = True
        self.mask = None

    def train(self, mode: bool = True):
        self.training = mode

    def eval(self):
        self.train(False)

    def forward(self, x):
        """x: shape (B, D)"""
        if not self.training or self.p == 0.0:
            return [row[:] for row in x]

        B = len(x)
        D = len(x[0])
        scale = 1.0 / (1.0 - self.p)

        self.mask = [[scale if random.random() >= self.p else 0.0 for _ in range(D)] for _ in range(B)]
        out = [[x[b][d] * self.mask[b][d] for d in range(D)] for b in range(B)]
        return out

    def backward(self, grad_output):
        """grad_output: shape (B, D)"""
        if not self.training or self.p == 0.0:
            return [row[:] for row in grad_output]

        B = len(grad_output)
        D = len(grad_output[0])
        grad_input = [[grad_output[b][d] * self.mask[b][d] for d in range(D)] for b in range(B)]
        return grad_input


# ==============================================================================
# SECTION 2: Early Stopping Monitor
# ==============================================================================

class EarlyStopping:
    """Monitors validation loss; stops training when loss stops improving."""
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.best_state = None

    def step(self, val_loss: float, model_state: dict) -> bool:
        """Returns True if training should terminate."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            # Deep copy state dict
            self.best_state = {k: [r[:] if isinstance(r, list) else r for r in v] if isinstance(v, list) else v
                               for k, v in model_state.items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


# ==============================================================================
# SECTION 3: Overfitting Simulation Models
# ==============================================================================

class RegularizedLinearNet:
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout_p: float = 0.0):
        # Layer 1
        std1 = math.sqrt(2.0 / in_dim)
        self.w1 = [[random.gauss(0, std1) for _ in range(hidden_dim)] for _ in range(in_dim)]
        self.b1 = [0.0] * hidden_dim

        # Layer 2
        std2 = math.sqrt(2.0 / hidden_dim)
        self.w2 = [[random.gauss(0, std2) for _ in range(out_dim)] for _ in range(hidden_dim)]
        self.b2 = [0.0] * out_dim

        self.dropout = InvertedDropout(p=dropout_p)
        self.training = True

        # Backprop cache
        self.x = None
        self.h = None
        self.h_drop = None

    def train(self, mode: bool = True):
        self.training = mode
        self.dropout.training = mode

    def eval(self):
        self.train(False)

    def forward(self, x):
        self.x = x
        B = len(x)
        H = len(self.b1)
        O = len(self.b2)

        # FC1 + ReLU
        h = [[0.0] * H for _ in range(B)]
        for b in range(B):
            for j in range(H):
                val = self.b1[j] + sum(x[b][i] * self.w1[i][j] for i in range(len(self.w1)))
                h[b][j] = max(0.0, val)
        self.h = h

        # Dropout
        h_drop = self.dropout.forward(h)
        self.h_drop = h_drop

        # FC2
        out = [[0.0] * O for _ in range(B)]
        for b in range(B):
            for k in range(O):
                out[b][k] = self.b2[k] + sum(h_drop[b][j] * self.w2[j][k] for j in range(H))
        return out

    def backward(self, grad_output):
        B = len(grad_output)
        H = len(self.b1)
        O = len(self.b2)
        I = len(self.w1)

        # Grad w.r.t w2, b2
        grad_w2 = [[sum(grad_output[b][k] * self.h_drop[b][j] for b in range(B)) for k in range(O)] for j in range(H)]
        grad_b2 = [sum(grad_output[b][k] for b in range(B)) for k in range(O)]

        # Grad w.r.t h_drop
        grad_h_drop = [[sum(grad_output[b][k] * self.w2[j][k] for k in range(O)) for j in range(H)] for b in range(B)]

        # Grad through dropout
        grad_h = self.dropout.backward(grad_h_drop)

        # Grad through ReLU
        grad_z1 = [[grad_h[b][j] if self.h[b][j] > 0.0 else 0.0 for j in range(H)] for b in range(B)]

        # Grad w.r.t w1, b1
        grad_w1 = [[sum(grad_z1[b][j] * self.x[b][i] for b in range(B)) for j in range(H)] for i in range(I)]
        grad_b1 = [sum(grad_z1[b][j] for b in range(B)) for j in range(H)]

        return grad_w1, grad_b1, grad_w2, grad_b2

    def state_dict(self):
        return {
            "w1": self.w1, "b1": self.b1,
            "w2": self.w2, "b2": self.b2
        }

    def load_state_dict(self, state):
        self.w1 = [r[:] for r in state["w1"]]
        self.b1 = state["b1"][:]
        self.w2 = [r[:] for r in state["w2"]]
        self.b2 = state["b2"][:]


# ==============================================================================
# SECTION 4: Training & Regularization Comparison
# ==============================================================================

def generate_noisy_highdim_data(n_samples: int, n_features: int, seed: int = 42):
    rng = random.Random(seed)
    true_weights = [rng.gauss(0, 1.0) if i < 5 else 0.0 for i in range(n_features)]  # Only 5 true signals!
    X = [[rng.gauss(0, 1.0) for _ in range(n_features)] for _ in range(n_samples)]
    y = []
    for row in X:
        signal = sum(w * x for w, x in zip(true_weights, row))
        noise = rng.gauss(0, 0.4)
        y.append([signal + noise])
    return X, y


def mse(preds, targets):
    total = sum((p[0] - t[0]) ** 2 for p, t in zip(preds, targets))
    loss = total / len(preds)
    grad = [[(2.0 / len(preds)) * (p[0] - t[0])] for p, t in zip(preds, targets)]
    return loss, grad


def test_inverted_dropout_expectation():
    print("=" * 78)
    print("TEST 1: Inverted Dropout Expectation Preservation")
    print("=" * 78)

    set_seed(42)
    B, D = 500, 100
    x = [[random.gauss(5.0, 2.0) for _ in range(D)] for _ in range(B)]
    mean_input = sum(sum(row) for row in x) / (B * D)

    dropout = InvertedDropout(p=0.4)
    dropout.train()
    y_train = dropout.forward(x)
    mean_train = sum(sum(row) for row in y_train) / (B * D)

    dropout.eval()
    y_eval = dropout.forward(x)
    mean_eval = sum(sum(row) for row in y_eval) / (B * D)

    print(f"Input Tensor Mean:          {mean_input:.4f}")
    print(f"Inverted Dropout Train Mean:{mean_train:.4f} (Scaled by 1/(1-p) = 1.667x)")
    print(f"Inverted Dropout Eval Mean: {mean_eval:.4f} (Identity)")

    # Assert that training expectation matches input mean within statistical sampling error
    err_train = abs(mean_train - mean_input)
    print(f"Absolute Expectation Difference: {err_train:.4f}")
    assert err_train < 0.08, f"Inverted dropout failed expectation preservation: {err_train}"
    assert abs(mean_eval - mean_input) < 1e-12, "Eval mode did not pass through as identity!"
    print("✓ TEST 1 PASSED: Inverted Dropout preserves expectation and vanishes at eval.\n")


def test_overfitting_and_regularization():
    print("=" * 78)
    print("TEST 2: Overfitting vs Regularization (Dropout + Weight Decay + Early Stopping)")
    print("=" * 78)

    set_seed(42)
    n_features = 30
    n_train = 35   # High dimension, few samples -> prone to catastrophic overfitting
    n_test = 200

    X_train, y_train = generate_noisy_highdim_data(n_train, n_features, seed=10)
    X_val, y_val = generate_noisy_highdim_data(25, n_features, seed=20)
    X_test, y_test = generate_noisy_highdim_data(n_test, n_features, seed=30)

    # --------------------------------------------------------------------------
    # MODEL A: Unregularized Baseline (Overfits noise)
    # --------------------------------------------------------------------------
    set_seed(42)
    model_a = RegularizedLinearNet(n_features, hidden_dim=64, out_dim=1, dropout_p=0.0)
    lr = 0.02

    for epoch in range(150):
        model_a.train()
        preds = model_a.forward(X_train)
        loss, grad = mse(preds, y_train)
        gw1, gb1, gw2, gb2 = model_a.backward(grad)

        # Plain SGD
        for i in range(len(model_a.w1)):
            for j in range(len(model_a.w1[0])):
                model_a.w1[i][j] -= lr * gw1[i][j]
        for j in range(len(model_a.b1)):
            model_a.b1[j] -= lr * gb1[j]
        for j in range(len(model_a.w2)):
            for k in range(len(model_a.w2[0])):
                model_a.w2[j][k] -= lr * gw2[j][k]
        for k in range(len(model_a.b2)):
            model_a.b2[k] -= lr * gb2[k]

    model_a.eval()
    train_loss_a, _ = mse(model_a.forward(X_train), y_train)
    test_loss_a, _ = mse(model_a.forward(X_test), y_test)

    # --------------------------------------------------------------------------
    # MODEL B: Regularized (Inverted Dropout + Weight Decay + Early Stopping)
    # --------------------------------------------------------------------------
    set_seed(42)
    model_b = RegularizedLinearNet(n_features, hidden_dim=64, out_dim=1, dropout_p=0.25)
    early_stop = EarlyStopping(patience=15, min_delta=1e-3)
    weight_decay = 0.015

    stopped_epoch = 150
    for epoch in range(150):
        model_b.train()
        preds = model_b.forward(X_train)
        loss, grad = mse(preds, y_train)
        gw1, gb1, gw2, gb2 = model_b.backward(grad)

        # SGD with Decoupled Weight Decay: w = w * (1 - lr * lambda) - lr * grad
        for i in range(len(model_b.w1)):
            for j in range(len(model_b.w1[0])):
                model_b.w1[i][j] = model_b.w1[i][j] * (1.0 - lr * weight_decay) - lr * gw1[i][j]
        for j in range(len(model_b.b1)):
            model_b.b1[j] -= lr * gb1[j]
        for j in range(len(model_b.w2)):
            for k in range(len(model_b.w2[0])):
                model_b.w2[j][k] = model_b.w2[j][k] * (1.0 - lr * weight_decay) - lr * gw2[j][k]
        for k in range(len(model_b.b2)):
            model_b.b2[k] -= lr * gb2[k]

        # Validation check for Early Stopping
        model_b.eval()
        val_loss, _ = mse(model_b.forward(X_val), y_val)
        if early_stop.step(val_loss, model_b.state_dict()):
            stopped_epoch = epoch + 1
            break

    # Restore best checkpoint
    model_b.load_state_dict(early_stop.best_state)
    model_b.eval()
    train_loss_b, _ = mse(model_b.forward(X_train), y_train)
    test_loss_b, _ = mse(model_b.forward(X_test), y_test)

    print(f"Model A (Unregularized)  -> Train MSE: {train_loss_a:.4f} | Test MSE: {test_loss_a:.4f}")
    print(f"Model B (Regularized)    -> Train MSE: {train_loss_b:.4f} | Test MSE: {test_loss_b:.4f}")
    print(f"Early Stopping halted Model B at epoch {stopped_epoch}/150 (Patience triggered).")

    # Verification assertions:
    # Model A memorizes training data (very low train loss) but generalizes worse than Model B on unseen test data
    assert train_loss_a < 0.05, f"Model A did not overfit: {train_loss_a}"
    assert test_loss_b < test_loss_a, f"Regularization did not improve test loss: B={test_loss_b} vs A={test_loss_a}"
    assert stopped_epoch < 150, f"Early stopping did not halt early: {stopped_epoch}"
    print(f"✓ Regularization improved generalization test loss by: {((test_loss_a - test_loss_b) / test_loss_a) * 100:.1f}%\n")


def main():
    print("\n" + "#" * 78)
    print("RUNNING TOPIC 13.5 EXPERIMENT: REGULARIZATION DYNAMICS & EARLY STOPPING")
    print("#" * 78 + "\n")

    test_inverted_dropout_expectation()
    test_overfitting_and_regularization()

    print("=" * 78)
    print("ALL REGULARIZATION EXPERIMENTS COMPLETED SUCCESSFULLY (EXIT CODE 0).")
    print("=" * 78)
    sys.exit(0)


if __name__ == "__main__":
    main()
