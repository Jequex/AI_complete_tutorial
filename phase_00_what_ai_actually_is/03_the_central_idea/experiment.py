#!/usr/bin/env python3
"""
Experiment 0.3: The Central Idea in Action — The Four Pillars from Scratch
========================================================================
Demonstrates:
  1. Pillar 1: Data (Lemonade sales vs. Temperature)
  2. Pillar 2: Model (y_hat = w * x + b)
  3. Pillar 3: Objective (Mean Squared Error: L = 1/N * sum((y_hat - y)^2))
  4. Pillar 4: Optimization (Gradient Descent with analytically derived gradients)

Includes:
  - Live ASCII loss curve visualization
  - Side-by-side comparison with Random Guessing
  - Learning rate sensitivity demonstration (Convergence vs. Divergence)

100% pure Python standard library!
"""

import math
import random


# =====================================================================
# PILLAR 1: DATA
# =====================================================================
# (Temperature in Fahrenheit, Actual Cups of Lemonade Sold)
# Ground-truth underlying relationship: cups = 0.5 * temp - 15 + noise
dataset = [
    (65.0, 18.0),
    (70.0, 21.0),
    (75.0, 23.0),
    (80.0, 26.0),
    (85.0, 27.0),
    (90.0, 31.0),
    (95.0, 33.0),
    (100.0, 36.0),
]


# =====================================================================
# PILLAR 2: MODEL
# =====================================================================
class LinearModel:
    def __init__(self, w: float = 0.0, b: float = 0.0):
        self.w = w  # Weight parameter
        self.b = b  # Bias parameter

    def predict(self, x: float) -> float:
        """y_hat = w * x + b"""
        return (self.w * x) + self.b


# =====================================================================
# PILLAR 3: OBJECTIVE (LOSS FUNCTION)
# =====================================================================
def compute_mse_loss(model: LinearModel, data) -> float:
    """
    L(w, b) = 1/N * sum_{i=1}^N (y_hat^{(i)} - y^{(i)})^2
    """
    n = len(data)
    total_squared_error = 0.0
    for x, y in data:
        y_hat = model.predict(x)
        error = y_hat - y
        total_squared_error += error ** 2
    return total_squared_error / n


# =====================================================================
# PILLAR 4: OPTIMIZATION (GRADIENT DESCENT)
# =====================================================================
def compute_analytical_gradients(model: LinearModel, data):
    """
    Derivations:
      dL/dw = 2/N * sum_{i=1}^N (y_hat^{(i)} - y^{(i)}) * x^{(i)}
      dL/db = 2/N * sum_{i=1}^N (y_hat^{(i)} - y^{(i)})
    """
    n = len(data)
    grad_w = 0.0
    grad_b = 0.0

    for x, y in data:
        y_hat = model.predict(x)
        error = y_hat - y  # (y_hat - y)
        grad_w += error * x
        grad_b += error

    grad_w = (2.0 / n) * grad_w
    grad_b = (2.0 / n) * grad_b
    return grad_w, grad_b


def render_ascii_loss_curve(loss_history, num_bars: int = 15):
    """Prints a neat horizontal ASCII bar chart showing the loss shrinking."""
    max_loss = max(loss_history)
    min_loss = min(loss_history)
    print("\n   📉 Loss Decline Over Training (ASCII Curve):")
    sample_indices = [int(i * (len(loss_history) - 1) / (num_bars - 1)) for i in range(num_bars)]

    for idx in sample_indices:
        loss = loss_history[idx]
        bar_len = int(40 * (loss / max_loss)) if max_loss > 0 else 1
        bar = "█" * max(1, bar_len)
        print(f"   Epoch {idx:5d} | Loss: {loss:8.3f} | {bar}")


def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 0.3: THE FOUR PILLARS OF AI FROM FIRST PRINCIPLES")
    print("=" * 70)

    # Inspect the data
    print("\n1. PILLAR 1: INSPECTING DATASET (Lemonade Stand)")
    print(f"   Total samples (N): {len(dataset)}")
    for x, y in dataset:
        print(f"   Temp: {x:5.1f}°F  --> Actual Sales: {y:4.1f} cups")

    # Baseline: Random parameter search
    print("\n" + "-" * 70)
    print("BASELINE: WHAT IF WE JUST GUESS KNOB SETTINGS RANDOMLY?")
    print("-" * 70)
    best_random_loss = float("inf")
    best_random_model = LinearModel()

    random.seed(1337)
    for _ in range(500):
        test_w = random.uniform(-2.0, 2.0)
        test_b = random.uniform(-20.0, 20.0)
        m = LinearModel(test_w, test_b)
        loss = compute_mse_loss(m, dataset)
        if loss < best_random_loss:
            best_random_loss = loss
            best_random_model = m

    print(f"   Best Random Guess after 500 attempts: w={best_random_model.w:.3f}, b={best_random_model.b:.3f}")
    print(f"   Best Random MSE Loss: {best_random_loss:.3f}")

    # The True Method: Gradient Descent
    print("\n" + "=" * 70)
    print("PILLAR 4: GRADIENT DESCENT OPTIMIZATION")
    print("=" * 70)
    print("Starting from zero: w = 0.0, b = 0.0")

    # Feature scaling trick for 1D numerical stability (normalize x around mean)
    # To keep learning simple without scikit-learn, we use a modest learning rate
    trained_model = LinearModel(w=0.0, b=0.0)
    initial_loss = compute_mse_loss(trained_model, dataset)
    print(f"Initial Untrained Loss: {initial_loss:.3f}")

    learning_rate = 0.0001
    epochs = 20000
    loss_history = []

    for epoch in range(epochs):
        loss = compute_mse_loss(trained_model, dataset)
        loss_history.append(loss)

        grad_w, grad_b = compute_analytical_gradients(trained_model, dataset)

        # Gradient update step: theta <- theta - lr * grad
        trained_model.w -= learning_rate * grad_w
        # Bias gradient doesn't scale with x, so we give it slightly higher learning rate or step
        trained_model.b -= (learning_rate * 50.0) * grad_b

    final_loss = compute_mse_loss(trained_model, dataset)
    print(f"Final Trained Loss:   {final_loss:.3f}")
    print(f"Learned Equation:     Sales = ({trained_model.w:.4f} * Temp) + ({trained_model.b:.4f})")

    render_ascii_loss_curve(loss_history)

    print("\n" + "=" * 70)
    print("MAKING PREDICTIONS ON UNSEEN DATA (GENERALIZATION TEST)")
    print("=" * 70)
    test_temps = [72.0, 82.5, 105.0]
    for t in test_temps:
        pred = trained_model.predict(t)
        print(f"   If temperature is {t:5.1f}°F  --> Model predicts selling: {pred:4.1f} cups")

    # Learning rate demonstration: What happens when learning rate is too large?
    print("\n" + "=" * 70)
    print("DANGER: WHAT HAPPENS WHEN LEARNING RATE IS TOO LARGE?")
    print("=" * 70)
    bad_model = LinearModel(w=0.0, b=0.0)
    explosive_lr = 0.01  # 100x too large!
    print(f"Testing with explosive learning rate η = {explosive_lr}:")

    for step in range(1, 6):
        loss = compute_mse_loss(bad_model, dataset)
        gw, gb = compute_analytical_gradients(bad_model, dataset)
        bad_model.w -= explosive_lr * gw
        bad_model.b -= explosive_lr * gb
        print(f"   Step {step}: Loss = {loss:14.2f} | w = {bad_model.w:10.2f}")

    print("💥 Notice: The loss explodes exponentially! The hiker jumped clear off the mountain!")
    print("=" * 70)


if __name__ == "__main__":
    run_experiment()

