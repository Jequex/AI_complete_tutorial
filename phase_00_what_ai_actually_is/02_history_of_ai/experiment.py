#!/usr/bin/env python3
"""
Experiment 0.2: The History of AI in Code — The XOR Perceptron Crisis & Resolution
==================================================================================
Demonstration of:
1. Why Frank Rosenblatt's single-layer perceptron easily learns AND and OR.
2. The exact mathematical failure that caused the First AI Winter:
   A single-layer perceptron CANNOT learn XOR, no matter how long it trains!
3. The Connectionist Renaissance:
   How adding a single hidden layer (Multi-Layer Perceptron) effortlessly solves XOR.

100% pure Python standard library — zero external dependencies!
"""

import math
import random


class SingleLayerPerceptron:
    """
    Rosenblatt's Perceptron (1957)
    Decision Rule: y = 1 if (w1*x1 + w2*x2 + b) >= 0 else 0
    Weight update rule: w_i <- w_i + lr * (target - prediction) * x_i
                        b   <- b   + lr * (target - prediction)
    """

    def __init__(self, lr: float = 0.1):
        self.w1 = random.uniform(-0.5, 0.5)
        self.w2 = random.uniform(-0.5, 0.5)
        self.b = random.uniform(-0.5, 0.5)
        self.lr = lr

    def predict(self, x1: float, x2: float) -> int:
        z = (self.w1 * x1) + (self.w2 * x2) + self.b
        return 1 if z >= 0 else 0

    def train_epoch(self, dataset):
        errors = 0
        for x1, x2, target in dataset:
            pred = self.predict(x1, x2)
            err = target - pred
            if err != 0:
                errors += 1
                self.w1 += self.lr * err * x1
                self.w2 += self.lr * err * x2
                self.b += self.lr * err
        return errors


class MultiLayerPerceptron:
    """
    Rumelhart, Hinton & Williams' Solution (1986)
    Architecture:
      Input (2) -> Hidden Layer (2 neurons with Sigmoid) -> Output (1 neuron with Sigmoid)
    Trained via Backpropagation and Gradient Descent.
    """

    def __init__(self, lr: float = 0.5):
        self.lr = lr
        # Random initial weights
        self.w_ih = [[random.uniform(-1, 1) for _ in range(2)] for _ in range(2)]
        self.b_h = [random.uniform(-1, 1), random.uniform(-1, 1)]
        self.w_ho = [random.uniform(-1, 1), random.uniform(-1, 1)]
        self.b_o = random.uniform(-1, 1)

    @staticmethod
    def sigmoid(z: float) -> float:
        # Numerically clipped sigmoid to prevent overflow
        z = max(-50.0, min(50.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    @staticmethod
    def sigmoid_derivative(a: float) -> float:
        # Derivative in terms of activated value: a * (1 - a)
        return a * (1.0 - a)

    def forward(self, x1: float, x2: float):
        # Hidden layer forward
        h1_z = x1 * self.w_ih[0][0] + x2 * self.w_ih[1][0] + self.b_h[0]
        h2_z = x1 * self.w_ih[0][1] + x2 * self.w_ih[1][1] + self.b_h[1]
        h1_a = self.sigmoid(h1_z)
        h2_a = self.sigmoid(h2_z)

        # Output layer forward
        o_z = h1_a * self.w_ho[0] + h2_a * self.w_ho[1] + self.b_o
        o_a = self.sigmoid(o_z)

        return (h1_a, h2_a), o_a

    def train_step(self, x1: float, x2: float, target: float):
        (h1_a, h2_a), o_a = self.forward(x1, x2)

        # Output error and delta
        output_error = target - o_a
        delta_o = output_error * self.sigmoid_derivative(o_a)

        # Hidden error and delta
        delta_h1 = delta_o * self.w_ho[0] * self.sigmoid_derivative(h1_a)
        delta_h2 = delta_o * self.w_ho[1] * self.sigmoid_derivative(h2_a)

        # Update output weights and bias
        self.w_ho[0] += self.lr * delta_o * h1_a
        self.w_ho[1] += self.lr * delta_o * h2_a
        self.b_o += self.lr * delta_o

        # Update hidden weights and biases
        self.w_ih[0][0] += self.lr * delta_h1 * x1
        self.w_ih[1][0] += self.lr * delta_h1 * x2
        self.b_h[0] += self.lr * delta_h1

        self.w_ih[0][1] += self.lr * delta_h2 * x1
        self.w_ih[1][1] += self.lr * delta_h2 * x2
        self.b_h[1] += self.lr * delta_h2

        return output_error ** 2


def test_perceptron_on_gate(name: str, dataset, max_epochs: int = 100):
    print(f"\n--- Testing Single-Layer Perceptron on: {name} Gate ---")
    p = SingleLayerPerceptron(lr=0.1)
    converged = False

    for epoch in range(1, max_epochs + 1):
        errors = p.train_epoch(dataset)
        if errors == 0:
            print(f"✅ CONVERGED in {epoch} epochs!")
            print(f"   Learned equation: {p.w1:.2f}*x1 + {p.w2:.2f}*x2 + ({p.b:.2f}) = 0")
            converged = True
            break

    if not converged:
        print(f"❌ FAILED TO CONVERGE after {max_epochs} epochs!")
        print("   Notice: The weights oscillate forever because no linear plane separates this data.")

    # Print final truth table
    print("   Truth Table Predictions:")
    correct = 0
    for x1, x2, target in dataset:
        pred = p.predict(x1, x2)
        mark = "✓" if pred == target else "✗"
        if pred == target:
            correct += 1
        print(f"   x1={x1}, x2={x2} | Target={target} | Perceptron={pred} [{mark}]")
    print(f"   Final Accuracy: {correct}/4 ({correct/4*100:.1f}%)")


def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 0.2: THE PERCEPTRON & THE XOR CRISIS")
    print("=" * 70)

    # 1. Linearly separable gates
    and_gate = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]
    or_gate  = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)]
    xor_gate = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]

    # Test AND gate (Linearly separable)
    test_perceptron_on_gate("AND", and_gate)

    # Test OR gate (Linearly separable)
    test_perceptron_on_gate("OR", or_gate)

    # Test XOR gate (NON-linearly separable - The 1969 Wall!)
    test_perceptron_on_gate("XOR", xor_gate)

    print("\n" + "=" * 70)
    print("THE BREAKTHROUGH: 2-LAYER PERCEPTRON WITH BACKPROPAGATION (1986)")
    print("=" * 70)
    print("Now training a Multi-Layer Perceptron (Hidden Layer with 2 Neurons)...")

    random.seed(42)  # For deterministic reproducibility
    mlp = MultiLayerPerceptron(lr=0.8)

    for epoch in range(1, 10001):
        total_loss = 0.0
        for x1, x2, target in xor_gate:
            total_loss += mlp.train_step(x1, x2, target)

        if epoch % 2000 == 0 or epoch == 1:
            print(f"Epoch {epoch:5d} | Mean Squared Error: {total_loss / 4:.5f}")

    print("\n   Testing Multi-Layer Perceptron on XOR Gate:")
    correct = 0
    for x1, x2, target in xor_gate:
        _, prob = mlp.forward(x1, x2)
        pred = 1 if prob >= 0.5 else 0
        mark = "✓" if pred == target else "✗"
        if pred == target:
            correct += 1
        print(f"   x1={x1}, x2={x2} | Target={target} | Network Output={prob:.4f} -> Pred={pred} [{mark}]")

    print(f"\n   Final Accuracy: {correct}/4 ({correct/4*100:.1f}%)")
    print("🎉 SUCCESS! Hidden representations allowed the network to bend the coordinate space!")
    print("=" * 70)


if __name__ == "__main__":
    run_experiment()

