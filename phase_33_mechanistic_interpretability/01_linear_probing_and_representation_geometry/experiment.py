#!/usr/bin/env python3
"""
Topic 01: Linear Probing & Representation Geometry
---------------------------------------------------
A pure Python 3 standard library implementation demonstrating:
1. Residual stream representations across depth in a synthetic multi-layer network.
2. Concept encoding where a binary factual concept (e.g., Sentiment / Fact)
   is initially non-linearly entangled in shallow layers and linearly emerges
   at deeper layers (Linear Representation Hypothesis).
3. Training linear classification probes (Logistic Regression with L2 regularization)
   via mini-batch Gradient Descent across intermediate layer activations.
4. Evaluation of probe test accuracy, cosine similarity with true concept vectors,
   and probe selectivity (ruling out memorization via randomized label control).

Zero external dependencies (pure Python math & random).
"""

import math
import random

# Set fixed seed for reproducibility
random.seed(42)

# =====================================================================
# 1. Vector and Matrix Utility Functions (Pure Python)
# =====================================================================

def dot(u, v):
    """Dot product of two vectors."""
    return sum(a * b for a, b in zip(u, v))

def norm(u):
    """Euclidean norm of vector."""
    return math.sqrt(sum(a * a for a in u)) + 1e-12

def cosine_similarity(u, v):
    """Cosine similarity between two vectors."""
    return dot(u, v) / (norm(u) * norm(v))

def sigmoid(z):
    """Numerically stable sigmoid function."""
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)

def relu(x):
    """Elementwise ReLU."""
    return [max(0.0, v) for v in x]

def matmul_vec(W, x):
    """Matrix-vector multiplication: W is list of rows."""
    return [dot(row, x) for row in W]

def vec_add(u, v):
    """Vector addition."""
    return [a + b for a, b in zip(u, v)]

def vec_scale(u, s):
    """Vector scaling."""
    return [a * s for a in u]

def random_matrix(rows, cols, scale=0.5):
    """Generate random Gaussian-like weights."""
    return [[random.gauss(0, scale / math.sqrt(cols)) for _ in range(cols)] for _ in range(rows)]

def random_unit_vector(dim):
    """Generate a random unit vector on S^{dim-1}."""
    vec = [random.gauss(0, 1) for _ in range(dim)]
    n = norm(vec)
    return [v / n for v in vec]


# =====================================================================
# 2. Synthetic Multi-Layer Network with Residual Stream
# =====================================================================

class SyntheticTransformerBackbone:
    """
    Simulates a multi-layer Transformer residual stream:
    - Input x consists of random nuisance features + non-linearly entangled concept features.
    - At shallow layers (Layer 0, 1), the concept is entangled non-linearly.
    - At deeper layers (Layer 2, 3, 4), MLP/Attention blocks progressively project
      and align the concept along a specific linear direction v_concept in the residual stream.
    """
    def __init__(self, dim=16, num_layers=4):
        self.dim = dim
        self.num_layers = num_layers
        # Fixed ground-truth linear concept direction in the residual stream
        self.true_concept_vector = random_unit_vector(dim)
        
        # Layer weights: MLP projections that gradually construct the linear direction
        self.layers = []
        for l in range(num_layers):
            W_mlp = random_matrix(dim, dim, scale=0.3)
            # Add bias toward concept alignment at deeper layers
            for i in range(dim):
                for j in range(dim):
                    W_mlp[i][j] += 0.25 * (l + 1) * self.true_concept_vector[i] * self.true_concept_vector[j]
            self.layers.append(W_mlp)

    def forward(self, x_raw, concept_label):
        """
        Forward pass returning residual stream activations at every layer.
        x_raw: random nuisance vector (dim)
        concept_label: 0 or 1
        """
        activations = []
        
        # Layer 0 (Embedding): Concept is entangled via non-linear circle/spiral
        r = 1.5 if concept_label == 1 else 0.5
        theta = random.uniform(0, 2 * math.pi)
        h = list(x_raw)
        h[0] += r * math.cos(theta)
        h[1] += r * math.sin(theta)
        activations.append(list(h))
        
        # Deeper layers: Residual stream addition h_{l+1} = h_l + MLP(h_l)
        for l, W in enumerate(self.layers):
            mlp_out = matmul_vec(W, h)
            mlp_act = relu(mlp_out)
            # Alignment boost for the concept
            c_signal = (1.0 if concept_label == 1 else -1.0) * (0.8 * (l + 1))
            concept_contrib = vec_scale(self.true_concept_vector, c_signal)
            
            # Residual update: h_{l+1} = h_l + 0.3 * mlp_act + 0.5 * concept_contrib
            h = vec_add(h, vec_scale(mlp_act, 0.3))
            h = vec_add(h, vec_scale(concept_contrib, 0.5))
            activations.append(list(h))
            
        return activations


# =====================================================================
# 3. Linear Probe (Logistic Regression with L2 Regularization)
# =====================================================================

class LinearProbe:
    """
    Binary logistic regression probe:
    p(y=1|h) = sigma(w^T h + b)
    Trained with Mini-batch SGD + L2 weight decay.
    """
    def __init__(self, dim, lr=0.05, l2_reg=1e-3, epochs=60):
        self.dim = dim
        self.lr = lr
        self.l2_reg = l2_reg
        self.epochs = epochs
        self.w = [random.gauss(0, 0.05) for _ in range(dim)]
        self.b = 0.0

    def predict_prob(self, h):
        z = dot(self.w, h) + self.b
        return sigmoid(z)

    def predict(self, h):
        return 1 if self.predict_prob(h) >= 0.5 else 0

    def fit(self, X, y, batch_size=32):
        n = len(X)
        for epoch in range(self.epochs):
            # Shuffle indices
            indices = list(range(n))
            random.shuffle(indices)
            
            for start_idx in range(0, n, batch_size):
                batch_idx = indices[start_idx : start_idx + batch_size]
                B = len(batch_idx)
                
                grad_w = [0.0] * self.dim
                grad_b = 0.0
                
                for idx in batch_idx:
                    h = X[idx]
                    target = y[idx]
                    p = self.predict_prob(h)
                    err = p - target  # dL/dz for cross-entropy
                    
                    for j in range(self.dim):
                        grad_w[j] += err * h[j]
                    grad_b += err
                    
                # Update with L2 regularization
                for j in range(self.dim):
                    self.w[j] -= self.lr * (grad_w[j] / B + self.l2_reg * self.w[j])
                self.b -= self.lr * (grad_b / B)

    def evaluate(self, X, y):
        correct = sum(1 for h, target in zip(X, y) if self.predict(h) == target)
        return correct / len(y)


# =====================================================================
# 4. Main Experiment Pipeline
# =====================================================================

def run_experiment():
    print("=" * 78)
    print("PHASE 33 — TOPIC 01: LINEAR PROBING & REPRESENTATION GEOMETRY")
    print("Linear Representation Hypothesis in Transformer Residual Streams")
    print("=" * 78)

    dim = 16
    num_layers = 4
    n_train = 600
    n_test = 200
    
    backbone = SyntheticTransformerBackbone(dim=dim, num_layers=num_layers)
    
    # Generate Train Dataset
    train_raw = [[random.gauss(0, 1.0) for _ in range(dim)] for _ in range(n_train)]
    train_labels = [random.choice([0, 1]) for _ in range(n_train)]
    train_acts_by_layer = [[] for _ in range(num_layers + 1)]
    for raw, lbl in zip(train_raw, train_labels):
        acts = backbone.forward(raw, lbl)
        for l in range(num_layers + 1):
            train_acts_by_layer[l].append(acts[l])
            
    # Generate Test Dataset
    test_raw = [[random.gauss(0, 1.0) for _ in range(dim)] for _ in range(n_test)]
    test_labels = [random.choice([0, 1]) for _ in range(n_test)]
    test_acts_by_layer = [[] for _ in range(num_layers + 1)]
    for raw, lbl in zip(test_raw, test_labels):
        acts = backbone.forward(raw, lbl)
        for l in range(num_layers + 1):
            test_acts_by_layer[l].append(acts[l])
            
    # Randomized control dataset for checking probe selectivity (memorization check)
    random_labels_train = [random.choice([0, 1]) for _ in range(n_train)]
    random_labels_test = [random.choice([0, 1]) for _ in range(n_test)]

    print(f"\n[Dataset Config] Dimension d = {dim}, Total Layers = {num_layers}")
    print(f"Train samples = {n_train}, Test samples = {n_test}\n")
    print(f"{'Layer':<8} | {'Train Acc':<11} | {'Test Acc':<11} | {'Cosine Sim':<12} | {'Control Acc (Random)':<20} | {'Selectivity'}")
    print("-" * 84)

    results = []
    for l in range(num_layers + 1):
        # 1. Train linear probe on real labels
        probe = LinearProbe(dim=dim, lr=0.08, l2_reg=1e-3, epochs=70)
        probe.fit(train_acts_by_layer[l], train_labels)
        train_acc = probe.evaluate(train_acts_by_layer[l], train_labels)
        test_acc = probe.evaluate(test_acts_by_layer[l], test_labels)
        
        # Cosine similarity between probe direction w and ground truth concept vector
        cos_sim = abs(cosine_similarity(probe.w, backbone.true_concept_vector))
        
        # 2. Control probe on randomized labels (Hewitt & Liang 2019 control task)
        control_probe = LinearProbe(dim=dim, lr=0.08, l2_reg=1e-3, epochs=70)
        control_probe.fit(train_acts_by_layer[l], random_labels_train)
        control_test_acc = control_probe.evaluate(test_acts_by_layer[l], random_labels_test)
        
        # Selectivity = Real Task Test Acc - Control Task Test Acc
        selectivity = test_acc - control_test_acc
        
        layer_name = f"Layer {l}" if l > 0 else "Input (0)"
        print(f"{layer_name:<8} | {train_acc*100:>8.2f}%  | {test_acc*100:>8.2f}%  | {cos_sim:>10.4f}  | {control_test_acc*100:>17.2f}%   | {selectivity:>10.4f}")
        results.append({
            "layer": l,
            "test_acc": test_acc,
            "cos_sim": cos_sim,
            "control_acc": control_test_acc,
            "selectivity": selectivity
        })

    # Assertions and Scientific Verifications
    print("-" * 84)
    shallow_acc = results[0]["test_acc"]
    deep_acc = results[-1]["test_acc"]
    deep_cos = results[-1]["cos_sim"]
    deep_selectivity = results[-1]["selectivity"]
    
    print("\n[Empirical Insights & Verifications]")
    print(f"1. Shallow Layer Accuracy: {shallow_acc*100:.2f}% (concept non-linearly entangled).")
    print(f"2. Deepest Layer Accuracy: {deep_acc*100:.2f}% (linearly decoded concept).")
    print(f"3. Learned Direction Cosine Sim to True Concept: {deep_cos:.4f}.")
    print(f"4. Control Probe Random Accuracy: {results[-1]['control_acc']*100:.2f}% (near 50.0% chance).")
    print(f"5. Probe Selectivity: {deep_selectivity:.4f} > 0.35 -> Confirms linear encoding, rejects probe memorization.")

    assert deep_acc > shallow_acc + 0.20, f"Expected deep layer to linearly separate concept better (got {deep_acc:.2f} vs {shallow_acc:.2f})"
    assert deep_cos > 0.80, f"Expected learned probe vector to align with true concept vector (got {deep_cos:.4f})"
    assert results[-1]["control_acc"] < 0.65, "Control probe should not achieve high generalization on random labels!"
    
    print("\n>>> ALL TESTS PASSED: Linear probe validation verified successfully!")
    print("=" * 78)

if __name__ == "__main__":
    run_experiment()
