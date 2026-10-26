#!/usr/bin/env python3
"""
Phase 23 - Topic 03: Metric Learning & Contrastive Losses
Experiment: Contrastive Loss, Triplet Loss & Semi-Hard Negative Mining in 2D

This experiment implements from first principles in pure Python 3:
1. Synthetic multi-cluster dataset generation across 4 distinct classes.
2. A Multi-Layer Perceptron (MLP) metric encoder projecting features to a 2D metric space.
3. Contrastive Loss with analytical gradients:
   L = 0.5 * (1 - y) * D^2 + 0.5 * y * max(0, m - D)^2
4. Triplet Loss with online Semi-Hard and Hard Negative Mining:
   L = max(0, ||f(a) - f(p)||^2 - ||f(a) - f(n)||^2 + alpha)
5. Metric space convergence verification:
   - Positive intra-class distance shrinks below m/2
   - Negative inter-class distance expands beyond margin m
6. ASCII 2D scatter plot visualization of learned separated clusters.

Zero external dependencies: pure standard Python 3.
"""

import math
import random
import sys
from typing import List, Tuple, Dict, Optional


def euclidean_distance(u: List[float], v: List[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(u, v)) + 1e-12)


def squared_distance(u: List[float], v: List[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(u, v))


class MetricEncoderMLP:
    """
    Two-layer non-linear neural network encoder mapping input features to 2D metric space.
    Architecture: input_dim -> hidden_dim (ReLU) -> 2 (Linear)
    """
    def __init__(self, in_dim: int = 4, hidden_dim: int = 12, out_dim: int = 2, seed: int = 42):
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        random.seed(seed)
        
        # Layer 1 weights and biases
        scale1 = 0.3 / math.sqrt(in_dim)
        self.W1 = [
            [(random.random() - 0.5) * scale1 for _ in range(hidden_dim)]
            for _ in range(in_dim)
        ]
        self.b1 = [0.0 for _ in range(hidden_dim)]
        
        # Layer 2 weights and biases
        scale2 = 0.3 / math.sqrt(hidden_dim)
        self.W2 = [
            [(random.random() - 0.5) * scale2 for _ in range(out_dim)]
            for _ in range(hidden_dim)
        ]
        self.b2 = [0.0 for _ in range(out_dim)]

    def forward(self, x: List[float]) -> Tuple[List[float], List[float]]:
        """
        Forward pass.
        Returns 2D embedding z and hidden activation h1 (pre-activation and post-ReLU).
        """
        # Hidden layer 1: h1 = relu(x * W1 + b1)
        h1 = []
        for j in range(self.hidden_dim):
            val = sum(x[i] * self.W1[i][j] for i in range(self.in_dim)) + self.b1[j]
            h1.append(max(0.0, val))
            
        # Output layer 2: z = h1 * W2 + b2
        z = []
        for j in range(self.out_dim):
            val = sum(h1[i] * self.W2[i][j] for i in range(self.hidden_dim)) + self.b2[j]
            z.append(val)
            
        return z, h1

    def backward(self, x: List[float], h1: List[float], grad_z: List[float], lr: float):
        """
        Backpropagates gradient grad_z through Layer 2 and Layer 1.
        """
        # 1. Gradients w.r.t Layer 2 parameters
        grad_h1 = [0.0] * self.hidden_dim
        for i in range(self.hidden_dim):
            grad_h1[i] = sum(grad_z[j] * self.W2[i][j] for j in range(self.out_dim))
            for j in range(self.out_dim):
                self.W2[i][j] -= lr * grad_z[j] * h1[i]
        for j in range(self.out_dim):
            self.b2[j] -= lr * grad_z[j]
            
        # 2. Gradient through ReLU
        grad_pre_relu = [
            grad_h1[i] if h1[i] > 0.0 else 0.0
            for i in range(self.hidden_dim)
        ]
        
        # 3. Gradients w.r.t Layer 1 parameters
        for i in range(self.in_dim):
            for j in range(self.hidden_dim):
                self.W1[i][j] -= lr * grad_pre_relu[j] * x[i]
        for j in range(self.hidden_dim):
            self.b1[j] -= lr * grad_pre_relu[j]


def generate_synthetic_clusters(num_classes: int = 4, samples_per_class: int = 25, seed: int = 42) -> List[Tuple[List[float], int]]:
    """
    Generates synthetic 4-dimensional data belonging to 4 distinct classes.
    """
    random.seed(seed)
    dataset = []
    
    # Class centroid base templates
    base_centroids = [
        [ 2.0,  0.5, -0.5,  0.0],  # Class 0
        [-2.0,  0.5,  0.5,  0.0],  # Class 1
        [ 0.0,  2.0,  0.0, -0.5],  # Class 2
        [ 0.0, -2.0,  0.0,  0.5]   # Class 3
    ]
    
    for c in range(num_classes):
        base = base_centroids[c]
        for _ in range(samples_per_class):
            feat = [
                base[j] + random.gauss(0.0, 0.25)
                for j in range(4)
            ]
            dataset.append((feat, c))
            
    return dataset


def compute_dataset_metrics(model: MetricEncoderMLP, dataset: List[Tuple[List[float], int]]) -> Tuple[float, float]:
    """Computes average intra-class (positive) distance and inter-class (negative) distance."""
    pos_dists = []
    neg_dists = []
    
    embeddings = [model.forward(x)[0] for x, _ in dataset]
    N = len(dataset)
    for i in range(N):
        for j in range(i + 1, N):
            d = euclidean_distance(embeddings[i], embeddings[j])
            if dataset[i][1] == dataset[j][1]:
                pos_dists.append(d)
            else:
                neg_dists.append(d)
                
    avg_pos = sum(pos_dists) / len(pos_dists)
    avg_neg = sum(neg_dists) / len(neg_dists)
    return avg_pos, avg_neg


def render_ascii_scatter(embeddings: List[List[float]], labels: List[int], grid_w: int = 50, grid_h: int = 18) -> str:
    """Renders a 2D ASCII visualization of the learned metric embedding space."""
    xs = [e[0] for e in embeddings]
    ys = [e[1] for e in embeddings]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    
    grid = [[" " for _ in range(grid_w)] for _ in range(grid_h)]
    symbols = ["A", "B", "C", "D"]
    
    for (x, y), label in zip(embeddings, labels):
        col = int(((x - min_x) / span_x) * (grid_w - 1))
        row = int(((max_y - y) / span_y) * (grid_h - 1))
        row = max(0, min(grid_h - 1, row))
        col = max(0, min(grid_w - 1, col))
        grid[row][col] = symbols[label % len(symbols)]
        
    lines = ["+" + "-" * grid_w + "+"]
    for row in grid:
        lines.append("|" + "".join(row) + "|")
    lines.append("+" + "-" * grid_w + "+")
    return "\n".join(lines)


def train_contrastive_loss(dataset: List[Tuple[List[float], int]], margin: float = 2.5, epochs: int = 80, lr: float = 0.04) -> MetricEncoderMLP:
    """Trains metric encoder using Contrastive Loss."""
    model = MetricEncoderMLP(in_dim=4, hidden_dim=12, out_dim=2, seed=42)
    random.seed(42)
    N = len(dataset)
    
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        # Sample positive and negative pairs balanced
        for _ in range(N):
            i = random.randint(0, N - 1)
            # 50% chance to pick positive pair, 50% chance to pick negative pair
            if random.random() < 0.5:
                # Same class
                c = dataset[i][1]
                candidates = [k for k in range(N) if dataset[k][1] == c and k != i]
                j = random.choice(candidates)
                y = 0  # positive
            else:
                # Different class
                c = dataset[i][1]
                candidates = [k for k in range(N) if dataset[k][1] != c]
                j = random.choice(candidates)
                y = 1  # negative
                
            x1, _ = dataset[i]
            x2, _ = dataset[j]
            z1, h1_1 = model.forward(x1)
            z2, h1_2 = model.forward(x2)
            D = euclidean_distance(z1, z2)
            
            if y == 0:
                # L = 0.5 * D^2
                loss = 0.5 * (D ** 2)
                grad_z1 = [z1[k] - z2[k] for k in range(2)]
                grad_z2 = [z2[k] - z1[k] for k in range(2)]
            else:
                # L = 0.5 * max(0, m - D)^2
                if D < margin:
                    loss = 0.5 * ((margin - D) ** 2)
                    coeff = -(margin - D) / D
                    grad_z1 = [coeff * (z1[k] - z2[k]) for k in range(2)]
                    grad_z2 = [coeff * (z2[k] - z1[k]) for k in range(2)]
                else:
                    loss = 0.0
                    grad_z1 = [0.0, 0.0]
                    grad_z2 = [0.0, 0.0]
                    
            total_loss += loss
            model.backward(x1, h1_1, grad_z1, lr)
            model.backward(x2, h1_2, grad_z2, lr)
            
    return model


def train_triplet_loss_with_mining(dataset: List[Tuple[List[float], int]], alpha: float = 1.5, epochs: int = 70, lr: float = 0.03) -> MetricEncoderMLP:
    """Trains metric encoder using Triplet Loss with online Semi-Hard negative mining."""
    model = MetricEncoderMLP(in_dim=4, hidden_dim=12, out_dim=2, seed=123)
    random.seed(123)
    N = len(dataset)
    
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        # Compute current representations
        embeddings = [model.forward(x)[0] for x, _ in dataset]
        
        # Mine triplets
        triplets_count = 0
        for a_idx in range(N):
            x_a, c_a = dataset[a_idx]
            z_a, h1_a = model.forward(x_a)
            
            # Find a positive
            pos_indices = [k for k in range(N) if dataset[k][1] == c_a and k != a_idx]
            p_idx = random.choice(pos_indices)
            x_p, _ = dataset[p_idx]
            z_p, h1_p = model.forward(x_p)
            D_ap = squared_distance(z_a, z_p)
            
            # Semi-hard negative mining:
            # Look for negative where: D_ap < D_an < D_ap + alpha
            neg_candidates = [k for k in range(N) if dataset[k][1] != c_a]
            semi_hard_negs = []
            hard_negs = []
            for n_idx in neg_candidates:
                z_n = embeddings[n_idx]
                D_an = squared_distance(z_a, z_n)
                if D_ap < D_an < D_ap + alpha:
                    semi_hard_negs.append((n_idx, D_an))
                elif D_an <= D_ap:
                    hard_negs.append((n_idx, D_an))
                    
            if semi_hard_negs:
                # Select the semi-hard negative closest to D_ap
                chosen_n_idx = sorted(semi_hard_negs, key=lambda item: item[1])[0][0]
            elif hard_negs:
                # Fallback to hard negative
                chosen_n_idx = sorted(hard_negs, key=lambda item: -item[1])[0][0]
            else:
                # Trivial / easy, pick random
                chosen_n_idx = random.choice(neg_candidates)
                
            x_n, _ = dataset[chosen_n_idx]
            z_n, h1_n = model.forward(x_n)
            D_an = squared_distance(z_a, z_n)
            
            loss_val = D_ap - D_an + alpha
            if loss_val > 0.0:
                loss = loss_val
                triplets_count += 1
                # Gradients:
                # dL / dz_a = 2 * (z_n - z_p)
                # dL / dz_p = 2 * (z_p - z_a)
                # dL / dz_n = 2 * (z_a - z_n)
                grad_za = [2.0 * (z_n[k] - z_p[k]) for k in range(2)]
                grad_zp = [2.0 * (z_p[k] - z_a[k]) for k in range(2)]
                grad_zn = [2.0 * (z_a[k] - z_n[k]) for k in range(2)]
                
                model.backward(x_a, h1_a, grad_za, lr)
                model.backward(x_p, h1_p, grad_zp, lr)
                model.backward(x_n, h1_n, grad_zn, lr)
                total_loss += loss
                
    return model


def main():
    print("=" * 80)
    print("PHASE 23 - TOPIC 03: METRIC LEARNING, CONTRASTIVE LOSS & TRIPLET MINING")
    print("=" * 80)

    # 1. Dataset Generation
    num_classes = 4
    samples_per_class = 20
    dataset = generate_synthetic_clusters(num_classes=num_classes, samples_per_class=samples_per_class, seed=42)
    print(f"[+] Generated synthetic dataset: {len(dataset)} samples across {num_classes} classes (A, B, C, D).")

    # 2. Pre-training Baseline Evaluation
    untrained_model = MetricEncoderMLP(in_dim=4, hidden_dim=12, out_dim=2, seed=999)
    pre_pos, pre_neg = compute_dataset_metrics(untrained_model, dataset)
    print("\n[*] Pre-Training Baseline Metrics:")
    print(f"    - Intra-Class (Positive) Distance : {pre_pos:.4f}")
    print(f"    - Inter-Class (Negative) Distance : {pre_neg:.4f}")
    print(f"    - Separation Ratio (Neg / Pos)    : {pre_neg / pre_pos:.4f} (Poor separation)")

    # 3. Train with Contrastive Loss
    margin = 3.0
    print(f"\n[+] Training MetricEncoder with Contrastive Loss (Margin m = {margin})...")
    contrastive_model = train_contrastive_loss(dataset, margin=margin, epochs=100, lr=0.04)
    cont_pos, cont_neg = compute_dataset_metrics(contrastive_model, dataset)
    print(f"[*] Post-Contrastive Training Metrics:")
    print(f"    - Intra-Class (Positive) Distance : {cont_pos:.4f} (Contracted)")
    print(f"    - Inter-Class (Negative) Distance : {cont_neg:.4f} (Separated > Margin)")
    print(f"    - Separation Ratio (Neg / Pos)    : {cont_neg / cont_pos:.4f} (High Discrimination)")

    # Assertions for Contrastive Loss
    assert cont_pos < margin / 2, f"Failed: Intra-class distance ({cont_pos:.4f}) should be < m/2 ({margin/2})"
    assert cont_neg > margin, f"Failed: Inter-class distance ({cont_neg:.4f}) should exceed margin m ({margin})"

    # 4. Train with Triplet Loss & Semi-Hard Negative Mining
    alpha = 2.0
    print(f"\n[+] Training MetricEncoder with Triplet Loss & Semi-Hard Negative Mining (Alpha = {alpha})...")
    triplet_model = train_triplet_loss_with_mining(dataset, alpha=alpha, epochs=80, lr=0.03)
    trip_pos, trip_neg = compute_dataset_metrics(triplet_model, dataset)
    print(f"[*] Post-Triplet Training Metrics:")
    print(f"    - Intra-Class (Positive) Distance : {trip_pos:.4f}")
    print(f"    - Inter-Class (Negative) Distance : {trip_neg:.4f}")
    print(f"    - Separation Ratio (Neg / Pos)    : {trip_neg / trip_pos:.4f}")

    # Assertions for Triplet Loss
    assert trip_pos < trip_neg / 2, "Failed: Triplet positive distance must be much smaller than negative distance"
    assert trip_neg > alpha, f"Failed: Negative distance ({trip_neg:.4f}) should exceed relative margin ({alpha})"

    # 5. ASCII 2D Cluster Visualization
    print("\n" + "=" * 80)
    print("LEARNED 2D METRIC SPACE SCATTER PLOT (Classes A, B, C, D)")
    print("=" * 80)
    embeddings = [contrastive_model.forward(x)[0] for x, _ in dataset]
    labels = [c for _, c in dataset]
    ascii_plot = render_ascii_scatter(embeddings, labels)
    print(ascii_plot)
    print("[*] Legend: A = Class 0, B = Class 1, C = Class 2, D = Class 3")

    print("\n[✓] Contrastive & Triplet Margin Separation Assertions PASSED!")
    print("\n[★] EXPERIMENT COMPLETE: ALL TESTS PASSED SUCCESSFULLY (Exit Code 0).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
