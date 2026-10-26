#!/usr/bin/env python3
"""
Topic 7.1 Experiment: Gradient Descent Variants Benchmark
Author: AI Complete Tutorial
Dependencies: Pure Python 3 Standard Library (math, random, time)

Compares:
1. Batch Gradient Descent (BGD)
2. Pure Stochastic Gradient Descent (SGD, B=1)
3. Mini-Batch SGD (B=32)

Evaluates on a multi-feature noisy linear regression problem:
    y = 2.5 * x1 - 1.8 * x2 + 0.7 * x3 + 0.5 + epsilon
"""

import math
import random
import time

def generate_synthetic_data(num_samples=1000, seed=42):
    random.seed(seed)
    # True weights: w = [2.5, -1.8, 0.7], bias = 0.5
    true_w = [2.5, -1.8, 0.7]
    true_b = 0.5
    
    X = []
    y = []
    for _ in range(num_samples):
        # Features drawn from standard normal
        x = [random.gauss(0.0, 1.0) for _ in range(3)]
        # Add Gaussian noise sigma = 0.3
        noise = random.gauss(0.0, 0.3)
        target = sum(w_i * x_i for w_i, x_i in zip(true_w, x)) + true_b + noise
        X.append(x)
        y.append(target)
    return X, y, true_w, true_b

def compute_mse_loss(X, y, w, b):
    N = len(X)
    total_loss = 0.0
    for x_i, y_i in zip(X, y):
        pred = sum(w_j * x_j for w_j, x_j in zip(w, x_i)) + b
        diff = pred - y_i
        total_loss += diff * diff
    return total_loss / (2.0 * N)

def compute_gradient(X_subset, y_subset, w, b):
    B = len(X_subset)
    grad_w = [0.0] * len(w)
    grad_b = 0.0
    for x_i, y_i in zip(X_subset, y_subset):
        pred = sum(w_j * x_j for w_j, x_j in zip(w, x_i)) + b
        diff = pred - y_i
        for j in range(len(w)):
            grad_w[j] += diff * x_i[j]
        grad_b += diff
    grad_w = [gw / B for gw in grad_w]
    grad_b = grad_b / B
    return grad_w, grad_b

def run_batch_gd(X, y, epochs=100, lr=0.05):
    w = [0.0, 0.0, 0.0]
    b = 0.0
    history = []
    start_time = time.perf_counter()
    
    for epoch in range(epochs):
        gw, gb = compute_gradient(X, y, w, b)
        for j in range(len(w)):
            w[j] -= lr * gw[j]
        b -= lr * gb
        loss = compute_mse_loss(X, y, w, b)
        history.append(loss)
        
    duration = time.perf_counter() - start_time
    return w, b, history, duration

def run_pure_sgd(X, y, epochs=100, lr=0.01, seed=42):
    random.seed(seed)
    N = len(X)
    w = [0.0, 0.0, 0.0]
    b = 0.0
    history = []
    start_time = time.perf_counter()
    
    for epoch in range(epochs):
        # Shuffle indices each epoch
        indices = list(range(N))
        random.shuffle(indices)
        for idx in indices:
            x_i = [X[idx]]
            y_i = [y[idx]]
            gw, gb = compute_gradient(x_i, y_i, w, b)
            for j in range(len(w)):
                w[j] -= lr * gw[j]
            b -= lr * gb
        loss = compute_mse_loss(X, y, w, b)
        history.append(loss)
        
    duration = time.perf_counter() - start_time
    return w, b, history, duration

def run_minibatch_sgd(X, y, batch_size=32, epochs=100, lr=0.03, seed=42):
    random.seed(seed)
    N = len(X)
    w = [0.0, 0.0, 0.0]
    b = 0.0
    history = []
    start_time = time.perf_counter()
    
    for epoch in range(epochs):
        indices = list(range(N))
        random.shuffle(indices)
        for start_idx in range(0, N, batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]
            X_batch = [X[i] for i in batch_indices]
            y_batch = [y[i] for i in batch_indices]
            gw, gb = compute_gradient(X_batch, y_batch, w, b)
            for j in range(len(w)):
                w[j] -= lr * gw[j]
            b -= lr * gb
        loss = compute_mse_loss(X, y, w, b)
        history.append(loss)
        
    duration = time.perf_counter() - start_time
    return w, b, history, duration

def measure_gradient_variance(X, y, w, b, trials=100):
    """Computes empirical variance of gradient estimates for B=1 vs B=32."""
    N = len(X)
    # True gradient over full dataset
    true_gw, true_gb = compute_gradient(X, y, w, b)
    
    # Single sample gradients
    sgd_norms = []
    for _ in range(trials):
        idx = random.randint(0, N - 1)
        gw, gb = compute_gradient([X[idx]], [y[idx]], w, b)
        diff_sq = sum((g - tg) ** 2 for g, tg in zip(gw, true_gw)) + (gb - true_gb) ** 2
        sgd_norms.append(diff_sq)
        
    # Mini-batch gradients (B=32)
    mb_norms = []
    for _ in range(trials):
        indices = [random.randint(0, N - 1) for _ in range(32)]
        X_b = [X[i] for i in indices]
        y_b = [y[i] for i in indices]
        gw, gb = compute_gradient(X_b, y_b, w, b)
        diff_sq = sum((g - tg) ** 2 for g, tg in zip(gw, true_gw)) + (gb - true_gb) ** 2
        mb_norms.append(diff_sq)
        
    var_sgd = sum(sgd_norms) / trials
    var_mb = sum(mb_norms) / trials
    return var_sgd, var_mb

def main():
    print("=" * 78)
    print("TOPIC 7.1 EXPERIMENT: GRADIENT DESCENT VARIANTS BENCHMARK")
    print("=" * 78)
    
    X, y, true_w, true_b = generate_synthetic_data(num_samples=1000, seed=42)
    print(f"Dataset generated: N = {len(X)} samples, 3 features")
    print(f"Ground Truth: w1={true_w[0]}, w2={true_w[1]}, w3={true_w[2]}, bias={true_b}")
    print("-" * 78)
    
    # 1. Gradient Variance Demonstration
    init_w, init_b = [0.0, 0.0, 0.0], 0.0
    var_sgd, var_mb = measure_gradient_variance(X, y, init_w, init_b, trials=200)
    print(f"Gradient Variance Test at Initial Point:")
    print(f"  • Single-sample SGD (B=1)  Variance: {var_sgd:.4f}")
    print(f"  • Mini-batch SGD (B=32)    Variance: {var_mb:.4f}")
    print(f"  • Empirical Variance Ratio (SGD / Mini-batch): {var_sgd / var_mb:.2f}x (Theoretical: ~32x)")
    print("-" * 78)
    
    epochs = 60
    print(f"Running optimization for {epochs} epochs each...\n")
    
    # Run BGD
    bgd_w, bgd_b, bgd_hist, bgd_time = run_batch_gd(X, y, epochs=epochs, lr=0.1)
    # Run Pure SGD
    sgd_w, sgd_b, sgd_hist, sgd_time = run_pure_sgd(X, y, epochs=epochs, lr=0.01)
    # Run Mini-Batch SGD
    mb_w, mb_b, mb_hist, mb_time = run_minibatch_sgd(X, y, batch_size=32, epochs=epochs, lr=0.04)
    
    def param_error(w, b):
        diff = sum((w_i - tw_i) ** 2 for w_i, tw_i in zip(w, true_w)) + (b - true_b) ** 2
        return math.sqrt(diff)
    
    bgd_err = param_error(bgd_w, bgd_b)
    sgd_err = param_error(sgd_w, sgd_b)
    mb_err = param_error(mb_w, mb_b)
    
    print(f"{'Method':<18} | {'Final Loss':<12} | {'Param Error':<14} | {'Updates/Epoch':<14} | {'Time (ms)':<10}")
    print("-" * 78)
    print(f"{'Batch GD':<18} | {bgd_hist[-1]:<12.6f} | {bgd_err:<14.6f} | {'1':<14} | {bgd_time * 1000:<10.2f}")
    print(f"{'Pure SGD (B=1)':<18} | {sgd_hist[-1]:<12.6f} | {sgd_err:<14.6f} | {'1000':<14} | {sgd_time * 1000:<10.2f}")
    print(f"{'Mini-Batch (B=32)':<18} | {mb_hist[-1]:<12.6f} | {mb_err:<14.6f} | {'32':<14} | {mb_time * 1000:<10.2f}")
    print("-" * 78)
    
    print("\nLoss Progression Across Epochs:")
    print(f"{'Epoch':<8} | {'Batch GD Loss':<16} | {'Pure SGD Loss':<16} | {'Mini-Batch Loss':<16}")
    print("-" * 62)
    for ep in [1, 5, 10, 20, 40, 60]:
        print(f"{ep:<8} | {bgd_hist[ep-1]:<16.6f} | {sgd_hist[ep-1]:<16.6f} | {mb_hist[ep-1]:<16.6f}")
    print("-" * 62)
    
    print("\nKey Takeaways:")
    print("1. Mini-Batch SGD combines the variance reduction of batching with high update frequency.")
    print("2. Batch GD makes only 1 update per epoch, converging much slower in epoch counts.")
    print("3. Pure SGD makes 1000 noisy updates per epoch, showing quick early drops but jitter near the optimum.")
    print("4. Mini-batch is the undisputed industry standard for GPU hardware parallelism and generalization.")
    print("=" * 78)

if __name__ == "__main__":
    main()
