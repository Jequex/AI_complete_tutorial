#!/usr/bin/env python3
"""
Topic 8.2 Experiment: Logistic Regression & Binary Classification from Scratch
Author: AI Complete Tutorial
Dependencies: Pure Python 3 Standard Library (math, random)

Demonstrates:
1. Synthetic 2D binary classification dataset generation (Class 0 and Class 1 clusters).
2. Numerically stable Sigmoid function with exponent clamping.
3. Binary Cross-Entropy (BCE) loss computation with epsilon safeguards.
4. From-scratch Batch Gradient Descent using exact matrix-vector calculus.
5. Decision boundary tracking and classification evaluation:
   - Confusion Matrix (TP, FP, TN, FN)
   - Accuracy, Precision, Recall, and F1-Score metrics.
6. Rigorous assertions validating training convergence and classification performance.
"""

import math
import random
import sys

# =====================================================================
# 1. Mathematical Functions & Numerics
# =====================================================================

def sigmoid(z):
    """
    Numerically stable sigmoid function:
    sigma(z) = 1 / (1 + exp(-z))
    Clamped to [-500, 500] to prevent floating point overflow.
    """
    if z < -500:
        return 0.0
    elif z > 500:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))

def compute_bce_loss(X, y, w, eps=1e-15):
    """
    Computes Binary Cross-Entropy loss:
    L = - (1 / N) * sum( y_i * log(p_i) + (1 - y_i) * log(1 - p_i) )
    """
    N = len(X)
    total_loss = 0.0
    for i in range(N):
        # Logit: z = w^T x
        z = sum(w[j] * X[i][j] for j in range(len(w)))
        p = sigmoid(z)
        # Numerical clamping to avoid log(0)
        p = max(eps, min(1.0 - eps, p))
        total_loss += y[i] * math.log(p) + (1.0 - y[i]) * math.log(1.0 - p)
    return -total_loss / N

# =====================================================================
# 2. Synthetic Dataset Generation
# =====================================================================

def generate_binary_dataset(num_samples=300, seed=42):
    """
    Generates two 2D Gaussian clusters:
    - Class 0 centered at (-1.5, -1.5)
    - Class 1 centered at (+1.5, +1.5)
    Augments each sample with a leading 1.0 for the bias term.
    """
    random.seed(seed)
    X = []
    y = []
    
    samples_per_class = num_samples // 2
    
    # Class 0: Center (-1.5, -1.5), std = 0.8
    for _ in range(samples_per_class):
        x1 = random.gauss(-1.5, 0.8)
        x2 = random.gauss(-1.5, 0.8)
        X.append([1.0, x1, x2])
        y.append(0.0)
        
    # Class 1: Center (+1.5, +1.5), std = 0.8
    for _ in range(samples_per_class):
        x1 = random.gauss(+1.5, 0.8)
        x2 = random.gauss(+1.5, 0.8)
        X.append([1.0, x1, x2])
        y.append(1.0)
        
    # Shuffle dataset
    combined = list(zip(X, y))
    random.shuffle(combined)
    X, y = zip(*combined)
    return list(X), list(y)

# =====================================================================
# 3. Logistic Regression Classifier Class
# =====================================================================

class LogisticRegressionScratch:
    def __init__(self, lr=0.1, max_epochs=1000, tol=1e-6):
        self.lr = lr
        self.max_epochs = max_epochs
        self.tol = tol
        self.w = None
        
    def fit(self, X, y):
        N = len(X)
        d = len(X[0])
        # Initialize weights to zeros
        self.w = [0.0] * d
        
        loss_history = []
        for epoch in range(self.max_epochs):
            # Compute predictions
            p = [sigmoid(sum(self.w[j] * X[i][j] for j in range(d))) for i in range(N)]
            
            # Compute gradient: (1 / N) * X^T @ (p - y)
            errors = [p[i] - y[i] for i in range(N)]
            grad = [0.0] * d
            for i in range(N):
                err = errors[i]
                for j in range(d):
                    grad[j] += err * X[i][j]
            grad = [g / N for g in grad]
            
            # Update weights
            for j in range(d):
                self.w[j] -= self.lr * grad[j]
                
            loss = compute_bce_loss(X, y, self.w)
            loss_history.append(loss)
            
            # Convergence check
            grad_norm = math.sqrt(sum(g * g for g in grad))
            if grad_norm < self.tol:
                break
                
        return loss_history
        
    def predict_proba(self, X):
        d = len(self.w)
        return [sigmoid(sum(self.w[j] * x[j] for j in range(d))) for x in X]
        
    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return [1.0 if p >= threshold else 0.0 for p in probs]

# =====================================================================
# 4. Metrics & Evaluation
# =====================================================================

def evaluate_classification(y_true, y_pred):
    tp = fp = tn = fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1.0 and yp == 1.0:
            tp += 1
        elif yt == 0.0 and yp == 1.0:
            fp += 1
        elif yt == 0.0 and yp == 0.0:
            tn += 1
        elif yt == 1.0 and yp == 0.0:
            fn += 1
            
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }

# =====================================================================
# 5. Main Experiment
# =====================================================================

def run_experiment():
    print("=" * 70)
    print("Topic 8.2 Experiment: Logistic Regression from Scratch")
    print("=" * 70)

    # 1. Dataset Generation
    X, y = generate_binary_dataset(num_samples=400, seed=42)
    print(f"Generated {len(X)} synthetic samples (2 classes, balanced).\n")

    # Train / Test split (80 / 20)
    split_idx = int(0.8 * len(X))
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]

    # 2. Model Training
    clf = LogisticRegressionScratch(lr=0.2, max_epochs=1200, tol=1e-6)
    initial_loss = compute_bce_loss(X_train, y_train, [0.0, 0.0, 0.0])
    loss_history = clf.fit(X_train, y_train)
    final_loss = loss_history[-1]

    print("1. Training Convergence:")
    print(f"   Initial BCE Loss : {initial_loss:.4f}")
    print(f"   Final BCE Loss   : {final_loss:.4f} (over {len(loss_history)} iterations)")
    print(f"   Learned Bias (w0): {clf.w[0]:+.4f}")
    print(f"   Learned Weight w1: {clf.w[1]:+.4f}")
    print(f"   Learned Weight w2: {clf.w[2]:+.4f}\n")

    # Loss must strictly decrease
    assert final_loss < initial_loss, "Training failed to decrease loss!"
    assert final_loss < 0.15, f"Loss is suspiciously high: {final_loss}"

    # 3. Decision Boundary Analysis
    # Boundary: w0 + w1*x1 + w2*x2 = 0 => x2 = - (w1/w2)*x1 - (w0/w2)
    slope = -clf.w[1] / clf.w[2]
    intercept = -clf.w[0] / clf.w[2]
    print("2. Linear Decision Boundary:")
    print(f"   Hyperplane: {clf.w[0]:+.3f} + ({clf.w[1]:+.3f})*x1 + ({clf.w[2]:+.3f})*x2 = 0")
    print(f"   Slope-intercept form: x2 = {slope:.4f} * x1 + {intercept:.4f}\n")

    # 4. Evaluation on Test Set
    test_preds = clf.predict(X_test)
    metrics = evaluate_classification(y_test, test_preds)

    print("3. Test Set Performance Metrics:")
    print(f"   Confusion Matrix: [TN={metrics['TN']}, FP={metrics['FP']}, FN={metrics['FN']}, TP={metrics['TP']}]")
    print(f"   Accuracy        : {metrics['Accuracy'] * 100:.2f}%")
    print(f"   Precision       : {metrics['Precision'] * 100:.2f}%")
    print(f"   Recall          : {metrics['Recall'] * 100:.2f}%")
    print(f"   F1-Score        : {metrics['F1']:.4f}\n")

    # Assertions
    assert metrics["Accuracy"] >= 0.95, f"Expected accuracy >= 95%, got {metrics['Accuracy']}"
    assert metrics["F1"] >= 0.95, f"Expected F1 >= 0.95, got {metrics['F1']}"
    print("-> All performance assertions verified!")

    print("\n" + "=" * 70)
    print("All assertions PASSED successfully! [Exit Code 0]")
    print("=" * 70)

if __name__ == "__main__":
    run_experiment()
    sys.exit(0)
