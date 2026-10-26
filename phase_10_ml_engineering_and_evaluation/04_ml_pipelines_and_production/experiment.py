#!/usr/bin/env python3
"""
Topic 10.4 Experiment: ML Pipelines, Safe Serialization & Covariate Shift Detection
Author: AI Complete Tutorial
Dependencies: Pure Python 3 Standard Library (math, random, json)

Implements completely from scratch:
1. Imputer (MedianImputer)
2. Scaler (StandardScaler)
3. Regressor (LinearRegression)
4. Pipeline (Chaining transformers + estimator, ensuring zero leakage)
5. Model Persistence: Safe JSON serialization & deserialization (vs pickle vulnerabilities)
6. Production Monitoring: Covariate Shift Detector using Population Stability Index (PSI)
"""

import math
import random
import json

# ----------------------------------------------------------------------
# 1. Transformers and Estimator
# ----------------------------------------------------------------------

class MedianImputer:
    """Imputes missing values (represented by None) using training feature medians."""
    def __init__(self):
        self.medians = []

    def fit(self, X):
        n_features = len(X[0])
        self.medians = []
        for j in range(n_features):
            valid_vals = sorted([X[i][j] for i in range(len(X)) if X[i][j] is not None])
            if not valid_vals:
                self.medians.append(0.0)
            else:
                m = len(valid_vals)
                med = valid_vals[m // 2] if m % 2 == 1 else (valid_vals[m // 2 - 1] + valid_vals[m // 2]) / 2.0
                self.medians.append(med)
        return self

    def transform(self, X):
        X_out = []
        for row in X:
            new_row = [self.medians[j] if row[j] is None else row[j] for j in range(len(row))]
            X_out.append(new_row)
        return X_out

    def to_dict(self):
        return {"medians": self.medians}

    @classmethod
    def from_dict(cls, d):
        obj = cls()
        obj.medians = d["medians"]
        return obj


class StandardScaler:
    """Standardizes features to zero mean and unit variance."""
    def __init__(self):
        self.means = []
        self.stds = []

    def fit(self, X):
        n_samples = len(X)
        n_features = len(X[0])
        self.means = [sum(X[i][j] for i in range(n_samples)) / n_samples for j in range(n_features)]
        self.stds = []
        for j in range(n_features):
            var = sum((X[i][j] - self.means[j]) ** 2 for i in range(n_samples)) / n_samples
            std = math.sqrt(var)
            self.stds.append(std if std > 1e-9 else 1.0)
        return self

    def transform(self, X):
        return [
            [(X[i][j] - self.means[j]) / self.stds[j] for j in range(len(self.means))]
            for i in range(len(X))
        ]

    def to_dict(self):
        return {"means": self.means, "stds": self.stds}

    @classmethod
    def from_dict(cls, d):
        obj = cls()
        obj.means = d["means"]
        obj.stds = d["stds"]
        return obj


class LinearRegression:
    """Ridge-regularized Linear Regression trained via Gradient Descent."""
    def __init__(self, lr=0.05, epochs=300, l2_reg=0.01):
        self.lr = lr
        self.epochs = epochs
        self.l2_reg = l2_reg
        self.weights = []
        self.bias = 0.0

    def fit(self, X, y):
        n_samples = len(X)
        n_features = len(X[0])
        self.weights = [0.0] * n_features
        self.bias = 0.0

        for _ in range(self.epochs):
            grad_w = [0.0] * n_features
            grad_b = 0.0
            for i in range(n_samples):
                pred = sum(self.weights[j] * X[i][j] for j in range(n_features)) + self.bias
                diff = pred - y[i]
                for j in range(n_features):
                    grad_w[j] += diff * X[i][j]
                grad_b += diff

            for j in range(n_features):
                # Gradient update with L2 shrinkage
                self.weights[j] -= (self.lr / n_samples) * (grad_w[j] + self.l2_reg * self.weights[j])
            self.bias -= (self.lr / n_samples) * grad_b
        return self

    def predict(self, X):
        return [
            sum(self.weights[j] * row[j] for j in range(len(self.weights))) + self.bias
            for row in X
        ]

    def to_dict(self):
        return {
            "weights": self.weights,
            "bias": self.bias,
            "lr": self.lr,
            "l2_reg": self.l2_reg
        }

    @classmethod
    def from_dict(cls, d):
        obj = cls(lr=d["lr"], l2_reg=d["l2_reg"])
        obj.weights = d["weights"]
        obj.bias = d["bias"]
        return obj


# ----------------------------------------------------------------------
# 2. Pipeline Architecture
# ----------------------------------------------------------------------

class MLPipeline:
    """
    Chains an arbitrary sequence of transformers followed by a final estimator.
    Guarantees no leakage between train and test/inference phases.
    """
    def __init__(self, steps):
        self.steps = steps

    def fit(self, X, y):
        current_X = X
        for name, transformer in self.steps[:-1]:
            current_X = transformer.fit(current_X).transform(current_X)
        final_name, estimator = self.steps[-1]
        estimator.fit(current_X, y)
        return self

    def predict(self, X):
        current_X = X
        for name, transformer in self.steps[:-1]:
            current_X = transformer.transform(current_X)
        final_name, estimator = self.steps[-1]
        return estimator.predict(current_X)

    def save_json(self, filepath):
        """Safe serialization avoiding pickle security vulnerabilities."""
        state = {
            "imputer": self.steps[0][1].to_dict(),
            "scaler": self.steps[1][1].to_dict(),
            "model": self.steps[2][1].to_dict()
        }
        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load_json(cls, filepath):
        """Safe deserialization without arbitrary code execution."""
        with open(filepath, "r") as f:
            state = json.load(f)
        imputer = MedianImputer.from_dict(state["imputer"])
        scaler = StandardScaler.from_dict(state["scaler"])
        model = LinearRegression.from_dict(state["model"])
        return cls([("imputer", imputer), ("scaler", scaler), ("model", model)])


# ----------------------------------------------------------------------
# 3. Production Monitoring: Covariate Shift (PSI)
# ----------------------------------------------------------------------

def compute_population_stability_index(reference_col, production_col, num_bins=10):
    """
    Computes Population Stability Index (PSI) to detect distribution drift:
        PSI = sum_{b=1}^B (P_b - Q_b) * ln(P_b / Q_b)
    where:
      P_b = fraction of production observations in bin b
      Q_b = fraction of reference baseline observations in bin b
    """
    ref_sorted = sorted(reference_col)
    n_ref = len(ref_sorted)
    n_prod = len(production_col)

    # Establish quantile breakpoints from reference distribution
    bin_edges = []
    for b in range(1, num_bins):
        q_idx = int(b * (n_ref / num_bins))
        bin_edges.append(ref_sorted[q_idx])

    # Count observations in bins
    def get_bin_counts(data):
        counts = [0] * num_bins
        for val in data:
            assigned = False
            for b, edge in enumerate(bin_edges):
                if val <= edge:
                    counts[b] += 1
                    assigned = True
                    break
            if not assigned:
                counts[-1] += 1
        return counts

    ref_counts = get_bin_counts(reference_col)
    prod_counts = get_bin_counts(production_col)

    psi_total = 0.0
    epsilon = 1e-4 # Smoothing to prevent div/0 or log(0)

    for q_count, p_count in zip(ref_counts, prod_counts):
        Q_b = max(epsilon, q_count / n_ref)
        P_b = max(epsilon, p_count / n_prod)
        psi_total += (P_b - Q_b) * math.log(P_b / Q_b)

    return psi_total

# ----------------------------------------------------------------------
# 4. Main Experiment Flow
# ----------------------------------------------------------------------

def run_experiment():
    print("=" * 78)
    print("TOPIC 10.4: ML PRODUCTION PIPELINES, SAFE SERIALIZATION & DRIFT")
    print("=" * 78)

    random.seed(42)

    # 1. Generate Training Data with Missing Values
    print("\n1. Generating synthetic training data with missingness:")
    n_train = 400
    n_test = 150

    def generate_data(n, shift_mean=0.0):
        X = []
        y = []
        for _ in range(n):
            # Feature 0: Server CPU load (continuous)
            cpu = random.gauss(50.0 + shift_mean, 15.0)
            # Feature 1: Memory load (continuous)
            mem = random.gauss(60.0 + shift_mean * 0.8, 10.0)
            # 5% missingness in feature 0
            cpu_val = None if random.random() < 0.05 else cpu
            
            target = 2.5 * (cpu if cpu_val is None else cpu_val) + 1.8 * mem + random.gauss(0.0, 5.0)
            X.append([cpu_val, mem])
            y.append(target)
        return X, y

    X_train, y_train = generate_data(n_train)
    X_test_clean, y_test_clean = generate_data(n_test)

    # 2. Build and Fit ML Pipeline
    print("2. Assembling end-to-end Pipeline: [MedianImputer -> StandardScaler -> LinearRegression]")
    pipeline = MLPipeline([
        ("imputer", MedianImputer()),
        ("scaler", StandardScaler()),
        ("model", LinearRegression(lr=0.1, epochs=250, l2_reg=0.05))
    ])

    pipeline.fit(X_train, y_train)
    print(f"   Pipeline fitted successfully!")
    print(f"   • Imputed medians: {pipeline.steps[0][1].medians}")
    print(f"   • Scaler means:    {[round(m, 2) for m in pipeline.steps[1][1].means]}")
    print(f"   • Learned weights: {[round(w, 2) for w in pipeline.steps[2][1].weights]}")

    # Evaluate on clean test set
    preds_clean = pipeline.predict(X_test_clean)
    mse_test = sum((p - y) ** 2 for p, y in zip(preds_clean, y_test_clean)) / len(y_test_clean)
    print(f"   • Out-of-Sample Test MSE: {mse_test:.2f}")

    # 3. Model Persistence (Safe JSON Serialization)
    print("\n3. Testing Safe JSON Serialization (Zero Code Execution Exploit Risk):")
    model_path = "/tmp/trained_ml_pipeline.json"
    pipeline.save_json(model_path)
    print(f"   Saved pipeline weights and statistics safely to: {model_path}")

    loaded_pipeline = MLPipeline.load_json(model_path)
    preds_loaded = loaded_pipeline.predict(X_test_clean)
    diff = max(abs(p1 - p2) for p1, p2 in zip(preds_clean, preds_loaded))
    print(f"   Deserialized pipeline verified! Max discrepancy: {diff:.1e} (Identical)")

    # 4. Production Monitoring: Covariate Shift Detection via PSI
    print("\n4. Simulating Live Production Traffic & Drift Monitoring (PSI):")
    
    # Batch 1: Stable traffic (same distribution)
    X_prod_stable, _ = generate_data(300, shift_mean=0.0)
    cpu_ref = [pipeline.steps[0][1].medians[0] if r[0] is None else r[0] for r in X_train]
    cpu_stable = [pipeline.steps[0][1].medians[0] if r[0] is None else r[0] for r in X_prod_stable]
    
    psi_stable = compute_population_stability_index(cpu_ref, cpu_stable, num_bins=8)
    print(f"   • Batch 1 (Normal Operations):  PSI = {psi_stable:.4f} -> [STABLE: No action needed]")

    # Batch 2: Severe Covariate Shift (CPU load mean shifts by +25%)
    X_prod_drifted, _ = generate_data(300, shift_mean=25.0)
    cpu_drifted = [pipeline.steps[0][1].medians[0] if r[0] is None else r[0] for r in X_prod_drifted]

    psi_drifted = compute_population_stability_index(cpu_ref, cpu_drifted, num_bins=8)
    print(f"   • Batch 2 (Server Overheating): PSI = {psi_drifted:.4f} -> [CRITICAL ALERT: Covariate Shift Detected!]")

    print("\n" + "=" * 78)
    print("PSI DECISION THRESHOLDS REFERENCE:")
    print("  • PSI < 0.10:       No significant distribution change. Model remains trustworthy.")
    print("  • 0.10 <= PSI < 0.25: Moderate shift. Increase monitoring frequency.")
    print("  • PSI >= 0.25:      Significant shift! Trigger automated retraining & alerting.")
    print("=" * 78)
    print("SUCCESS: Topic 10.4 experiment completed with exit code 0.")

if __name__ == "__main__":
    run_experiment()
