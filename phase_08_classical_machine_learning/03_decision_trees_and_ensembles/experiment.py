#!/usr/bin/env python3
"""
Topic 8.3 Experiment: Decision Trees & Random Forest Ensemble from Scratch
Author: AI Complete Tutorial
Dependencies: Pure Python 3 Standard Library (math, random)

Demonstrates:
1. Exact mathematical computation of Gini Impurity and Shannon Entropy.
2. From-scratch CART Binary Decision Tree:
   - Exhaustive split evaluation across all features and continuous thresholds.
   - Information Gain / Gini reduction maximization.
   - Recursive tree building with depth and sample constraints.
3. Random Forest Classifier:
   - Bootstrap dataset sampling with replacement.
   - Random feature subsampling (m = sqrt(d)).
   - Majority voting aggregation across M trees.
4. Synthetic non-linear XOR/quadrant dataset evaluation.
5. Strict performance and sanity assertions (accuracy, impurity bounds, exit code 0).
"""

import math
import random
import sys

# =====================================================================
# 1. Impurity Metrics from Scratch
# =====================================================================

def compute_gini(labels):
    """
    Computes Gini Impurity: G = 1 - sum(p_i^2)
    """
    if not labels:
        return 0.0
    N = len(labels)
    counts = {}
    for y in labels:
        counts[y] = counts.get(y, 0) + 1
    sum_sq = sum((count / N) ** 2 for count in counts.values())
    return 1.0 - sum_sq

def compute_entropy(labels):
    """
    Computes Shannon Entropy (in bits): H = - sum(p_i * log2(p_i))
    """
    if not labels:
        return 0.0
    N = len(labels)
    counts = {}
    for y in labels:
        counts[y] = counts.get(y, 0) + 1
    entropy = 0.0
    for count in counts.values():
        p = count / N
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy

# =====================================================================
# 2. Decision Tree Node & Classifier
# =====================================================================

class Node:
    def __init__(self, is_leaf=False, prediction=None, feature=None, threshold=None, left=None, right=None, gain=0.0):
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.gain = gain

class DecisionTreeScratch:
    def __init__(self, max_depth=5, min_samples_split=2, criterion="gini", max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features  # For Random Forest feature subsampling
        self.root = None

    def _impurity(self, labels):
        if self.criterion == "gini":
            return compute_gini(labels)
        else:
            return compute_entropy(labels)

    def _best_split(self, X, y):
        best_gain = -1.0
        best_feature = None
        best_threshold = None
        
        n_samples = len(X)
        n_features = len(X[0])
        current_impurity = self._impurity(y)
        
        # Feature subsampling if requested (Random Forest mode)
        feature_indices = list(range(n_features))
        if self.max_features is not None and self.max_features < n_features:
            feature_indices = random.sample(feature_indices, self.max_features)
            
        for feat in feature_indices:
            values = sorted(set(X[i][feat] for i in range(n_samples)))
            if len(values) <= 1:
                continue
                
            # Test midpoints between consecutive values as candidate thresholds
            for i in range(len(values) - 1):
                thresh = (values[i] + values[i + 1]) / 2.0
                
                left_y = [y[j] for j in range(n_samples) if X[j][feat] <= thresh]
                right_y = [y[j] for j in range(n_samples) if X[j][feat] > thresh]
                
                if not left_y or not right_y:
                    continue
                    
                p_left = len(left_y) / n_samples
                p_right = len(right_y) / n_samples
                weighted_impurity = p_left * self._impurity(left_y) + p_right * self._impurity(right_y)
                gain = current_impurity - weighted_impurity
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat
                    best_threshold = thresh
                    
        return best_feature, best_threshold, best_gain

    def _build_tree(self, X, y, depth=0):
        # Determine majority class
        counts = {}
        for label in y:
            counts[label] = counts.get(label, 0) + 1
        majority_class = max(counts.keys(), key=lambda k: counts[k])
        
        # Stopping criteria
        if (depth >= self.max_depth or 
            len(y) < self.min_samples_split or 
            len(counts) == 1):
            return Node(is_leaf=True, prediction=majority_class)
            
        feat, thresh, gain = self._best_split(X, y)
        if feat is None or gain <= 1e-7:
            return Node(is_leaf=True, prediction=majority_class)
            
        # Partition data
        left_X, left_y, right_X, right_y = [], [], [], []
        for i in range(len(X)):
            if X[i][feat] <= thresh:
                left_X.append(X[i])
                left_y.append(y[i])
            else:
                right_X.append(X[i])
                right_y.append(y[i])
                
        left_child = self._build_tree(left_X, left_y, depth + 1)
        right_child = self._build_tree(right_X, right_y, depth + 1)
        
        return Node(is_leaf=False, feature=feat, threshold=thresh, left=left_child, right=right_child, gain=gain)

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _predict_single(self, node, x):
        if node.is_leaf:
            return node.prediction
        if x[node.feature] <= node.threshold:
            return self._predict_single(node.left, x)
        else:
            return self._predict_single(node.right, x)

    def predict(self, X):
        return [self._predict_single(self.root, x) for x in X]

# =====================================================================
# 3. Random Forest Ensemble
# =====================================================================

class RandomForestScratch:
    def __init__(self, n_estimators=15, max_depth=5, min_samples_split=2, max_features=None, seed=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.seed = seed
        self.trees = []

    def fit(self, X, y):
        random.seed(self.seed)
        N = len(X)
        d = len(X[0])
        feat_subsample = self.max_features if self.max_features is not None else max(1, int(math.sqrt(d)))
        
        self.trees = []
        for _ in range(self.n_estimators):
            # Bootstrap sampling (sample N with replacement)
            boot_indices = [random.randint(0, N - 1) for _ in range(N)]
            X_boot = [X[i] for i in boot_indices]
            y_boot = [y[i] for i in boot_indices]
            
            tree = DecisionTreeScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criterion="gini",
                max_features=feat_subsample
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
        return self

    def predict(self, X):
        # Collect predictions from all trees
        all_preds = [tree.predict(X) for tree in self.trees]
        # Majority voting per instance
        final_preds = []
        for i in range(len(X)):
            votes = {}
            for t in range(self.n_estimators):
                vote = all_preds[t][i]
                votes[vote] = votes.get(vote, 0) + 1
            majority = max(votes.keys(), key=lambda k: votes[k])
            final_preds.append(majority)
        return final_preds

# =====================================================================
# 4. Synthetic Non-Linear Dataset Generation
# =====================================================================

def generate_xor_dataset(n_samples=400, seed=42):
    """
    Generates non-linear 2D XOR quadrant data:
    Class 1: Quadrants I & III (x1 > 0 and x2 > 0) OR (x1 < 0 and x2 < 0)
    Class 0: Quadrants II & IV (x1 > 0 and x2 < 0) OR (x1 < 0 and x2 > 0)
    """
    random.seed(seed)
    X = []
    y = []
    for _ in range(n_samples):
        x1 = random.uniform(-2.0, 2.0)
        x2 = random.uniform(-2.0, 2.0)
        # Add slight boundary margin to avoid ambiguity at exactly 0
        if abs(x1) < 0.1 or abs(x2) < 0.1:
            continue
        label = 1 if (x1 * x2 > 0) else 0
        X.append([x1, x2])
        y.append(label)
    return X, y

# =====================================================================
# 5. Main Experiment
# =====================================================================

def run_experiment():
    print("=" * 70)
    print("Topic 8.3 Experiment: Decision Trees & Random Forest from Scratch")
    print("=" * 70)

    # 1. Impurity Verification
    test_node_y = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  # 4 pos, 6 neg
    gini = compute_gini(test_node_y)
    entropy = compute_entropy(test_node_y)
    print("1. Mathematical Verification of Impurity Metrics:")
    print(f"   Node Distribution: 4 Positive, 6 Negative (N=10)")
    print(f"   Gini Impurity     : {gini:.4f} (Expected: 0.4800)")
    print(f"   Shannon Entropy   : {entropy:.4f} bits (Expected: 0.9710)")
    assert abs(gini - 0.4800) < 1e-4, f"Gini mismatch: {gini}"
    assert abs(entropy - 0.9710) < 1e-3, f"Entropy mismatch: {entropy}"
    print("   -> Impurity calculations verified with high precision!\n")

    # 2. Dataset Generation
    X, y = generate_xor_dataset(n_samples=450, seed=42)
    print(f"2. Synthetic Non-Linear XOR Dataset:")
    print(f"   Total valid instances: {len(X)}")
    
    split_idx = int(0.75 * len(X))
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]
    print(f"   Train samples: {len(X_train)}, Test samples: {len(X_test)}\n")

    # 3. Train Single Decision Tree (CART)
    print("3. Training Single CART Decision Tree (max_depth=6)...")
    single_tree = DecisionTreeScratch(max_depth=6, criterion="gini")
    single_tree.fit(X_train, y_train)
    tree_preds = single_tree.predict(X_test)
    tree_acc = sum(1 for yt, yp in zip(y_test, tree_preds) if yt == yp) / len(y_test)
    print(f"   Root split feature   : Feature {single_tree.root.feature} at threshold {single_tree.root.threshold:.3f}")
    print(f"   Root information gain: {single_tree.root.gain:.4f}")
    print(f"   Single Tree Test Acc : {tree_acc * 100:.2f}%\n")

    # 4. Train Random Forest Ensemble
    print("4. Training Random Forest Ensemble (M=25 trees, max_depth=6)...")
    forest = RandomForestScratch(n_estimators=25, max_depth=6, seed=42)
    forest.fit(X_train, y_train)
    forest_preds = forest.predict(X_test)
    forest_acc = sum(1 for yt, yp in zip(y_test, forest_preds) if yt == yp) / len(y_test)
    print(f"   Random Forest Test Acc: {forest_acc * 100:.2f}%\n")

    # 5. Assertions
    assert tree_acc >= 0.95, f"Single tree accuracy too low: {tree_acc}"
    assert forest_acc >= 0.95, f"Random Forest accuracy too low: {forest_acc}"
    print("5. Performance & Generalization Sanity Check:")
    print(f"   Single Tree Accuracy : {tree_acc * 100:.2f}% (>= 88%)")
    print(f"   Random Forest Acc    : {forest_acc * 100:.2f}% (>= 92%)")
    print("   -> Successfully modeled non-linear quadrant partition!")

    print("\n" + "=" * 70)
    print("All assertions PASSED successfully! [Exit Code 0]")
    print("=" * 70)

if __name__ == "__main__":
    run_experiment()
    sys.exit(0)
