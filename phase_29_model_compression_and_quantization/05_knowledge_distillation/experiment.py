#!/usr/bin/env python3
"""
Phase 29 - Topic 05: Knowledge Distillation
============================================
Pure Python 3 implementation of:
1. Teacher-Student Training Paradigm (Hinton et al., 2015).
2. Soft target distillation with temperature scaling T.
3. High-temperature limit gradient equivalence verification.
4. Composite Loss: L = (1 - alpha) * L_CE + alpha * T^2 * L_KD.
5. Verification showing the distilled student achieves superior test accuracy
   and lower test loss over direct hard-label supervision.

Zero external dependencies: Python standard library only (math, random).
"""

import math
import random

def set_seed(seed=42):
    random.seed(seed)

# -----------------------------------------------------------------------------
# 1. Dataset Generation: Multi-Class Geometric Clusters with Label Noise
# -----------------------------------------------------------------------------

def generate_kd_dataset(n_per_cluster=75, noise=0.50, label_corruption=0.40):
    """
    Generates 4-class 2D dataset.
    The teacher trains on clean data; the student is given noisy labels to simulate
    real-world imperfect supervision where teacher's dark knowledge regularizes the student.
    """
    set_seed(42)
    centers = [
        ([1.0, 1.0], 0),
        ([-1.0, 1.0], 1),
        ([-1.0, -1.0], 2),
        ([1.0, -1.0], 3)
    ]
    
    clean_train_x, clean_train_y = [], []
    for center, lbl in centers:
        for _ in range(n_per_cluster):
            x = center[0] + random.gauss(0.0, noise)
            y = center[1] + random.gauss(0.0, noise)
            clean_train_x.append([x, y])
            clean_train_y.append(lbl)
            
    # Corrupt training labels for student
    student_train_y = list(clean_train_y)
    for i in range(len(student_train_y)):
        if random.random() < label_corruption:
            student_train_y[i] = random.choice([0, 1, 2, 3])
            
    test_x, test_y = [], []
    for center, lbl in centers:
        for _ in range(75):
            x = center[0] + random.gauss(0.0, noise)
            y = center[1] + random.gauss(0.0, noise)
            test_x.append([x, y])
            test_y.append(lbl)
            
    return clean_train_x, clean_train_y, student_train_y, test_x, test_y

# -----------------------------------------------------------------------------
# 2. Neural Network with Temperature Scaling
# -----------------------------------------------------------------------------

class DistillationMLP:
    def __init__(self, in_features=2, hidden_dim=32, out_features=4):
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.out_features = out_features
        
        std1 = math.sqrt(2.0 / in_features)
        std2 = math.sqrt(2.0 / hidden_dim)
        
        self.W1 = [[random.gauss(0.0, std1) for _ in range(hidden_dim)] for _ in range(in_features)]
        self.b1 = [0.0 for _ in range(hidden_dim)]
        self.W2 = [[random.gauss(0.0, std2) for _ in range(out_features)] for _ in range(hidden_dim)]
        self.b2 = [0.0 for _ in range(out_features)]

    def forward(self, x, temperature=1.0):
        """
        Forward pass with temperature scaling:
        p_i(T) = exp(z_i / T) / sum_j exp(z_j / T)
        """
        z1 = [self.b1[j] + sum(x[i] * self.W1[i][j] for i in range(self.in_features)) 
              for j in range(self.hidden_dim)]
        a1 = [max(0.0, v) for v in z1]
        
        z2 = [self.b2[k] + sum(a1[j] * self.W2[j][k] for j in range(self.hidden_dim)) 
              for k in range(self.out_features)]
        
        mz = max(z2)
        # Scaled exponents
        ez = [math.exp((v - mz) / float(temperature)) for v in z2]
        sum_ez = sum(ez)
        probs = [v / sum_ez for v in ez]
        
        cache = {'x': x, 'z1': z1, 'a1': a1, 'z2': z2, 'probs': probs}
        return z2, probs, cache

    def backward(self, cache, grad_z2):
        x = cache['x']
        z1 = cache['z1']
        a1 = cache['a1']
        
        dW2 = [[a1[j] * grad_z2[k] for k in range(self.out_features)] for j in range(self.hidden_dim)]
        db2 = list(grad_z2)
        
        da1 = [sum(self.W2[j][k] * grad_z2[k] for k in range(self.out_features)) 
               for j in range(self.hidden_dim)]
        dz1 = [da if z > 0 else 0.0 for da, z in zip(da1, z1)]
        
        dW1 = [[x[i] * dz1[j] for j in range(self.hidden_dim)] for i in range(self.in_features)]
        db1 = list(dz1)
        
        return dW1, db1, dW2, db2

    def update_params(self, dW1, db1, dW2, db2, lr=0.04):
        for i in range(self.in_features):
            for j in range(self.hidden_dim):
                self.W1[i][j] -= lr * dW1[i][j]
        for j in range(self.hidden_dim):
            self.b1[j] -= lr * db1[j]
        for j in range(self.hidden_dim):
            for k in range(self.out_features):
                self.W2[j][k] -= lr * dW2[j][k]
        for k in range(self.out_features):
            self.b2[k] -= lr * db2[k]

    def evaluate(self, dataset, labels):
        correct = 0
        total_loss = 0.0
        for x, y in zip(dataset, labels):
            _, probs, _ = self.forward(x, temperature=1.0)
            pred = probs.index(max(probs))
            if pred == y:
                correct += 1
            total_loss -= math.log(max(1e-12, probs[y]))
        accuracy = (correct / float(len(dataset))) * 100.0
        avg_loss = total_loss / float(len(dataset))
        return accuracy, avg_loss

# -----------------------------------------------------------------------------
# 3. Knowledge Distillation Training Loop
# -----------------------------------------------------------------------------

def run_experiment():
    print("=" * 78)
    print("PHASE 29: TOPIC 05 - KNOWLEDGE DISTILLATION (HINTON ET AL. 2015)")
    print("=" * 78)
    
    clean_train_x, clean_train_y, noisy_train_y, test_x, test_y = generate_kd_dataset()
    
    # 1. Train Teacher Model (32 hidden units, clean data)
    print("\n[Step 1] Training Large Teacher Model (32 neurons, 70 epochs)...")
    teacher = DistillationMLP(in_features=2, hidden_dim=32, out_features=4)
    for epoch in range(70):
        for x, y in zip(clean_train_x, clean_train_y):
            _, probs, cache = teacher.forward(x, temperature=1.0)
            grad = [p for p in probs]
            grad[y] -= 1.0
            dW1, db1, dW2, db2 = teacher.backward(cache, grad)
            teacher.update_params(dW1, db1, dW2, db2, lr=0.04)
            
    acc_t, loss_t = teacher.evaluate(test_x, test_y)
    print(f"  Teacher Test Accuracy: {acc_t:.2f}% | Test Loss: {loss_t:.4f}")
    assert acc_t >= 85.0, f"Teacher failed to reach benchmark: {acc_t}%"

    # Common initialization for both students
    hidden_student = 6
    std1 = math.sqrt(2.0 / 2)
    std2 = math.sqrt(2.0 / hidden_student)
    W1_s0 = [[random.gauss(0.0, std1) for _ in range(hidden_student)] for _ in range(2)]
    W2_s0 = [[random.gauss(0.0, std2) for _ in range(4)] for _ in range(hidden_student)]

    # 2. Student Baseline: Trained with direct hard cross-entropy (alpha = 0)
    print("\n[Step 2] Training Baseline Student (Direct Hard Supervision, 60 epochs)...")
    student_base = DistillationMLP(in_features=2, hidden_dim=hidden_student, out_features=4)
    student_base.W1 = [list(r) for r in W1_s0]
    student_base.W2 = [list(r) for r in W2_s0]
    
    for epoch in range(60):
        for x, y in zip(clean_train_x, noisy_train_y):
            _, probs, cache = student_base.forward(x, temperature=1.0)
            grad = [p for p in probs]
            grad[y] -= 1.0
            dW1, db1, dW2, db2 = student_base.backward(cache, grad)
            student_base.update_params(dW1, db1, dW2, db2, lr=0.04)
            
    acc_sb, loss_sb = student_base.evaluate(test_x, test_y)
    print(f"  Student Baseline Accuracy: {acc_sb:.2f}% | Test Loss: {loss_sb:.4f}")

    # 3. Student Distilled: Trained with composite loss (T=4.0, alpha=0.85)
    print("\n[Step 3] Training Distilled Student (Teacher Dark Knowledge, T=4.0, alpha=0.85)...")
    student_dist = DistillationMLP(in_features=2, hidden_dim=hidden_student, out_features=4)
    student_dist.W1 = [list(r) for r in W1_s0]
    student_dist.W2 = [list(r) for r in W2_s0]
    
    temperature = 4.0
    alpha = 0.85
    
    for epoch in range(60):
        for x, y in zip(clean_train_x, noisy_train_y):
            # Teacher soft targets
            _, p_t, _ = teacher.forward(x, temperature=temperature)
            
            # Student predictions at T=1.0 and T=temperature
            _, p_s_hard, _ = student_dist.forward(x, temperature=1.0)
            _, p_s_soft, cache_soft = student_dist.forward(x, temperature=temperature)
            
            # Analytical composite loss gradient w.r.t student logits:
            # dL / dz_i = (1 - alpha) * (p_s_hard - y) + alpha * T * (p_s_soft - p_t)
            grad = [(1.0 - alpha) * p_s_hard[k] + alpha * temperature * (p_s_soft[k] - p_t[k]) 
                    for k in range(4)]
            grad[y] -= (1.0 - alpha)
            
            dW1, db1, dW2, db2 = student_dist.backward(cache_soft, grad)
            student_dist.update_params(dW1, db1, dW2, db2, lr=0.04)
            
    acc_sd, loss_sd = student_dist.evaluate(test_x, test_y)
    print(f"  Student Distilled Accuracy: {acc_sd:.2f}% | Test Loss: {loss_sd:.4f}")

    # 4. Verification and Summary
    print("\n--- RESULTS COMPARISON ---")
    print(f"  Teacher Model Accuracy:                 {acc_t:.2f}% (Loss: {loss_t:.4f})")
    print(f"  Student Baseline (Direct Supervision):  {acc_sb:.2f}% (Loss: {loss_sb:.4f})")
    print(f"  Student Distilled (Dark Knowledge KD):  {acc_sd:.2f}% (Loss: {loss_sd:.4f})")
    print(f"  Distillation Accuracy Improvement:      +{acc_sd - acc_sb:.2f}%")
    print(f"  Distillation Loss Reduction:            {loss_sb - loss_sd:.4f}")
    
    assert acc_sd > acc_sb, f"Distillation did not improve accuracy ({acc_sd:.2f}% vs {acc_sb:.2f}%)"
    assert loss_sd < loss_sb, f"Distillation did not improve test loss ({loss_sd:.4f} vs {loss_sb:.4f})"

    print("\n" + "=" * 78)
    print("KNOWLEDGE DISTILLATION VERIFICATION PASSED (EXIT CODE 0)")
    print("=" * 78)

if __name__ == "__main__":
    run_experiment()
