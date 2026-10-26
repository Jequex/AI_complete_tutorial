#!/usr/bin/env python3
"""
Phase 29 - Topic 02: Post-Training Quantization (PTQ) vs Quantization-Aware Training (QAT)
========================================================================================
Pure Python 3 implementation of:
1. PTQ vs QAT on a non-linear neural network (Concentric Rings classification).
2. Simulated Fake Quantization of weights and activations with STE (Straight-Through Estimator).
3. Naive PTQ evaluation showing severe degradation under aggressive low-bit precision.
4. QAT fine-tuning with STE restoring near-baseline accuracy.

Zero external dependencies: Python standard library only (math, random).
"""

import math
import random

def set_seed(seed=42):
    random.seed(seed)

# -----------------------------------------------------------------------------
# 1. Dataset Generation: Concentric Rings (Non-linear classification)
# -----------------------------------------------------------------------------

def generate_dataset(n_samples=500, noise=0.15):
    """Generates 2D concentric rings: inner circle = Class 0, outer circle = Class 1."""
    data = []
    labels = []
    half = n_samples // 2
    
    # Class 0: Inner circle (radius ~ 0.5)
    for _ in range(half):
        theta = random.uniform(0, 2 * math.pi)
        r = 0.5 + random.gauss(0, noise)
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        data.append([x, y])
        labels.append(0)
        
    # Class 1: Outer circle (radius ~ 1.25)
    for _ in range(half):
        theta = random.uniform(0, 2 * math.pi)
        r = 1.25 + random.gauss(0, noise)
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        data.append([x, y])
        labels.append(1)
        
    # Shuffle
    combined = list(zip(data, labels))
    random.shuffle(combined)
    data, labels = zip(*combined)
    return list(data), list(labels)

# -----------------------------------------------------------------------------
# 2. Fake Quantization & Straight-Through Estimator (STE)
# -----------------------------------------------------------------------------

def fake_quantize_tensor(tensor, num_bits=4):
    """
    Simulates integer quantization in forward pass while keeping floating-point container:
    For signed b-bit: q_min = -2^(b-1), q_max = 2^(b-1) - 1.
    scale = max(|x|) / q_max
    x_hat = scale * clip(round(x / scale), q_min, q_max)
    STE gradient mask: 1.0 inside clipping boundaries, 0.0 outside.
    """
    q_max = (1 << (num_bits - 1)) - 1
    q_min = -(1 << (num_bits - 1))
    
    is_2d = isinstance(tensor[0], list)
    flat = [item for sub in tensor for item in sub] if is_2d else list(tensor)
    
    abs_max = max(abs(x) for x in flat) if flat else 1.0
    scale = abs_max / float(q_max) if abs_max > 1e-8 else 1.0
    
    def quant_val(x):
        scaled = x / scale
        rounded = round(scaled)
        clamped = max(q_min, min(q_max, rounded))
        x_hat = clamped * scale
        # STE Mask: gradient flows if inside clipping interval
        mask = 1.0 if (scaled >= q_min and scaled <= q_max) else 0.0
        return x_hat, mask

    if is_2d:
        hat_tensor = []
        mask_tensor = []
        for row in tensor:
            hat_row = []
            mask_row = []
            for val in row:
                xh, m = quant_val(val)
                hat_row.append(xh)
                mask_row.append(m)
            hat_tensor.append(hat_row)
            mask_tensor.append(mask_row)
        return hat_tensor, scale, mask_tensor
    else:
        hat_tensor = []
        mask_tensor = []
        for val in tensor:
            xh, m = quant_val(val)
            hat_tensor.append(xh)
            mask_tensor.append(m)
        return hat_tensor, scale, mask_tensor

# -----------------------------------------------------------------------------
# 3. 2-Layer Neural Network
# -----------------------------------------------------------------------------

class SmallMLP:
    def __init__(self, in_features=2, hidden_dim=8, out_features=2):
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.out_features = out_features
        
        # He initialization
        std1 = math.sqrt(2.0 / in_features)
        self.W1 = [[random.gauss(0, std1) for _ in range(hidden_dim)] for _ in range(in_features)]
        self.b1 = [0.0 for _ in range(hidden_dim)]
        
        std2 = math.sqrt(2.0 / hidden_dim)
        self.W2 = [[random.gauss(0, std2) for _ in range(out_features)] for _ in range(hidden_dim)]
        self.b2 = [0.0 for _ in range(out_features)]

    def forward(self, x, fake_quant_weights=False, fake_quant_acts=False, num_bits=4):
        w1_used = self.W1
        w2_used = self.W2
        self.w1_mask = None
        self.w2_mask = None
        
        if fake_quant_weights:
            w1_used, _, self.w1_mask = fake_quantize_tensor(self.W1, num_bits=num_bits)
            w2_used, _, self.w2_mask = fake_quantize_tensor(self.W2, num_bits=num_bits)
            
        z1 = [self.b1[j] + sum(x[i] * w1_used[i][j] for i in range(self.in_features)) 
              for j in range(self.hidden_dim)]
        
        a1 = [max(0.0, v) for v in z1]
        
        self.a1_mask = None
        a1_used = a1
        if fake_quant_acts:
            a1_used, _, self.a1_mask = fake_quantize_tensor(a1, num_bits=num_bits)
            
        z2 = [self.b2[k] + sum(a1_used[j] * w2_used[j][k] for j in range(self.hidden_dim))
              for k in range(self.out_features)]
        
        max_z = max(z2)
        exp_z = [math.exp(v - max_z) for v in z2]
        sum_exp = sum(exp_z)
        probs = [e / sum_exp for e in exp_z]
        
        cache = {
            'x': x,
            'z1': z1,
            'a1': a1,
            'a1_used': a1_used,
            'w1_used': w1_used,
            'w2_used': w2_used,
            'probs': probs
        }
        return probs, cache

    def backward(self, cache, target_class, use_ste=False):
        x = cache['x']
        z1 = cache['z1']
        a1_used = cache['a1_used']
        w2_used = cache['w2_used']
        probs = cache['probs']
        
        dz2 = [p for p in probs]
        dz2[target_class] -= 1.0
        
        dW2 = [[a1_used[j] * dz2[k] for k in range(self.out_features)] for j in range(self.hidden_dim)]
        db2 = [dz2[k] for k in range(self.out_features)]
        
        if use_ste and self.w2_mask is not None:
            for j in range(self.hidden_dim):
                for k in range(self.out_features):
                    dW2[j][k] *= self.w2_mask[j][k]
                    
        da1_used = [sum(w2_used[j][k] * dz2[k] for k in range(self.out_features)) 
                    for j in range(self.hidden_dim)]
        
        da1 = da1_used
        if use_ste and self.a1_mask is not None:
            da1 = [da * m for da, m in zip(da1_used, self.a1_mask)]
            
        dz1 = [da if z > 0 else 0.0 for da, z in zip(da1, z1)]
        
        dW1 = [[x[i] * dz1[j] for j in range(self.hidden_dim)] for i in range(self.in_features)]
        db1 = [dz1[j] for j in range(self.hidden_dim)]
        
        if use_ste and self.w1_mask is not None:
            for i in range(self.in_features):
                for j in range(self.hidden_dim):
                    dW1[i][j] *= self.w1_mask[i][j]
                    
        return dW1, db1, dW2, db2

    def update_params(self, dW1, db1, dW2, db2, lr=0.05):
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

    def evaluate(self, dataset, labels, fake_quant_weights=False, fake_quant_acts=False, num_bits=4):
        correct = 0
        total = len(dataset)
        total_loss = 0.0
        for x, y in zip(dataset, labels):
            probs, _ = self.forward(x, fake_quant_weights=fake_quant_weights, 
                                    fake_quant_acts=fake_quant_acts, num_bits=num_bits)
            pred = 1 if probs[1] >= probs[0] else 0
            if pred == y:
                correct += 1
            total_loss -= math.log(max(1e-12, probs[y]))
        accuracy = (correct / float(total)) * 100.0
        avg_loss = total_loss / float(total)
        return accuracy, avg_loss

# -----------------------------------------------------------------------------
# 4. Main Verification
# -----------------------------------------------------------------------------

def run_experiment():
    set_seed(42)
    print("=" * 78)
    print("PHASE 29: TOPIC 02 - POST-TRAINING QUANTIZATION (PTQ) VS QAT (INT4 & INT2)")
    print("=" * 78)
    
    train_x, train_y = generate_dataset(n_samples=600, noise=0.15)
    test_x, test_y = generate_dataset(n_samples=200, noise=0.15)
    
    # 1. Train FP32 Baseline
    print("\n[Step 1] Training FP32 Baseline Model (120 epochs)...")
    model = SmallMLP(in_features=2, hidden_dim=8, out_features=2)
    for epoch in range(120):
        for x, y in zip(train_x, train_y):
            probs, cache = model.forward(x)
            dW1, db1, dW2, db2 = model.backward(cache, y)
            model.update_params(dW1, db1, dW2, db2, lr=0.08)
            
    acc_fp32, loss_fp32 = model.evaluate(test_x, test_y)
    print(f"  FP32 Baseline Accuracy: {acc_fp32:.2f}% | Loss: {loss_fp32:.4f}")
    assert acc_fp32 >= 95.0, f"Baseline model failed to train: {acc_fp32}%"

    # 2. Evaluate PTQ at INT4
    acc_ptq_int4, loss_ptq_int4 = model.evaluate(test_x, test_y, fake_quant_weights=True, fake_quant_acts=True, num_bits=4)
    print(f"\n[Step 2] Naive INT4 PTQ Accuracy: {acc_ptq_int4:.2f}% | Loss: {loss_ptq_int4:.4f}")

    # 3. Stress Test: Naive PTQ at INT2 (Extreme Quantization)
    acc_ptq_int2, loss_ptq_int2 = model.evaluate(test_x, test_y, fake_quant_weights=True, fake_quant_acts=True, num_bits=2)
    print(f"\n[Step 3] Stress Test: Naive INT2 PTQ Accuracy: {acc_ptq_int2:.2f}% | Loss: {loss_ptq_int2:.4f}")
    print(f"  Severe Accuracy Collapse under Naive INT2 PTQ: -{acc_fp32 - acc_ptq_int2:.2f}%")
    assert acc_ptq_int2 < 75.0, "Expected significant drop under naive INT2 PTQ"

    # 4. Quantization-Aware Training (QAT) with STE at INT2
    print("\n[Step 4] Quantization-Aware Training (QAT) with STE at INT2 (80 epochs)...")
    for epoch in range(80):
        for x, y in zip(train_x, train_y):
            probs, cache = model.forward(x, fake_quant_weights=True, fake_quant_acts=True, num_bits=2)
            dW1, db1, dW2, db2 = model.backward(cache, y, use_ste=True)
            model.update_params(dW1, db1, dW2, db2, lr=0.02)
            
    acc_qat_int2, loss_qat_int2 = model.evaluate(test_x, test_y, fake_quant_weights=True, fake_quant_acts=True, num_bits=2)
    print(f"  QAT INT2 Accuracy: {acc_qat_int2:.2f}% | Loss: {loss_qat_int2:.4f}")
    print(f"  Accuracy Recovery via QAT: +{acc_qat_int2 - acc_ptq_int2:.2f}%")
    
    # 5. Engineering Assertions
    assert acc_qat_int2 >= 92.0, f"QAT failed to recover INT2 accuracy: {acc_qat_int2}%"
    assert acc_qat_int2 > acc_ptq_int2 + 20.0, "QAT should dramatically outperform naive PTQ"

    print("\n" + "=" * 78)
    print("ALL EXPERIMENTS & STE ASSERTIONS PASSED (EXIT CODE 0)")
    print("=" * 78)

if __name__ == "__main__":
    run_experiment()
