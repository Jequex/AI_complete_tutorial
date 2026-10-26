#!/usr/bin/env python3
"""
Topic 7.3 Experiment: Adaptive Optimizers Engine (AdaGrad, RMSProp, Adam, AdamW)
Author: AI Complete Tutorial
Dependencies: Pure Python 3 Standard Library (math, time)

Evaluates:
1. Standard SGD with Momentum
2. RMSProp
3. Adam (with standard L2 penalty)
4. AdamW (with decoupled weight decay)

Tested on two challenging 2D non-convex benchmark landscapes:
A. The Rosenbrock "Banana" Function:
   f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2
   Global minimum: (1.0, 1.0), f(1, 1) = 0

B. The Rastrigin Multimodal Function:
   f(x, y) = 20 + (x^2 - 10*cos(2*pi*x)) + (y^2 - 10*cos(2*pi*y))
   Global minimum: (0.0, 0.0), f(0, 0) = 0
"""

import math
import time

# --- Benchmark Functions ---

def rosenbrock(x, y):
    return (1.0 - x) ** 2 + 100.0 * (y - x * x) ** 2

def rosenbrock_grad(x, y):
    df_dx = -2.0 * (1.0 - x) - 400.0 * x * (y - x * x)
    df_dy = 200.0 * (y - x * x)
    return df_dx, df_dy

def rastrigin(x, y):
    return 20.0 + (x * x - 10.0 * math.cos(2.0 * math.pi * x)) + (y * y - 10.0 * math.cos(2.0 * math.pi * y))

def rastrigin_grad(x, y):
    df_dx = 2.0 * x + 20.0 * math.pi * math.sin(2.0 * math.pi * x)
    df_dy = 2.0 * y + 20.0 * math.pi * math.sin(2.0 * math.pi * y)
    return df_dx, df_dy


# --- Pure Python Optimizer Implementations ---

class OptimizerSGD:
    def __init__(self, lr=0.001, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.vx = 0.0
        self.vy = 0.0

    def step(self, x, y, gx, gy):
        self.vx = self.momentum * self.vx + gx
        self.vy = self.momentum * self.vy + gy
        new_x = x - self.lr * self.vx
        new_y = y - self.lr * self.vy
        return new_x, new_y

class OptimizerRMSProp:
    def __init__(self, lr=0.01, alpha=0.99, eps=1e-8):
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.vx = 0.0
        self.vy = 0.0

    def step(self, x, y, gx, gy):
        self.vx = self.alpha * self.vx + (1.0 - self.alpha) * (gx * gx)
        self.vy = self.alpha * self.vy + (1.0 - self.alpha) * (gy * gy)
        new_x = x - (self.lr / (math.sqrt(self.vx) + self.eps)) * gx
        new_y = y - (self.lr / (math.sqrt(self.vy) + self.eps)) * gy
        return new_x, new_y

class OptimizerAdam:
    def __init__(self, lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8, l2_reg=0.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.l2_reg = l2_reg
        self.mx, self.my = 0.0, 0.0
        self.vx, self.vy = 0.0, 0.0
        self.t = 0

    def step(self, x, y, gx, gy):
        self.t += 1
        # L2 regularization coupled directly with gradient
        gx_tot = gx + self.l2_reg * x
        gy_tot = gy + self.l2_reg * y
        
        # Biased moments
        self.mx = self.beta1 * self.mx + (1.0 - self.beta1) * gx_tot
        self.my = self.beta1 * self.my + (1.0 - self.beta1) * gy_tot
        self.vx = self.beta2 * self.vx + (1.0 - self.beta2) * (gx_tot * gx_tot)
        self.vy = self.beta2 * self.vy + (1.0 - self.beta2) * (gy_tot * gy_tot)
        
        # Bias corrections
        m_hat_x = self.mx / (1.0 - self.beta1 ** self.t)
        m_hat_y = self.my / (1.0 - self.beta1 ** self.t)
        v_hat_x = self.vx / (1.0 - self.beta2 ** self.t)
        v_hat_y = self.vy / (1.0 - self.beta2 ** self.t)
        
        new_x = x - (self.lr / (math.sqrt(v_hat_x) + self.eps)) * m_hat_x
        new_y = y - (self.lr / (math.sqrt(v_hat_y) + self.eps)) * m_hat_y
        return new_x, new_y

class OptimizerAdamW:
    def __init__(self, lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.mx, self.my = 0.0, 0.0
        self.vx, self.vy = 0.0, 0.0
        self.t = 0

    def step(self, x, y, gx, gy):
        self.t += 1
        # Decoupled weight decay: shrink weights directly
        x = (1.0 - self.lr * self.weight_decay) * x
        y = (1.0 - self.lr * self.weight_decay) * y
        
        # Pure data gradient for moments
        self.mx = self.beta1 * self.mx + (1.0 - self.beta1) * gx
        self.my = self.beta1 * self.my + (1.0 - self.beta1) * gy
        self.vx = self.beta2 * self.vx + (1.0 - self.beta2) * (gx * gx)
        self.vy = self.beta2 * self.vy + (1.0 - self.beta2) * (gy * gy)
        
        # Bias corrections
        m_hat_x = self.mx / (1.0 - self.beta1 ** self.t)
        m_hat_y = self.my / (1.0 - self.beta1 ** self.t)
        v_hat_x = self.vx / (1.0 - self.beta2 ** self.t)
        v_hat_y = self.vy / (1.0 - self.beta2 ** self.t)
        
        new_x = x - (self.lr / (math.sqrt(v_hat_x) + self.eps)) * m_hat_x
        new_y = y - (self.lr / (math.sqrt(v_hat_y) + self.eps)) * m_hat_y
        return new_x, new_y


def run_benchmark(opt_class, opt_kwargs, func, grad_func, start_x, start_y, steps=1000):
    opt = opt_class(**opt_kwargs)
    x, y = start_x, start_y
    loss_history = []
    
    start_time = time.perf_counter()
    for _ in range(steps):
        loss = func(x, y)
        loss_history.append(loss)
        gx, gy = grad_func(x, y)
        # Gradient clipping for extreme landscape stability
        norm = math.sqrt(gx * gx + gy * gy)
        if norm > 100.0:
            gx = (gx / norm) * 100.0
            gy = (gy / norm) * 100.0
        x, y = opt.step(x, y, gx, gy)
    duration = time.perf_counter() - start_time
    
    final_loss = func(x, y)
    return (x, y), final_loss, loss_history, duration


def main():
    print("=" * 82)
    print("TOPIC 7.3 EXPERIMENT: ADAPTIVE OPTIMIZERS BENCHMARK (ADAM & ADAMW)")
    print("=" * 82)
    
    # ---------------------------------------------------------
    # Benchmark 1: Rosenbrock Banana Valley
    # ---------------------------------------------------------
    print("\n[LANDSCAPE 1: Rosenbrock Banana Valley]")
    print("Function: f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2")
    print("Start: (-1.2, 1.0) | Target Minimum: (1.0, 1.0) with f(1, 1) = 0.0")
    print("-" * 82)
    
    steps = 1500
    configs_rosenbrock = [
        ("SGD + Momentum", OptimizerSGD, {"lr": 0.001, "momentum": 0.9}),
        ("RMSProp", OptimizerRMSProp, {"lr": 0.005, "alpha": 0.99}),
        ("Adam", OptimizerAdam, {"lr": 0.05, "beta1": 0.9, "beta2": 0.999}),
        ("AdamW", OptimizerAdamW, {"lr": 0.05, "beta1": 0.9, "beta2": 0.999, "weight_decay": 0.001}),
    ]
    
    print(f"{'Optimizer':<18} | {'Final (x, y)':<22} | {'Final Loss':<12} | {'Dist to Min':<12} | {'Time (ms)':<10}")
    print("-" * 82)
    for name, opt_cls, kwargs in configs_rosenbrock:
        (fx, fy), floss, hist, dur = run_benchmark(
            opt_cls, kwargs, rosenbrock, rosenbrock_grad, -1.2, 1.0, steps=steps
        )
        dist = math.sqrt((fx - 1.0) ** 2 + (fy - 1.0) ** 2)
        coord_str = f"({fx:.4f}, {fy:.4f})"
        print(f"{name:<18} | {coord_str:<22} | {floss:<12.6f} | {dist:<12.6f} | {dur*1000:<10.2f}")
    print("-" * 82)
    
    # ---------------------------------------------------------
    # Benchmark 2: Rastrigin Multimodal Landscape
    # ---------------------------------------------------------
    print("\n[LANDSCAPE 2: Rastrigin Highly Multimodal Surface]")
    print("Function: f(x, y) = 20 + (x^2 - 10*cos(2*pi*x)) + (y^2 - 10*cos(2*pi*y))")
    print("Start: (2.5, 2.5) | Global Minimum: (0.0, 0.0) with f(0, 0) = 0.0")
    print("-" * 82)
    
    configs_rastrigin = [
        ("SGD + Momentum", OptimizerSGD, {"lr": 0.005, "momentum": 0.9}),
        ("RMSProp", OptimizerRMSProp, {"lr": 0.02, "alpha": 0.99}),
        ("Adam", OptimizerAdam, {"lr": 0.08, "beta1": 0.9, "beta2": 0.999}),
        ("AdamW", OptimizerAdamW, {"lr": 0.08, "beta1": 0.9, "beta2": 0.999, "weight_decay": 0.01}),
    ]
    
    print(f"{'Optimizer':<18} | {'Final (x, y)':<22} | {'Final Loss':<12} | {'Dist to Min':<12} | {'Time (ms)':<10}")
    print("-" * 82)
    for name, opt_cls, kwargs in configs_rastrigin:
        (fx, fy), floss, hist, dur = run_benchmark(
            opt_cls, kwargs, rastrigin, rastrigin_grad, 2.5, 2.5, steps=800
        )
        dist = math.sqrt(fx * fx + fy * fy)
        coord_str = f"({fx:.4f}, {fy:.4f})"
        print(f"{name:<18} | {coord_str:<22} | {floss:<12.6f} | {dist:<12.6f} | {dur*1000:<10.2f}")
    print("-" * 82)
    
    # ---------------------------------------------------------
    # Benchmark 3: Adam vs AdamW Weight Decay Discrepancy
    # ---------------------------------------------------------
    print("\n[WEIGHT DECAY COMPARISON: Coupled L2 vs Decoupled AdamW]")
    print("Simulating 500 steps of sparse parameter training with regularization = 0.05")
    
    # Coordinate x has large frequent gradients, Coordinate y has tiny rare gradients
    x_adam, y_adam = 5.0, 5.0
    x_adamw, y_adamw = 5.0, 5.0
    
    adam = OptimizerAdam(lr=0.01, l2_reg=0.05)
    adamw = OptimizerAdamW(lr=0.01, weight_decay=0.05)
    
    for step in range(500):
        # x receives strong gradient, y receives faint gradient
        gx, gy = 2.0, 0.01
        x_adam, y_adam = adam.step(x_adam, y_adam, gx, gy)
        x_adamw, y_adamw = adamw.step(x_adamw, y_adamw, gx, gy)
        
    print(f"Initial weights: x = 5.0 (frequent gradient), y = 5.0 (rare gradient)")
    print(f"Standard Adam (Coupled L2):   Final x = {x_adam:.4f}, Final y = {y_adam:.4f}")
    print(f"AdamW (Decoupled Weight Decay): Final x = {x_adamw:.4f}, Final y = {y_adamw:.4f}")
    print("\nKey Insight: In standard Adam, the rare gradient on y causes its effective weight decay")
    print("to be inflated by 1 / sqrt(v), shrinking y aggressively while under-regularizing x.")
    print("AdamW applies uniform proportional shrinkage to both coordinates, preserving representations.")
    print("=" * 82)

if __name__ == "__main__":
    main()
