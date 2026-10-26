#!/usr/bin/env python3
"""
Topic 7.2 Experiment: Ill-Conditioned Ravine Trajectory Simulator
Author: AI Complete Tutorial
Dependencies: Pure Python 3 Standard Library (math, time)

Compares:
1. Standard Gradient Descent (SGD)
2. Polyak Heavy-Ball Momentum
3. Nesterov Accelerated Gradient (NAG)

Objective function: Anisotropic Quadratic Ravine
    f(x, y) = 100 * x^2 + 1 * y^2
    Hessian: [[200, 0], [0, 2]]  --> Condition Number kappa = 100
    Minimum at (0.0, 0.0) with f(0, 0) = 0.0
"""

import math
import time

def objective(x, y):
    return 100.0 * x * x + 1.0 * y * y

def gradient(x, y):
    return 200.0 * x, 2.0 * y

def simulate_standard_gd(x0, y0, lr=0.009, max_steps=2000, tol=1e-3):
    x, y = x0, y0
    path = [(x, y)]
    start_time = time.perf_counter()
    
    for step in range(1, max_steps + 1):
        gx, gy = gradient(x, y)
        x -= lr * gx
        y -= lr * gy
        path.append((x, y))
        if math.sqrt(x * x + y * y) < tol:
            break
            
    duration = time.perf_counter() - start_time
    return path, step, duration

def simulate_polyak_momentum(x0, y0, lr=0.005, beta=0.9, max_steps=2000, tol=1e-3):
    x, y = x0, y0
    vx, vy = 0.0, 0.0
    path = [(x, y)]
    start_time = time.perf_counter()
    
    for step in range(1, max_steps + 1):
        gx, gy = gradient(x, y)
        vx = beta * vx - lr * gx
        vy = beta * vy - lr * gy
        x += vx
        y += vy
        path.append((x, y))
        if math.sqrt(x * x + y * y) < tol:
            break
            
    duration = time.perf_counter() - start_time
    return path, step, duration

def simulate_nesterov(x0, y0, lr=0.005, beta=0.9, max_steps=2000, tol=1e-3):
    x, y = x0, y0
    vx, vy = 0.0, 0.0
    path = [(x, y)]
    start_time = time.perf_counter()
    
    for step in range(1, max_steps + 1):
        # Nesterov lookahead evaluation
        look_x = x + beta * vx
        look_y = y + beta * vy
        gx, gy = gradient(look_x, look_y)
        vx = beta * vx - lr * gx
        vy = beta * vy - lr * gy
        x += vx
        y += vy
        path.append((x, y))
        if math.sqrt(x * x + y * y) < tol:
            break
            
    duration = time.perf_counter() - start_time
    return path, step, duration

def analyze_trajectory(path):
    total_distance = 0.0
    x_sign_flips = 0
    max_x_overshoot = 0.0
    
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        total_distance += dist
        if i > 0:
            prev_dx = path[i][0] - path[i - 1][0]
            curr_dx = x2 - x1
            if prev_dx * curr_dx < 0:
                x_sign_flips += 1
        if x2 < 0:
            max_x_overshoot = max(max_x_overshoot, abs(x2))
            
    return total_distance, x_sign_flips, max_x_overshoot

def main():
    print("=" * 80)
    print("TOPIC 7.2 EXPERIMENT: ILL-CONDITIONED RAVINE OPTIMIZER SIMULATOR")
    print("=" * 80)
    
    x0, y0 = 1.0, 10.0
    tol = 1e-3
    print(f"Objective Landscape: f(x, y) = 100 * x^2 + 1 * y^2 (Condition Number kappa = 100)")
    print(f"Starting Point: ({x0}, {y0}) | Target Minimum: (0.0, 0.0) | Tolerance: {tol}")
    print("-" * 80)
    
    # Run simulations
    gd_path, gd_steps, gd_time = simulate_standard_gd(x0, y0, lr=0.009, tol=tol)
    polyak_path, polyak_steps, polyak_time = simulate_polyak_momentum(x0, y0, lr=0.005, beta=0.9, tol=tol)
    nag_path, nag_steps, nag_time = simulate_nesterov(x0, y0, lr=0.005, beta=0.9, tol=tol)
    
    # Analyze paths
    gd_dist, gd_flips, gd_over = analyze_trajectory(gd_path)
    polyak_dist, polyak_flips, polyak_over = analyze_trajectory(polyak_path)
    nag_dist, nag_flips, nag_over = analyze_trajectory(nag_path)
    
    final_gd_x, final_gd_y = gd_path[-1]
    final_poly_x, final_poly_y = polyak_path[-1]
    final_nag_x, final_nag_y = nag_path[-1]
    
    print(f"{'Optimizer':<18} | {'Steps to Conv':<14} | {'Path Length':<12} | {'X Oscillations':<15} | {'Max X Overshoot':<15}")
    print("-" * 80)
    print(f"{'Standard GD':<18} | {gd_steps:<14} | {gd_dist:<12.4f} | {gd_flips:<15} | {gd_over:<15.4f}")
    print(f"{'Polyak Momentum':<18} | {polyak_steps:<14} | {polyak_dist:<12.4f} | {polyak_flips:<15} | {polyak_over:<15.4f}")
    print(f"{'Nesterov (NAG)':<18} | {nag_steps:<14} | {nag_dist:<12.4f} | {nag_flips:<15} | {nag_over:<15.4f}")
    print("-" * 80)
    
    print("\nIntermediate Coordinate Checkpoints at Key Steps:")
    print(f"{'Step':<6} | {'Standard GD (x, y)':<22} | {'Polyak Momentum (x, y)':<24} | {'Nesterov NAG (x, y)':<24}")
    print("-" * 82)
    sample_steps = [1, 5, 20, 50, 100, 200]
    for s in sample_steps:
        gd_coord = f"({gd_path[min(s, len(gd_path)-1)][0]:.3f}, {gd_path[min(s, len(gd_path)-1)][1]:.3f})"
        poly_coord = f"({polyak_path[min(s, len(polyak_path)-1)][0]:.3f}, {polyak_path[min(s, len(polyak_path)-1)][1]:.3f})"
        nag_coord = f"({nag_path[min(s, len(nag_path)-1)][0]:.3f}, {nag_path[min(s, len(nag_path)-1)][1]:.3f})"
        print(f"{s:<6} | {gd_coord:<22} | {poly_coord:<24} | {nag_coord:<24}")
    print("-" * 82)
    
    speedup_polyak = gd_steps / polyak_steps
    speedup_nag = gd_steps / nag_steps
    print(f"\nEmpirical Speedup Factors:")
    print(f"  • Polyak Momentum vs GD: {speedup_polyak:.2f}x faster")
    print(f"  • Nesterov NAG vs GD:    {speedup_nag:.2f}x faster")
    print(f"  • NAG Path Length Efficiency: NAG traversed {nag_dist:.2f} units vs Polyak's {polyak_dist:.2f} units")
    print(f"    (NAG lookahead saved {(polyak_dist - nag_dist) / polyak_dist * 100:.1f}% wasted movement from overshoot)")
    print("=" * 80)

if __name__ == "__main__":
    main()
