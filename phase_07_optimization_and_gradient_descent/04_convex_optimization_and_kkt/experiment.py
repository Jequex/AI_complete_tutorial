#!/usr/bin/env python3
"""
Topic 7.4 Experiment: Constrained Optimization & KKT Verification via Projected Gradient Descent (PGD)
Author: AI Complete Tutorial
Dependencies: Pure Python 3 Standard Library (math, time)

Implements:
1. Projected Gradient Descent (PGD) engine for arbitrary convex sets
2. Euclidean Box Projection: Pi_box(x, lower, upper)
3. Euclidean L2 Ball Projection: Pi_ball(x, radius)
4. Halfspace Hyperplane Projection: Pi_halfspace(x, a, b) [a^T x >= b]

Verifies all four Karush-Kuhn-Tucker (KKT) conditions numerically at convergence:
- Primal Feasibility: g_i(x*) <= 0
- Dual Feasibility: lambda_i* >= 0
- Complementary Slackness: lambda_i* * g_i(x*) == 0
- Stationarity: -grad_f(x*) == sum(lambda_i* * grad_g_i(x*))
"""

import math
import time

# --- Projection Operators ---

def project_box(x, lower, upper):
    """Projects vector x onto hypercube [lower_i, upper_i]."""
    return [max(l, min(u, xi)) for xi, l, u in zip(x, lower, upper)]

def project_ball(x, radius):
    """Projects vector x onto Euclidean ball {y : ||y||_2 <= radius}."""
    norm = math.sqrt(sum(xi * xi for xi in x))
    if norm <= radius or norm == 0.0:
        return list(x)
    scale = radius / norm
    return [xi * scale for xi in x]

def project_halfspace(x, a, b):
    """
    Projects vector x onto halfspace {y : a^T y >= b}.
    Constraint g(y) = b - a^T y <= 0.
    """
    dot_ax = sum(ai * xi for ai, xi in zip(a, x))
    if dot_ax >= b:
        return list(x)  # Already feasible
    # Otherwise project onto hyper-plane a^T y = b
    norm_sq = sum(ai * ai for ai in a)
    dist = (b - dot_ax) / norm_sq
    return [xi + dist * ai for xi, ai in zip(x, a)]


# --- Constrained Solvers ---

def solve_pgd(grad_fn, proj_fn, x_init, lr=0.05, max_steps=1000, tol=1e-6):
    """Pure Python Projected Gradient Descent."""
    x = list(x_init)
    start_time = time.perf_counter()
    
    for step in range(1, max_steps + 1):
        gx = grad_fn(x)
        # Gradient step in unconstrained space
        unconstrained_step = [xi - lr * gxi for xi, gxi in zip(x, gx)]
        # Projection step back onto convex feasible set
        x_next = proj_fn(unconstrained_step)
        
        # Check step convergence
        diff = math.sqrt(sum((xn - xi) ** 2 for xn, xi in zip(x_next, x)))
        x = x_next
        if diff < tol:
            break
            
    duration = time.perf_counter() - start_time
    return x, step, duration


def main():
    print("=" * 82)
    print("TOPIC 7.4 EXPERIMENT: CONSTRAINED OPTIMIZATION & KKT VERIFIER VIA PGD")
    print("=" * 82)
    
    # -------------------------------------------------------------------------
    # Case 1: Box-Constrained Quadratic Program
    # Objective: f(x, y) = (x - 3)^2 + (y - 4)^2
    # Constraints: 0 <= x <= 2, 0 <= y <= 2
    # Unconstrained Minimum: (3.0, 4.0)
    # Constrained Analytical Solution: (2.0, 2.0)
    # -------------------------------------------------------------------------
    print("\n[EXPERIMENT 1: Box-Constrained Quadratic Minimization]")
    print("Objective: f(x, y) = (x - 3)^2 + (y - 4)^2")
    print("Constraints: 0.0 <= x <= 2.0,  0.0 <= y <= 2.0")
    print("Unconstrained Optimum: (3.0, 4.0)  --> Feasible Optimum: (2.0, 2.0)")
    
    def grad1(x):
        return [2.0 * (x[0] - 3.0), 2.0 * (x[1] - 4.0)]
        
    def proj1(x):
        return project_box(x, [0.0, 0.0], [2.0, 2.0])
        
    x1_star, steps1, dur1 = solve_pgd(grad1, proj1, x_init=[0.0, 0.0], lr=0.1)
    f1_val = (x1_star[0] - 3.0) ** 2 + (x1_star[1] - 4.0) ** 2
    
    print(f"  • PGD Converged in: {steps1} steps ({dur1*1000:.2f} ms)")
    print(f"  • Computed Optimum: x* = ({x1_star[0]:.6f}, {x1_star[1]:.6f})")
    print(f"  • Objective Value:  f(x*) = {f1_val:.6f}")
    
    # KKT Check: Active upper bounds on both x and y
    # Stationarity: -grad_f = [-2(2-3), -2(2-4)] = [2, 4]
    # Outward normal of box bounds: lambda_x * [1, 0] + lambda_y * [0, 1]
    lambda1_x = 2.0 * (3.0 - x1_star[0])
    lambda1_y = 2.0 * (4.0 - x1_star[1])
    print(f"  • KKT Multipliers:  lambda_x = {lambda1_x:.4f}, lambda_y = {lambda1_y:.4f} (Both >= 0, Dual Feasible)")
    print(f"  • Boundary Check:   g_x = x - 2 = {x1_star[0] - 2.0:.1e}, g_y = y - 2 = {x1_star[1] - 2.0:.1e}")
    print("-" * 82)
    
    # -------------------------------------------------------------------------
    # Case 2: Euclidean Ball Constrained Optimization
    # Objective: f(x, y) = (x - 5)^2 + (y - 5)^2
    # Constraint: ||(x, y)||_2 <= 3.0
    # Analytical Solution: 3.0 * (5/sqrt(50), 5/sqrt(50)) = (3/sqrt(2), 3/sqrt(2)) ~= (2.121320, 2.121320)
    # -------------------------------------------------------------------------
    print("\n[EXPERIMENT 2: L2 Ball-Constrained Optimization]")
    print("Objective: f(x, y) = (x - 5)^2 + (y - 5)^2")
    print("Constraint: ||(x, y)||_2 <= 3.0")
    expected_x2 = 3.0 / math.sqrt(2.0)
    print(f"Theoretical Optimum: ({expected_x2:.6f}, {expected_x2:.6f})")
    
    def grad2(x):
        return [2.0 * (x[0] - 5.0), 2.0 * (x[1] - 5.0)]
        
    def proj2(x):
        return project_ball(x, radius=3.0)
        
    x2_star, steps2, dur2 = solve_pgd(grad2, proj2, x_init=[0.0, 0.0], lr=0.05)
    f2_val = (x2_star[0] - 5.0) ** 2 + (x2_star[1] - 5.0) ** 2
    r_actual = math.sqrt(x2_star[0] ** 2 + x2_star[1] ** 2)
    
    print(f"  • PGD Converged in: {steps2} steps ({dur2*1000:.2f} ms)")
    print(f"  • Computed Optimum: x* = ({x2_star[0]:.6f}, {x2_star[1]:.6f})")
    print(f"  • Distance to Theoretical: {math.sqrt((x2_star[0]-expected_x2)**2 + (x2_star[1]-expected_x2)**2):.2e}")
    print(f"  • Ball Radius at Optimum:  ||x*|| = {r_actual:.6f} (Active boundary = 3.0)")
    print("-" * 82)
    
    # -------------------------------------------------------------------------
    # Case 3: Halfspace Constraint & Direct KKT System Verification
    # Objective: f(x, y) = x^2 + y^2
    # Constraint: x + y >= 2  ==>  g(x, y) = 2 - x - y <= 0
    # Analytical Solution from Problem 2: x* = (1.0, 1.0), lambda* = 2.0
    # -------------------------------------------------------------------------
    print("\n[EXPERIMENT 3: Halfspace Constraint & Complete KKT Verification]")
    print("Objective: f(x, y) = x^2 + y^2")
    print("Constraint: x + y >= 2.0  (Standard form: g(x) = 2 - x - y <= 0)")
    print("Analytical Solution: x* = (1.0, 1.0), lambda* = 2.0, f(x*) = 2.0")
    
    def grad3(x):
        return [2.0 * x[0], 2.0 * x[1]]
        
    def proj3(x):
        return project_halfspace(x, a=[1.0, 1.0], b=2.0)
        
    x3_star, steps3, dur3 = solve_pgd(grad3, proj3, x_init=[5.0, 5.0], lr=0.1)
    
    # Check KKT Conditions
    primal_slack = 2.0 - (x3_star[0] + x3_star[1])
    # Stationarity: 2x - lambda = 0 ==> lambda = 2x
    lambda3_est = 2.0 * x3_star[0]
    comp_slackness = abs(lambda3_est * primal_slack)
    # Norm of stationarity residual
    stat_residual = math.sqrt((2.0 * x3_star[0] - lambda3_est * 1.0)**2 + (2.0 * x3_star[1] - lambda3_est * 1.0)**2)
    
    print(f"  • PGD Converged in: {steps3} steps ({dur3*1000:.2f} ms)")
    print(f"  • Computed Optimum: x* = ({x3_star[0]:.6f}, {x3_star[1]:.6f})")
    print(f"  • Objective Loss:   f(x*) = {x3_star[0]**2 + x3_star[1]**2:.6f}")
    print("\n[KKT Certificate of Optimality]:")
    print(f"  1. Primal Feasibility:      g(x*) = {primal_slack:.2e} <= 0       --> {'PASS' if primal_slack <= 1e-5 else 'FAIL'}")
    print(f"  2. Dual Feasibility:        lambda* = {lambda3_est:.4f} >= 0         --> {'PASS' if lambda3_est >= 0 else 'FAIL'}")
    print(f"  3. Complementary Slackness: |lambda* * g(x*)| = {comp_slackness:.2e}      --> {'PASS' if comp_slackness <= 1e-5 else 'FAIL'}")
    print(f"  4. Stationarity Residual:   ||grad_L|| = {stat_residual:.2e}             --> {'PASS' if stat_residual <= 1e-5 else 'FAIL'}")
    print("=" * 82)

if __name__ == "__main__":
    main()
