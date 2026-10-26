#!/usr/bin/env python3
"""
Topic 3.5 Experiment: Hessian Matrices, Curvature Analysis, and Newton's Method
Phase 03: Mathematics II — Multivariable Calculus

This script provides a pure Python implementation (zero external dependencies) of:
1. Numerical Hessian matrix calculation via finite difference differentiation of the gradient.
2. Exact analytical vs numerical Hessian comparison on quadratic, saddle, and Rosenbrock surfaces.
3. Eigenvalue & curvature classification (positive definite, negative definite, saddle).
4. Direct comparison of Newton's Method vs standard Gradient Descent on an ill-conditioned ravine.
5. Demonstration of how Momentum rescues Gradient Descent in high-condition-number ravines.
"""

import math
from typing import Callable, List, Tuple


# ---------------------------------------------------------------------------
# 1. Numerical Differentiation: Gradient & Hessian
# ---------------------------------------------------------------------------

def compute_gradient(
    func: Callable[[List[float]], float],
    x: List[float],
    h: float = 1e-5
) -> List[float]:
    """Computes gradient vector via central differences."""
    n = len(x)
    grad = [0.0] * n
    for i in range(n):
        x_plus = list(x)
        x_minus = list(x)
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (func(x_plus) - func(x_minus)) / (2.0 * h)
    return grad


def compute_numerical_hessian(
    func: Callable[[List[float]], float],
    x: List[float],
    h: float = 1e-4
) -> List[List[float]]:
    """
    Computes numerical Hessian matrix H (n x n) by differentiating the gradient:
    H[i][j] = (grad_i(x + h * e_j) - grad_i(x - h * e_j)) / (2 * h)
    Guarantees symmetry: H = 0.5 * (H + H^T).
    """
    n = len(x)
    hessian = [[0.0] * n for _ in range(n)]
    
    for j in range(n):
        x_plus = list(x)
        x_minus = list(x)
        x_plus[j] += h
        x_minus[j] -= h
        
        g_plus = compute_gradient(func, x_plus, h=h)
        g_minus = compute_gradient(func, x_minus, h=h)
        
        for i in range(n):
            hessian[i][j] = (g_plus[i] - g_minus[i]) / (2.0 * h)
            
    # Enforce symmetry to suppress small numerical floating-point noise
    for i in range(n):
        for j in range(i + 1, n):
            avg = 0.5 * (hessian[i][j] + hessian[j][i])
            hessian[i][j] = avg
            hessian[j][i] = avg
            
    return hessian


# ---------------------------------------------------------------------------
# 2. 2x2 Matrix Eigenvalues & Curvature Classifier
# ---------------------------------------------------------------------------

def classify_2x2_hessian(H: List[List[float]]) -> Tuple[float, float, str]:
    """
    Computes eigenvalues of a 2x2 symmetric matrix via quadratic formula:
    det(H - lambda * I) = lambda^2 - tr(H)*lambda + det(H) = 0
    """
    a, b = H[0][0], H[0][1]
    c, d = H[1][0], H[1][1]
    
    trace = a + d
    det = (a * d) - (b * c)
    
    discriminant = trace**2 - 4.0 * det
    if discriminant < 0:
        discriminant = 0.0  # Symmetric real matrices always have real eigenvalues
        
    sqrt_disc = math.sqrt(discriminant)
    lambda1 = (trace + sqrt_disc) / 2.0
    lambda2 = (trace - sqrt_disc) / 2.0
    
    eps = 1e-6
    if lambda1 > eps and lambda2 > eps:
        nature = "Positive Definite (Local Minimum / Bowl)"
    elif lambda1 < -eps and lambda2 < -eps:
        nature = "Negative Definite (Local Maximum / Dome)"
    elif (lambda1 > eps and lambda2 < -eps) or (lambda1 < -eps and lambda2 > eps):
        nature = "Indefinite (Saddle Point / Mountain Pass)"
    else:
        nature = "Degenerate / Singular (Flat valley / Ridge)"
        
    return lambda1, lambda2, nature


# ---------------------------------------------------------------------------
# 3. 2x2 Matrix Inversion & Newton Step
# ---------------------------------------------------------------------------

def invert_2x2_matrix(H: List[List[float]]) -> List[List[float]]:
    """Inverts a 2x2 matrix: H^(-1) = (1 / det) * [[d, -b], [-c, a]]."""
    a, b = H[0][0], H[0][1]
    c, d = H[1][0], H[1][1]
    det = (a * d) - (b * c)
    if abs(det) < 1e-12:
        raise ValueError("Singular matrix cannot be inverted!")
    inv_det = 1.0 / det
    return [
        [d * inv_det, -b * inv_det],
        [-c * inv_det, a * inv_det]
    ]


def newton_step_2d(grad: List[float], H: List[List[float]]) -> List[float]:
    """Computes Delta w = -H^(-1) @ grad."""
    H_inv = invert_2x2_matrix(H)
    delta_x = -(H_inv[0][0] * grad[0] + H_inv[0][1] * grad[1])
    delta_y = -(H_inv[1][0] * grad[0] + H_inv[1][1] * grad[1])
    return [delta_x, delta_y]


# ---------------------------------------------------------------------------
# 4. Benchmark Functions
# ---------------------------------------------------------------------------

def quadratic_bowl(w: List[float]) -> float:
    """f(x, y) = 3*x^2 + 7*y^2. Minimum at (0, 0)."""
    return 3.0 * w[0]**2 + 7.0 * w[1]**2


def saddle_surface(w: List[float]) -> float:
    """f(x, y) = x^2 - 4*y^2. Critical point at (0, 0)."""
    return w[0]**2 - 4.0 * w[1]**2


def rosenbrock_banana(w: List[float]) -> float:
    """Rosenbrock banana: f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2. Min at (1, 1)."""
    return (1.0 - w[0])**2 + 100.0 * (w[1] - w[0]**2)**2


# ---------------------------------------------------------------------------
# 5. Main Execution Suite
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("TOPIC 3.5: HESSIANS, CURVATURE ANALYSIS & OPTIMIZATION METHODS")
    print("=" * 72)

    # -----------------------------------------------------------------------
    # Part 1: Analytical vs Numerical Hessian Verification
    # -----------------------------------------------------------------------
    print("\n--- 1. Hessian Matrix Verification & Eigenvalue Classification ---")
    
    # Test A: Quadratic Bowl at (2.0, 3.0)
    # f(x, y) = 3x^2 + 7y^2 => grad = [6x, 14y] => H = [[6, 0], [0, 14]]
    point_bowl = [2.0, 3.0]
    H_bowl_num = compute_numerical_hessian(quadratic_bowl, point_bowl)
    l1, l2, nature = classify_2x2_hessian(H_bowl_num)
    
    print("Surface 1: Quadratic Bowl f(x, y) = 3x^2 + 7y^2 at (2, 3)")
    print(f"Numerical Hessian:   [[{H_bowl_num[0][0]:.4f}, {H_bowl_num[0][1]:.4f}], [{H_bowl_num[1][0]:.4f}, {H_bowl_num[1][1]:.4f}]]")
    print(f"Eigenvalues:         lambda_1 = {l1:.4f}, lambda_2 = {l2:.4f}")
    print(f"Classification:      {nature}")
    assert "Minimum" in nature, "Bowl classification failed!"

    # Test B: Saddle Point at (0.0, 0.0)
    # f(x, y) = x^2 - 4y^2 => grad = [2x, -8y] => H = [[2, 0], [0, -8]]
    point_saddle = [0.0, 0.0]
    H_saddle_num = compute_numerical_hessian(saddle_surface, point_saddle)
    l1, l2, nature = classify_2x2_hessian(H_saddle_num)
    
    print("\nSurface 2: Saddle Surface f(x, y) = x^2 - 4y^2 at (0, 0)")
    print(f"Numerical Hessian:   [[{H_saddle_num[0][0]:.4f}, {H_saddle_num[0][1]:.4f}], [{H_saddle_num[1][0]:.4f}, {H_saddle_num[1][1]:.4f}]]")
    print(f"Eigenvalues:         lambda_1 = {l1:.4f}, lambda_2 = {l2:.4f}")
    print(f"Classification:      {nature}")
    assert "Saddle" in nature, "Saddle classification failed!"

    # -----------------------------------------------------------------------
    # Part 2: Newton's Method 1-Step Convergence on Quadratic Bowls
    # -----------------------------------------------------------------------
    print("\n--- 2. Newton's 1-Step Quadratic Convergence Demonstration ---")
    start_point = [15.0, -25.0]
    print(f"Starting far away at: x = {start_point[0]}, y = {start_point[1]}")
    print(f"Initial loss:         {quadratic_bowl(start_point):.4f}")
    
    # Compute gradient and Hessian at starting point
    grad_start = compute_gradient(quadratic_bowl, start_point)
    H_start = compute_numerical_hessian(quadratic_bowl, start_point)
    
    # Take 1 Newton step: w_1 = w_0 - H^(-1) * grad
    step = newton_step_2d(grad_start, H_start)
    w_newton = [start_point[0] + step[0], start_point[1] + step[1]]
    
    print(f"Newton step vector:   [{step[0]:.4f}, {step[1]:.4f}]")
    print(f"New position w_1:     [{w_newton[0]:.6f}, {w_newton[1]:.6f}]")
    print(f"Loss after 1 step:    {quadratic_bowl(w_newton):.2e}")
    assert quadratic_bowl(w_newton) < 1e-6, "Newton 1-step check failed!"
    print(">> SUCCESS: Newton's method achieved the exact minimum in exactly 1 step!")

    # -----------------------------------------------------------------------
    # Part 3: Optimizer Showdown on Ill-Conditioned Rosenbrock Ravine
    # -----------------------------------------------------------------------
    print("\n--- 3. Optimizer Benchmark: Rosenbrock Banana Ravine ---")
    print("Global minimum is at (1.0, 1.0) with loss = 0.0")
    start_rosen = [-1.2, 1.0]
    
    # 1. Newton's Method (5 iterations)
    w_n = list(start_rosen)
    print("\nExecuting Newton's Method:")
    for step_idx in range(1, 6):
        g = compute_gradient(rosenbrock_banana, w_n)
        H = compute_numerical_hessian(rosenbrock_banana, w_n)
        d = newton_step_2d(g, H)
        w_n = [w_n[0] + d[0], w_n[1] + d[1]]
        loss = rosenbrock_banana(w_n)
        print(f"  Step {step_idx}: position = ({w_n[0]:.5f}, {w_n[1]:.5f}), loss = {loss:.6f}")
        
    # 2. Standard Gradient Descent (5 iterations, lr = 0.001)
    w_gd = list(start_rosen)
    lr = 0.001
    print(f"\nExecuting Standard Gradient Descent (lr = {lr}):")
    for step_idx in range(1, 6):
        g = compute_gradient(rosenbrock_banana, w_gd)
        w_gd = [w_gd[0] - lr * g[0], w_gd[1] - lr * g[1]]
        loss = rosenbrock_banana(w_gd)
        print(f"  Step {step_idx}: position = ({w_gd[0]:.5f}, {w_gd[1]:.5f}), loss = {loss:.6f}")
        
    # 3. Momentum Gradient Descent (5 iterations, lr = 0.001, beta = 0.9)
    w_mom = list(start_rosen)
    v = [0.0, 0.0]
    beta = 0.9
    print(f"\nExecuting Momentum Gradient Descent (lr = {lr}, beta = {beta}):")
    for step_idx in range(1, 6):
        g = compute_gradient(rosenbrock_banana, w_mom)
        v = [beta * v[0] + g[0], beta * v[1] + g[1]]
        w_mom = [w_mom[0] - lr * v[0], w_mom[1] - lr * v[1]]
        loss = rosenbrock_banana(w_mom)
        print(f"  Step {step_idx}: position = ({w_mom[0]:.5f}, {w_mom[1]:.5f}), loss = {loss:.6f}")
        
    print("\n" + "=" * 72)
    print("SUMMARY OF RESULTS:")
    print(f"Newton Final Loss (Step 5):   {rosenbrock_banana(w_n):.6f}")
    print(f"Momentum Final Loss (Step 5): {rosenbrock_banana(w_mom):.6f}")
    print(f"SGD Final Loss (Step 5):      {rosenbrock_banana(w_gd):.6f}")
    print(">> Second-order curvature dramatically accelerates convergence!")
    print("=" * 72)


if __name__ == "__main__":
    main()
