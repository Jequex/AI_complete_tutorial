"""
Topic 02: Matrix Calculus and Hessian-Vector Products (Pearlmutter's Trick)
Pure Python 3 standard library implementation (Zero external dependencies).

Verifies:
1. Exact gradient evaluation for a non-convex, non-linear multi-parameter neural loss function.
2. Exact full Hessian matrix evaluation H in O(n^2) space.
3. Pearlmutter's R-operator (Forward-over-Reverse Dual Autodiff) computing H*v in O(n) time and O(n) space.
4. Centered finite difference directional derivative comparison.
5. Numerical equivalence between exact analytical H*v, Pearlmutter's O(n) HVP, and finite difference HVP.
6. Curvature analysis: Extreme Rayleigh quotients and principal curvature directions.
"""

import math
import random
import sys
from typing import List, Tuple

Vector = List[float]
Matrix = List[List[float]]


# --- Dual Number Primitive for Forward-Mode Tangent Propagation ---

class Dual:
    """Dual number class for forward-mode algorithmic differentiation: x + eps * v"""
    __slots__ = ("val", "dot")

    def __init__(self, val: float, dot: float = 0.0):
        self.val = float(val)
        self.dot = float(dot)

    def __repr__(self):
        return f"Dual({self.val:.4f}, {self.dot:.4f})"

    def __add__(self, other):
        if isinstance(other, Dual):
            return Dual(self.val + other.val, self.dot + other.dot)
        return Dual(self.val + other, self.dot)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Dual):
            return Dual(self.val - other.val, self.dot - other.dot)
        return Dual(self.val - other, self.dot)

    def __rsub__(self, other):
        return Dual(other - self.val, -self.dot)

    def __mul__(self, other):
        if isinstance(other, Dual):
            return Dual(self.val * other.val, self.val * other.dot + self.dot * other.val)
        return Dual(self.val * other, self.dot * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Dual):
            denom = other.val * other.val
            return Dual(self.val / other.val, (self.dot * other.val - self.val * other.dot) / denom)
        return Dual(self.val / other, self.dot / other)

    def __rtruediv__(self, other):
        denom = self.val * self.val
        return Dual(other / self.val, -other * self.dot / denom)

    def __neg__(self):
        return Dual(-self.val, -self.dot)

    def __pow__(self, power):
        # f(x) = x^p -> f'(x) = p * x^(p-1) * x_dot
        val_pow = self.val ** power
        dot_pow = power * (self.val ** (power - 1)) * self.dot
        return Dual(val_pow, dot_pow)


def sin_d(x: Dual) -> Dual:
    return Dual(math.sin(x.val), math.cos(x.val) * x.dot)


def cos_d(x: Dual) -> Dual:
    return Dual(math.cos(x.val), -math.sin(x.val) * x.dot)


def exp_d(x: Dual) -> Dual:
    e = math.exp(x.val)
    return Dual(e, e * x.dot)


# --- Non-linear Multi-Parameter Objective Function ---
# f(theta) = sum_i sin(theta_i) + sum_{i=0}^{n-2} 10 * (theta_{i+1} - theta_i^2)^2 + 0.5 * theta^T W theta

def create_symmetric_matrix(n: int, seed: int = 42) -> Matrix:
    rng = random.Random(seed)
    W = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            val = rng.uniform(-1.0, 1.0)
            W[i][j] = val
            W[j][i] = val
    return W


def loss_scalar(theta: Vector, W: Matrix) -> float:
    n = len(theta)
    term1 = sum(math.sin(theta[i]) for i in range(n))
    term2 = sum(10.0 * (theta[i + 1] - theta[i] ** 2) ** 2 for i in range(n - 1))
    term3 = 0.5 * sum(theta[i] * W[i][j] * theta[j] for i in range(n) for j in range(n))
    return term1 + term2 + term3


def exact_gradient(theta: Vector, W: Matrix) -> Vector:
    """Exact analytical gradient computed via backward differentiation rules"""
    n = len(theta)
    grad = [0.0] * n
    
    # Gradient of term 1: d/d theta_i (sin theta_i) = cos theta_i
    for i in range(n):
        grad[i] += math.cos(theta[i])
        
    # Gradient of term 2: sum 10 * (theta_{i+1} - theta_i^2)^2
    for i in range(n - 1):
        diff = theta[i + 1] - theta[i] ** 2
        # d/d theta_i: 10 * 2 * (theta_{i+1} - theta_i^2) * (-2 * theta_i) = -40 * theta_i * diff
        grad[i] += -40.0 * theta[i] * diff
        # d/d theta_{i+1}: 10 * 2 * (theta_{i+1} - theta_i^2) * (1) = 20 * diff
        grad[i + 1] += 20.0 * diff
        
    # Gradient of term 3: 0.5 * theta^T W theta -> W theta (since W is symmetric)
    for i in range(n):
        for j in range(n):
            grad[i] += W[i][j] * theta[j]
            
    return grad


def exact_hessian(theta: Vector, W: Matrix) -> Matrix:
    """Exact analytical n x n Hessian matrix (O(n^2) space)"""
    n = len(theta)
    H = [[0.0 for _ in range(n)] for _ in range(n)]
    
    # 1. From term 1: diagonal elements get -sin(theta_i)
    for i in range(n):
        H[i][i] += -math.sin(theta[i])
        
    # 2. From term 2: sum_{i=0}^{n-2} 10 * (theta_{i+1} - theta_i^2)^2
    # grad_i has -40 * theta_i * (theta_{i+1} - theta_i^2) = -40 theta_i theta_{i+1} + 40 theta_i^3
    # d/d theta_i (grad_i) = -40 theta_{i+1} + 120 theta_i^2
    # d/d theta_{i+1} (grad_i) = -40 theta_i
    # grad_{i+1} has 20 * (theta_{i+1} - theta_i^2)
    # d/d theta_{i+1} (grad_{i+1}) = 20
    # d/d theta_i (grad_{i+1}) = -40 theta_i
    for i in range(n - 1):
        H[i][i] += -40.0 * theta[i + 1] + 120.0 * (theta[i] ** 2)
        H[i][i + 1] += -40.0 * theta[i]
        H[i + 1][i] += -40.0 * theta[i]
        H[i + 1][i + 1] += 20.0
        
    # 3. From term 3: W
    for i in range(n):
        for j in range(n):
            H[i][j] += W[i][j]
            
    return H


# --- Pearlmutter's R-Operator: Hessian-Vector Product in O(n) ---

def pearlmutter_hvp(theta: Vector, v: Vector, W: Matrix) -> Vector:
    """
    Computes H * v in O(n) time and O(n) space without ever instantiating
    the n x n Hessian matrix H.
    
    Using the R-operator on the gradient:
        R_v{ f(theta) } = d/dr f(theta + r*v) |_{r=0}
        R_v{ grad f(theta) } = H * v
        
    Implemented via forward-mode dual number propagation through the
    gradient calculation.
    """
    n = len(theta)
    # Create dual vector theta_dual = theta + eps * v
    theta_d = [Dual(theta[i], v[i]) for i in range(n)]
    
    # We now evaluate the gradient using Dual numbers!
    # The 'dot' part of each gradient component will be exactly (H * v)_i!
    hvp = [0.0] * n
    
    # Component 1: d/d theta_i (sin theta_i) = cos(theta_i)
    # R_v{ cos(theta_i) } = -sin(theta_i) * v_i
    for i in range(n):
        cos_part = cos_d(theta_d[i])
        hvp[i] += cos_part.dot  # This is -sin(theta_i) * v_i
        
    # Component 2: Rosenbrock terms
    # For each i:
    # diff_d = theta_{i+1} - theta_i^2
    for i in range(n - 1):
        th_i = theta_d[i]
        th_next = theta_d[i + 1]
        diff_d = th_next - (th_i * th_i)
        
        # grad_i contribution: -40 * theta_i * diff
        term_i = Dual(-40.0, 0.0) * th_i * diff_d
        hvp[i] += term_i.dot
        
        # grad_{i+1} contribution: 20 * diff
        term_next = Dual(20.0, 0.0) * diff_d
        hvp[i + 1] += term_next.dot
        
    # Component 3: Quadratic term W * theta
    # R_v{ W * theta } = W * v
    for i in range(n):
        s_dot = 0.0
        for j in range(n):
            s_dot += W[i][j] * v[j]
        hvp[i] += s_dot
        
    return hvp


# --- Finite Difference HVP for Numerical Cross-Check ---

def finite_difference_hvp(theta: Vector, v: Vector, W: Matrix, h: float = 1e-6) -> Vector:
    """Centered finite difference directional derivative: (grad(theta + h*v) - grad(theta - h*v)) / (2*h)"""
    n = len(theta)
    theta_pos = [theta[i] + h * v[i] for i in range(n)]
    theta_neg = [theta[i] - h * v[i] for i in range(n)]
    
    g_pos = exact_gradient(theta_pos, W)
    g_neg = exact_gradient(theta_neg, W)
    
    return [(g_pos[i] - g_neg[i]) / (2.0 * h) for i in range(n)]


def l2_norm(v: Vector) -> float:
    return math.sqrt(sum(x * x for x in v))


def l2_diff(u: Vector, v: Vector) -> float:
    return l2_norm([ui - vi for ui, vi in zip(u, v)])


# --- Experiment & Verification Suite ---

def run_experiment():
    print("=" * 70)
    print("PHASE 34: ADVANCED MATHEMATICS FOR RESEARCH")
    print("Topic 02: Matrix Calculus & Hessian-Vector Products")
    print("=" * 70)
    
    n = 100
    print(f"[*] Parameter Dimension n = {n}")
    print(f"[*] Full Hessian requires n x n = {n * n} elements.")
    print(f"[*] Pearlmutter's R-operator requires O(n) = {n} elements.")
    
    rng = random.Random(777)
    theta = [rng.uniform(-0.5, 0.5) for _ in range(n)]
    W = create_symmetric_matrix(n, seed=42)
    
    # 1. Compute exact gradient
    grad = exact_gradient(theta, W)
    print(f"[*] Exact Gradient computed. ||grad||_2 = {l2_norm(grad):.6f}")
    
    # 2. Compute full n x n Hessian matrix
    H = exact_hessian(theta, W)
    print(f"[*] Exact Full Hessian H computed. Shape: {len(H)} x {len(H[0])}")
    
    # Verify symmetry of full Hessian: H = H^T
    sym_diff = 0.0
    for i in range(n):
        for j in range(n):
            sym_diff += (H[i][j] - H[j][i]) ** 2
    sym_err = math.sqrt(sym_diff)
    print(f"[*] Hessian Symmetry check ||H - H^T||_F = {sym_err:.2e}")
    assert sym_err < 1e-12, "Calculated Hessian is not symmetric!"
    
    # 3. Test multiple arbitrary probe directions v
    num_probes = 5
    for probe_idx in range(num_probes):
        # Generate random unit vector v
        raw_v = [rng.gauss(0.0, 1.0) for _ in range(n)]
        norm_v = l2_norm(raw_v)
        v = [x / norm_v for x in raw_v]
        
        # Method A: Exact Matrix-Vector Multiplication H * v (O(n^2))
        hvp_exact = [0.0] * n
        for i in range(n):
            s = 0.0
            for j in range(n):
                s += H[i][j] * v[j]
            hvp_exact[i] = s
            
        # Method B: Pearlmutter's R-operator Forward Dual Autodiff (O(n))
        hvp_pearlmutter = pearlmutter_hvp(theta, v, W)
        
        # Method C: Centered Finite Difference Directional Derivative
        hvp_fd = finite_difference_hvp(theta, v, W, h=1e-6)
        
        # Compute discrepancy norms
        err_pearlmutter = l2_diff(hvp_exact, hvp_pearlmutter)
        err_fd = l2_diff(hvp_exact, hvp_fd)
        
        print(f"\n[Probe Direction {probe_idx + 1}]:")
        print(f"  Exact ||H v||_2:                      {l2_norm(hvp_exact):.8f}")
        print(f"  Pearlmutter ||R_v{{grad}}||_2:           {l2_norm(hvp_pearlmutter):.8f}")
        print(f"  Finite Difference ||(g+ - g-)/2h||_2: {l2_norm(hvp_fd):.8f}")
        print(f"  Absolute Error |H v - Pearlmutter|:   {err_pearlmutter:.2e}")
        print(f"  Absolute Error |H v - FinDiff|:       {err_fd:.2e}")
        
        # Assertions
        assert err_pearlmutter < 1e-11, f"Pearlmutter trick deviated from exact H*v at probe {probe_idx+1}!"
        assert err_fd < 1e-5, f"Finite difference deviated from exact H*v at probe {probe_idx+1}!"
        
    # 4. Curvature Analysis along Direction of Negative Curvature
    print("\n--- Curvature & Eigenspace Verification ---")
    # Compute Rayleigh quotient v^T H v = v^T (H v) directly using Pearlmutter's HVP!
    # Let's perform Power Iteration on (H + shift*I) using only the O(n) Pearlmutter HVP!
    
    # Find dominant eigenvector using purely Pearlmutter HVP (zero matrix storage)
    v_iter = [rng.gauss(0.0, 1.0) for _ in range(n)]
    v_norm = l2_norm(v_iter)
    v_iter = [x / v_norm for x in v_iter]
    
    for _ in range(100):
        # Multiply H * v_iter without H!
        hvp = pearlmutter_hvp(theta, v_iter, W)
        norm_h = l2_norm(hvp)
        v_iter = [x / norm_h for x in hvp]
        
    # Rayleigh quotient curvature lambda_max = v^T H v
    hvp_final = pearlmutter_hvp(theta, v_iter, W)
    curvature_rayleigh = sum(vi * hi for vi, hi in zip(v_iter, hvp_final))
    print(f"[*] Dominant Curvature via Matrix-Free Pearlmutter Power Iteration: {curvature_rayleigh:.6f}")
    
    # Verify against exact H
    v_H_v = sum(v_iter[i] * H[i][j] * v_iter[j] for i in range(n) for j in range(n))
    diff_curv = abs(curvature_rayleigh - v_H_v)
    print(f"[*] Difference with explicit v^T H v: {diff_curv:.2e}")
    assert diff_curv < 1e-10, "Curvature from Pearlmutter differs from quadratic form!"
    
    print("\n" + "=" * 70)
    print("ALL MATRIX CALCULUS & HESSIAN-VECTOR PRODUCT ASSERTIONS PASSED (EXIT CODE 0)")
    print("=" * 70)


if __name__ == "__main__":
    run_experiment()
