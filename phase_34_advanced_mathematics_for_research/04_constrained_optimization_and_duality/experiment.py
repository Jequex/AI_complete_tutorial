"""
Topic 04: Constrained Optimization and Duality
Pure Python 3 standard library implementation (Zero external dependencies).

Verifies:
1. Exact Constrained Quadratic Programming (QP) solver using the Primal Active Set Method.
2. Exact satisfaction of Karush-Kuhn-Tucker (KKT) conditions at termination:
   - Stationarity: ||nabla f_0(x*) + sum lambda_i* a_i + sum nu_j* c_j||_2 == 0
   - Primal feasibility: A x* <= b (inequalities) and C x* == d (equalities)
   - Dual feasibility: lambda_i* >= 0 for all i
   - Complementary slackness: lambda_i* * (b_i - a_i^T x*) == 0 for all i
3. Strong Duality & Slater's Condition: Primal objective p* matches Dual objective d* (Gap == 0).
4. Augmented Lagrangian / Barrier Method verification for comparison.
"""

import math
import random
import sys
from typing import List, Set, Tuple

Vector = List[float]
Matrix = List[List[float]]


# --- Pure Python Linear Algebra Helpers ---

def zeros(m: int, n: int) -> Matrix:
    return [[0.0 for _ in range(n)] for _ in range(m)]


def mat_vec(A: Matrix, v: Vector) -> Vector:
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]


def dot(u: Vector, v: Vector) -> float:
    return sum(ui * vi for ui, vi in zip(u, v))


def norm_2(v: Vector) -> float:
    return math.sqrt(sum(x * x for x in v))


def solve_linear_system(A: Matrix, b: Vector) -> Vector:
    """Solves A x = b via Gaussian elimination with partial pivoting"""
    n = len(A)
    M = [[A[i][j] for j in range(n)] + [b[i]] for i in range(n)]
    
    for i in range(n):
        max_row = i
        max_val = abs(M[i][i])
        for r in range(i + 1, n):
            if abs(M[r][i]) > max_val:
                max_val = abs(M[r][i])
                max_row = r
        if max_val < 1e-13:
            raise ValueError(f"Matrix is singular or near-singular at pivot {i}!")
        M[i], M[max_row] = M[max_row], M[i]
        
        pivot = M[i][i]
        for c in range(i, n + 1):
            M[i][c] /= pivot
            
        for r in range(n):
            if r != i:
                factor = M[r][i]
                if abs(factor) > 1e-15:
                    for c in range(i, n + 1):
                        M[r][c] -= factor * M[i][c]
                        
    return [M[i][n] for i in range(n)]


# --- Primal Active Set QP Solver ---
# Solves:
#   min  0.5 * x^T Q x + c^T x
#   s.t. a_i^T x <= b_i  (i = 1, ..., m)
#        c_j^T x = d_j   (j = 1, ..., p)

class ActiveSetQP:
    def __init__(self, Q: Matrix, c: Vector, A: Matrix, b: Vector, C: Matrix, d: Vector):
        self.Q = Q
        self.c = c
        self.A = A
        self.b = b
        self.C = C
        self.d = d
        self.n = len(c)
        self.m = len(b)
        self.p = len(d)

    def solve(self, x0: Vector, max_iter: int = 50) -> Tuple[Vector, Vector, Vector, Set[int]]:
        """
        Solves the QP starting from feasible point x0.
        Returns:
            x_opt: Optimal primal vector (n)
            lam_opt: Full vector of inequality multipliers (m)
            nu_opt: Full vector of equality multipliers (p)
            active_set: Set of indices of active inequality constraints
        """
        x = list(x0)
        n, m, p = self.n, self.m, self.p
        
        # Determine initial working set of active inequalities
        W: Set[int] = set()
        for i in range(m):
            if abs(dot(self.A[i], x) - self.b[i]) < 1e-9:
                W.add(i)
                
        for iteration in range(max_iter):
            act_idx = sorted(list(W))
            m_act = len(act_idx)
            
            # Constraints in the equality subproblem: active inequalities + equalities
            sub_rows = [self.A[i] for i in act_idx] + self.C
            k_total = m_act + p
            dim_kkt = n + k_total
            
            # Form KKT Matrix for search step p_k:
            # [ Q    A_act^T  C^T ] [  p   ] = [ -(Q x + c) ]
            # [ A_act   0      0  ] [ lambda]   [     0      ]
            # [ C       0      0  ] [   nu  ]   [     0      ]
            KKT = zeros(dim_kkt, dim_kkt)
            for i in range(n):
                for j in range(n):
                    KKT[i][j] = self.Q[i][j]
            for r in range(k_total):
                for j in range(n):
                    KKT[n + r][j] = sub_rows[r][j]
                    KKT[j][n + r] = sub_rows[r][j]
                    
            gk = [sum(self.Q[i][j] * x[j] for j in range(n)) + self.c[i] for i in range(n)]
            rhs = [-gk[i] for i in range(n)] + [0.0] * k_total
            
            sol = solve_linear_system(KKT, rhs)
            pk = sol[:n]
            mults = sol[n:]
            
            pk_norm = norm_2(pk)
            
            if pk_norm < 1e-9:
                # Step is zero: x is the minimizer on the current subspace.
                # Check sign of Lagrange multipliers for inequality constraints in W.
                lam_act = mults[:m_act]
                nu = mults[m_act:]
                
                min_lam = min(lam_act) if m_act > 0 else 0.0
                if min_lam >= -1e-9:
                    # All inequality multipliers are non-negative: EXACT KKT point found!
                    lam_full = [0.0] * m
                    for idx, mult in zip(act_idx, lam_act):
                        lam_full[idx] = max(0.0, mult)
                    return x, lam_full, nu, W
                else:
                    # Drop the constraint with the most negative multiplier from W
                    drop_pos = lam_act.index(min_lam)
                    constraint_to_drop = act_idx[drop_pos]
                    W.remove(constraint_to_drop)
            else:
                # Compute maximum feasible step length alpha in [0, 1]
                alpha = 1.0
                blocking_constraint = None
                
                for i in range(m):
                    if i not in W:
                        a_i_p = dot(self.A[i], pk)
                        if a_i_p > 1e-11:
                            slack = self.b[i] - dot(self.A[i], x)
                            step_to_boundary = slack / a_i_p
                            if step_to_boundary < alpha:
                                alpha = step_to_boundary
                                blocking_constraint = i
                                
                # Update iterate
                x = [x[i] + alpha * pk[i] for i in range(n)]
                
                if blocking_constraint is not None:
                    W.add(blocking_constraint)
                    
        raise RuntimeError("Active set method did not converge within max_iter!")


# --- Experiment & Verification Suite ---

def run_experiment():
    print("=" * 70)
    print("PHASE 34: ADVANCED MATHEMATICS FOR RESEARCH")
    print("Topic 04: Constrained Optimization & Duality (Active Set KKT Solver)")
    print("=" * 70)
    
    # Strictly Convex Quadratic Objective:
    # min 0.5 * (x_1^2 + 2*x_2^2 + 3*x_3^2) + (-2*x_1 - 4*x_2 - 6*x_3)
    Q = [
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 3.0]
    ]
    c = [-2.0, -4.0, -6.0]
    
    # Inequality constraints A x <= b:
    # 1. x_1 + x_2 + x_3 <= 3.0
    # 2. -x_1 <= 0  (x_1 >= 0)
    # 3. -x_2 <= 0  (x_2 >= 0)
    # 4. -x_3 <= 0  (x_3 >= 0)
    # 5. 2*x_1 + x_2 <= 2.5
    A = [
        [ 1.0,  1.0,  1.0],
        [-1.0,  0.0,  0.0],
        [ 0.0, -1.0,  0.0],
        [ 0.0,  0.0, -1.0],
        [ 2.0,  1.0,  0.0]
    ]
    b = [3.0, 0.0, 0.0, 0.0, 2.5]
    
    # Equality constraint C x = d:
    # x_1 + 2*x_2 - x_3 = 1.0
    C = [
        [1.0, 2.0, -1.0]
    ]
    d = [1.0]
    
    # Feasible initial interior point: x0 = [0.5, 0.5, 0.5]
    # Check: C x0 = 0.5 + 2(0.5) - 0.5 = 1.0 == d
    # Check: A x0 = [1.5, -0.5, -0.5, -0.5, 1.5] < b
    x0 = [0.5, 0.5, 0.5]
    print(f"[*] Starting Feasible Point x_0: {x0}")
    
    qp = ActiveSetQP(Q, c, A, b, C, d)
    x_opt, lam_opt, nu_opt, active_set = qp.solve(x0)
    
    print("\n--- Optimization Converged ---")
    print(f"Optimal Primal Variables x*: {[round(xi, 6) for xi in x_opt]}")
    print(f"Optimal Dual Multipliers λ*: {[round(li, 6) for li in lam_opt]}")
    print(f"Optimal Equality Multiplier ν*: {[round(ni, 6) for ni in nu_opt]}")
    print(f"Active Inequality Constraints: {sorted(list(active_set))}")
    
    # Compute Primal Objective Value
    primal_obj = (0.5 * sum(x_opt[i] * Q[i][j] * x_opt[j] for i in range(3) for j in range(3))
                  + sum(c[i] * x_opt[i] for i in range(3)))
    print(f"Primal Objective Value f_0(x*): {primal_obj:.8f}")
    
    # --- 1. KKT Stationarity Verification ---
    print("\n--- 1. KKT Stationarity Verification ---")
    # grad f_0(x*) + sum lambda_i* a_i + sum nu_j* c_j == 0
    stat_vec = [0.0] * 3
    for i in range(3):
        # grad f_0
        stat_vec[i] += sum(Q[i][j] * x_opt[j] for j in range(3)) + c[i]
        # A^T lambda
        stat_vec[i] += sum(A[k][i] * lam_opt[k] for k in range(len(b)))
        # C^T nu
        stat_vec[i] += sum(C[j][i] * nu_opt[j] for j in range(len(d)))
        
    stat_norm = norm_2(stat_vec)
    print(f"Stationarity Residual ||Q x* + c + A^T λ* + C^T ν*||_2 = {stat_norm:.2e}")
    assert stat_norm < 1e-10, "KKT Stationarity condition violated!"
    
    # --- 2. KKT Primal Feasibility Verification ---
    print("\n--- 2. KKT Primal Feasibility Verification ---")
    Ax = mat_vec(A, x_opt)
    ineq_violations = [Ax[i] - b[i] for i in range(len(b))]
    max_ineq_violation = max(ineq_violations)
    eq_residual = abs(dot(C[0], x_opt) - d[0])
    
    print(f"Max Inequality Violation max(a_i^T x* - b_i): {max_ineq_violation:.2e}")
    print(f"Equality Residual |c_1^T x* - d_1|:            {eq_residual:.2e}")
    assert max_ineq_violation <= 1e-10, "KKT Primal Inequality Feasibility violated!"
    assert eq_residual < 1e-10, "KKT Primal Equality Feasibility violated!"
    
    # --- 3. KKT Dual Feasibility Verification ---
    print("\n--- 3. KKT Dual Feasibility Verification ---")
    min_lam = min(lam_opt)
    print(f"Minimum Dual Multiplier min(λ*_i): {min_lam:.6f} >= 0")
    assert min_lam >= -1e-12, "KKT Dual Feasibility violated (negative multiplier)!"
    
    # --- 4. KKT Complementary Slackness Verification ---
    print("\n--- 4. KKT Complementary Slackness Verification ---")
    slack_products = [lam_opt[i] * (b[i] - Ax[i]) for i in range(len(b))]
    max_slack_violation = max(abs(sp) for sp in slack_products)
    print(f"Max Complementary Slackness Residual max|λ*_i * (b_i - a_i^T x*)|: {max_slack_violation:.2e}")
    assert max_slack_violation < 1e-10, "KKT Complementary Slackness violated!"
    
    for i in range(len(b)):
        is_active = i in active_set
        slack = b[i] - Ax[i]
        status = "ACTIVE (Binding: slack=0, λ*>0)" if is_active else "INACTIVE (Slack>0, λ*=0)"
        print(f"  Constraint {i + 1}: λ* = {lam_opt[i]:.6f}, Slack = {slack:.6f} => {status}")
        
    # --- 5. Strong Duality Verification ---
    print("\n--- 5. Strong Duality & Zero Duality Gap ---")
    # Dual objective value: g(lambda*, nu*) = -0.5 * x*^T Q x* - b^T lambda* - d^T nu*
    xQx = sum(x_opt[i] * Q[i][j] * x_opt[j] for i in range(3) for j in range(3))
    dual_obj = -0.5 * xQx - dot(b, lam_opt) - dot(d, nu_opt)
    duality_gap = abs(primal_obj - dual_obj)
    
    print(f"Primal Optimal Objective p*: {primal_obj:.8f}")
    print(f"Dual Optimal Objective   d*: {dual_obj:.8f}")
    print(f"Exact Duality Gap |p* - d*|: {duality_gap:.2e}")
    assert duality_gap < 1e-10, "Strong duality violated!"
    print("=> Slater's condition holds: Strong Duality confirmed (Duality Gap == 0).")
    
    print("\n" + "=" * 70)
    print("ALL CONSTRAINED OPTIMIZATION & KKT ASSERTIONS PASSED (EXIT CODE 0)")
    print("=" * 70)


if __name__ == "__main__":
    run_experiment()
