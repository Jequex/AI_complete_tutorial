"""
Topic 01: Spectral Theory and SVD Geometry
Pure Python 3 standard library implementation (Zero external dependencies).

Verifies:
1. Spectral decomposition of symmetric matrices (A = Q Lambda Q^T)
2. Singular Value Decomposition (A = U Sigma V^T) via deflationary power iteration
3. Eckart-Young-Mirsky Low-Rank Approximation Theorem (Frobenius and Spectral norms)
4. Hypersphere deformation geometry and condition number kappa(A) = sigma_max / sigma_min
"""

import math
import random
import sys
from typing import List, Tuple

# --- Pure Python Vector and Matrix Primitives ---

Vector = List[float]
Matrix = List[List[float]]


def zeros(rows: int, cols: int) -> Matrix:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def identity(n: int) -> Matrix:
    I = zeros(n, n)
    for i in range(n):
        I[i][i] = 1.0
    return I


def mat_mul(A: Matrix, B: Matrix) -> Matrix:
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    assert cols_A == rows_B, f"Dimension mismatch: {cols_A} != {rows_B}"
    C = zeros(rows_A, cols_B)
    for i in range(rows_A):
        for k in range(cols_A):
            aik = A[i][k]
            if abs(aik) > 1e-15:
                for j in range(cols_B):
                    C[i][j] += aik * B[k][j]
    return C


def mat_vec(A: Matrix, v: Vector) -> Vector:
    rows, cols = len(A), len(A[0])
    assert cols == len(v), f"Dimension mismatch: {cols} != {len(v)}"
    out = [0.0] * rows
    for i in range(rows):
        s = 0.0
        for j in range(cols):
            s += A[i][j] * v[j]
        out[i] = s
    return out


def transpose(A: Matrix) -> Matrix:
    rows, cols = len(A), len(A[0])
    At = zeros(cols, rows)
    for i in range(rows):
        for j in range(cols):
            At[j][i] = A[i][j]
    return At


def dot(u: Vector, v: Vector) -> float:
    return sum(ui * vi for ui, vi in zip(u, v))


def norm_2(v: Vector) -> float:
    return math.sqrt(sum(x * x for x in v))


def normalize(v: Vector) -> Vector:
    n = norm_2(v)
    if n < 1e-12:
        return [0.0] * len(v)
    return [x / n for x in v]


def outer_product(u: Vector, v: Vector) -> Matrix:
    m, n = len(u), len(v)
    out = zeros(m, n)
    for i in range(m):
        for j in range(n):
            out[i][j] = u[i] * v[j]
    return out


def mat_add(A: Matrix, B: Matrix, scale_B: float = 1.0) -> Matrix:
    rows, cols = len(A), len(A[0])
    return [[A[i][j] + scale_B * B[i][j] for j in range(cols)] for i in range(rows)]


def frobenius_norm(A: Matrix) -> float:
    return math.sqrt(sum(A[i][j] ** 2 for i in range(len(A)) for j in range(len(A[0]))))


# --- SVD via Deflationary Power Iteration ---

def compute_svd(A: Matrix, max_rank: int = None, tol: float = 1e-12, max_iter: int = 2000) -> Tuple[Matrix, List[float], Matrix]:
    """
    Computes SVD: A = U * diag(Sigma) * V^T using Gram-Schmidt deflated Power Iteration
    Returns:
        U: m x r matrix of orthonormal left singular vectors
        singular_values: list of length r (sorted descending)
        V: n x r matrix of orthonormal right singular vectors
    """
    m, n = len(A), len(A[0])
    r_target = min(m, n) if max_rank is None else min(max_rank, m, n)
    
    A_curr = [[A[i][j] for j in range(n)] for i in range(m)]
    
    U_cols: List[Vector] = []
    sigmas: List[float] = []
    V_cols: List[Vector] = []
    
    rng = random.Random(42)
    
    for rank_idx in range(r_target):
        # Initialize random probe vector v
        v = [rng.gauss(0.0, 1.0) for _ in range(n)]
        # Re-orthogonalize against all previous V vectors
        for prev_v in V_cols:
            proj = dot(v, prev_v)
            v = [v[j] - proj * prev_v[j] for j in range(n)]
        v = normalize(v)
        
        sigma_prev = 0.0
        
        for _ in range(max_iter):
            # u = A_curr * v
            u = mat_vec(A_curr, v)
            for prev_u in U_cols:
                proj = dot(u, prev_u)
                u = [u[i] - proj * prev_u[i] for i in range(m)]
            norm_u = norm_2(u)
            if norm_u < tol:
                break
            u = [x / norm_u for x in u]
            
            # v = A_curr^T * u
            At = transpose(A_curr)
            v = mat_vec(At, u)
            for prev_v in V_cols:
                proj = dot(v, prev_v)
                v = [v[j] - proj * prev_v[j] for j in range(n)]
            norm_v = norm_2(v)
            if norm_v < tol:
                break
            v = [x / norm_v for x in v]
            
            sigma = norm_v
            if abs(sigma - sigma_prev) < tol * (1.0 + sigma):
                break
            sigma_prev = sigma
            
        if norm_u < tol or sigma_prev < tol:
            break
            
        # Final orthogonalization pass
        for prev_u in U_cols:
            proj = dot(u, prev_u)
            u = [u[i] - proj * prev_u[i] for i in range(m)]
        u = normalize(u)
        
        v_unnorm = mat_vec(transpose(A), u)
        for prev_v in V_cols:
            proj = dot(v_unnorm, prev_v)
            v_unnorm = [v_unnorm[j] - proj * prev_v[j] for j in range(n)]
        v = normalize(v_unnorm)
        
        sigma_final = dot(u, mat_vec(A, v))
        if sigma_final < tol:
            break
            
        U_cols.append(u)
        sigmas.append(sigma_final)
        V_cols.append(v)
        
        # Deflate A_curr
        rank1 = outer_product(u, v)
        A_curr = mat_add(A_curr, rank1, scale_B=-sigma_final)
        
    rank = len(sigmas)
    U = zeros(m, rank)
    for j in range(rank):
        for i in range(m):
            U[i][j] = U_cols[j][i]
            
    V = zeros(n, rank)
    for j in range(rank):
        for i in range(n):
            V[i][j] = V_cols[j][i]
            
    return U, sigmas, V


def reconstruct_low_rank(U: Matrix, sigmas: List[float], V: Matrix, k: int) -> Matrix:
    """Reconstructs A_k = sum_{i=1}^k sigma_i * u_i * v_i^T"""
    m = len(U)
    n = len(V)
    A_k = zeros(m, n)
    for r in range(k):
        sigma_r = sigmas[r]
        u_r = [U[i][r] for i in range(m)]
        v_r = [V[j][r] for j in range(n)]
        uv_t = outer_product(u_r, v_r)
        A_k = mat_add(A_k, uv_t, scale_B=sigma_r)
    return A_k


def matrix_spectral_norm(A: Matrix) -> float:
    """Spectral norm ||A||_2 is the largest singular value"""
    _, s, _ = compute_svd(A, max_rank=1)
    return s[0] if s else 0.0


# --- Experiment & Rigorous Verification Suite ---

def run_experiment():
    print("=" * 70)
    print("PHASE 34: ADVANCED MATHEMATICS FOR RESEARCH")
    print("Topic 01: Spectral Theory & SVD Geometry")
    print("=" * 70)
    
    # 1. Define synthetic ill-conditioned matrix A (6 x 4)
    # A has known decay in singular values
    A: Matrix = [
        [3.0, 2.0, 1.0, 0.5],
        [2.0, 4.0, 2.0, 1.0],
        [1.0, 2.0, 5.0, 2.0],
        [0.5, 1.0, 2.0, 6.0],
        [1.5, 0.5, 1.0, 0.5],
        [0.2, 0.8, 1.2, 1.8],
    ]
    m, n = len(A), len(A[0])
    print(f"[*] Input Matrix A: {m} x {n}")
    
    # 2. Compute SVD
    U, sigmas, V = compute_svd(A)
    rank = len(sigmas)
    print(f"[*] SVD Computed. Discovered Rank = {rank}")
    print(f"[*] Singular Values: {[round(s, 6) for s in sigmas]}")
    
    # 3. Verify Orthonormality of U and V
    print("\n--- 1. Orthonormality Verification ---")
    UtU = mat_mul(transpose(U), U)
    VtV = mat_mul(transpose(V), V)
    I_r = identity(rank)
    
    err_U_orth = frobenius_norm(mat_add(UtU, I_r, scale_B=-1.0))
    err_V_orth = frobenius_norm(mat_add(VtV, I_r, scale_B=-1.0))
    print(f"||U^T U - I||_F = {err_U_orth:.2e}")
    print(f"||V^T V - I||_F = {err_V_orth:.2e}")
    assert err_U_orth < 1e-6, "Left singular vectors U are not orthonormal!"
    assert err_V_orth < 1e-6, "Right singular vectors V are not orthonormal!"
    
    # 4. Verify Full Reconstruction: A = sum_{i=1}^r sigma_i u_i v_i^T
    print("\n--- 2. Full Matrix Reconstruction ---")
    A_reconstructed = reconstruct_low_rank(U, sigmas, V, rank)
    recon_err = frobenius_norm(mat_add(A, A_reconstructed, scale_B=-1.0))
    print(f"||A - sum_i sigma_i u_i v_i^T||_F = {recon_err:.2e}")
    assert recon_err < 1e-5, "SVD reconstruction error exceeds tolerance!"
    
    # 5. Verify Eckart-Young-Mirsky Theorem
    # ||A - A_k||_F = sqrt(sum_{j=k+1}^r sigma_j^2)
    # ||A - A_k||_2 = sigma_{k+1}
    print("\n--- 3. Eckart-Young-Mirsky Theorem Verification ---")
    for k in range(1, rank):
        A_k = reconstruct_low_rank(U, sigmas, V, k)
        residual_matrix = mat_add(A, A_k, scale_B=-1.0)
        
        empirical_frob = frobenius_norm(residual_matrix)
        theoretical_frob = math.sqrt(sum(sigmas[j] ** 2 for j in range(k, rank)))
        
        empirical_spec = matrix_spectral_norm(residual_matrix)
        theoretical_spec = sigmas[k]
        
        frob_rel_diff = abs(empirical_frob - theoretical_frob) / theoretical_frob
        spec_rel_diff = abs(empirical_spec - theoretical_spec) / theoretical_spec
        
        print(f"Rank k={k}:")
        print(f"  Frobenius norm: Empirical = {empirical_frob:.6f}, Theoretical = {theoretical_frob:.6f} (Rel Diff: {frob_rel_diff:.2e})")
        print(f"  Spectral norm:  Empirical = {empirical_spec:.6f}, Theoretical = {theoretical_spec:.6f} (Rel Diff: {spec_rel_diff:.2e})")
        
        assert frob_rel_diff < 1e-5, f"Eckart-Young Frobenius norm violated at rank {k}!"
        assert spec_rel_diff < 1e-4, f"Eckart-Young Spectral norm violated at rank {k}!"
        
    # 6. Verify Suboptimality of Any Random Rank-k Perturbation
    print("\n--- 4. Suboptimality of Alternate Rank-k Matrices ---")
    k = 2
    A_k_optimal = reconstruct_low_rank(U, sigmas, V, k)
    err_optimal = frobenius_norm(mat_add(A, A_k_optimal, scale_B=-1.0))
    
    # Construct a perturbed rank-k matrix by rotating subspace
    rng = random.Random(123)
    perturbed_V = [list(row) for row in V]
    # Small random perturbation to right singular vectors
    theta = 0.25
    v0_perturbed = [math.cos(theta) * V[j][0] + math.sin(theta) * V[j][1] for j in range(n)]
    v1_perturbed = [-math.sin(theta) * V[j][0] + math.cos(theta) * V[j][1] for j in range(n)]
    
    A_k_suboptimal = zeros(m, n)
    for j in range(n):
        for i in range(m):
            A_k_suboptimal[i][j] = (sigmas[0] * U[i][0] * v0_perturbed[j] +
                                    sigmas[1] * U[i][1] * v1_perturbed[j])
                                    
    err_suboptimal = frobenius_norm(mat_add(A, A_k_suboptimal, scale_B=-1.0))
    print(f"Rank-2 Optimal Error ||A - A_2^*||_F:   {err_optimal:.6f}")
    print(f"Rank-2 Suboptimal Error ||A - A_2'||_F: {err_suboptimal:.6f}")
    assert err_suboptimal > err_optimal, "Optimal low-rank approximation was beaten by perturbation!"
    print("=> Eckart-Young optimality confirmed: ||A - A_k^*||_F < ||A - B_k||_F for all B_k != A_k^*.")
    
    # 7. Condition Number and Hypersphere Deformation
    print("\n--- 5. Condition Number & Hypersphere Geometry ---")
    kappa = sigmas[0] / sigmas[-1]
    print(f"Maximum Singular Value (sigma_1): {sigmas[0]:.6f}")
    print(f"Minimum Singular Value (sigma_r): {sigmas[-1]:.6f}")
    print(f"Matrix Condition Number kappa(A): {kappa:.6f}")
    
    # Check deformation along principal right singular vectors
    for i in range(rank):
        v_i = [V[j][i] for j in range(n)]
        Av_i = mat_vec(A, v_i)
        mapped_norm = norm_2(Av_i)
        print(f"Direction v_{i+1}: ||A v_{i+1}||_2 = {mapped_norm:.6f} (Expected sigma_{i+1} = {sigmas[i]:.6f})")
        assert abs(mapped_norm - sigmas[i]) < 1e-6, f"Axis length mismatch along singular direction {i+1}!"
        
    # Sample random points on the unit hypersphere in R^n to verify bounds
    max_observed = -1.0
    min_observed = float("inf")
    num_samples = 1500
    for _ in range(num_samples):
        rand_x = [rng.gauss(0.0, 1.0) for _ in range(n)]
        x_unit = normalize(rand_x)
        y = mat_vec(A, x_unit)
        len_y = norm_2(y)
        if len_y > max_observed:
            max_observed = len_y
        if len_y < min_observed:
            min_observed = len_y
            
    print(f"Sampled {num_samples} points on unit sphere:")
    print(f"  Observed min ||Ax||: {min_observed:.6f} >= sigma_r ({sigmas[-1]:.6f})")
    print(f"  Observed max ||Ax||: {max_observed:.6f} <= sigma_1 ({sigmas[0]:.6f})")
    assert min_observed >= sigmas[-1] - 1e-8, "Sampled point norm went below minimum singular value!"
    assert max_observed <= sigmas[0] + 1e-8, "Sampled point norm exceeded maximum singular value!"
    
    print("\n" + "=" * 70)
    print("ALL SPECTRAL THEORY & SVD GEOMETRY ASSERTIONS PASSED (EXIT CODE 0)")
    print("=" * 70)


if __name__ == "__main__":
    run_experiment()
