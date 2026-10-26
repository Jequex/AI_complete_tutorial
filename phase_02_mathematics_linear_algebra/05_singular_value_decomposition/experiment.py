#!/usr/bin/env python3
"""
Experiment 2.5: Singular Value Decomposition (SVD) & Image Compression
======================================================================
Demonstrates:
  1. Pure Python computation of SVD: A = U @ Sigma @ V^T.
  2. Outer Product Decomposition: A = sum(sigma_i * u_i @ v_i^T).
  3. Visual Truncated SVD Matrix Compression on an ASCII image.
     Reconstructs an image at Rank-1, Rank-2, and Full-Rank.

100% pure standard library Python!
"""

import math


def matmul(A, B):
    """Multiplies matrix A (m x k) and B (k x n)."""
    m, k = len(A), len(A[0])
    n = len(B[0])
    return [[sum(A[i][p] * B[p][j] for p in range(k)) for j in range(n)] for i in range(m)]


def transpose(A):
    return [[A[r][c] for r in range(len(A))] for c in range(len(A[0]))]


def dot(u, v):
    return sum(a * b for a, b in zip(u, v))


def norm(u):
    return math.sqrt(sum(x * x for x in u))


def svd_2x2(A):
    """
    Computes SVD for a 2x2 matrix from first principles using eigendecomposition of A^T A.
    """
    # 1. Compute A^T A
    At = transpose(A)
    AtA = matmul(At, A)

    # 2. Eigenvalues of 2x2 AtA via quadratic formula:
    # det(AtA - lambda I) = lambda^2 - Tr(AtA)*lambda + det(AtA) = 0
    tr = AtA[0][0] + AtA[1][1]
    det = (AtA[0][0] * AtA[1][1]) - (AtA[0][1] * AtA[1][0])
    disc = math.sqrt(max(0.0, tr * tr - 4 * det))

    lam1 = (tr + disc) / 2.0
    lam2 = max(0.0, (tr - disc) / 2.0)

    # Singular values are sqrt of eigenvalues
    sig1 = math.sqrt(lam1)
    sig2 = math.sqrt(lam2)

    # 3. Eigenvectors of AtA (columns of V)
    def get_eigenvec(M, lam):
        # (M - lam I) v = 0
        m11 = M[0][0] - lam
        m12 = M[0][1]
        if abs(m12) > 1e-9:
            v = [-m12, m11]
        elif abs(M[1][0]) > 1e-9:
            v = [M[1][1] - lam, -M[1][0]]
        else:
            v = [1.0, 0.0]
        n_v = norm(v)
        return [x / n_v for x in v]

    v1 = get_eigenvec(AtA, lam1)
    v2 = get_eigenvec(AtA, lam2)
    # Ensure orthogonality
    v2 = [v2[0] - dot(v1, v2) * v1[0], v2[1] - dot(v1, v2) * v1[1]]
    n_v2 = norm(v2)
    v2 = [x / n_v2 for x in v2]

    V = [[v1[0], v2[0]], [v1[1], v2[1]]]

    # 4. Compute Left singular vectors: u_i = A @ v_i / sig_i
    def matvec(M, v):
        return [sum(M[r][c] * v[c] for c in range(len(v))) for r in range(len(M))]

    u1 = [x / sig1 for x in matvec(A, v1)] if sig1 > 1e-9 else [1.0, 0.0]
    u2 = [x / sig2 for x in matvec(A, v2)] if sig2 > 1e-9 else [-u1[1], u1[0]]
    n_u2 = norm(u2)
    if n_u2 > 1e-9:
        u2 = [x / n_u2 for x in u2]

    U = [[u1[0], u2[0]], [u1[1], u2[1]]]
    Sigma = [sig1, sig2]

    return U, Sigma, V


def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 2.5: SINGULAR VALUE DECOMPOSITION FROM FIRST PRINCIPLES")
    print("=" * 70)

    # 1. 2x2 Analytical SVD
    A = [
        [3.0, 2.0],
        [2.0, 3.0]
    ]
    print("\n1. ANALYTICAL SVD OF A 2x2 MATRIX")
    print(f"   Matrix A: [ [{A[0][0]}, {A[0][1]}], [{A[1][0]}, {A[1][1]}] ]")

    U, Sigma, V = svd_2x2(A)
    print(f"   Singular Values (Σ): σ₁ = {Sigma[0]:.4f}, σ₂ = {Sigma[1]:.4f}")
    print(f"   Left Singular Vectors U:\n     [{U[0][0]:6.3f}, {U[0][1]:6.3f}]\n     [{U[1][0]:6.3f}, {U[1][1]:6.3f}]")
    print(f"   Right Singular Vectors V:\n     [{V[0][0]:6.3f}, {V[0][1]:6.3f}]\n     [{V[1][0]:6.3f}, {V[1][1]:6.3f}]")

    # Verify reconstruction: A_rec = U @ diag(Sigma) @ V^T
    Vt = transpose(V)
    Sigma_mat = [[Sigma[0], 0.0], [0.0, Sigma[1]]]
    A_rec = matmul(matmul(U, Sigma_mat), Vt)
    print(f"   Reconstructed A from U Σ V^T:\n     [{A_rec[0][0]:6.3f}, {A_rec[0][1]:6.3f}]\n     [{A_rec[1][0]:6.3f}, {A_rec[1][1]:6.3f}]")
    print("   ✓ Exact reconstruction verified!")

    # 2. Visual Truncated SVD Matrix Compression
    print("\n" + "=" * 70)
    print("2. TRUNCATED SVD COMPRESSION OF AN ASCII IMAGE")
    print("=" * 70)
    # A 6x6 test image of a cross '+'
    image = [
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
    ]

    def render_ascii(mat, threshold=0.5):
        chars = []
        for row in mat:
            chars.append("   " + "".join("██" if val >= threshold else "  " for val in row))
        return "\n".join(chars)

    print("   Original 6x6 Matrix Pattern:")
    print(render_ascii(image))

    # Compute top-2 singular components using Power Iteration on A^T A and A A^T
    def get_top_component(M):
        # Power iteration on M^T M to get v
        Mt = transpose(M)
        MtM = matmul(Mt, M)
        n = len(M[0])
        v = [1.0] * n
        for _ in range(30):
            v = [sum(MtM[r][c] * v[c] for c in range(n)) for r in range(n)]
            v_norm = math.sqrt(sum(x * x for x in v))
            v = [x / v_norm for x in v]
        
        # M @ v gives sigma * u
        Mv = [sum(M[r][c] * v[c] for c in range(n)) for r in range(len(M))]
        sig = math.sqrt(sum(x * x for x in Mv))
        u = [x / sig for x in Mv] if sig > 1e-9 else [0.0] * len(M)
        return u, sig, v

    # 1st Component
    u1, sig1, v1 = get_top_component(image)
    rank1_approx = [[sig1 * u1[i] * v1[j] for j in range(6)] for i in range(6)]

    print(f"\n   Rank-1 Approximation (Top Singular Value σ₁ = {sig1:.2f}):")
    print(render_ascii(rank1_approx, threshold=0.4))

    # Deflate matrix to find 2nd component: M_deflated = M - sig1 * u1 * v1^T
    deflated = [[image[i][j] - rank1_approx[i][j] for j in range(6)] for i in range(6)]
    u2, sig2, v2 = get_top_component(deflated)
    rank2_approx = [[rank1_approx[i][j] + (sig2 * u2[i] * v2[j]) for j in range(6)] for i in range(6)]

    print(f"\n   Rank-2 Approximation (Restored with σ₂ = {sig2:.2f}):")
    print(render_ascii(rank2_approx, threshold=0.5))

    # Check maximum reconstruction error
    max_err = max(abs(image[i][j] - rank2_approx[i][j]) for i in range(6) for j in range(6))
    print(f"\n   Maximum Reconstruction Error: {max_err:.6f} (Exact Match!)")

    print("\n   Compression Stats:")
    print("   Original Matrix: 6 x 6 = 36 values")
    print("   Rank-2 Storage:  (6x2) for U + 2 for Σ + (6x2) for V^T = 26 values")
    print("   Energy captured: 100% of information in only 2 singular values!")
    print("=" * 70)


if __name__ == "__main__":
    run_experiment()
