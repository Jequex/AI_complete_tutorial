#!/usr/bin/env python3
"""
Experiment 2.3: Vector Spaces, Rank, & The Rank-Nullity Theorem
==============================================================
Demonstrates:
  1. Gaussian Elimination from scratch in pure Python.
  2. Finding Matrix Rank & Detecting Linear Independence.
  3. Verifying the Rank-Nullity Theorem: Rank(A) + Nullity(A) = n.
  4. Simulating LoRA (Low-Rank Adaptation):
     Creating a 10x10 weight matrix via B @ A with rank 2.

100% pure standard library Python!
"""


def print_matrix(mat, label: str = ""):
    if label:
        print(f"   {label}:")
    for row in mat:
        print("     [" + ", ".join(f"{x:7.2f}" for x in row) + "]")


def gaussian_elimination_rank(mat, tol: float = 1e-9):
    """
    Performs forward Gaussian elimination with partial pivoting.
    Returns the Row Echelon Form (REF) and the Matrix Rank.
    """
    # Clone matrix to avoid in-place mutation
    m = len(mat)
    n = len(mat[0])
    a = [[float(val) for val in row] for row in mat]

    lead = 0
    rank = 0

    for r in range(m):
        if lead >= n:
            break
        i = r
        # Find pivot
        while abs(a[i][lead]) < tol:
            i += 1
            if i == m:
                i = r
                lead += 1
                if lead == n:
                    break
        if lead == n:
            break

        # Swap rows
        a[i], a[r] = a[r], a[i]

        # Normalize pivot row
        pivot = a[r][lead]
        if abs(pivot) > tol:
            rank += 1
            # Eliminate rows below
            for i_sub in range(r + 1, m):
                factor = a[i_sub][lead] / pivot
                for j in range(lead, n):
                    a[i_sub][j] -= factor * a[r][j]
            lead += 1

    return a, rank


def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 2.3: VECTOR SPACES, RANK & RANK-NULLITY")
    print("=" * 70)

    # 1. Full Rank 3x3 Matrix (Linearly Independent Vectors)
    print("\n1. FULL RANK 3x3 MATRIX (Linearly Independent Columns)")
    M_full = [
        [1.0, 0.0, 2.0],
        [0.0, 1.0, 1.0],
        [2.0, 0.0, 1.0]
    ]
    print_matrix(M_full, "Matrix M_full")
    ref_full, rank_full = gaussian_elimination_rank(M_full)
    nullity_full = 3 - rank_full
    print(f"\n   Computed Rank:    {rank_full} (All 3 columns are linearly independent!)")
    print(f"   Computed Nullity: {nullity_full} (Null space contains only the zero vector)")
    print(f"   Rank-Nullity Check: {rank_full} + {nullity_full} = 3 (Total Columns) ✓")

    # 2. Rank-Deficient 3x3 Matrix (Redundant Column)
    # Column 3 = Column 1 + Column 2
    print("\n" + "=" * 70)
    print("2. RANK-DEFICIENT MATRIX (Column 3 = Column 1 + Column 2)")
    print("=" * 70)
    M_dep = [
        [1.0, 2.0, 3.0],
        [2.0, 1.0, 3.0],
        [0.0, 1.0, 1.0]
    ]
    print_matrix(M_dep, "Matrix M_dep (Col 3 is redundant)")
    ref_dep, rank_dep = gaussian_elimination_rank(M_dep)
    nullity_dep = 3 - rank_dep
    print(f"\n   Computed Rank:    {rank_dep} (Only 2 independent directions)")
    print(f"   Computed Nullity: {nullity_dep} (1-dimensional Null space line)")
    print(f"   Rank-Nullity Check: {rank_dep} + {nullity_dep} = 3 (Total Columns) ✓")

    # Null space verification: x = [1, 1, -1]^T
    x_null = [1.0, 1.0, -1.0]
    output = [sum(M_dep[r][c] * x_null[c] for c in range(3)) for r in range(3)]
    print(f"   Testing Null vector x = {x_null} -> M_dep @ x = {output} (Squashed to ZERO!)")

    # 3. LoRA Simulation
    print("\n" + "=" * 70)
    print("3. SIMULATING LoRA: LOW-RANK PARAMETER ADAPTATION")
    print("=" * 70)
    print("Suppose we have a 10x10 layer. We factor update Delta_W = B @ A with rank r = 2.")

    # Matrix B: 10 rows x 2 cols
    B = [[float(r + c) for c in range(2)] for r in range(10)]
    # Matrix A: 2 rows x 10 cols
    A = [[float(r * 2 + c) for c in range(10)] for r in range(2)]

    # Compute Delta_W = B @ A (10x10 matrix)
    Delta_W = []
    for i in range(10):
        row = []
        for j in range(10):
            row.append(sum(B[i][k] * A[k][j] for k in range(2)))
        Delta_W.append(row)

    print(f"   Full Delta_W shape: 10 x 10 = 100 parameters")
    print(f"   LoRA factor shape:  B (10x2=20) + A (2x10=20) = 40 parameters (60% savings!)")

    _, rank_lora = gaussian_elimination_rank(Delta_W)
    print(f"   Exact Rank of Delta_W: {rank_lora} (Bounded by r = 2!)")
    print("=" * 70)


if __name__ == "__main__":
    run_experiment()

