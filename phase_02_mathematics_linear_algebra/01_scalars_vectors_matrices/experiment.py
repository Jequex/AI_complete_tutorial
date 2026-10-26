#!/usr/bin/env python3
"""
Experiment 2.1: Scalars, Vectors, and Matrices From Scratch
===========================================================
Demonstrates:
  1. Pure Python implementation of Vector and Matrix (Zero external dependencies).
  2. Norms: L1 (Manhattan) and L2 (Euclidean) length calculations.
  3. Geometric Dot Products and Cosine Similarity.
  4. Real-world AI application: Word Embeddings and Vector Analogies!
     ("king" - "man" + "woman" ≈ "queen")
  5. Matrix-Matrix and Matrix-Vector multiplication.

Run this script:
    python3 experiment.py
"""

import math


class Vector:
    """A mathematical 1D column vector built from first principles."""

    def __init__(self, elements):
        self.data = [float(x) for x in elements]
        self.dim = len(self.data)

    def __repr__(self):
        return f"Vector([{', '.join(f'{x:.2f}' for x in self.data)}])"

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.dim

    def __add__(self, other):
        if self.dim != other.dim:
            raise ValueError(f"Dimension mismatch: {self.dim} vs {other.dim}")
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        if self.dim != other.dim:
            raise ValueError(f"Dimension mismatch: {self.dim} vs {other.dim}")
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        """Scalar multiplication: c * v"""
        return Vector([x * scalar for x in self.data])

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def dot(self, other):
        """Dot product: sum(x_i * y_i)"""
        if self.dim != other.dim:
            raise ValueError(f"Dimension mismatch in dot product: {self.dim} vs {other.dim}")
        return sum(a * b for a, b in zip(self.data, other.data))

    def norm_l2(self) -> float:
        """Euclidean Norm: sqrt(sum(x_i^2))"""
        return math.sqrt(sum(x * x for x in self.data))

    def norm_l1(self) -> float:
        """Manhattan Norm: sum(|x_i|)"""
        return sum(abs(x) for x in self.data)

    def cosine_similarity(self, other) -> float:
        """cos(theta) = (x . y) / (||x|| * ||y||)"""
        denom = self.norm_l2() * other.norm_l2()
        if denom == 0.0:
            return 0.0
        return self.dot(other) / denom


class Matrix:
    """A mathematical 2D matrix built from first principles."""

    def __init__(self, rows):
        self.rows = [[float(val) for val in r] for r in rows]
        self.num_rows = len(rows)
        self.num_cols = len(rows[0]) if rows else 0

    def __repr__(self):
        lines = []
        for r in self.rows:
            lines.append("  [" + ", ".join(f"{x:6.2f}" for x in r) + "]")
        return "Matrix([\n" + "\n".join(lines) + "\n])"

    def transpose(self):
        """Flips rows into columns: A^T"""
        transposed_data = [
            [self.rows[r][c] for r in range(self.num_rows)]
            for c in range(self.num_cols)
        ]
        return Matrix(transposed_data)

    def matmul(self, other):
        """Matrix multiplication: (m x k) @ (k x n) -> (m x n)"""
        if self.num_cols != other.num_rows:
            raise ValueError(
                f"Cannot multiply ({self.num_rows}x{self.num_cols}) and ({other.num_rows}x{other.num_cols}). "
                f"Inner dimensions must match!"
            )
        result = []
        for i in range(self.num_rows):
            new_row = []
            for j in range(other.num_cols):
                cell_sum = sum(self.rows[i][k] * other.rows[k][j] for k in range(self.num_cols))
                new_row.append(cell_sum)
            result.append(new_row)
        return Matrix(result)

    def matvec(self, v: Vector) -> Vector:
        """Matrix-vector product: A @ v"""
        if self.num_cols != v.dim:
            raise ValueError(f"Matrix cols {self.num_cols} != Vector dim {v.dim}")
        return Vector([sum(self.rows[r][c] * v[c] for c in range(self.num_cols)) for r in range(self.num_rows)])


def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 2.1: LINEAR ALGEBRA ENGINE FROM FIRST PRINCIPLES")
    print("=" * 70)

    # 1. Basic Vector Operations
    print("\n1. VECTOR BASICS & NORMS")
    v1 = Vector([3.0, 4.0])
    v2 = Vector([1.0, 2.0])

    print(f"   Vector 1: {v1}")
    print(f"   Vector 2: {v2}")
    print(f"   v1 + v2:  {v1 + v2}")
    print(f"   v1 * 2:   {v1 * 2.0}")
    print(f"   L2 Norm of v1 (Euclidean):  {v1.norm_l2():.2f}  (sqrt(3^2 + 4^2) = 5.0)")
    print(f"   L1 Norm of v1 (Manhattan):  {v1.norm_l1():.2f}  (|3| + |4| = 7.0)")
    print(f"   Dot product (v1 . v2):      {v1.dot(v2):.2f}  ((3*1) + (4*2) = 11.0)")

    # 2. Real-World AI Embeddings Demo
    print("\n" + "=" * 70)
    print("2. REAL-WORLD AI APPLICATION: WORD EMBEDDINGS & COSINE SIMILARITY")
    print("=" * 70)
    print("Imagine a tiny 3D semantic embedding space:")
    print("Dimensions: [Royalty, Femininity, Edibility]")

    embeddings = {
        "king":   Vector([0.95, 0.05, 0.00]),
        "queen":  Vector([0.92, 0.96, 0.00]),
        "man":    Vector([0.05, 0.02, 0.00]),
        "woman":  Vector([0.04, 0.97, 0.00]),
        "apple":  Vector([0.00, 0.00, 0.98]),
        "banana": Vector([0.00, 0.00, 0.95]),
    }

    for word, vec in embeddings.items():
        print(f"   {word:7s}: {vec}")

    print("\n   Cosine Similarities:")
    pairs = [
        ("king", "queen"),
        ("king", "man"),
        ("apple", "banana"),
        ("queen", "apple"),
        ("man", "banana"),
    ]
    for w1, w2 in pairs:
        sim = embeddings[w1].cosine_similarity(embeddings[w2])
        print(f"   cos({w1:6s}, {w2:6s}) = {sim:6.4f}")

    print("\n   Vector Analogy Test:")
    print("   Evaluating: Vector('king') - Vector('man') + Vector('woman')")
    analogy = embeddings["king"] - embeddings["man"] + embeddings["woman"]
    print(f"   Resulting Vector: {analogy}")

    # Compare analogy vector against all words
    print("   Distances to candidate words in vocabulary:")
    for word, vec in embeddings.items():
        sim = analogy.cosine_similarity(vec)
        print(f"   Similarity to '{word:6s}': {sim:6.4f}")

    best_match = max(embeddings.keys(), key=lambda w: analogy.cosine_similarity(embeddings[w]))
    print(f"   ⭐ Best semantic match: '{best_match.upper()}'!")

    # 3. Matrix Multiplication
    print("\n" + "=" * 70)
    print("3. MATRIX MULTIPLICATION & TRANSFORMATIONS")
    print("=" * 70)
    A = Matrix([
        [1.0, 2.0],
        [3.0, 4.0]
    ])
    B = Matrix([
        [5.0, 6.0],
        [7.0, 8.0]
    ])
    print("Matrix A:")
    print(A)
    print("Matrix B:")
    print(B)

    C = A.matmul(B)
    print("\nProduct C = A @ B:")
    print(C)

    print("\nTranspose of A (A^T):")
    print(A.transpose())

    print("\nMatrix-Vector Product A @ v1 (Transforming an arrow):")
    transformed_v = A.matvec(v1)
    print(f"Original v1:    {v1}")
    print(f"Transformed Av: {transformed_v}")
    print("=" * 70)


if __name__ == "__main__":
    run_experiment()

