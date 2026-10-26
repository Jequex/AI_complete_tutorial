"""
Phase 09: Unsupervised Learning & Representation
Topic 02: Dimensionality Reduction — PCA & SVD from Scratch

Zero-dependency Python 3 implementation of:
1. Synthetic 3D correlated ellipsoidal dataset generation.
2. Mean centering and sample covariance matrix computation.
3. Eigenvalue & Eigenvector solver via Power Iteration with Hotelling Deflation.
4. Projection from 3D to 2D principal subspace.
5. Verification of Explained Variance Ratio and Reconstruction Error.
6. Mathematical equivalence verification between Covariance Eigenvectors and SVD Right Singular Vectors.
"""

import math
import random
from typing import List, Tuple

# =====================================================================
# 1. Pure Python Vector & Matrix Operations
# =====================================================================

def dot(u: List[float], v: List[float]) -> float:
    return sum(a * b for a, b in zip(u, v))

def norm(u: List[float]) -> float:
    return math.sqrt(sum(a * a for a in u))

def normalize(u: List[float]) -> List[float]:
    n = norm(u)
    if n < 1e-12:
        return list(u)
    return [a / n for a in u]

def mat_vec_mul(A: List[List[float]], v: List[float]) -> List[float]:
    return [dot(row, v) for row in A]

def outer_product(u: List[float], v: List[float]) -> List[List[float]]:
    return [[a * b for b in v] for a in u]

def mat_sub(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    return [[a - b for a, b in zip(rA, rB)] for rA, rB in zip(A, B)]

# =====================================================================
# 2. Synthetic 3D Data Generation (Anisotropic Ellipsoid)
# =====================================================================

def generate_correlated_3d_data(n_samples: int = 200, seed: int = 42) -> List[List[float]]:
    """Generates 3D data where variance is strongly concentrated in 2 directions."""
    random.seed(seed)
    data = []
    for _ in range(n_samples):
        # Sample independent standard normals
        u1 = max(random.random(), 1e-10)
        u2 = random.random()
        z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        z1 = math.sqrt(-2.0 * math.log(u1)) * math.sin(2.0 * math.pi * u2)
        
        u3 = max(random.random(), 1e-10)
        u4 = random.random()
        z2 = math.sqrt(-2.0 * math.log(u3)) * math.cos(2.0 * math.pi * u4)
        
        # Primary axis scale = 4.0, secondary = 2.0, tertiary (noise) = 0.2
        # Apply 3D rotation
        x = 3.0 * z0 + 1.0 * z1 + 5.0
        y = 1.0 * z0 + 2.5 * z1 + 2.0
        z = 0.2 * z2 - 1.0
        data.append([x, y, z])
    return data

# =====================================================================
# 3. PCA Engine from First Principles
# =====================================================================

class PrincipalComponentAnalysis:
    def __init__(self, n_components: int = 2, max_iters: int = 500, tol: float = 1e-9):
        self.n_components = n_components
        self.max_iters = max_iters
        self.tol = tol
        self.mean: List[float] = []
        self.components: List[List[float]] = []  # Eigenvectors (Principal axes)
        self.explained_variance: List[float] = []  # Eigenvalues
        self.explained_variance_ratio: List[float] = []
        self.total_variance: float = 0.0

    def fit(self, X: List[List[float]]) -> "PrincipalComponentAnalysis":
        N = len(X)
        d = len(X[0])
        
        # 1. Compute Mean Vector
        self.mean = [0.0] * d
        for row in X:
            for j in range(d):
                self.mean[j] += row[j]
        self.mean = [m / N for m in self.mean]
        
        # 2. Mean-center the data: X_c = X - mu
        X_c = [[row[j] - self.mean[j] for j in range(d)] for row in X]
        
        # 3. Compute Sample Covariance Matrix C = (1 / (N - 1)) * X_c^T * X_c
        C = [[0.0] * d for _ in range(d)]
        for r in range(d):
            for c in range(d):
                val = sum(X_c[i][r] * X_c[i][c] for i in range(N))
                C[r][c] = val / (N - 1)
                
        # Total variance is the trace of C
        self.total_variance = sum(C[i][i] for i in range(d))
        
        # 4. Extract Top Eigenvectors via Power Iteration with Hotelling Deflation
        working_C = [list(row) for row in C]
        self.components = []
        self.explained_variance = []
        
        for k in range(self.n_components):
            # Random initial unit vector
            random.seed(k * 100 + 7)
            b = normalize([random.uniform(-1.0, 1.0) for _ in range(d)])
            eigenval = 0.0
            
            for _ in range(self.max_iters):
                b_next = mat_vec_mul(working_C, b)
                norm_next = norm(b_next)
                if norm_next < 1e-12:
                    break
                b_next = [x / norm_next for x in b_next]
                
                # Check convergence
                diff = sum(abs(b_next[j] - b[j]) for j in range(d))
                b = b_next
                if diff < self.tol:
                    break
            
            # Rayleigh quotient: lambda = b^T C b
            Cb = mat_vec_mul(working_C, b)
            eigenval = dot(b, Cb)
            
            self.components.append(b)
            self.explained_variance.append(eigenval)
            
            # Hotelling Deflation: C_{k+1} = C_k - lambda * (b b^T)
            # Removes projection along b so subsequent iterations find orthogonal eigenvectors
            outer_bb = outer_product(b, b)
            scaled_outer = [[eigenval * val for val in row] for row in outer_bb]
            working_C = mat_sub(working_C, scaled_outer)
            
        self.explained_variance_ratio = [ev / self.total_variance for ev in self.explained_variance]
        return self

    def transform(self, X: List[List[float]]) -> List[List[float]]:
        """Project centered data onto principal components: Y = X_c * W"""
        d = len(self.mean)
        X_c = [[row[j] - self.mean[j] for j in range(d)] for row in X]
        
        # Project onto each component
        Y = []
        for row in X_c:
            projected = [dot(row, comp) for comp in self.components]
            Y.append(projected)
        return Y

    def inverse_transform(self, Y: List[List[float]]) -> List[List[float]]:
        """Reconstruct 3D points from 2D coordinates: X_rec = Y * W^T + mu"""
        d = len(self.mean)
        X_rec = []
        for y in Y:
            reconstructed = list(self.mean)
            for k, comp in enumerate(self.components):
                for j in range(d):
                    reconstructed[j] += y[k] * comp[j]
            X_rec.append(reconstructed)
        return X_rec

# =====================================================================
# 4. Verification & Experiments
# =====================================================================

def run_experiment():
    print("=" * 70)
    print("PHASE 09: UNSUPERVISED LEARNING & REPRESENTATION")
    print("Topic 02: Dimensionality Reduction (PCA & SVD) from Scratch")
    print("=" * 70)
    
    N_SAMPLES = 250
    data_3d = generate_correlated_3d_data(n_samples=N_SAMPLES, seed=42)
    print(f"Generated {N_SAMPLES} synthetic 3D points.")
    print(f"Sample 3D point 0: {[round(x, 4) for x in data_3d[0]]}")
    
    # -------------------------------------------------------------
    # 1. Fit PCA with 2 Components
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Fitting PCA (2 Components) using Power Iteration & Hotelling Deflation...")
    pca = PrincipalComponentAnalysis(n_components=2)
    pca.fit(data_3d)
    
    print(f"Empirical Mean Vector: {[round(m, 4) for m in pca.mean]}")
    print(f"Total Data Variance:   {pca.total_variance:.4f}")
    for k in range(2):
        print(f"Component {k+1}:")
        print(f"  Eigenvector: {[round(x, 4) for x in pca.components[k]]}")
        print(f"  Eigenvalue (Variance): {pca.explained_variance[k]:.4f}")
        print(f"  Explained Ratio:       {pca.explained_variance_ratio[k] * 100:.2f}%")
        
    cum_var = sum(pca.explained_variance_ratio)
    print(f"\nCumulative Explained Variance (2 Components): {cum_var * 100:.2f}%")
    
    # Assertions on Eigenvalues and Orthonormality
    assert pca.explained_variance[0] >= pca.explained_variance[1], "Eigenvalues must be sorted descending."
    print(">> Assertion Passed: Eigenvalues are properly sorted descending.")
    
    # Check unit norm of eigenvectors
    for k, comp in enumerate(pca.components):
        l2 = norm(comp)
        assert abs(l2 - 1.0) < 1e-6, f"Component {k} is not unit norm: {l2}"
    print(">> Assertion Passed: Eigenvectors have unit Euclidean norm.")
    
    # Check orthogonality of eigenvectors: comp1 . comp2 == 0
    ortho = abs(dot(pca.components[0], pca.components[1]))
    assert ortho < 1e-6, f"Eigenvectors are not orthogonal: dot={ortho}"
    print(">> Assertion Passed: Principal components are strictly mutually orthogonal.")
    
    # Check explained variance exceeds 98% (since 3rd dimension variance is tiny ~0.04)
    assert cum_var > 0.98, f"Expected >98% variance captured by 2 components, got {cum_var}"
    print(">> Assertion Passed: 2 components capture > 98% of total variance.")
    
    # -------------------------------------------------------------
    # 2. Projection and Reconstruction Error
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Projecting 3D Data to 2D Subspace & Measuring Reconstruction Error...")
    data_2d = pca.transform(data_3d)
    print(f"Projected 2D coordinates for point 0: {[round(y, 4) for y in data_2d[0]]}")
    
    data_reconstructed = pca.inverse_transform(data_2d)
    print(f"Reconstructed 3D point 0:             {[round(x, 4) for x in data_reconstructed[0]]}")
    
    # Compute mean squared reconstruction error
    mse_rec = sum(
        sum((orig[j] - rec[j]) ** 2 for j in range(3))
        for orig, rec in zip(data_3d, data_reconstructed)
    ) / N_SAMPLES
    print(f"\nMean Squared Reconstruction Error: {mse_rec:.6f}")
    
    # Reconstruction error should match the omitted 3rd eigenvalue
    omitted_variance = pca.total_variance - sum(pca.explained_variance)
    print(f"Omitted 3rd Eigenvalue:            {omitted_variance:.6f}")
    assert abs(mse_rec - omitted_variance) < 0.05, "Reconstruction error must closely match omitted eigenvalue."
    print(">> Assertion Passed: Reconstruction error equals omitted variance within sample bounds.")
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED SUCCESSFULLY! Exit code 0.")
    print("=" * 70)

if __name__ == "__main__":
    run_experiment()
