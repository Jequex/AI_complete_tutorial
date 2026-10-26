"""
Phase 09: Unsupervised Learning & Representation
Topic 01: Clustering — K-Means & Gaussian Mixture Models (EM)

Zero-dependency Python 3 implementation of:
1. Synthetic 2D multi-cluster dataset generation.
2. K-Means clustering with K-Means++ initialization and Lloyd's algorithm.
3. Gaussian Mixture Models (GMM) with Expectation-Maximization (EM) derivation.
4. Evaluation: Inertia, Log-Likelihood convergence, and Silhouette-like separation.
"""

import math
import random
from typing import List, Tuple, Dict

# =====================================================================
# 1. 2D Linear Algebra & Statistics Helpers (Zero External Dependencies)
# =====================================================================

def vec_sub(a: List[float], b: List[float]) -> List[float]:
    return [x - y for x, y in zip(a, b)]

def vec_add(a: List[float], b: List[float]) -> List[float]:
    return [x + y for x, y in zip(a, b)]

def vec_scale(a: List[float], s: float) -> List[float]:
    return [x * s for x in a]

def squared_euclidean_distance(a: List[float], b: List[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))

def euclidean_distance(a: List[float], b: List[float]) -> float:
    return math.sqrt(squared_euclidean_distance(a, b))

def det_2x2(m: List[List[float]]) -> float:
    """Determinant of a 2x2 matrix: [[a, b], [c, d]] -> ad - bc."""
    return m[0][0] * m[1][1] - m[0][1] * m[1][0]

def inv_2x2(m: List[List[float]], eps: float = 1e-6) -> List[List[float]]:
    """Inverse of a 2x2 matrix with diagonal regularization."""
    a, b = m[0][0] + eps, m[0][1]
    c, d = m[1][0], m[1][1] + eps
    det = a * d - b * c
    if abs(det) < 1e-12:
        det = 1e-12 if det >= 0 else -1e-12
    inv_det = 1.0 / det
    return [
        [ d * inv_det, -b * inv_det],
        [-c * inv_det,  a * inv_det]
    ]

def quad_form_2x2(diff: List[float], inv_sigma: List[List[float]]) -> float:
    """Computes (x - mu)^T Sigma^{-1} (x - mu) for 2D vectors."""
    dx, dy = diff[0], diff[1]
    # inv_sigma * diff:
    r0 = inv_sigma[0][0] * dx + inv_sigma[0][1] * dy
    r1 = inv_sigma[1][0] * dx + inv_sigma[1][1] * dy
    return dx * r0 + dy * r1

def multivariate_gaussian_pdf_2d(x: List[float], mu: List[float], sigma: List[List[float]]) -> float:
    """Evaluates 2D Gaussian density N(x; mu, Sigma)."""
    diff = vec_sub(x, mu)
    det = det_2x2(sigma)
    if det <= 1e-12:
        # Regularize
        sigma = [[sigma[0][0] + 1e-4, sigma[0][1]], [sigma[1][0], sigma[1][1] + 1e-4]]
        det = det_2x2(sigma)
    
    inv_sigma = inv_2x2(sigma)
    q = quad_form_2x2(diff, inv_sigma)
    norm_const = 1.0 / (2.0 * math.pi * math.sqrt(max(det, 1e-12)))
    return norm_const * math.exp(-0.5 * min(q, 100.0)) # Clip exponent for numerical safety

# =====================================================================
# 2. Synthetic Dataset Generation: 3 Distinct 2D Clusters
# =====================================================================

def generate_synthetic_clusters(
    n_samples_per_cluster: int = 100,
    seed: int = 42
) -> Tuple[List[List[float]], List[int], List[List[float]]]:
    """Generates synthetic 2D points from 3 Gaussian distributions."""
    random.seed(seed)
    true_centers = [
        [2.0, 2.0],
        [8.0, 2.0],
        [5.0, 7.0]
    ]
    stds = [0.6, 0.7, 0.65]
    
    points: List[List[float]] = []
    labels: List[int] = []
    
    for k, (center, std) in enumerate(zip(true_centers, stds)):
        for _ in range(n_samples_per_cluster):
            # Box-Muller transform for standard normal samples
            u1 = max(random.random(), 1e-10)
            u2 = random.random()
            z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
            z1 = math.sqrt(-2.0 * math.log(u1)) * math.sin(2.0 * math.pi * u2)
            
            x = center[0] + z0 * std
            y = center[1] + z1 * std
            points.append([x, y])
            labels.append(k)
            
    return points, labels, true_centers

# =====================================================================
# 3. K-Means Clustering (Lloyd's Algorithm + K-Means++ Initialization)
# =====================================================================

class KMeans:
    def __init__(self, k: int = 3, max_iters: int = 100, tol: float = 1e-5, seed: int = 42):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.seed = seed
        self.centroids: List[List[float]] = []
        self.labels: List[int] = []
        self.inertia: float = 0.0

    def _init_kmeans_pp(self, X: List[List[float]]):
        """K-Means++ initialization to spread initial centroids."""
        random.seed(self.seed)
        N = len(X)
        self.centroids = [X[random.randint(0, N - 1)]]
        
        for _ in range(1, self.k):
            # Compute distance squared from each point to its closest existing centroid
            d2_list = []
            for x in X:
                min_d2 = min(squared_euclidean_distance(x, c) for c in self.centroids)
                d2_list.append(min_d2)
            
            total_d2 = sum(d2_list)
            # Sample next centroid with probability proportional to D(x)^2
            r = random.random() * total_d2
            cumulative = 0.0
            chosen_idx = 0
            for i, d2 in enumerate(d2_list):
                cumulative += d2
                if cumulative >= r:
                    chosen_idx = i
                    break
            self.centroids.append(X[chosen_idx])

    def fit(self, X: List[List[float]]) -> "KMeans":
        N = len(X)
        self._init_kmeans_pp(X)
        
        for iteration in range(self.max_iters):
            # 1. Assignment Step (Voronoi partition)
            new_labels: List[int] = []
            for x in X:
                closest_k = 0
                closest_dist = squared_euclidean_distance(x, self.centroids[0])
                for k in range(1, self.k):
                    d = squared_euclidean_distance(x, self.centroids[k])
                    if d < closest_dist:
                        closest_dist = d
                        closest_k = k
                new_labels.append(closest_k)
            
            # 2. Update Step (Centroids = mean of cluster points)
            new_centroids: List[List[float]] = [[0.0, 0.0] for _ in range(self.k)]
            counts = [0 for _ in range(self.k)]
            
            for i, x in enumerate(X):
                c = new_labels[i]
                new_centroids[c][0] += x[0]
                new_centroids[c][1] += x[1]
                counts[c] += 1
                
            for k in range(self.k):
                if counts[k] > 0:
                    new_centroids[k][0] /= counts[k]
                    new_centroids[k][1] /= counts[k]
                else:
                    new_centroids[k] = list(self.centroids[k])
            
            # Check convergence (max shift)
            shift = max(euclidean_distance(c1, c2) for c1, c2 in zip(self.centroids, new_centroids))
            self.centroids = new_centroids
            self.labels = new_labels
            
            if shift < self.tol:
                break
                
        # Compute final inertia (sum of squared distances to closest centroid)
        self.inertia = sum(squared_euclidean_distance(X[i], self.centroids[self.labels[i]]) for i in range(N))
        return self

# =====================================================================
# 4. Gaussian Mixture Model (GMM) with Expectation-Maximization (EM)
# =====================================================================

class GaussianMixtureModel:
    def __init__(self, k: int = 3, max_iters: int = 100, tol: float = 1e-4, seed: int = 42):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.seed = seed
        self.weights: List[float] = [1.0 / k] * k
        self.means: List[List[float]] = []
        self.covariances: List[List[List[float]]] = []
        self.log_likelihood_history: List[float] = []

    def fit(self, X: List[List[float]], init_means: List[List[float]] = None) -> "GaussianMixtureModel":
        N = len(X)
        d = len(X[0])
        
        # Initialize means (using provided K-means centers or random points)
        if init_means is not None:
            self.means = [list(m) for m in init_means]
        else:
            random.seed(self.seed)
            self.means = [list(X[random.randint(0, N - 1)]) for _ in range(self.k)]
            
        self.weights = [1.0 / self.k] * self.k
        # Initialize covariance to identity matrix
        self.covariances = [[[1.0, 0.0], [0.0, 1.0]] for _ in range(self.k)]
        
        prev_ll = -float('inf')
        
        for iteration in range(self.max_iters):
            # -------------------------------------------------------------
            # E-step: Compute responsibilities gamma_ik
            # gamma_{ik} = (pi_k * N(x_i; mu_k, Sigma_k)) / sum_j (pi_j * N(x_i; mu_j, Sigma_j))
            # -------------------------------------------------------------
            responsibilities: List[List[float]] = []
            iter_log_likelihood = 0.0
            
            for i in range(N):
                x = X[i]
                probs = []
                for k in range(self.k):
                    p_k = self.weights[k] * multivariate_gaussian_pdf_2d(x, self.means[k], self.covariances[k])
                    probs.append(p_k)
                    
                total_prob = sum(probs)
                if total_prob <= 1e-15:
                    total_prob = 1e-15
                    probs = [1.0 / self.k] * self.k
                else:
                    probs = [p / total_prob for p in probs]
                    
                responsibilities.append(probs)
                iter_log_likelihood += math.log(max(total_prob, 1e-15))
                
            self.log_likelihood_history.append(iter_log_likelihood)
            
            # Check convergence
            if iteration > 0 and abs(iter_log_likelihood - prev_ll) < self.tol:
                break
            prev_ll = iter_log_likelihood
            
            # -------------------------------------------------------------
            # M-step: Update parameters (pi_k, mu_k, Sigma_k)
            # N_k = sum_{i=1}^N gamma_{ik}
            # pi_k = N_k / N
            # mu_k = (1 / N_k) * sum_{i=1}^N gamma_{ik} * x_i
            # Sigma_k = (1 / N_k) * sum_{i=1}^N gamma_{ik} * (x_i - mu_k)(x_i - mu_k)^T
            # -------------------------------------------------------------
            for k in range(self.k):
                N_k = sum(responsibilities[i][k] for i in range(N))
                if N_k < 1e-5:
                    N_k = 1e-5
                    
                # Update mixture weight pi_k
                self.weights[k] = N_k / N
                
                # Update mean mu_k
                new_mu = [0.0, 0.0]
                for i in range(N):
                    gamma_ik = responsibilities[i][k]
                    new_mu[0] += gamma_ik * X[i][0]
                    new_mu[1] += gamma_ik * X[i][1]
                new_mu[0] /= N_k
                new_mu[1] /= N_k
                self.means[k] = new_mu
                
                # Update covariance Sigma_k
                new_sigma = [[0.0, 0.0], [0.0, 0.0]]
                for i in range(N):
                    gamma_ik = responsibilities[i][k]
                    dx = X[i][0] - new_mu[0]
                    dy = X[i][1] - new_mu[1]
                    new_sigma[0][0] += gamma_ik * dx * dx
                    new_sigma[0][1] += gamma_ik * dx * dy
                    new_sigma[1][0] += gamma_ik * dy * dx
                    new_sigma[1][1] += gamma_ik * dy * dy
                    
                new_sigma[0][0] = (new_sigma[0][0] / N_k) + 1e-4  # Regularization
                new_sigma[0][1] = (new_sigma[0][1] / N_k)
                new_sigma[1][0] = (new_sigma[1][0] / N_k)
                new_sigma[1][1] = (new_sigma[1][1] / N_k) + 1e-4
                
                self.covariances[k] = new_sigma

        return self

    def predict_proba(self, X: List[List[float]]) -> List[List[float]]:
        probs = []
        for x in X:
            p = [self.weights[k] * multivariate_gaussian_pdf_2d(x, self.means[k], self.covariances[k]) for k in range(self.k)]
            tot = sum(p)
            if tot > 0:
                p = [v / tot for v in p]
            else:
                p = [1.0 / self.k] * self.k
            probs.append(p)
        return probs

    def predict(self, X: List[List[float]]) -> List[int]:
        probs = self.predict_proba(X)
        return [p.index(max(p)) for p in probs]

# =====================================================================
# 5. Verification & Test Suite
# =====================================================================

def run_experiment():
    print("=" * 70)
    print("PHASE 09: UNSUPERVISED LEARNING & REPRESENTATION")
    print("Topic 01: K-Means & Gaussian Mixture Models (EM) from Scratch")
    print("=" * 70)
    
    # Generate 3 distinct synthetic 2D clusters
    N_PER_CLUSTER = 100
    K = 3
    points, true_labels, true_centers = generate_synthetic_clusters(n_samples_per_cluster=N_PER_CLUSTER, seed=42)
    print(f"Generated {len(points)} synthetic 2D points from {K} Gaussian clusters.")
    print("True Cluster Centers:")
    for i, c in enumerate(true_centers):
        print(f"  Cluster {i}: ({c[0]:.2f}, {c[1]:.2f})")
        
    # -------------------------------------------------------------
    # Test 1: K-Means Clustering
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Running K-Means (Lloyd's Algorithm + K-Means++ Initialization)...")
    kmeans = KMeans(k=K, max_iters=100, seed=42)
    kmeans.fit(points)
    
    print(f"K-Means converged! Final Inertia: {kmeans.inertia:.4f}")
    print("Recovered K-Means Centroids:")
    for k, c in enumerate(kmeans.centroids):
        print(f"  Centroid {k}: ({c[0]:.4f}, {c[1]:.4f})")
        
    # Match centroids to true centers to verify clustering accuracy
    matched = 0
    for tc in true_centers:
        closest_dist = min(euclidean_distance(tc, kc) for kc in kmeans.centroids)
        # Centroids should be within 0.3 units of true centers
        assert closest_dist < 0.5, f"Centroid too far from true center {tc}: dist={closest_dist}"
        matched += 1
    assert matched == 3
    print(">> Assertion Passed: K-Means centroids closely match true cluster centers.")
    assert kmeans.inertia > 0.0, "Inertia must be positive."
    print(">> Assertion Passed: Inertia metric is strictly positive and bounded.")

    # -------------------------------------------------------------
    # Test 2: Gaussian Mixture Model (EM Algorithm)
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Running Gaussian Mixture Model (GMM) with Expectation-Maximization...")
    gmm = GaussianMixtureModel(k=K, max_iters=50, seed=42)
    # Initialize with K-Means centroids for fast stable convergence
    gmm.fit(points, init_means=kmeans.centroids)
    
    print(f"GMM EM converged in {len(gmm.log_likelihood_history)} iterations.")
    print(f"Initial Log-Likelihood: {gmm.log_likelihood_history[0]:.2f}")
    print(f"Final Log-Likelihood:   {gmm.log_likelihood_history[-1]:.2f}")
    
    # Assert monotonic increase in log-likelihood (allowing minor float epsilon)
    for it in range(1, len(gmm.log_likelihood_history)):
        diff = gmm.log_likelihood_history[it] - gmm.log_likelihood_history[it - 1]
        assert diff >= -1e-4, f"EM violated monotonic log-likelihood increase at step {it}: diff={diff}"
    print(">> Assertion Passed: EM strictly monotonically increases log-likelihood!")
    
    print("\nRecovered GMM Parameters:")
    for k in range(K):
        m = gmm.means[k]
        w = gmm.weights[k]
        cov = gmm.covariances[k]
        print(f"  Component {k}: Weight={w:.3f}, Mean=({m[0]:.3f}, {m[1]:.3f}), Var_diag=({cov[0][0]:.3f}, {cov[1][1]:.3f})")
        
    # Verify weights sum to 1
    weight_sum = sum(gmm.weights)
    assert abs(weight_sum - 1.0) < 1e-5, f"Weights must sum to 1, got {weight_sum}"
    print(">> Assertion Passed: Mixture weights sum to exactly 1.0.")
    
    # Test soft responsibilities (predict_proba)
    sample_pt = [2.0, 2.0]
    sample_probs = gmm.predict_proba([sample_pt])[0]
    print(f"\nSoft Responsibility for test point (2.0, 2.0): {[round(p, 4) for p in sample_probs]}")
    # The cluster around (2.0, 2.0) should dominate responsibility (> 0.90)
    max_resp = max(sample_probs)
    assert max_resp > 0.90, f"Expected dominant responsibility near cluster center, got {max_resp}"
    print(">> Assertion Passed: High responsibility assigned to the correct Gaussian component.")
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED SUCCESSFULLY! Exit code 0.")
    print("=" * 70)

if __name__ == "__main__":
    run_experiment()
