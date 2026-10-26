"""
Phase 09: Unsupervised Learning & Representation
Topic 04: Density Estimation & Anomaly Detection from Scratch

Zero-dependency Python 3 implementation of:
1. Kernel Density Estimation (KDE) with Gaussian kernel and Silverman's rule of thumb.
2. Numerical integration of KDE density to verify valid probability distribution (integrates to 1.0).
3. 2D Correlated Gaussian data generator and synthetic anomaly injection.
4. Mahalanobis distance anomaly detector via empirical mean and covariance inversion.
5. Verification: Detection of correlation-breaking anomalies that evade Euclidean thresholds.
"""

import math
import random
from typing import List, Tuple

# =====================================================================
# 1. 1D Kernel Density Estimation (KDE)
# =====================================================================

def gaussian_kernel(u: float) -> float:
    """Standard normal Gaussian kernel K(u) = 1/sqrt(2pi) * exp(-u^2 / 2)."""
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * u * u)

def silverman_bandwidth(data: List[float]) -> float:
    """Computes Silverman's rule of thumb bandwidth: h = 1.06 * sigma * N^(-1/5)."""
    N = len(data)
    mean = sum(data) / N
    variance = sum((x - mean) ** 2 for x in data) / (N - 1)
    std = math.sqrt(variance)
    return 1.06 * std * (N ** (-0.2))

class KernelDensityEstimator1D:
    def __init__(self, bandwidth: float = None):
        self.bandwidth = bandwidth
        self.data: List[float] = []

    def fit(self, data: List[float]) -> "KernelDensityEstimator1D":
        self.data = list(data)
        if self.bandwidth is None:
            self.bandwidth = silverman_bandwidth(self.data)
        return self

    def score_samples(self, points: List[float]) -> List[float]:
        """Evaluates density f_hat(x) = (1 / (N * h)) * sum_i K((x - x_i) / h)."""
        N = len(self.data)
        h = self.bandwidth
        inv_Nh = 1.0 / (N * h)
        
        densities = []
        for x in points:
            val = sum(gaussian_kernel((x - xi) / h) for xi in self.data)
            densities.append(inv_Nh * val)
        return densities

# =====================================================================
# 2. 2D Mahalanobis Distance Anomaly Detection
# =====================================================================

def det_2x2(m: List[List[float]]) -> float:
    return m[0][0] * m[1][1] - m[0][1] * m[1][0]

def inv_2x2(m: List[List[float]]) -> List[List[float]]:
    det = det_2x2(m)
    if abs(det) < 1e-12:
        det = 1e-12
    inv_det = 1.0 / det
    return [
        [ m[1][1] * inv_det, -m[0][1] * inv_det],
        [-m[1][0] * inv_det,  m[0][0] * inv_det]
    ]

class MahalanobisAnomalyDetector:
    def __init__(self, threshold_p_value: float = 0.99):
        # For 2 degrees of freedom, Chi-squared 99% critical value = -2 * ln(1 - 0.99) = 9.2103
        # D_M = sqrt(chi2) = sqrt(9.2103) approx 3.035
        self.threshold = math.sqrt(-2.0 * math.log(1.0 - threshold_p_value))
        self.mean: List[float] = []
        self.cov: List[List[float]] = []
        self.inv_cov: List[List[float]] = []

    def fit(self, X: List[List[float]]) -> "MahalanobisAnomalyDetector":
        N = len(X)
        d = len(X[0])
        assert d == 2, "Only 2D implemented in this mini engine."
        
        # 1. Compute Empirical Mean
        self.mean = [sum(row[j] for row in X) / N for j in range(d)]
        
        # 2. Compute Empirical Covariance
        self.cov = [[0.0] * d for _ in range(d)]
        for row in X:
            dx = row[0] - self.mean[0]
            dy = row[1] - self.mean[1]
            self.cov[0][0] += dx * dx
            self.cov[0][1] += dx * dy
            self.cov[1][0] += dy * dx
            self.cov[1][1] += dy * dy
            
        self.cov[0][0] /= (N - 1)
        self.cov[0][1] /= (N - 1)
        self.cov[1][0] /= (N - 1)
        self.cov[1][1] /= (N - 1)
        
        # 3. Invert Covariance
        self.inv_cov = inv_2x2(self.cov)
        return self

    def mahalanobis_distance(self, x: List[float]) -> float:
        """D_M(x) = sqrt((x - mu)^T Sigma^-1 (x - mu))."""
        dx = x[0] - self.mean[0]
        dy = x[1] - self.mean[1]
        
        # inv_cov * [dx, dy]
        rx = self.inv_cov[0][0] * dx + self.inv_cov[0][1] * dy
        ry = self.inv_cov[1][0] * dx + self.inv_cov[1][1] * dy
        
        dist_sq = dx * rx + dy * ry
        return math.sqrt(max(0.0, dist_sq))

    def predict(self, X: List[List[float]]) -> List[bool]:
        """Returns True if anomaly, False if normal inlier."""
        return [self.mahalanobis_distance(x) > self.threshold for x in X]

# =====================================================================
# 3. Verification & Experiments
# =====================================================================

def run_experiment():
    print("=" * 70)
    print("PHASE 09: UNSUPERVISED LEARNING & REPRESENTATION")
    print("Topic 04: Density Estimation & Anomaly Detection from Scratch")
    print("=" * 70)
    
    # -------------------------------------------------------------
    # Experiment 1: 1D Multimodal KDE
    # -------------------------------------------------------------
    print("\n--- Experiment 1: 1D Multimodal Kernel Density Estimation ---")
    random.seed(42)
    # Generate mixture of two Gaussians: N(2.0, 0.4^2) and N(6.0, 0.5^2)
    data_1d = []
    for _ in range(150):
        # Cluster 1
        u1, u2 = max(random.random(), 1e-10), random.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        data_1d.append(2.0 + z * 0.4)
    for _ in range(150):
        # Cluster 2
        u1, u2 = max(random.random(), 1e-10), random.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        data_1d.append(6.0 + z * 0.5)
        
    kde = KernelDensityEstimator1D()
    kde.fit(data_1d)
    print(f"Data size: {len(data_1d)} points.")
    print(f"Silverman's rule of thumb optimal bandwidth h: {kde.bandwidth:.4f}")
    
    # Evaluate KDE on fine grid from -1.0 to 9.0 to verify integration to ~ 1.0
    GRID_STEPS = 1000
    x_min, x_max = -1.0, 9.0
    dx = (x_max - x_min) / GRID_STEPS
    grid = [x_min + i * dx for i in range(GRID_STEPS)]
    densities = kde.score_samples(grid)
    
    # Riemann sum integral
    integral = sum(densities) * dx
    print(f"Riemann sum integral of KDE over [-1, 9]: {integral:.4f}")
    assert abs(integral - 1.0) < 0.01, f"KDE density does not integrate to 1.0: {integral}"
    print(">> Assertion Passed: KDE probability density integrates to 1.0.")
    
    # Verify two peaks around x = 2.0 and x = 6.0
    d_at_2 = kde.score_samples([2.0])[0]
    d_at_4 = kde.score_samples([4.0])[0]
    d_at_6 = kde.score_samples([6.0])[0]
    print(f"Density at Peak 1 (x=2.0): {d_at_2:.4f}")
    print(f"Density at Valley (x=4.0): {d_at_4:.4f}")
    print(f"Density at Peak 2 (x=6.0): {d_at_6:.4f}")
    assert d_at_2 > d_at_4 and d_at_6 > d_at_4, "KDE failed to capture bimodal valley structure."
    print(">> Assertion Passed: KDE captures distinct bimodal peaks and central valley.")
    
    # -------------------------------------------------------------
    # Experiment 2: 2D Mahalanobis Distance Anomaly Detection
    # -------------------------------------------------------------
    print("\n--- Experiment 2: 2D Mahalanobis Distance Anomaly Detection ---")
    # Generate 200 points with strong positive covariance (x and y co-vary along y = x)
    inliers = []
    for _ in range(200):
        u1, u2 = max(random.random(), 1e-10), random.random()
        z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        u3, u4 = max(random.random(), 1e-10), random.random()
        z1 = math.sqrt(-2.0 * math.log(u3)) * math.cos(2.0 * math.pi * u4)
        
        # Correlated dimensions: var(main) = 4.0, var(cross) = 0.25
        x = 2.0 * z0 + 0.5 * z1
        y = 2.0 * z0 - 0.5 * z1
        inliers.append([x, y])
        
    # Inject 2 known anomalies:
    # Anomaly 1: Breaks correlation! (x = 3.0, y = -3.0).
    # Euclidean distance from origin is sqrt(3^2 + (-3)^2) = 4.24 (well within normal range of inliers!).
    # But it strongly violates the positive correlation line y = x.
    anomaly_break_corr = [3.0, -3.0]
    
    # Anomaly 2: Extreme magnitude outlier (x = 10.0, y = 10.0)
    anomaly_magnitude = [10.0, 10.0]
    
    test_outliers = [anomaly_break_corr, anomaly_magnitude]
    
    detector = MahalanobisAnomalyDetector(threshold_p_value=0.99)
    detector.fit(inliers)
    
    print(f"Empirical Center mu: {[round(m, 3) for m in detector.mean]}")
    print(f"Empirical Covariance Matrix:")
    print(f"  [{detector.cov[0][0]:.3f}, {detector.cov[0][1]:.3f}]")
    print(f"  [{detector.cov[1][0]:.3f}, {detector.cov[1][1]:.3f}]")
    print(f"Mahalanobis Decision Threshold (99% Chi-sq): {detector.threshold:.3f}")
    
    # Evaluate Outliers
    d_m_corr = detector.mahalanobis_distance(anomaly_break_corr)
    d_m_mag = detector.mahalanobis_distance(anomaly_magnitude)
    print(f"\nAnomaly 1 (Correlation Breaker [3.0, -3.0]): Mahalanobis Dist = {d_m_corr:.3f}")
    print(f"Anomaly 2 (Extreme Magnitude [10.0, 10.0]):   Mahalanobis Dist = {d_m_mag:.3f}")
    
    # Evaluate a normal inlier near center
    d_m_inlier = detector.mahalanobis_distance([1.5, 1.5])
    print(f"Typical Inlier [1.5, 1.5]:                     Mahalanobis Dist = {d_m_inlier:.3f}")
    
    # Assertions
    assert d_m_corr > detector.threshold, f"Failed to detect correlation-breaking anomaly! D_M={d_m_corr}"
    assert d_m_mag > detector.threshold, f"Failed to detect extreme magnitude anomaly! D_M={d_m_mag}"
    assert d_m_inlier < detector.threshold, f"False alarm on normal inlier! D_M={d_m_inlier}"
    print(">> Assertion Passed: Mahalanobis distance successfully flagged all anomalies.")
    print(">> Assertion Passed: Normal inliers are safely recognized below threshold.")
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED SUCCESSFULLY! Exit code 0.")
    print("=" * 70)

if __name__ == "__main__":
    run_experiment()
