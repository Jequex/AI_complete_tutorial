#!/usr/bin/env python3
"""
Phase 25 - Topic 01: Score-Based Models and Langevin Dynamics
Experiment: Pure Python Langevin Dynamics & Annealed Score-Based Sampler

This experiment implements from first principles in pure Python 3:
1. Analytical score computation for multi-modal energy landscapes (Gaussian Mixture Models).
2. Exact score matching verification and score vector field evaluation.
3. Standard Langevin Dynamics (SGLD) sampling from noise particles.
4. Annealed Langevin Dynamics (Song & Ermon, 2019) across geometric noise scales:
   sigma_1 > sigma_2 > ... > sigma_L.
5. Mode collapse analysis: demonstrating how annealing overcomes low-density barriers
   to accurately capture all multimodal mixture components according to their true weights.
6. ASCII 2D spatial density scatter visualizer.

Zero external dependencies: pure standard Python 3.
"""

import math
import random
import sys
from typing import List, Tuple, Dict


# =====================================================================
# 1. 2D Vector Math Utilities
# =====================================================================

def vec_add(u: List[float], v: List[float]) -> List[float]:
    return [a + b for a, b in zip(u, v)]

def vec_sub(u: List[float], v: List[float]) -> List[float]:
    return [a - b for a, b in zip(u, v)]

def vec_scale(v: List[float], s: float) -> List[float]:
    return [a * s for a in v]

def vec_dot(u: List[float], v: List[float]) -> float:
    return sum(a * b for a, b in zip(u, v))

def vec_norm(v: List[float]) -> float:
    return math.sqrt(sum(a * a for a in v))

def sample_standard_normal_2d() -> List[float]:
    """Box-Muller transform for standard 2D Gaussian sample."""
    u1 = max(1e-12, random.random())
    u2 = random.random()
    r = math.sqrt(-2.0 * math.log(u1))
    theta = 2.0 * math.pi * u2
    return [r * math.cos(theta), r * math.sin(theta)]


# =====================================================================
# 2. Gaussian Mixture Model & Analytical Score
# =====================================================================

class GaussianMixture2D:
    """
    2D Gaussian Mixture Model with analytical density and score evaluation.
    p(x) = sum_k pi_k * N(x; mu_k, sigma_k^2 * I)
    """
    def __init__(self, weights: List[float], means: List[List[float]], stds: List[float]):
        assert len(weights) == len(means) == len(stds)
        total_w = sum(weights)
        self.weights = [w / total_w for w in weights]
        self.means = means
        self.stds = stds
        self.num_components = len(weights)

    def component_pdf(self, k: int, x: List[float], added_var: float = 0.0) -> float:
        """Evaluates N(x; mu_k, (sigma_k^2 + added_var) * I)."""
        mu = self.means[k]
        total_var = (self.stds[k] ** 2) + added_var
        dx = x[0] - mu[0]
        dy = x[1] - mu[1]
        dist_sq = dx * dx + dy * dy
        norm_const = 1.0 / (2.0 * math.pi * total_var)
        return norm_const * math.exp(-0.5 * dist_sq / total_var)

    def pdf(self, x: List[float], added_var: float = 0.0) -> float:
        """Evaluates total mixture density at x with optional additive noise variance."""
        return sum(self.weights[k] * self.component_pdf(k, x, added_var) for k in range(self.num_components))

    def score(self, x: List[float], added_var: float = 0.0) -> List[float]:
        """
        Computes exact score s(x) = grad_x log p(x).
        s(x) = sum_k [ w_k(x) * (mu_k - x) / var_k ]
        where w_k(x) = pi_k * N_k(x) / sum_j pi_j * N_j(x) (Bayesian posterior responsibility).
        """
        densities = [self.weights[k] * self.component_pdf(k, x, added_var) for k in range(self.num_components)]
        total_p = sum(densities)
        if total_p < 1e-30:
            # Fallback for extreme outliers: pull toward overall center of mass
            return [0.0, 0.0]

        grad = [0.0, 0.0]
        for k in range(self.num_components):
            resp = densities[k] / total_p
            total_var = (self.stds[k] ** 2) + added_var
            mu = self.means[k]
            # (mu_k - x) / total_var
            grad[0] += resp * (mu[0] - x[0]) / total_var
            grad[1] += resp * (mu[1] - x[1]) / total_var
        return grad


# =====================================================================
# 3. Langevin Dynamics Samplers
# =====================================================================

def standard_langevin_dynamics(
    gmm: GaussianMixture2D,
    num_particles: int,
    num_steps: int = 400,
    step_size: float = 0.05
) -> List[List[float]]:
    """
    Standard un-annealed Langevin Dynamics:
    x_{t+1} = x_t + (eps / 2) * s(x_t) + sqrt(eps) * z_t, z_t ~ N(0, I)
    Initial particles sampled from wide prior N(0, 9*I).
    """
    particles = []
    for _ in range(num_particles):
        z = sample_standard_normal_2d()
        particles.append(vec_scale(z, 3.0))

    sqrt_eps = math.sqrt(step_size)
    for _ in range(num_steps):
        for i in range(num_particles):
            x = particles[i]
            s = gmm.score(x)
            noise = sample_standard_normal_2d()
            # Drift term + Diffusion term
            drift = vec_scale(s, step_size * 0.5)
            diffusion = vec_scale(noise, sqrt_eps)
            particles[i] = vec_add(x, vec_add(drift, diffusion))

    return particles


def annealed_langevin_dynamics(
    gmm: GaussianMixture2D,
    num_particles: int,
    sigma_max: float = 6.0,
    sigma_min: float = 0.05,
    num_scales: int = 10,
    steps_per_scale: int = 40
) -> List[List[float]]:
    """
    Annealed Langevin Dynamics (Song & Ermon, 2019):
    Geometric noise sequence: sigma_1 > sigma_2 > ... > sigma_L
    Step size scaled proportionally: alpha_i = eps * (sigma_i / sigma_L)^2
    """
    # Geometric sequence of noise levels
    ratio = (sigma_min / sigma_max) ** (1.0 / (num_scales - 1))
    sigmas = [sigma_max * (ratio ** i) for i in range(num_scales)]

    particles = []
    for _ in range(num_particles):
        z = sample_standard_normal_2d()
        particles.append(vec_scale(z, sigma_max))

    eps_base = 0.02
    for sigma in sigmas:
        alpha = eps_base * ((sigma / sigma_min) ** 2)
        # Cap alpha to prevent explosive updates at high sigma
        alpha = min(alpha, 0.4)
        sqrt_alpha = math.sqrt(alpha)
        added_var = sigma ** 2

        for _ in range(steps_per_scale):
            for i in range(num_particles):
                x = particles[i]
                # Score evaluated on sigma-smoothed landscape
                s = gmm.score(x, added_var=added_var)
                noise = sample_standard_normal_2d()
                drift = vec_scale(s, alpha * 0.5)
                diffusion = vec_scale(noise, sqrt_alpha)
                particles[i] = vec_add(x, vec_add(drift, diffusion))

    return particles


# =====================================================================
# 4. Evaluation and Mode Attribution
# =====================================================================

def assign_modes(particles: List[List[float]], gmm: GaussianMixture2D) -> List[int]:
    """Assigns each particle to the nearest Gaussian component mean."""
    assignments = []
    for p in particles:
        best_k = 0
        min_dist = float("inf")
        for k, mu in enumerate(gmm.means):
            d = vec_norm(vec_sub(p, mu))
            if d < min_dist:
                min_dist = d
                best_k = k
        assignments.append(best_k)
    return assignments


def print_ascii_density_map(
    particles: List[List[float]],
    title: str,
    x_range: Tuple[float, float] = (-5.0, 5.0),
    y_range: Tuple[float, float] = (-5.0, 5.0),
    width: int = 50,
    height: int = 20
):
    """Renders 2D particle cloud as an ASCII density grid."""
    grid = [[0 for _ in range(width)] for _ in range(height)]
    x_min, x_max = x_range
    y_min, y_max = y_range

    for x, y in particles:
        if x_min <= x <= x_max and y_min <= y <= y_max:
            col = int((x - x_min) / (x_max - x_min) * (width - 1))
            row = int((y_max - y) / (y_max - y_min) * (height - 1))
            grid[row][col] += 1

    max_count = max(max(row) for row in grid) or 1
    # Density intensity ramp
    charset = " .:-=+*#%@"

    print(f"\n--- {title} ---")
    print("+" + "-" * width + "+")
    for r in range(height):
        line = []
        for c in range(width):
            cnt = grid[r][c]
            if cnt == 0:
                line.append(" ")
            else:
                idx = min(len(charset) - 1, int((cnt / max_count) * (len(charset) - 1)) + 1)
                line.append(charset[idx])
        print("|" + "".join(line) + "|")
    print("+" + "-" * width + "+")


# =====================================================================
# 5. Main Execution and Self-Verification
# =====================================================================

def main():
    random.seed(42)
    print("======================================================================")
    print("Phase 25 - Topic 01: Score-Based Models & Langevin Dynamics")
    print("Self-Contained Pure Python Verification Experiment")
    print("======================================================================")

    # Define a tri-modal GMM with separated clusters and different mixture weights
    # Component 0: Mode at (-3.0, -2.0), weight = 0.50
    # Component 1: Mode at (3.0, -2.0),  weight = 0.30
    # Component 2: Mode at (0.0, 3.0),   weight = 0.20
    true_weights = [0.50, 0.30, 0.20]
    true_means = [[-3.0, -2.0], [3.0, -2.0], [0.0, 3.0]]
    true_stds = [0.4, 0.4, 0.4]

    gmm = GaussianMixture2D(weights=true_weights, means=true_means, stds=true_stds)

    print(f"\n[1] Multi-Modal Target Landscape Initialized:")
    for k in range(gmm.num_components):
        print(f"    Mode {k}: mean = {gmm.means[k]}, std = {gmm.stds[k]}, weight = {gmm.weights[k]:.2f}")

    # Verify score points inward toward modes
    test_pt = [-3.0, -1.0] # 1 unit above mode 0
    score_pt = gmm.score(test_pt)
    print(f"\n[2] Analytical Score Verification:")
    print(f"    Evaluating score at test point {test_pt}:")
    print(f"    s(x) = [{score_pt[0]:.4f}, {score_pt[1]:.4f}]")
    print(f"    Vector direction points down towards mode 0 at [-3.0, -2.0]: y-grad = {score_pt[1]:.4f} (< 0 confirmed)")
    assert score_pt[1] < -0.5, "Score direction test failed!"

    num_samples = 400

    # 1. Standard Langevin Dynamics (suffers from barrier entrapment)
    print(f"\n[3] Running Standard (Un-annealed) Langevin Dynamics (N={num_samples} particles)...")
    std_particles = standard_langevin_dynamics(gmm, num_particles=num_samples, num_steps=300, step_size=0.08)
    std_assignments = assign_modes(std_particles, gmm)
    std_counts = [std_assignments.count(k) for k in range(gmm.num_components)]
    std_props = [c / num_samples for c in std_counts]

    print("    Standard Langevin Mode Allocation:")
    for k in range(gmm.num_components):
        print(f"      Mode {k}: empirical {std_props[k]*100:5.1f}% (true: {gmm.weights[k]*100:5.1f}%)")

    # 2. Annealed Langevin Dynamics (Song & Ermon 2019)
    print(f"\n[4] Running Annealed Langevin Dynamics (Song & Ermon, N={num_samples} particles)...")
    ann_particles = annealed_langevin_dynamics(
        gmm,
        num_particles=num_samples,
        sigma_max=5.0,
        sigma_min=0.1,
        num_scales=10,
        steps_per_scale=35
    )
    ann_assignments = assign_modes(ann_particles, gmm)
    ann_counts = [ann_assignments.count(k) for k in range(gmm.num_components)]
    ann_props = [c / num_samples for c in ann_counts]

    print("    Annealed Langevin Mode Allocation:")
    for k in range(gmm.num_components):
        print(f"      Mode {k}: empirical {ann_props[k]*100:5.1f}% (true: {gmm.weights[k]*100:5.1f}%)")

    # Display ASCII density map of annealed samples
    print_ascii_density_map(ann_particles, title="Annealed Langevin Particle Distribution (2D Plane)")

    # 3. Quantitative Error Comparison
    std_l1_err = sum(abs(p - w) for p, w in zip(std_props, gmm.weights))
    ann_l1_err = sum(abs(p - w) for p, w in zip(ann_props, gmm.weights))
    print(f"\n[5] Quantitative Metrics:")
    print(f"    Standard Langevin Total Variation Error: {std_l1_err:.4f}")
    print(f"    Annealed Langevin Total Variation Error: {ann_l1_err:.4f}")

    # Check that Annealed Langevin discovers all three modes
    for k in range(gmm.num_components):
        assert ann_counts[k] > 0.05 * num_samples, f"Annealed Langevin failed to explore Mode {k}!"

    print("\n[SUCCESS] Experiment completed successfully with zero external dependencies (Exit Code: 0).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
