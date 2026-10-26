#!/usr/bin/env python3
"""
Topic 24.5: Wasserstein GAN & The Gradient Penalty (WGAN-GP)
============================================================

This experiment demonstrates why Earth Mover's (Wasserstein-1) Distance
solves the fundamental flaw of Jensen-Shannon Divergence on low-dimensional
manifolds, and verifies the 1-Lipschitz continuity penalty from first principles:

1. Non-Overlapping Manifold Test (The Arjovsky Parallel Lines):
   P_0 = (0, y), P_theta = (theta, y) where y ~ Uniform(0, 1).
   - Shows KL divergence is infinite: D_KL = +inf
   - Shows JS divergence is completely constant: JSD = log(2) ~= 0.69315 for all theta > 0
     (Gradient d(JSD)/d(theta) = 0, completely uninformative!)
   - Shows Wasserstein distance is smooth and linear: W(P_0, P_theta) = |theta|
     (Gradient dW/d(theta) = 1, continuous learning signal!).

2. Pure Python 1-Lipschitz Critic & Exact Gradient Penalty Optimization
   Evaluates a critic network f(x), samples random line interpolations
   x_hat = eps * x_real + (1 - eps) * x_fake, computes spatial critic
   gradients ||nabla_{x_hat} f(x_hat)||_2, and optimizes the 1-Lipschitz
   Kantorovich-Rubinstein objective with gradient penalty.

Zero external dependencies: Pure Python 3 standard library.
"""

import math
import random

random.seed(42)


# =====================================================================
# 1. Parallel Line Manifold Analytical Divergences
# =====================================================================

def evaluate_manifold_divergences(theta_values):
    """
    Given distribution P_0 supported on {(0, y) : y in [0, 1]}
    and P_theta supported on {(theta, y) : y in [0, 1]},
    evaluates KL, JSD, and Wasserstein-1 distances.
    """
    results = []
    for theta in theta_values:
        abs_theta = abs(theta)
        if abs_theta == 0.0:
            kl = 0.0
            jsd = 0.0
            w1 = 0.0
        else:
            # Supports are completely disjoint
            kl = float('inf')
            jsd = math.log(2.0)  # ln(2) ~= 0.693147
            w1 = abs_theta       # Earth mover transport cost is exactly |theta|
        results.append((theta, kl, jsd, w1))
    return results


# =====================================================================
# 2. 1-Lipschitz Critic & Exact Gradient Penalty Optimization
# =====================================================================

class WGAN_Critic_1D:
    """
    A 1D continuous critic MLP: x in R^1 -> Linear(1->6) -> LeakyReLU(0.2) -> Linear(6->1)
    Parameters: W1 (1x6), b1 (6), W2 (6x1), b2 (1) = 19 parameters.
    """
    def __init__(self, hidden_dim=6, lr=0.04, alpha=0.2):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.alpha = alpha

        # Xavier initialization
        lim1 = math.sqrt(6.0 / (1 + hidden_dim))
        self.W1 = [random.uniform(-lim1, lim1) for _ in range(hidden_dim)]
        self.b1 = [0.0] * hidden_dim

        lim2 = math.sqrt(6.0 / (hidden_dim + 1))
        self.W2 = [random.uniform(-lim2, lim2) for _ in range(hidden_dim)]
        self.b2 = 0.0

    def forward(self, x):
        h = [0.0] * self.hidden_dim
        for j in range(self.hidden_dim):
            s = x * self.W1[j] + self.b1[j]
            h[j] = s if s > 0 else self.alpha * s
        score = sum(h[i] * self.W2[i] for i in range(self.hidden_dim)) + self.b2
        return h, score

    def spatial_gradient(self, x):
        """Computes exact derivative df(x)/dx via chain rule."""
        grad_x = 0.0
        for i in range(self.hidden_dim):
            s = x * self.W1[i] + self.b1[i]
            d_act = 1.0 if s > 0 else self.alpha
            grad_x += self.W2[i] * d_act * self.W1[i]
        return grad_x

    def get_params(self):
        return self.W1[:] + self.b1[:] + self.W2[:] + [self.b2]

    def set_params(self, p):
        idx = 0
        self.W1 = p[idx : idx + self.hidden_dim]
        idx += self.hidden_dim
        self.b1 = p[idx : idx + self.hidden_dim]
        idx += self.hidden_dim
        self.W2 = p[idx : idx + self.hidden_dim]
        idx += self.hidden_dim
        self.b2 = p[idx]

    def compute_loss(self, reals, fakes, eps_list, lambda_gp=10.0):
        """
        Critic Loss: E[f(x_fake)] - E[f(x_real)] + lambda * E[( |grad| - 1 )^2]
        """
        # Real score
        score_real = sum(self.forward(r)[1] for r in reals) / len(reals)
        # Fake score
        score_fake = sum(self.forward(f)[1] for f in fakes) / len(fakes)

        # Gradient penalty on interpolations
        gp_terms = []
        for r, f, eps in zip(reals, fakes, eps_list):
            x_hat = eps * r + (1.0 - eps) * f
            gn = abs(self.spatial_gradient(x_hat))
            gp_terms.append((gn - 1.0) ** 2)

        gp = sum(gp_terms) / len(gp_terms)
        loss = score_fake - score_real + lambda_gp * gp
        return loss, score_real, score_fake, gp

    def train_step(self, reals, fakes, lambda_gp=10.0):
        eps_list = [random.random() for _ in range(len(reals))]

        curr_params = self.get_params()
        delta = 1e-5
        grads = []

        base_loss, score_real, score_fake, gp = self.compute_loss(reals, fakes, eps_list, lambda_gp)

        for i in range(len(curr_params)):
            perturbed_plus = curr_params[:]
            perturbed_plus[i] += delta
            self.set_params(perturbed_plus)
            loss_plus, _, _, _ = self.compute_loss(reals, fakes, eps_list, lambda_gp)

            perturbed_minus = curr_params[:]
            perturbed_minus[i] -= delta
            self.set_params(perturbed_minus)
            loss_minus, _, _, _ = self.compute_loss(reals, fakes, eps_list, lambda_gp)

            g = (loss_plus - loss_minus) / (2.0 * delta)
            grads.append(g)

        # Restore original and step along gradient
        new_params = [p - self.lr * g for p, g in zip(curr_params, grads)]
        self.set_params(new_params)

        return base_loss, score_real, score_fake, gp


# =====================================================================
# Main Execution & Verification
# =====================================================================

def run_experiment():
    print("=" * 76)
    print("TOPIC 24.5: WASSERSTEIN GAN & GRADIENT PENALTY (WGAN-GP)")
    print("=" * 76)

    # -------------------------------------------------------------
    # 1. Manifold Geometry & Divergence Comparison
    # -------------------------------------------------------------
    print("\n--- [1] Divergences under Disjoint Manifolds: P_0 vs P_theta ---")
    print("  Two parallel vertical line segments in R^2 separated by distance theta.")
    print(f"  {'Theta (Distance)':<18} | {'KL(P_0 || P_theta)':<20} | {'JSD(P_0 || P_theta)':<22} | {'Wasserstein-1 (W1)'}")
    print("  " + "-" * 80)

    thetas = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    results = evaluate_manifold_divergences(thetas)

    for theta, kl, jsd, w1 in results:
        kl_str = "0.00000" if kl == 0.0 else "+infinity"
        print(f"  {theta:<18.2f} | {kl_str:<20} | {jsd:<22.5f} | {w1:<18.2f}")

    print("\n  Diagnostic Analysis:")
    print("  * When theta > 0, JSD is COMPLETELY FLAT at log(2) ~= 0.69315.")
    print("  * The gradient d(JSD)/d(theta) = 0.0. A standard GAN receives ZERO learning signal!")
    print("  * In contrast, Wasserstein Distance W1 scales LINEARLY with distance: W1(theta) = theta.")
    print("  * The gradient d(W1)/d(theta) = 1.0 everywhere, providing a steady directional slope!")

    # -------------------------------------------------------------
    # 2. 1-Lipschitz Critic & Gradient Penalty Verification
    # -------------------------------------------------------------
    print("\n--- [2] 1-Lipschitz Continuity & Gradient Penalty Enforcement ---")
    critic = WGAN_Critic_1D(hidden_dim=6, lr=0.04)

    # Real data around +2.0, Fake data around -2.0
    reals = [2.0 + random.gauss(0, 0.2) for _ in range(25)]
    fakes = [-2.0 + random.gauss(0, 0.2) for _ in range(25)]

    print("  Training Critic with Gradient Penalty (lambda = 5.0)...")
    for step in range(120):
        loss, s_real, s_fake, gp = critic.train_step(reals, fakes, lambda_gp=5.0)
        if (step + 1) % 30 == 0:
            print(f"    Step {step+1:3d}/120 | E[f(real)]: {s_real:6.3f} | E[f(fake)]: {s_fake:6.3f} | GP Loss: {gp:.5f}")

    # Inspect gradient norms along the line connecting fake and real modes
    print("\n  Evaluating Spatial Gradient Norms ||df/dx|| across space:")
    test_xs = [-2.0, -1.0, 0.0, 1.0, 2.0]
    grad_norms = []
    for tx in test_xs:
        gn = abs(critic.spatial_gradient(tx))
        grad_norms.append(gn)
        _, score = critic.forward(tx)
        print(f"    x = {tx:5.1f} | Critic Score f(x) = {score:7.3f} | Gradient Norm ||df/dx|| = {gn:.4f}")

    avg_gn = sum(grad_norms) / len(grad_norms)
    print(f"\n  Average Spatial Gradient Norm across test points: {avg_gn:.4f}")
    print(f"  Target 1-Lipschitz Bound: 1.0")
    # Verify that the gradient penalty kept the critic close to 1-Lipschitz continuity
    assert 0.4 <= avg_gn <= 1.8, f"Gradient norm out of expected 1-Lipschitz bounds: {avg_gn}"

    # Wasserstein distance estimate E[f(real)] - E[f(fake)]
    real_scores = [critic.forward(xr)[1] for xr in reals]
    fake_scores = [critic.forward(xf)[1] for xf in fakes]
    w1_estimate = (sum(real_scores) / len(real_scores)) - (sum(fake_scores) / len(fake_scores))
    print(f"\n  Wasserstein Distance Metric E[f(real)] - E[f(fake)]: {w1_estimate:.4f}")
    assert w1_estimate > 0.0, "Critic must assign strictly higher score to real samples."

    print("\n" + "=" * 76)
    print("WGAN-GP EXPERIMENT COMPLETE: Exit code 0.")
    print("=" * 76)
    return True


if __name__ == "__main__":
    run_experiment()
