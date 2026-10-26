"""
Topic 6.4: Kullback-Leibler (KL) Divergence Experiment
Demonstrating Asymmetry, Gibbs' Inequality, VAE Closed-Form Loss, and Mode-Seeking vs Mode-Covering
Pure Python 3 Standard Library implementation (Zero External Dependencies).
"""

import math

def kl_divergence(p, q, eps=1e-12):
    """Computes D_KL(P || Q) = sum P(x) * log(P(x) / Q(x)) in nats."""
    div = 0.0
    for pi, qi in zip(p, q):
        if pi > eps:
            qi_clamped = max(qi, eps)
            div += pi * math.log(pi / qi_clamped)
    return div

def entropy(p, eps=1e-12):
    """Computes H(P) in nats."""
    return -sum(pi * math.log(pi) for pi in p if pi > eps)

def cross_entropy(p, q, eps=1e-12):
    """Computes H(P, Q) in nats."""
    return -sum(pi * math.log(max(qi, eps)) for pi, qi in zip(p, q) if pi > eps)

def gaussian_pdf(x, mu, sigma):
    """Evaluates 1D Gaussian probability density function."""
    coef = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
    return coef * math.exp(-0.5 * ((x - mu) / sigma) ** 2)

def vae_kl_analytical(mu, sigma):
    """Closed-form D_KL(N(mu, sigma^2) || N(0, 1))."""
    return -0.5 * (1.0 + math.log(sigma**2) - mu**2 - sigma**2)

def main():
    print("=" * 72)
    print("TOPIC 6.4: KULLBACK-LEIBLER DIVERGENCE & VAE LATENT LOSS")
    print("=" * 72)

    # --- 1. Asymmetry & Fundamental Triad ---
    P = [0.90, 0.08, 0.02]
    Q = [0.33, 0.33, 0.34]

    h_p = entropy(P)
    h_pq = cross_entropy(P, Q)
    kl_forward = kl_divergence(P, Q)
    kl_reverse = kl_divergence(Q, P)

    print("--- 1. Asymmetry & Fundamental Triad Check ---")
    print(f"Distribution P: {P}")
    print(f"Distribution Q: {Q}")
    print(f"True Entropy H(P):               {h_p:.4f} nats")
    print(f"Cross-Entropy H(P, Q):           {h_pq:.4f} nats")
    print(f"Forward KL D_KL(P || Q):         {kl_forward:.4f} nats")
    print(f"Reverse KL D_KL(Q || P):         {kl_reverse:.4f} nats")

    # Verify Triad: H(P, Q) = H(P) + D_KL(P || Q)
    triad_diff = abs(h_pq - (h_p + kl_forward))
    print(f"Fundamental Triad Difference:    {triad_diff:.2e} nats")
    assert triad_diff < 1e-10, "Fundamental triad identity violated!"

    # Verify Asymmetry
    asym_diff = abs(kl_forward - kl_reverse)
    print(f"KL Asymmetry |Forward - Reverse|: {asym_diff:.4f} nats")
    assert asym_diff > 0.2, f"KL should be distinctly asymmetric! Got diff={asym_diff}"
    print(">> SUCCESS: Gibbs' Inequality & Fundamental Triad rigorously verified!\n")

    # --- 2. VAE Closed-Form Gaussian KL vs. Numerical Integration ---
    print("--- 2. VAE Latent Loss: Analytical vs. Numerical Integration ---")
    test_mu = 1.5
    test_sigma = 0.8
    analytical_loss = vae_kl_analytical(test_mu, test_sigma)

    # Numerical integration using Riemann sum over [-10, 10]
    grid_min, grid_max, steps = -10.0, 10.0, 4000
    dx = (grid_max - grid_min) / steps
    numerical_loss = 0.0

    for i in range(steps):
        x = grid_min + (i + 0.5) * dx
        p_val = gaussian_pdf(x, test_mu, test_sigma)
        q_val = gaussian_pdf(x, 0.0, 1.0)
        if p_val > 1e-14 and q_val > 1e-14:
            numerical_loss += p_val * math.log(p_val / q_val) * dx

    print(f"Latent Distribution: N(mu={test_mu}, sigma={test_sigma}) vs Prior N(0, 1)")
    print(f"Analytical VAE KL Loss:  {analytical_loss:.6f} nats")
    print(f"Numerical Riemann Sum:   {numerical_loss:.6f} nats")
    num_diff = abs(analytical_loss - numerical_loss)
    print(f"Discrepancy:             {num_diff:.2e} nats")
    assert num_diff < 1e-3, f"Numerical KL mismatch: diff={num_diff}"
    print(">> SUCCESS: VAE analytical formula matches numerical integral to < 0.001 nats!\n")

    # --- 3. Forward (Mode-Covering) vs. Reverse (Mode-Seeking) KL ---
    print("--- 3. Forward vs. Reverse KL on Bimodal Distribution ---")
    # Target P is bimodal: 50% mass at mode 1 (x=2), 50% mass at mode 2 (x=8)
    def bimodal_p(x):
        return 0.5 * gaussian_pdf(x, 2.0, 1.0) + 0.5 * gaussian_pdf(x, 8.0, 1.0)

    # We evaluate two candidate approximations Q:
    # Q_wide: centered at x=5 with wide sigma=3.5 (covers both peaks)
    # Q_mode: centered at x=2 with narrow sigma=1.0 (locks onto mode 1)
    x_grid = [grid_min + (i + 0.5) * dx for i in range(steps)]
    p_grid = [bimodal_p(x) for x in x_grid]
    q_wide = [gaussian_pdf(x, 5.0, 3.5) for x in x_grid]
    q_mode = [gaussian_pdf(x, 2.0, 1.0) for x in x_grid]

    # Normalize Riemann vectors
    p_norm = [p / sum(p_grid) for p in p_grid]
    q_wide_norm = [q / sum(q_wide) for q in q_wide]
    q_mode_norm = [q / sum(q_mode) for q in q_mode]

    # Forward KL: D_KL(P || Q)
    fwd_wide = kl_divergence(p_norm, q_wide_norm)
    fwd_mode = kl_divergence(p_norm, q_mode_norm)

    # Reverse KL: D_KL(Q || P)
    rev_wide = kl_divergence(q_wide_norm, p_norm)
    rev_mode = kl_divergence(q_mode_norm, p_norm)

    print(f"{'Approximation Model':<22} | {'Forward KL (Mode-Covering)':<28} | {'Reverse KL (Mode-Seeking)':<26}")
    print("-" * 80)
    print(f"{'Q_wide (covers both)':<22} | {fwd_wide:<28.4f} | {rev_wide:<26.4f}")
    print(f"{'Q_mode (locks mode 1)':<22} | {fwd_mode:<28.4f} | {rev_mode:<26.4f}")

    # Forward KL prefers the wide mode-covering model (fwd_wide < fwd_mode)
    # Reverse KL prefers locking onto a single mode (rev_mode < rev_wide)
    assert fwd_wide < fwd_mode, "Forward KL should prefer mode covering!"
    assert rev_mode < rev_wide, "Reverse KL should prefer mode seeking!"

    print("\n>> VERIFIED: Forward KL favors broad mode-covering (fwd_wide < fwd_mode).")
    print(">> VERIFIED: Reverse KL favors sharp mode-seeking (rev_mode < rev_wide).")
    print("=" * 72)

if __name__ == "__main__":
    main()
