#!/usr/bin/env python3
"""
Topic 24.3: The Reparameterization Trick & Latent Space Interpolation
====================================================================

This experiment implements a Variational Autoencoder (VAE) completely from
first principles in pure Python 3:
1. Encoder MLP: x (dim 4) -> Hidden (dim 6, Tanh) -> mu (dim 2), log_var (dim 2)
2. Reparameterization Trick:
     epsilon ~ N(0, I)
     z = mu + exp(0.5 * log_var) * epsilon
3. Decoder MLP: z (dim 2) -> Hidden (dim 6, Tanh) -> x_hat (dim 4)
4. Total Loss = Reconstruction MSE + beta * Analytical KL Divergence
5. Exact analytical backpropagation through the stochastic reparameterized node.
6. Manifold Continuity & Latent Space Interpolation (LERP & SLERP):
   Demonstrating smooth semantic transitions across the latent space.

Zero external dependencies: Pure Python 3 standard library.
"""

import math
import random

random.seed(42)


# =====================================================================
# 1. Math & Sampling Primitives
# =====================================================================

def sample_standard_normal():
    """Generates a standard normal sample epsilon ~ N(0, 1) using Box-Muller."""
    u1 = max(random.random(), 1e-12)
    u2 = random.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

def slerp(z0, z1, alpha):
    """
    Spherical Linear Interpolation (SLERP) between two latent vectors in R^d.
    Preserves constant norm along the great circle arc on the Gaussian shell.
    """
    norm0 = math.sqrt(sum(v ** 2 for v in z0))
    norm1 = math.sqrt(sum(v ** 2 for v in z1))
    if norm0 < 1e-8 or norm1 < 1e-8:
        return [(1.0 - alpha) * a + alpha * b for a, b in zip(z0, z1)]

    u0 = [v / norm0 for v in z0]
    u1 = [v / norm1 for v in z1]
    dot = max(-1.0, min(1.0, sum(a * b for a, b in zip(u0, u1))))

    if abs(dot) > 0.9995:
        # Vectors nearly collinear, fall back to LERP
        res = [(1.0 - alpha) * a + alpha * b for a, b in zip(z0, z1)]
        return res

    omega = math.acos(dot)
    sin_omega = math.sin(omega)
    scale0 = math.sin((1.0 - alpha) * omega) / sin_omega
    scale1 = math.sin(alpha * omega) / sin_omega

    interp_norm = (1.0 - alpha) * norm0 + alpha * norm1
    interp_dir = [scale0 * a + scale1 * b for a, b in zip(u0, u1)]
    return [v * interp_norm for v in interp_dir]


# =====================================================================
# 2. Pure Python Variational Autoencoder
# =====================================================================

class VariationalAutoencoder:
    """
    Architecture:
      Encoder:
        x (in_dim) -> [W_enc, b_enc] -> h_enc (hidden_dim) with Tanh
        h_enc -> [W_mu, b_mu] -> mu (latent_dim)
        h_enc -> [W_lv, b_lv] -> log_var (latent_dim)
      Reparameterization:
        z = mu + exp(0.5 * log_var) * eps
      Decoder:
        z -> [W_dec, b_dec] -> h_dec (hidden_dim) with Tanh
        h_dec -> [W_out, b_out] -> x_hat (in_dim)
    """
    def __init__(self, in_dim=4, hidden_dim=6, latent_dim=2, lr=0.03, beta=0.1):
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.beta = beta

        def xavier(din, dout):
            limit = math.sqrt(6.0 / (din + dout))
            return [[random.uniform(-limit, limit) for _ in range(dout)] for _ in range(din)]

        # Encoder weights
        self.W_enc = xavier(in_dim, hidden_dim)
        self.b_enc = [0.0] * hidden_dim
        self.W_mu = xavier(hidden_dim, latent_dim)
        self.b_mu = [0.0] * latent_dim
        self.W_lv = xavier(hidden_dim, latent_dim)
        self.b_lv = [0.0] * latent_dim

        # Decoder weights
        self.W_dec = xavier(latent_dim, hidden_dim)
        self.b_dec = [0.0] * hidden_dim
        self.W_out = xavier(hidden_dim, in_dim)
        self.b_out = [0.0] * in_dim

    def encode(self, x):
        h_enc = [0.0] * self.hidden_dim
        for j in range(self.hidden_dim):
            s = sum(x[i] * self.W_enc[i][j] for i in range(self.in_dim)) + self.b_enc[j]
            h_enc[j] = math.tanh(s)

        mu = [0.0] * self.latent_dim
        log_var = [0.0] * self.latent_dim
        for j in range(self.latent_dim):
            mu[j] = sum(h_enc[i] * self.W_mu[i][j] for i in range(self.hidden_dim)) + self.b_mu[j]
            log_var[j] = sum(h_enc[i] * self.W_lv[i][j] for i in range(self.hidden_dim)) + self.b_lv[j]
            # Clamp log_var for stability
            log_var[j] = max(-6.0, min(6.0, log_var[j]))

        return h_enc, mu, log_var

    def reparameterize(self, mu, log_var):
        eps = [sample_standard_normal() for _ in range(self.latent_dim)]
        std = [math.exp(0.5 * lv) for lv in log_var]
        z = [m + s * e for m, s, e in zip(mu, std, eps)]
        return z, eps, std

    def decode(self, z):
        h_dec = [0.0] * self.hidden_dim
        for j in range(self.hidden_dim):
            s = sum(z[i] * self.W_dec[i][j] for i in range(self.latent_dim)) + self.b_dec[j]
            h_dec[j] = math.tanh(s)

        x_hat = [0.0] * self.in_dim
        for j in range(self.in_dim):
            x_hat[j] = sum(h_dec[i] * self.W_out[i][j] for i in range(self.hidden_dim)) + self.b_out[j]

        return h_dec, x_hat

    def train_step(self, x):
        # Forward pass
        h_enc, mu, log_var = self.encode(x)
        z, eps, std = self.reparameterize(mu, log_var)
        h_dec, x_hat = self.decode(z)

        # 1. Reconstruction loss: MSE = sum_j (x_hat_j - x_j)^2
        recon_loss = sum((x_hat[j] - x[j]) ** 2 for j in range(self.in_dim))

        # 2. Gaussian KL loss: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        kl_loss = -0.5 * sum(1.0 + log_var[j] - mu[j] ** 2 - math.exp(log_var[j]) for j in range(self.latent_dim))

        total_loss = recon_loss + self.beta * kl_loss

        # -------------------------------------------------------------
        # Backpropagation
        # -------------------------------------------------------------
        # Output error: dL/dx_hat = 2 * (x_hat - x)
        d_xhat = [2.0 * (x_hat[j] - x[j]) for j in range(self.in_dim)]

        # Decoder Layer 2 (Output)
        d_W_out = [[0.0] * self.in_dim for _ in range(self.hidden_dim)]
        d_b_out = [0.0] * self.in_dim
        d_hdec = [0.0] * self.hidden_dim

        for j in range(self.in_dim):
            d_b_out[j] = d_xhat[j]
            for i in range(self.hidden_dim):
                d_W_out[i][j] = h_dec[i] * d_xhat[j]
                d_hdec[i] += self.W_out[i][j] * d_xhat[j]

        # Decoder Tanh activation
        d_s_dec = [d_hdec[i] * (1.0 - h_dec[i] ** 2) for i in range(self.hidden_dim)]

        # Decoder Layer 1
        d_W_dec = [[0.0] * self.hidden_dim for _ in range(self.latent_dim)]
        d_b_dec = [0.0] * self.hidden_dim
        d_z = [0.0] * self.latent_dim

        for j in range(self.hidden_dim):
            d_b_dec[j] = d_s_dec[j]
            for i in range(self.latent_dim):
                d_W_dec[i][j] = z[i] * d_s_dec[j]
                d_z[i] += self.W_dec[i][j] * d_s_dec[j]

        # --- PATHWISE GRADIENT THROUGH REPARAMETERIZATION NODE ---
        # z = mu + std * eps
        # dz/dmu = 1.0
        # dz/d_logvar = std * 0.5 * eps
        # KL gradients:
        #   d(KL)/dmu = mu
        #   d(KL)/d_logvar = -0.5 * (1.0 - exp(log_var)) = 0.5 * (exp(log_var) - 1.0)

        d_mu = [0.0] * self.latent_dim
        d_lv = [0.0] * self.latent_dim

        for j in range(self.latent_dim):
            # Combined pathwise gradient from recon + analytical gradient from KL
            d_mu[j] = d_z[j] * 1.0 + self.beta * mu[j]
            d_lv[j] = d_z[j] * (0.5 * std[j] * eps[j]) + self.beta * (0.5 * (math.exp(log_var[j]) - 1.0))

        # Encoder mu and log_var heads
        d_W_mu = [[0.0] * self.latent_dim for _ in range(self.hidden_dim)]
        d_b_mu = [0.0] * self.latent_dim
        d_W_lv = [[0.0] * self.latent_dim for _ in range(self.hidden_dim)]
        d_b_lv = [0.0] * self.latent_dim
        d_henc = [0.0] * self.hidden_dim

        for j in range(self.latent_dim):
            d_b_mu[j] = d_mu[j]
            d_b_lv[j] = d_lv[j]
            for i in range(self.hidden_dim):
                d_W_mu[i][j] = h_enc[i] * d_mu[j]
                d_W_lv[i][j] = h_enc[i] * d_lv[j]
                d_henc[i] += self.W_mu[i][j] * d_mu[j] + self.W_lv[i][j] * d_lv[j]

        # Encoder Tanh activation
        d_s_enc = [d_henc[i] * (1.0 - h_enc[i] ** 2) for i in range(self.hidden_dim)]

        # Encoder Layer 1
        d_W_enc = [[0.0] * self.hidden_dim for _ in range(self.in_dim)]
        d_b_enc = [0.0] * self.hidden_dim

        for j in range(self.hidden_dim):
            d_b_enc[j] = d_s_enc[j]
            for i in range(self.in_dim):
                d_W_enc[i][j] = x[i] * d_s_enc[j]

        # SGD Updates
        for i in range(self.in_dim):
            for j in range(self.hidden_dim):
                self.W_enc[i][j] -= self.lr * d_W_enc[i][j]
        for j in range(self.hidden_dim):
            self.b_enc[j] -= self.lr * d_b_enc[j]

        for i in range(self.hidden_dim):
            for j in range(self.latent_dim):
                self.W_mu[i][j] -= self.lr * d_W_mu[i][j]
                self.W_lv[i][j] -= self.lr * d_W_lv[i][j]
        for j in range(self.latent_dim):
            self.b_mu[j] -= self.lr * d_b_mu[j]
            self.b_lv[j] -= self.lr * d_b_lv[j]

        for i in range(self.latent_dim):
            for j in range(self.hidden_dim):
                self.W_dec[i][j] -= self.lr * d_W_dec[i][j]
        for j in range(self.hidden_dim):
            self.b_dec[j] -= self.lr * d_b_dec[j]

        for i in range(self.hidden_dim):
            for j in range(self.in_dim):
                self.W_out[i][j] -= self.lr * d_W_out[i][j]
        for j in range(self.in_dim):
            self.b_out[j] -= self.lr * d_b_out[j]

        return total_loss, recon_loss, kl_loss


# =====================================================================
# Main Experiment Execution
# =====================================================================

def run_experiment():
    print("=" * 76)
    print("TOPIC 24.3: THE REPARAMETERIZATION TRICK & LATENT SPACE CONTINUITY")
    print("=" * 76)

    # Synthetic dataset: Two geometric continuous patterns in 4 dimensions
    # Class A: Ascending ramp [0.2, 0.4, 0.6, 0.8] + noise
    # Class B: Descending valley [0.9, 0.1, 0.1, 0.9] + noise
    dataset_A = [[0.2 + random.gauss(0, 0.03), 0.4 + random.gauss(0, 0.03),
                  0.6 + random.gauss(0, 0.03), 0.8 + random.gauss(0, 0.03)] for _ in range(40)]
    dataset_B = [[0.9 + random.gauss(0, 0.03), 0.1 + random.gauss(0, 0.03),
                  0.1 + random.gauss(0, 0.03), 0.9 + random.gauss(0, 0.03)] for _ in range(40)]
    all_data = dataset_A + dataset_B

    vae = VariationalAutoencoder(in_dim=4, hidden_dim=8, latent_dim=2, lr=0.03, beta=0.05)

    print("\n--- [1] Training VAE with Pathwise Reparameterization Gradients ---")
    epochs = 400
    for epoch in range(epochs):
        random.shuffle(all_data)
        tot_loss, tot_rec, tot_kl = 0.0, 0.0, 0.0
        for x in all_data:
            l, r, k = vae.train_step(x)
            tot_loss += l
            tot_rec += r
            tot_kl += k

        if (epoch + 1) % 100 == 0:
            avg_rec = tot_rec / len(all_data)
            avg_kl = tot_kl / len(all_data)
            print(f"    Epoch {epoch+1:3d}/{epochs} | Recon MSE: {avg_rec:.5f} | KL Divergence: {avg_kl:.5f}")

    # Check reconstruction fidelity
    test_sample_A = dataset_A[0]
    test_sample_B = dataset_B[0]
    _, mu_A, _ = vae.encode(test_sample_A)
    _, mu_B, _ = vae.encode(test_sample_B)
    _, rec_A = vae.decode(mu_A)
    _, rec_B = vae.decode(mu_B)

    mse_A = sum((test_sample_A[i] - rec_A[i]) ** 2 for i in range(4)) / 4.0
    print(f"\n  Trained VAE Reconstruction Quality:")
    print(f"    Class A Target: {[round(v, 2) for v in test_sample_A]}")
    print(f"    Class A Recon : {[round(v, 2) for v in rec_A]}  (MSE: {mse_A:.5f})")
    print(f"    Class A Latent: {[round(v, 3) for v in mu_A]}")
    print(f"    Class B Latent: {[round(v, 3) for v in mu_B]}")
    assert mse_A < 0.1, f"Reconstruction MSE exceeds acceptable threshold: {mse_A}"

    # -------------------------------------------------------------
    # Part 2: Latent Space Interpolation (LERP vs SLERP)
    # -------------------------------------------------------------
    print("\n--- [2] Manifold Continuity: Latent Interpolation from Class A to Class B ---")
    print(f"  Interpolating smoothly between Class A {tuple(round(v, 2) for v in mu_A)} and Class B {tuple(round(v, 2) for v in mu_B)}:")
    print(f"  {'Alpha':<7} | {'Interpolated Latent z':<22} | {'Decoded Pattern x_hat':<32} | {'Visual Morph'}")
    print("  " + "-" * 75)

    steps = 5
    for s in range(steps + 1):
        alpha = s / steps
        # Spherical interpolation along latent sphere
        z_interp = slerp(mu_A, mu_B, alpha)
        _, x_interp = vae.decode(z_interp)

        # Quick ASCII visualization of intermediate pattern
        bar = "".join("#" if v > 0.45 else "." for v in x_interp)
        z_str = f"[{z_interp[0]:.2f}, {z_interp[1]:.2f}]"
        x_str = f"[{', '.join(f'{v:.2f}' for v in x_interp)}]"
        print(f"  {alpha:5.2f}   | {z_str:<22} | {x_str:<32} | {bar}")

    # Random generation from Gaussian prior p(z) ~ N(0, I)
    print("\n--- [3] Generative Sampling Directly from Prior p(z) = N(0, I) ---")
    print("  Drawing 3 random samples from standard normal distribution:")
    for i in range(3):
        z_prior = [sample_standard_normal(), sample_standard_normal()]
        _, x_gen = vae.decode(z_prior)
        print(f"    Sample {i+1}: z = {[round(v, 2) for v in z_prior]} -> Decoded x = {[round(v, 2) for v in x_gen]}")

    print("\n" + "=" * 76)
    print("REPARAMETERIZATION & INTERPOLATION EXPERIMENT COMPLETE: Exit code 0.")
    print("=" * 76)
    return True


if __name__ == "__main__":
    run_experiment()
