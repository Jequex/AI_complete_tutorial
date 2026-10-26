#!/usr/bin/env python3
"""
Phase 25 - Topic 04: Classifier-Free Guidance & Latent Diffusion Models
Experiment: Pure Python Conditional Diffusion Model with Classifier-Free Guidance (CFG)

This experiment implements from first principles in pure Python 3:
1. Two-class conditional diffusion setup: Class 0 (target center: -3.0) vs Class 1 (target center: +3.0).
2. Unconditional condition dropout during training (p_uncond = 0.20) mapping conditioning to null token.
3. Conditional Denoising Network with analytical backpropagation.
4. Classifier-Free Guidance (CFG, Ho & Salimans 2021) sampling:
   eps_tilde(x_t, c) = eps_theta(x_t, null) + s * (eps_theta(x_t, c) - eps_theta(x_t, null)).
5. Systematic sweep over guidance scales s in {0.0, 1.0, 3.0} showing the fidelity-diversity trade-off
   and class steering dynamics.
6. ASCII distribution comparison.

Zero external dependencies: pure standard Python 3.
"""

import math
import random
import sys
from typing import List, Tuple, Dict


def sample_normal(mean: float = 0.0, std: float = 1.0) -> float:
    """Box-Muller Gaussian sampler."""
    u1 = max(1e-15, random.random())
    u2 = random.random()
    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return mean + std * z


# =====================================================================
# 1. Diffusion Schedule & Positional / Class Embeddings
# =====================================================================

class DiffusionSchedule:
    def __init__(self, T: int = 20, beta_start: float = 0.01, beta_end: float = 0.10):
        self.T = T
        self.betas = [beta_start + (beta_end - beta_start) * (t / (T - 1)) for t in range(T)]
        self.alphas = [1.0 - b for b in self.betas]
        self.alpha_bars = []
        prod = 1.0
        for a in self.alphas:
            prod *= a
            self.alpha_bars.append(prod)


def time_embedding(t: int, T: int) -> List[float]:
    """2D normalized harmonic time encoding."""
    tau = t / max(1, T - 1)
    return [math.sin(tau * math.pi), math.cos(tau * math.pi)]


def class_embedding(c: int) -> List[float]:
    """One-hot class indicator: Class 0 -> [1.0, 0.0], Class 1 -> [0.0, 1.0], Null -> [0.0, 0.0]."""
    if c == 0:
        return [1.0, 0.0]
    elif c == 1:
        return [0.0, 1.0]
    else:  # Null token (unconditional)
        return [0.0, 0.0]


# =====================================================================
# 2. Conditional Denoising Neural Network
# =====================================================================

class ConditionalMLP:
    """
    Two-layer MLP for conditional noise estimation:
    Input: [x_t (1), time_emb (2), class_emb (2)] -> total 5 features.
    Hidden: 32 units with tanh activation.
    Output: scalar noise prediction eps_hat.
    """
    def __init__(self, in_dim: int = 5, hidden_dim: int = 32, seed: int = 42):
        random.seed(seed)
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        scale = 0.2
        self.W1 = [[(random.random() * 2.0 - 1.0) * scale for _ in range(in_dim)] for _ in range(hidden_dim)]
        self.b1 = [0.0] * hidden_dim
        self.W2 = [[(random.random() * 2.0 - 1.0) * scale for _ in range(hidden_dim)]]
        self.b2 = [0.0]

    def forward(self, x: List[float]) -> Tuple[float, List[float]]:
        h = [math.tanh(sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(self.W1, self.b1)]
        out = sum(w * hi for w, hi in zip(self.W2[0], h)) + self.b2[0]
        return out, h

    def step(self, x: List[float], target: float, lr: float = 0.02) -> float:
        """Online stochastic gradient descent step with exact analytical backpropagation."""
        out, h = self.forward(x)
        diff = out - target
        
        # Gradients for layer 2
        grad_W2 = [diff * hi for hi in h]
        grad_b2 = diff

        # Gradients for layer 1 (backprop through tanh: d/dz tanh(z) = 1 - h^2)
        grad_h = [self.W2[0][i] * diff * (1.0 - h[i] ** 2) for i in range(self.hidden_dim)]

        # Parameter updates
        for i in range(self.hidden_dim):
            for j in range(self.in_dim):
                self.W1[i][j] -= lr * grad_h[i] * x[j]
            self.b1[i] -= lr * grad_h[i]

        for i in range(self.hidden_dim):
            self.W2[0][i] -= lr * grad_W2[i]
        self.b2[0] -= lr * grad_b2

        return diff * diff


# =====================================================================
# 3. Training Loop with Unconditional Dropout
# =====================================================================

def sample_training_data(c: int) -> float:
    """Class 0: N(-3.0, 0.4^2), Class 1: N(+3.0, 0.4^2)."""
    mu = -3.0 if c == 0 else 3.0
    return sample_normal(mu, 0.4)


def train_conditional_model(
    model: ConditionalMLP,
    sched: DiffusionSchedule,
    num_steps: int = 8000,
    lr: float = 0.025,
    p_uncond: float = 0.20
) -> List[float]:
    """
    Trains the conditional model using joint conditional and unconditional data:
    With probability p_uncond, the class conditioning label is replaced by the null token c = -1.
    """
    losses = []
    accum_loss = 0.0
    
    for step in range(num_steps):
        # 1. Sample uniform class c in {0, 1}
        true_c = 0 if random.random() < 0.5 else 1
        x0 = sample_training_data(true_c)
        t = random.randint(0, sched.T - 1)
        eps = sample_normal(0.0, 1.0)

        # 2. Closed-form forward jump
        ab = sched.alpha_bars[t]
        xt = math.sqrt(ab) * x0 + math.sqrt(1.0 - ab) * eps

        # 3. Apply unconditional dropout
        train_c = -1 if random.random() < p_uncond else true_c

        # 4. Construct input vector [xt, time_features, class_features]
        inp = [xt] + time_embedding(t, sched.T) + class_embedding(train_c)

        loss = model.step(inp, target=eps, lr=lr)
        accum_loss += loss

        if (step + 1) % 2000 == 0:
            avg_loss = accum_loss / 2000.0
            losses.append(avg_loss)
            print(f"    Step {step+1:4d}/{num_steps} | L_simple Loss: {avg_loss:.4f}")
            accum_loss = 0.0

    return losses


# =====================================================================
# 4. Classifier-Free Guidance Sampling
# =====================================================================

def sample_with_cfg(
    model: ConditionalMLP,
    sched: DiffusionSchedule,
    target_class: int,
    guidance_scale: float,
    num_samples: int = 200
) -> List[float]:
    """
    Ancestral reverse diffusion sampling guided by CFG:
    eps_guided = eps_null + s * (eps_cond - eps_null)
    """
    samples = [sample_normal(0.0, 1.0) for _ in range(num_samples)]

    for t in range(sched.T - 1, -1, -1):
        ab = sched.alpha_bars[t]
        a = sched.alphas[t]
        b = sched.betas[t]
        t_vec = time_embedding(t, sched.T)
        sqrt_a = math.sqrt(a)
        sqrt_one_minus_ab = math.sqrt(1.0 - ab)

        new_samples = []
        for xt in samples:
            # Unconditional prediction (c = null)
            eps_null, _ = model.forward([xt] + t_vec + class_embedding(-1))
            # Conditional prediction (c = target_class)
            eps_cond, _ = model.forward([xt] + t_vec + class_embedding(target_class))

            # CFG Extrapolation formula
            eps_guided = eps_null + guidance_scale * (eps_cond - eps_null)

            # Posterior mean
            mu = (1.0 / sqrt_a) * (xt - (b / sqrt_one_minus_ab) * eps_guided)

            if t > 0:
                xt_prev = mu + math.sqrt(b) * sample_normal(0.0, 1.0)
            else:
                xt_prev = mu
            new_samples.append(xt_prev)
        samples = new_samples

    return samples


# =====================================================================
# 5. Main Execution and Guidance Scale Sweeps
# =====================================================================

def main():
    random.seed(42)
    print("======================================================================")
    print("Phase 25 - Topic 04: Classifier-Free Guidance & Latent Diffusion")
    print("Self-Contained Pure Python Verification Experiment")
    print("======================================================================")

    sched = DiffusionSchedule(T=20, beta_start=0.01, beta_end=0.10)
    model = ConditionalMLP(in_dim=5, hidden_dim=32, seed=42)

    print("\n[1] Training Conditional Model with 20% Unconditional Dropout (c = null)...")
    losses = train_conditional_model(model, sched, num_steps=8000, lr=0.025, p_uncond=0.20)
    assert losses[-1] < losses[0], "Training failed to reduce loss!"

    scales = [0.0, 1.0, 3.0]
    num_eval = 200

    print("\n[2] Executing Guidance Scale Sweeps (s in {0.0, 1.0, 3.0}) Targeting Class 0 (Mode: -3.0):")
    print("    Scale s = 0.0 (Unconditional: draws from both modes freely)")
    print("    Scale s = 1.0 (Standard Conditional: guides toward requested mode)")
    print("    Scale s = 3.0 (CFG Boosted: sharpens mode with high fidelity)")

    results = {}
    for s in scales:
        samples = sample_with_cfg(model, sched, target_class=0, guidance_scale=s, num_samples=num_eval)
        # Class 0 fidelity: count proportion of samples on the left (x < 0)
        c0_count = sum(1 for x in samples if x < 0.0)
        target_pct = (c0_count / num_eval) * 100.0
        
        # Calculate mean of samples
        mean_val = sum(samples) / num_eval
        var_val = sum((x - mean_val) ** 2 for x in samples) / (num_eval - 1)
        std_val = math.sqrt(var_val)

        results[s] = {"pct": target_pct, "mean": mean_val, "std": std_val}
        print(f"\n    Guidance Scale s = {s:.1f}:")
        print(f"      Target Class 0 Accuracy: {target_pct:5.1f}% ({c0_count}/{num_eval})")
        print(f"      Overall Sample Mean:     {mean_val:6.2f} (Target mode center: -3.00)")
        print(f"      Sample Std Deviation:    {std_val:6.2f}")

    # Assertions
    # 1. At s=0, unconditional generation mixes modes (target accuracy between 30% and 75%)
    assert 30.0 <= results[0.0]["pct"] <= 75.0, f"Unconditional generation failed to mix modes ({results[0.0]['pct']}%)!"
    # 2. At s=1.0, conditional generation significantly favors class 0 (> 75%)
    assert results[1.0]["pct"] >= 75.0, f"Conditional generation failed to steer toward class 0 ({results[1.0]['pct']}%)!"
    # 3. At s=3.0, CFG strongly boosts accuracy (> 90%) and shifts mean close to -3.0
    assert results[3.0]["pct"] >= 90.0, f"CFG s=3.0 failed to achieve high fidelity ({results[3.0]['pct']}%)!"
    assert results[3.0]["mean"] < -2.0, f"CFG s=3.0 mean did not shift sufficiently toward target mode!"

    # 4. Latent Diffusion Dimensionality Analysis
    print("\n[3] Latent Diffusion Model (LDM / Stable Diffusion) Complexity Analysis:")
    pixel_dims = (512, 512, 3)
    latent_dims = (64, 64, 4)
    pixel_volume = pixel_dims[0] * pixel_dims[1] * pixel_dims[2]
    latent_volume = latent_dims[0] * latent_dims[1] * latent_dims[2]
    compression_ratio = pixel_volume / latent_volume
    print(f"    Pixel Space:  {pixel_dims[0]}x{pixel_dims[1]}x{pixel_dims[2]} = {pixel_volume:,} dimensions")
    print(f"    Latent Space: {latent_dims[0]}x{latent_dims[1]}x{latent_dims[4-2]} = {latent_volume:,} dimensions")
    print(f"    Dimensionality Reduction Factor: {compression_ratio:.1f}x reduction")
    print(f"    Attention Flop Reduction (O(N^2)): {(pixel_dims[0]*pixel_dims[1] / (latent_dims[0]*latent_dims[1]))**2:.0f}x theoretical self-attention speedup!")

    print("\n[4] ASCII Guidance Scale Steering Comparison:")
    for s in scales:
        neg_bars = int(results[s]["pct"] / 2.5)
        pos_bars = int((100.0 - results[s]["pct"]) / 2.5)
        vis = ("<" * neg_bars) + "|" + (">" * pos_bars)
        print(f"    s = {s:.1f} | Class 0: {results[s]['pct']:5.1f}% | {vis}")

    print("\n[SUCCESS] Topic 04 experiment completed successfully (Exit Code: 0).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
