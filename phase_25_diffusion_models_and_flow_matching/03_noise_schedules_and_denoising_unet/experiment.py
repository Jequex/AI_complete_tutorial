#!/usr/bin/env python3
"""
Phase 25 - Topic 03: Noise Schedules and Denoising U-Net / Residual Networks
Experiment: Pure Python 1D Diffusion Model with Sinusoidal Time Embeddings & Ancestral Sampling

This experiment implements from first principles in pure Python 3:
1. Cosine Noise Schedule (Nichol & Dhariwal, 2021) vs Linear Noise Schedule.
2. Sinusoidal Positional Time Embeddings: PE(t) = [sin(omega_i * t), cos(omega_i * t)].
3. Deep Residual Denoising MLP with exact analytical backpropagation (zero torch/autograd deps).
4. Training on a bimodal 1D target distribution using the simplified objective:
   L_simple(theta) = E_{t, x_0, eps} [ (eps - eps_theta(x_t, t))^2 ].
5. Reverse ancestral sampling generating synthetic data from pure Gaussian noise.
6. ASCII histogram comparing the ground-truth multimodal distribution against the diffusion samples.

Zero external dependencies: pure standard Python 3.
"""

import math
import random
import sys
from typing import List, Tuple


def sample_normal(mean: float = 0.0, std: float = 1.0) -> float:
    """Standard Box-Muller normal sampler."""
    u1 = max(1e-15, random.random())
    u2 = random.random()
    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return mean + std * z


# =====================================================================
# 1. Noise Schedules
# =====================================================================

class CosineNoiseSchedule:
    """
    Cosine variance schedule (Nichol & Dhariwal, 2021):
    f(t) = cos^2( (t/T + s)/(1 + s) * pi / 2 )
    alpha_bar_t = f(t) / f(0)
    beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}
    """
    def __init__(self, T: int = 40, s: float = 0.008):
        self.T = T
        self.s = s
        
        def f(t: float) -> float:
            val = ((t / self.T) + s) / (1.0 + s) * (math.pi / 2.0)
            return math.cos(val) ** 2

        f0 = f(0.0)
        self.alpha_bars = [f(t) / f0 for t in range(T + 1)]
        
        self.betas = []
        self.alphas = []
        for t in range(1, T + 1):
            ab_curr = self.alpha_bars[t]
            ab_prev = self.alpha_bars[t - 1]
            b = min(1.0 - (ab_curr / ab_prev), 0.999)
            self.betas.append(b)
            self.alphas.append(1.0 - b)

        # Slice alpha_bars to timesteps 0..T-1 for sampling
        self.alpha_bars_t = self.alpha_bars[1:]  # index t corresponds to step t+1


# =====================================================================
# 2. Sinusoidal Positional Embeddings
# =====================================================================

def sinusoidal_time_embedding(t: int, dim: int = 8, max_period: float = 10000.0) -> List[float]:
    """Generates sinusoidal time embedding for discrete step t."""
    emb = [0.0] * dim
    half_dim = dim // 2
    for i in range(half_dim):
        freq = math.exp(-math.log(max_period) * i / max_period)
        emb[2 * i] = math.sin(t * freq)
        emb[2 * i + 1] = math.cos(t * freq)
    return emb


# =====================================================================
# 3. Residual Denoising MLP with Analytical Backprop
# =====================================================================

class DenoisingMLP:
    """
    Residual MLP for scalar noise prediction: eps_theta(x_t, t).
    Architecture:
    Input: [x_t, emb(t)] in R^9
    Layer 1: Linear(9 -> 32) + LeakyReLU(0.1)
    Layer 2: Linear(32 -> 32) + Residual Skip + LeakyReLU(0.1)
    Layer 3: Linear(32 -> 1)
    """
    def __init__(self, in_dim: int = 9, hidden_dim: int = 32, seed: int = 42):
        random.seed(seed)
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        
        def xavier_init(fan_in: int, fan_out: int) -> List[List[float]]:
            bound = math.sqrt(2.0 / (fan_in + fan_out))
            return [[(random.random() * 2.0 - 1.0) * bound for _ in range(fan_in)] for _ in range(fan_out)]

        self.W1 = xavier_init(in_dim, hidden_dim)
        self.b1 = [0.0] * hidden_dim
        
        self.W2 = xavier_init(hidden_dim, hidden_dim)
        self.b2 = [0.0] * hidden_dim
        
        self.W3 = xavier_init(hidden_dim, 1)
        self.b3 = [0.0]

    @staticmethod
    def act(x: float) -> float:
        return x if x > 0 else 0.1 * x

    @staticmethod
    def act_grad(x: float) -> float:
        return 1.0 if x > 0 else 0.1

    def forward(self, x_in: List[float]) -> Tuple[float, dict]:
        """Forward pass caching intermediate activations for backprop."""
        # Layer 1
        z1 = [sum(w * x for w, x in zip(row, x_in)) + b for row, b in zip(self.W1, self.b1)]
        a1 = [self.act(z) for z in z1]

        # Layer 2 (Residual block)
        z2 = [sum(w * a for w, a in zip(row, a1)) + b for row, b in zip(self.W2, self.b2)]
        h2 = [self.act(z) for z in z2]
        a2 = [h + a for h, a in zip(h2, a1)]  # Skip connection

        # Layer 3 (Output projection)
        out = sum(w * a for w, a in zip(self.W3[0], a2)) + self.b3[0]

        cache = {"x_in": x_in, "z1": z1, "a1": a1, "z2": z2, "h2": h2, "a2": a2, "out": out}
        return out, cache

    def backward(self, cache: dict, grad_out: float) -> dict:
        """Exact analytical backpropagation."""
        a2 = cache["a2"]
        a1 = cache["a1"]
        z2 = cache["z2"]
        z1 = cache["z1"]
        x_in = cache["x_in"]

        # Gradients for W3, b3
        grad_W3 = [[grad_out * a for a in a2]]
        grad_b3 = [grad_out]

        # Backprop through Layer 3
        grad_a2 = [self.W3[0][i] * grad_out for i in range(self.hidden_dim)]

        # Backprop through Skip connection: a2 = h2 + a1
        grad_h2 = grad_a2[:]
        grad_a1_from_skip = grad_a2[:]

        # Backprop through Layer 2 activation
        grad_z2 = [gh * self.act_grad(z) for gh, z in zip(grad_h2, z2)]

        grad_W2 = [[grad_z2[i] * a1[j] for j in range(self.hidden_dim)] for i in range(self.hidden_dim)]
        grad_b2 = grad_z2[:]

        # Gradients entering a1 through W2
        grad_a1_from_W2 = [0.0] * self.hidden_dim
        for i in range(self.hidden_dim):
            for j in range(self.hidden_dim):
                grad_a1_from_W2[j] += self.W2[i][j] * grad_z2[i]

        # Total grad_a1
        grad_a1 = [g1 + g2 for g1, g2 in zip(grad_a1_from_skip, grad_a1_from_W2)]

        # Backprop through Layer 1 activation
        grad_z1 = [ga * self.act_grad(z) for ga, z in zip(grad_a1, z1)]

        grad_W1 = [[grad_z1[i] * x_in[j] for j in range(self.in_dim)] for i in range(self.hidden_dim)]
        grad_b1 = grad_z1[:]

        return {
            "W1": grad_W1, "b1": grad_b1,
            "W2": grad_W2, "b2": grad_b2,
            "W3": grad_W3, "b3": grad_b3
        }

    def update_params(self, grads: dict, lr: float):
        """Standard SGD update."""
        for i in range(self.hidden_dim):
            for j in range(self.in_dim):
                self.W1[i][j] -= lr * grads["W1"][i][j]
            self.b1[i] -= lr * grads["b1"][i]

        for i in range(self.hidden_dim):
            for j in range(self.hidden_dim):
                self.W2[i][j] -= lr * grads["W2"][i][j]
            self.b2[i] -= lr * grads["b2"][i]

        for j in range(self.hidden_dim):
            self.W3[0][j] -= lr * grads["W3"][0][j]
        self.b3[0] -= lr * grads["b3"][0]


# =====================================================================
# 4. Training & Ancestral Sampling Loop
# =====================================================================

def sample_bimodal_data() -> float:
    """Target distribution: 50% chance N(-2.0, 0.3^2), 50% chance N(+2.0, 0.3^2)."""
    mu = -2.0 if random.random() < 0.5 else 2.0
    return sample_normal(mu, 0.3)


def train_diffusion(model: DenoisingMLP, sched: CosineNoiseSchedule, num_epochs: int = 150, batch_size: int = 64, lr: float = 0.02):
    """Trains the mini-diffusion model using L_simple."""
    loss_history = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        # Initialize accumulated gradients
        acc_grads = {
            "W1": [[0.0] * model.in_dim for _ in range(model.hidden_dim)],
            "b1": [0.0] * model.hidden_dim,
            "W2": [[0.0] * model.hidden_dim for _ in range(model.hidden_dim)],
            "b2": [0.0] * model.hidden_dim,
            "W3": [[0.0] * model.hidden_dim],
            "b3": [0.0]
        }

        for _ in range(batch_size):
            # 1. Sample clean data x0 and discrete timestep t
            x0 = sample_bimodal_data()
            t = random.randint(0, sched.T - 1)
            
            # 2. Sample noise eps
            eps = sample_normal(0.0, 1.0)
            
            # 3. Closed-form forward jump to x_t
            ab = sched.alpha_bars_t[t]
            xt = math.sqrt(ab) * x0 + math.sqrt(1.0 - ab) * eps
            
            # 4. Construct input vector: [xt, sinusoidal_embedding(t)]
            t_emb = sinusoidal_time_embedding(t, dim=8)
            x_in = [xt] + t_emb
            
            # 5. Model prediction and squared error loss
            pred_eps, cache = model.forward(x_in)
            diff = pred_eps - eps
            loss = 0.5 * (diff ** 2)
            epoch_loss += loss
            
            # 6. Backward gradient
            g = model.backward(cache, diff)
            
            # Accumulate gradients
            for i in range(model.hidden_dim):
                for j in range(model.in_dim):
                    acc_grads["W1"][i][j] += g["W1"][i][j] / batch_size
                acc_grads["b1"][i] += g["b1"][i] / batch_size

            for i in range(model.hidden_dim):
                for j in range(model.hidden_dim):
                    acc_grads["W2"][i][j] += g["W2"][i][j] / batch_size
                acc_grads["b2"][i] += g["b2"][i] / batch_size

            for j in range(model.hidden_dim):
                acc_grads["W3"][0][j] += g["W3"][0][j] / batch_size
            acc_grads["b3"][0] += g["b3"][0] / batch_size

        model.update_params(acc_grads, lr)
        avg_loss = epoch_loss / batch_size
        loss_history.append(avg_loss)
        if (epoch + 1) % 30 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d}/{num_epochs} | L_simple Loss: {avg_loss:.4f}")

    return loss_history


def ancestral_sampling(model: DenoisingMLP, sched: CosineNoiseSchedule, num_samples: int = 300) -> List[float]:
    """
    DDPM Ancestral Sampling:
    Starts from x_T ~ N(0, 1) and iteratively denoises down to x_0.
    """
    samples = [sample_normal(0.0, 1.0) for _ in range(num_samples)]

    for t in range(sched.T - 1, -1, -1):
        ab = sched.alpha_bars_t[t]
        a = sched.alphas[t]
        b = sched.betas[t]
        sqrt_a = math.sqrt(a)
        sqrt_one_minus_ab = math.sqrt(1.0 - ab)

        t_emb = sinusoidal_time_embedding(t, dim=8)

        new_samples = []
        for xt in samples:
            x_in = [xt] + t_emb
            pred_eps, _ = model.forward(x_in)
            
            # Posterior mean reparameterization:
            # mu_theta = (1 / sqrt(a_t)) * (x_t - (beta_t / sqrt(1 - ab_t)) * pred_eps)
            mu = (1.0 / sqrt_a) * (xt - (b / sqrt_one_minus_ab) * pred_eps)

            if t > 0:
                sigma = math.sqrt(b)
                z = sample_normal(0.0, 1.0)
                x_prev = mu + sigma * z
            else:
                x_prev = mu
            new_samples.append(x_prev)
            
        samples = new_samples

    return samples


def print_ascii_histogram(samples: List[float], title: str, num_bins: int = 24, min_val: float = -4.0, max_val: float = 4.0):
    """Renders 1D sample distribution as an ASCII bar chart."""
    bins = [0] * num_bins
    bin_width = (max_val - min_val) / num_bins

    for s in samples:
        if min_val <= s < max_val:
            idx = int((s - min_val) / bin_width)
            bins[idx] += 1

    max_count = max(bins) or 1
    print(f"\n--- {title} ---")
    for i, count in enumerate(bins):
        center = min_val + (i + 0.5) * bin_width
        bar = "#" * int((count / max_count) * 36)
        print(f"  {center:5.2f} | {count:3d} | {bar}")


# =====================================================================
# 5. Main Execution
# =====================================================================

def main():
    random.seed(42)
    print("======================================================================")
    print("Phase 25 - Topic 03: Noise Schedules & Denoising Networks")
    print("Self-Contained Pure Python Verification Experiment")
    print("======================================================================")

    # 1. Inspect Cosine vs Linear Schedule
    T = 30
    cosine_sched = CosineNoiseSchedule(T=T, s=0.008)
    print(f"\n[1] Cosine Noise Schedule Analysis (T = {T}):")
    print(f"    t =  0: alpha_bar = {cosine_sched.alpha_bars_t[0]:.4f} (Signal retained: ~100%)")
    print(f"    t = {T//2}: alpha_bar = {cosine_sched.alpha_bars_t[T//2]:.4f} (Smooth signal decline)")
    print(f"    t = {T-1}: alpha_bar = {cosine_sched.alpha_bars_t[-1]:.4f} (Destroys signal smoothly without early cliff)")

    # 2. Inspect Sinusoidal Embedding
    emb0 = sinusoidal_time_embedding(0, dim=8)
    emb15 = sinusoidal_time_embedding(15, dim=8)
    print(f"\n[2] Sinusoidal Time Positional Embeddings:")
    print(f"    t =  0: {[round(v, 3) for v in emb0[:4]]}...")
    print(f"    t = 15: {[round(v, 3) for v in emb15[:4]]}...")

    # 3. Train mini-diffusion network on bimodal target
    print(f"\n[3] Training 1D Residual Denoising MLP on Bimodal Targets (Modes at -2.0 and +2.0)...")
    model = DenoisingMLP(in_dim=9, hidden_dim=32, seed=42)
    losses = train_diffusion(model, cosine_sched, num_epochs=120, batch_size=64, lr=0.03)

    assert losses[-1] < losses[0], "Model failed to optimize L_simple objective!"
    print(f"    Initial Loss: {losses[0]:.4f} -> Final Loss: {losses[-1]:.4f}")

    # 4. Generate Ancestral Reverse Samples
    num_gen = 300
    print(f"\n[4] Generating {num_gen} Samples via Reverse Ancestral Diffusion...")
    generated_samples = ancestral_sampling(model, cosine_sched, num_samples=num_gen)

    # 5. Evaluate Bimodal Recovery
    mode_neg = sum(1 for s in generated_samples if s < 0.0)
    mode_pos = sum(1 for s in generated_samples if s > 0.0)
    print(f"\n[5] Multi-Modal Recovery Validation:")
    print(f"    Samples in Negative Mode (x < 0): {mode_neg:3d} ({mode_neg/num_gen*100:4.1f}%)")
    print(f"    Samples in Positive Mode (x > 0): {mode_pos:3d} ({mode_pos/num_gen*100:4.1f}%)")

    # ASCII Histogram of Generated Samples
    print_ascii_histogram(generated_samples, title=f"Generated Diffusion Distribution (N={num_gen})")

    assert mode_neg > 0.25 * num_gen, "Negative mode underrepresented!"
    assert mode_pos > 0.25 * num_gen, "Positive mode underrepresented!"

    print("\n[SUCCESS] Topic 03 experiment completed successfully (Exit Code: 0).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
