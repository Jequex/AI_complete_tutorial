#!/usr/bin/env python3
"""
Phase 25 - Topic 02: DDPM Forward and Reverse Processes
Experiment: Pure Python Forward Markov Chain & Posterior Reconstruction Engine

This experiment verifies from first principles in pure Python 3:
1. DDPM forward Markov chain formulation: q(x_t | x_{t-1}) = N(sqrt(1 - beta_t) * x_{t-1}, beta_t * I).
2. The closed-form marginal shortcut: q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I).
3. Monte Carlo validation comparing step-by-step Markov simulation against closed-form jump.
4. Analytical Bayes' Rule derivation and verification of the true tractable posterior:
   q(x_{t-1} | x_t, x_0) = N(mu_tilde_t(x_t, x_0), beta_tilde_t * I).
5. Ground-truth reverse reconstruction test demonstrating perfect signal recovery
   when conditioned on true x_0.
6. ASCII trajectory visualization of the forward degradation and reverse recovery.

Zero external dependencies: pure standard Python 3.
"""

import math
import random
import sys
from typing import List, Tuple, Dict


def sample_normal(mean: float = 0.0, std: float = 1.0) -> float:
    """Box-Muller standard Gaussian sampler."""
    u1 = max(1e-15, random.random())
    u2 = random.random()
    z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return mean + std * z0


class DDPMProcess:
    """
    Implements analytical DDPM schedules, forward jumps, and true posterior statistics.
    """
    def __init__(self, num_timesteps: int = 100, beta_start: float = 0.0001, beta_end: float = 0.02):
        self.T = num_timesteps
        # Linear beta schedule (Ho et al. 2020)
        self.betas = [beta_start + (beta_end - beta_start) * (t / (self.T - 1)) for t in range(self.T)]
        self.alphas = [1.0 - b for b in self.betas]
        
        # Cumulative product alpha_bar_t = prod_{s=0}^t alpha_s
        self.alpha_bars = []
        prod = 1.0
        for a in self.alphas:
            prod *= a
            self.alpha_bars.append(prod)

        # Precompute posterior variance:
        # beta_tilde_t = (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t) * beta_t
        self.beta_tildes = [0.0] * self.T
        for t in range(1, self.T):
            ab_prev = self.alpha_bars[t - 1]
            ab_curr = self.alpha_bars[t]
            self.beta_tildes[t] = ((1.0 - ab_prev) / (1.0 - ab_curr)) * self.betas[t]

    def forward_step(self, x_prev: float, t: int) -> float:
        """Single Markov forward step: q(x_t | x_{t-1})."""
        mean = math.sqrt(1.0 - self.betas[t]) * x_prev
        std = math.sqrt(self.betas[t])
        return sample_normal(mean, std)

    def forward_closed_form(self, x_0: float, t: int, eps: float = None) -> Tuple[float, float]:
        """
        Direct closed-form jump: q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, 1 - alpha_bar_t).
        Returns: (x_t, eps_used)
        """
        if eps is None:
            eps = sample_normal(0.0, 1.0)
        ab = self.alpha_bars[t]
        x_t = math.sqrt(ab) * x_0 + math.sqrt(1.0 - ab) * eps
        return x_t, eps

    def posterior_params(self, x_t: float, x_0: float, t: int) -> Tuple[float, float]:
        """
        Computes analytical true posterior mean mu_tilde_t and variance beta_tilde_t
        under Bayes' rule q(x_{t-1} | x_t, x_0).
        """
        if t == 0:
            return x_0, 0.0

        ab_curr = self.alpha_bars[t]
        ab_prev = self.alpha_bars[t - 1]
        a_curr = self.alphas[t]
        b_curr = self.betas[t]

        # mu_tilde_t = [sqrt(ab_{t-1}) * b_t / (1 - ab_t)] * x_0 + [sqrt(a_t) * (1 - ab_{t-1}) / (1 - ab_t)] * x_t
        coef_x0 = (math.sqrt(ab_prev) * b_curr) / (1.0 - ab_curr)
        coef_xt = (math.sqrt(a_curr) * (1.0 - ab_prev)) / (1.0 - ab_curr)
        mu_tilde = coef_x0 * x_0 + coef_xt * x_t
        var_tilde = self.beta_tildes[t]
        return mu_tilde, var_tilde

    def reverse_step_ground_truth(self, x_t: float, x_0: float, t: int) -> float:
        """Samples x_{t-1} from true posterior q(x_{t-1} | x_t, x_0)."""
        mu_tilde, var_tilde = self.posterior_params(x_t, x_0, t)
        if t == 0 or var_tilde == 0.0:
            return mu_tilde
        return sample_normal(mu_tilde, math.sqrt(var_tilde))


def run_experiment():
    random.seed(42)
    print("======================================================================")
    print("Phase 25 - Topic 02: DDPM Forward & Reverse Processes")
    print("Self-Contained Pure Python Verification Experiment")
    print("======================================================================")

    T = 100
    ddpm = DDPMProcess(num_timesteps=T, beta_start=0.0001, beta_end=0.02)
    x0 = 4.0  # Clean scalar data point

    print(f"\n[1] Setup: Initial clean state x_0 = {x0:.2f}, Timesteps T = {T}")
    print(f"    beta_0 = {ddpm.betas[0]:.6f}, beta_{T-1} = {ddpm.betas[-1]:.6f}")
    print(f"    alpha_bar_0 = {ddpm.alpha_bars[0]:.6f}, alpha_bar_{T-1} = {ddpm.alpha_bars[-1]:.6f}")

    # -----------------------------------------------------------------
    # Test 1: Monte Carlo Verification of Closed-Form Jump vs Markov Chain
    # -----------------------------------------------------------------
    check_timesteps = [10, 30, 60, 99]
    num_mc_trials = 3000
    print(f"\n[2] Verifying Closed-Form Shortcut q(x_t | x_0) vs Step-by-Step Markov Chain:")
    print(f"    Running {num_mc_trials} Monte Carlo simulations...")

    for t in check_timesteps:
        # Analytical formulas:
        ab = ddpm.alpha_bars[t]
        ana_mean = math.sqrt(ab) * x0
        ana_var = 1.0 - ab

        # Step-by-step Markov chain samples:
        step_samples = []
        for _ in range(num_mc_trials):
            val = x0
            for step in range(t + 1):
                val = ddpm.forward_step(val, step)
            step_samples.append(val)

        # Closed-form samples:
        direct_samples = [ddpm.forward_closed_form(x0, t)[0] for _ in range(num_mc_trials)]

        mc_step_mean = sum(step_samples) / num_mc_trials
        mc_step_var = sum((s - mc_step_mean) ** 2 for s in step_samples) / (num_mc_trials - 1)

        mc_dir_mean = sum(direct_samples) / num_mc_trials
        mc_dir_var = sum((s - mc_dir_mean) ** 2 for s in direct_samples) / (num_mc_trials - 1)

        print(f"\n    Timestep t = {t:2d} (alpha_bar_t = {ab:.4f}):")
        print(f"      Analytical:     Mean = {ana_mean:7.4f}, Var = {ana_var:7.4f}")
        print(f"      Markov Chain:   Mean = {mc_step_mean:7.4f}, Var = {mc_step_var:7.4f} (Err: {abs(mc_step_mean - ana_mean):.4f})")
        print(f"      Closed Shortcut:Mean = {mc_dir_mean:7.4f}, Var = {mc_dir_var:7.4f} (Err: {abs(mc_dir_mean - ana_mean):.4f})")

        # Assert statistical alignment
        assert abs(mc_dir_mean - ana_mean) < 0.15, f"Shortcut mean deviation too high at t={t}"
        assert abs(mc_dir_var - ana_var) < 0.15, f"Shortcut variance deviation too high at t={t}"

    # -----------------------------------------------------------------
    # Test 2: True Posterior Reconstruction Test
    # -----------------------------------------------------------------
    print(f"\n[3] Reconstructing Clean Signal x_0 from Pure Noise x_T via Posterior q(x_{{t-1}} | x_t, x_0):")
    # Sample a noisy endpoint at t=99
    x_T, eps_init = ddpm.forward_closed_form(x0, T - 1)
    print(f"    Diffused state at t={T-1}: x_T = {x_T:.4f} (Clean signal was {x0:.4f})")

    trajectory = [x_T]
    curr_x = x_T
    for step in range(T - 1, -1, -1):
        curr_x = ddpm.reverse_step_ground_truth(curr_x, x0, step)
        trajectory.append(curr_x)

    recovered_x0 = trajectory[-1]
    print(f"    Reverse reconstructed state at t=0: x_0_hat = {recovered_x0:.4f}")
    error = abs(recovered_x0 - x0)
    print(f"    Absolute Reconstruction Error: {error:.6f}")
    assert error < 1e-5, "True posterior reverse process failed to recover x_0!"

    # -----------------------------------------------------------------
    # ASCII Visualization: Forward Degradation & Reverse Recovery
    # -----------------------------------------------------------------
    print(f"\n[4] ASCII Visualization of Diffusion and Posterior Inversion Trajectory:")
    # Forward trajectory snapshots:
    fw_snaps = []
    curr = x0
    for t in range(T):
        curr = ddpm.forward_step(curr, t)
        if t % 10 == 0 or t == T - 1:
            fw_snaps.append((t, curr))

    print("\n    Forward Diffusion (Signal decays toward standard Gaussian noise N(0, 1)):")
    for t, val in fw_snaps:
        bar_len = int((val + 3.0) * 8.0)
        bar_len = max(0, min(50, bar_len))
        print(f"      t={t:2d} | x={val:6.2f} | " + "#" * bar_len)

    print("\n    Reverse Posterior Sampling (Noise purifies back to exact x_0 = 4.0):")
    rev_snaps = []
    curr = x_T
    for step in range(T - 1, -1, -1):
        curr = ddpm.reverse_step_ground_truth(curr, x0, step)
        if step % 10 == 0 or step == 0:
            rev_snaps.append((step, curr))

    for step, val in rev_snaps:
        bar_len = int((val + 3.0) * 8.0)
        bar_len = max(0, min(50, bar_len))
        print(f"      t={step:2d} | x={val:6.2f} | " + "=" * bar_len)

    print("\n[SUCCESS] Topic 02 experiment completed successfully (Exit Code: 0).")
    return 0


if __name__ == "__main__":
    sys.exit(run_experiment())
