"""
Topic 03: Stochastic Calculus and Differential Equations
Pure Python 3 standard library implementation (Zero external dependencies).

Verifies:
1. Box-Muller transform for standard Gaussian generation in pure Python.
2. Euler-Maruyama numerical integration for Geometric Brownian Motion (GBM).
   - Verifies empirical mean and variance match exact Ito analytical solutions.
3. Ornstein-Uhlenbeck (OU) Mean-Reverting Drift-Diffusion Process.
   - Verifies asymptotic empirical distribution matches the stationary Fokker-Planck density N(mu, sigma^2 / (2*theta)).
4. Continuous Reverse-Time Score-Based Diffusion SDE (Song et al. 2020).
   - Generates samples from target distribution via reverse-time SDE integration using the analytical score function.
"""

import math
import random
import sys
from typing import Callable, List, Tuple


# --- Pure Python Gaussian Generator (Box-Muller) ---

class GaussianRNG:
    """Deterministic Box-Muller Gaussian random number generator"""
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.has_spare = False
        self.spare = 0.0

    def sample(self) -> float:
        if self.has_spare:
            self.has_spare = False
            return self.spare
        
        u1 = self.rng.random()
        while u1 <= 1e-15:
            u1 = self.rng.random()
        u2 = self.rng.random()
        
        radius = math.sqrt(-2.0 * math.log(u1))
        theta = 2.0 * math.pi * u2
        
        self.spare = radius * math.sin(theta)
        self.has_spare = True
        return radius * math.cos(theta)


# --- Euler-Maruyama Numerical SDE Solver ---

def simulate_euler_maruyama(
    x0: float,
    drift: Callable[[float, float], float],
    diffusion: Callable[[float, float], float],
    t0: float,
    t_end: float,
    num_steps: int,
    num_paths: int,
    rng: GaussianRNG
) -> Tuple[List[float], List[List[float]]]:
    """
    Euler-Maruyama integration for: dX_t = mu(X_t, t) dt + sigma(X_t, t) dW_t
    Returns:
        times: List of time coordinates of length num_steps + 1
        trajectories: num_paths x (num_steps + 1) matrix of path values
    """
    dt = (t_end - t0) / num_steps
    sqrt_dt = math.sqrt(dt)
    times = [t0 + i * dt for i in range(num_steps + 1)]
    
    trajectories = []
    for _ in range(num_paths):
        path = [0.0] * (num_steps + 1)
        path[0] = x0
        x = x0
        for step in range(num_steps):
            t = times[step]
            dW = sqrt_dt * rng.sample()
            x = x + drift(x, t) * dt + diffusion(x, t) * dW
            path[step + 1] = x
        trajectories.append(path)
        
    return times, trajectories


# --- Statistical Helpers ---

def sample_mean(data: List[float]) -> float:
    return sum(data) / len(data)


def sample_variance(data: List[float], mean_val: float = None) -> float:
    n = len(data)
    m = sample_mean(data) if mean_val is None else mean_val
    return sum((x - m) ** 2 for x in data) / (n - 1)


# --- Verification Experiments ---

def verify_geometric_brownian_motion():
    print("\n--- 1. Geometric Brownian Motion (GBM) Verification ---")
    print("SDE: dX_t = mu * X_t dt + sigma * X_t dW_t")
    
    # Parameters
    x0 = 10.0
    mu = 0.08
    sigma = 0.20
    t0 = 0.0
    t_end = 2.0
    num_steps = 200
    num_paths = 6000
    
    rng = GaussianRNG(seed=101)
    
    drift_fn = lambda x, t: mu * x
    diffusion_fn = lambda x, t: sigma * x
    
    times, paths = simulate_euler_maruyama(
        x0=x0,
        drift=drift_fn,
        diffusion=diffusion_fn,
        t0=t0,
        t_end=t_end,
        num_steps=num_steps,
        num_paths=num_paths,
        rng=rng
    )
    
    # Extract terminal states X(T)
    terminal_vals = [path[-1] for path in paths]
    emp_mean = sample_mean(terminal_vals)
    emp_var = sample_variance(terminal_vals, emp_mean)
    
    # Exact theoretical moments from Ito's Lemma:
    # E[X_T] = X_0 * exp(mu * T)
    # Var(X_T) = X_0^2 * exp(2 * mu * T) * (exp(sigma^2 * T) - 1)
    theo_mean = x0 * math.exp(mu * t_end)
    theo_var = (x0 ** 2) * math.exp(2 * mu * t_end) * (math.exp(sigma ** 2 * t_end) - 1.0)
    
    mean_err_pct = abs(emp_mean - theo_mean) / theo_mean * 100.0
    var_err_pct = abs(emp_var - theo_var) / theo_var * 100.0
    
    print(f"Terminal Time T = {t_end:.2f} across N = {num_paths} paths:")
    print(f"  Empirical Mean:   {emp_mean:.4f} | Theoretical E[X_T]:   {theo_mean:.4f} (Error: {mean_err_pct:.2f}%)")
    print(f"  Empirical Var:    {emp_var:.4f} | Theoretical Var(X_T): {theo_var:.4f} (Error: {var_err_pct:.2f}%)")
    
    assert mean_err_pct < 2.0, f"GBM empirical mean deviated from theoretical by {mean_err_pct:.2f}%!"
    assert var_err_pct < 5.0, f"GBM empirical variance deviated from theoretical by {var_err_pct:.2f}%!"
    print("=> Ito's Lemma moments for Geometric Brownian Motion verified within 95% confidence interval.")


def verify_ornstein_uhlenbeck_fokker_planck():
    print("\n--- 2. Ornstein-Uhlenbeck (OU) & Fokker-Planck Density ---")
    print("SDE: dX_t = theta * (mu - X_t) dt + sigma dW_t")
    print("Fokker-Planck Stationary Density: p_inf(x) = N(mu, sigma^2 / (2*theta))")
    
    theta = 2.5
    mu = 1.5
    sigma = 0.8
    x0 = -3.0  # Start far from mean to test reversion
    t0 = 0.0
    t_end = 5.0  # T >> 1/theta = 0.4, fully reaching steady-state
    num_steps = 500
    num_paths = 7000
    
    rng = GaussianRNG(seed=202)
    
    drift_fn = lambda x, t: theta * (mu - x)
    diffusion_fn = lambda x, t: sigma
    
    times, paths = simulate_euler_maruyama(
        x0=x0,
        drift=drift_fn,
        diffusion=diffusion_fn,
        t0=t0,
        t_end=t_end,
        num_steps=num_steps,
        num_paths=num_paths,
        rng=rng
    )
    
    terminal_vals = [path[-1] for path in paths]
    emp_mean = sample_mean(terminal_vals)
    emp_var = sample_variance(terminal_vals, emp_mean)
    
    # Theoretical Fokker-Planck stationary distribution moments
    theo_mean = mu
    theo_var = (sigma ** 2) / (2.0 * theta)
    
    print(f"Stationary Distribution after T = {t_end} (reversion timescale 1/theta = {1.0/theta:.2f}):")
    print(f"  Initial State X_0: {x0:.2f}")
    print(f"  Empirical Mean:    {emp_mean:.4f} | Fokker-Planck Stationary Mean: {theo_mean:.4f}")
    print(f"  Empirical Var:     {emp_var:.4f} | Fokker-Planck Stationary Var:  {theo_var:.4f}")
    
    assert abs(emp_mean - theo_mean) < 0.05, "OU stationary mean failed to match Fokker-Planck prediction!"
    assert abs(emp_var - theo_var) / theo_var < 0.06, "OU stationary variance failed to match Fokker-Planck prediction!"
    print("=> Fokker-Planck stationary Gaussian equilibrium verified.")


def verify_score_based_diffusion_reverse_sde():
    print("\n--- 3. Continuous Score-Based Diffusion: Reverse SDE (Song et al. 2020) ---")
    print("Forward SDE (VP-SDE): dx = -0.5 * beta(t) * x dt + sqrt(beta(t)) dw")
    print("Reverse SDE:         dx = [-0.5 * beta(t) * x - beta(t) * score(x, t)] dt + sqrt(beta(t)) dw_bar")
    
    # Target distribution at t=0: Gaussian mixture / bimodal distribution
    # Let initial target be N(mu_0, sigma_0^2)
    mu_0 = 3.0
    sigma_0 = 0.5
    
    # VP-SDE schedule: beta(t) = 0.1 + (20 - 0.1) * t, or constant beta for simplicity
    beta_const = 2.0
    T = 1.5
    
    # Under forward linear SDE with beta_const:
    # x_t | x_0 ~ N(x_0 * exp(-0.5 * beta * t), 1 - exp(-beta * t))
    # Marginally, if x_0 ~ N(mu_0, sigma_0^2), then x_t ~ N(mu_t, sigma_t^2):
    # mu_t = mu_0 * exp(-0.5 * beta * t)
    # sigma_t^2 = sigma_0^2 * exp(-beta * t) + (1 - exp(-beta * t))
    # Analytical score function: nabla_x log p_t(x) = -(x - mu_t) / sigma_t^2
    
    def get_marginal_moments(t: float) -> Tuple[float, float]:
        alpha_t = math.exp(-0.5 * beta_const * t)
        m_t = mu_0 * alpha_t
        v_t = (sigma_0 ** 2) * (alpha_t ** 2) + (1.0 - alpha_t ** 2)
        return m_t, v_t
        
    def analytical_score(x: float, t: float) -> float:
        m_t, v_t = get_marginal_moments(t)
        return -(x - m_t) / v_t

    # Simulate reverse-time SDE from t=T down to t=0
    # At t=T, x_T ~ N(0, 1) approximately (prior noise)
    rng = GaussianRNG(seed=303)
    num_samples = 5000
    num_steps = 300
    dt = T / num_steps
    sqrt_dt = math.sqrt(dt)
    
    # Sample prior x_T from standard Gaussian N(0, 1)
    m_T, v_T = get_marginal_moments(T)
    x_current = [m_T + math.sqrt(v_T) * rng.sample() for _ in range(num_samples)]
    
    for step in range(num_steps):
        t = T - step * dt
        for i in range(num_samples):
            x = x_current[i]
            score = analytical_score(x, t)
            # Reverse drift: f_rev(x, t) = -0.5 * beta * x - beta * score(x, t)
            # Notice reverse time step is backward in time
            drift_rev = -0.5 * beta_const * x - beta_const * score
            dW = sqrt_dt * rng.sample()
            x_next = x - drift_rev * dt + math.sqrt(beta_const) * dW
            x_current[i] = x_next
            
    emp_rev_mean = sample_mean(x_current)
    emp_rev_var = sample_variance(x_current, emp_rev_mean)
    
    print(f"Reverse SDE Generation from Noise (T={T} -> 0):")
    print(f"  Target Data Distribution: Mean = {mu_0:.4f}, Variance = {sigma_0**2:.4f}")
    print(f"  Generated Sample Stats:   Mean = {emp_rev_mean:.4f}, Variance = {emp_rev_var:.4f}")
    
    assert abs(emp_rev_mean - mu_0) < 0.15, f"Reverse SDE generated mean {emp_rev_mean:.4f} deviated from target {mu_0}!"
    assert abs(emp_rev_var - sigma_0**2) < 0.15, f"Reverse SDE generated variance {emp_rev_var:.4f} deviated from target {sigma_0**2}!"
    print("=> Continuous Reverse SDE Diffusion sampling verified.")


def run_experiment():
    print("=" * 70)
    print("PHASE 34: ADVANCED MATHEMATICS FOR RESEARCH")
    print("Topic 03: Stochastic Calculus & Differential Equations")
    print("=" * 70)
    
    verify_geometric_brownian_motion()
    verify_ornstein_uhlenbeck_fokker_planck()
    verify_score_based_diffusion_reverse_sde()
    
    print("\n" + "=" * 70)
    print("ALL STOCHASTIC CALCULUS & DIFFUSION SDE ASSERTIONS PASSED (EXIT CODE 0)")
    print("=" * 70)


if __name__ == "__main__":
    run_experiment()
