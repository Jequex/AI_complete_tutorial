#!/usr/bin/env python3
"""
Topic 26.1 Experiment: Markov Decision Processes & Analytical vs Empirical Returns
Implements a stochastic GridWorld MDP from first principles in pure Python.
Computes exact analytical state values via linear algebra (Gaussian elimination)
and verifies convergence of Monte Carlo empirical trajectory rollouts.
"""

import math
import random

# Fix random seed for strict reproducibility
random.seed(42)

# GridWorld Definition: 4x4 grid (16 states)
GRID_ROWS = 4
GRID_COLS = 4
NUM_STATES = GRID_ROWS * GRID_COLS

GOAL_STATE = (3, 3)
PIT_STATE = (1, 1)
GAMMA = 0.9

ACTIONS = {
    0: (-1, 0),  # Up
    1: (0, 1),   # Right
    2: (1, 0),   # Down
    3: (0, -1),  # Left
}
ACTION_NAMES = {0: "Up", 1: "Right", 2: "Down", 3: "Left"}

def state_to_idx(r, c):
    return r * GRID_COLS + c

def idx_to_state(idx):
    return divmod(idx, GRID_COLS)

def step_dynamics(s, a):
    """
    Returns list of (next_state, prob, reward, done)
    Stochastic: 0.8 intended action, 0.1 slip orthogonal-left, 0.1 slip orthogonal-right
    """
    r, c = idx_to_state(s)
    if (r, c) == GOAL_STATE or (r, c) == PIT_STATE:
        return [(s, 1.0, 0.0, True)]

    # Slip transitions
    orthogonal = {
        0: [3, 1], # Up slips Left or Right
        1: [0, 2], # Right slips Up or Down
        2: [1, 3], # Down slips Right or Left
        3: [2, 0], # Left slips Down or Up
    }

    outcomes = []
    candidates = [(a, 0.8), (orthogonal[a][0], 0.1), (orthogonal[a][1], 0.1)]

    for act, prob in candidates:
        dr, dc = ACTIONS[act]
        nr = min(max(r + dr, 0), GRID_ROWS - 1)
        nc = min(max(c + dc, 0), GRID_COLS - 1)
        next_s = state_to_idx(nr, nc)

        if (nr, nc) == GOAL_STATE:
            reward = 10.0
            done = True
        elif (nr, nc) == PIT_STATE:
            reward = -5.0
            done = True
        else:
            reward = -0.1
            done = False

        outcomes.append((next_s, prob, reward, done))
    return outcomes

def gaussian_elimination(A, b):
    """Solve A x = b using Gaussian elimination with partial pivoting."""
    n = len(A)
    # Augment matrix [A | b]
    M = [[A[i][j] for j in range(n)] + [b[i]] for i in range(n)]

    for i in range(n):
        # Pivot search
        max_row = max(range(i, n), key=lambda r: abs(M[r][i]))
        if abs(M[max_row][i]) < 1e-12:
            raise ValueError("Singular matrix encountered.")
        M[i], M[max_row] = M[max_row], M[i]

        pivot = M[i][i]
        for j in range(i, n + 1):
            M[i][j] /= pivot

        for r in range(n):
            if r != i:
                factor = M[r][i]
                for c in range(i, n + 1):
                    M[r][c] -= factor * M[i][c]

    return [M[i][n] for i in range(n)]

def compute_analytical_values(policy, gamma=GAMMA):
    """
    Solves (I - gamma * P_pi) v = r_pi for exact analytical V^pi
    """
    n = NUM_STATES
    P_pi = [[0.0 for _ in range(n)] for _ in range(n)]
    r_pi = [0.0 for _ in range(n)]

    for s in range(n):
        r, c = idx_to_state(s)
        if (r, c) == GOAL_STATE or (r, c) == PIT_STATE:
            P_pi[s][s] = 1.0
            r_pi[s] = 0.0
            continue

        for a, p_act in enumerate(policy[s]):
            if p_act == 0.0:
                continue
            for next_s, p_trans, rew, done in step_dynamics(s, a):
                prob = p_act * p_trans
                r_pi[s] += prob * rew
                if not done:
                    P_pi[s][next_s] += prob

    # Form (I - gamma * P_pi)
    A = [[(1.0 if i == j else 0.0) - gamma * P_pi[i][j] for j in range(n)] for i in range(n)]
    v_exact = gaussian_elimination(A, r_pi)
    return v_exact

def simulate_trajectory(start_s, policy, gamma=GAMMA, max_steps=100):
    """Generates one Monte Carlo rollout return G_0."""
    s = start_s
    r, c = idx_to_state(s)
    if (r, c) == GOAL_STATE or (r, c) == PIT_STATE:
        return 0.0

    discounted_return = 0.0
    discount = 1.0

    for step in range(max_steps):
        # Sample action from policy
        probs = policy[s]
        rand_val = random.random()
        cumulative = 0.0
        chosen_a = 0
        for a, p in enumerate(probs):
            cumulative += p
            if rand_val <= cumulative:
                chosen_a = a
                break

        # Step environment
        outcomes = step_dynamics(s, chosen_a)
        rand_trans = random.random()
        cum_trans = 0.0
        for next_s, p_trans, rew, done in outcomes:
            cum_trans += p_trans
            if rand_trans <= cum_trans:
                discounted_return += discount * rew
                discount *= gamma
                s = next_s
                if done:
                    return discounted_return
                break
    return discounted_return

def run_experiment():
    print("=" * 70)
    print("TOPIC 26.1: MARKOV DECISION PROCESSES & ANALYTICAL VS EMPIRICAL RETURNS")
    print("=" * 70)

    # 1. Define Policies
    # Policy A: Uniform Random (explore all actions equally)
    uniform_policy = [[0.25, 0.25, 0.25, 0.25] for _ in range(NUM_STATES)]

    # Policy B: Goal-directed heuristic (prefer Down and Right)
    guided_policy = []
    for s in range(NUM_STATES):
        r, c = idx_to_state(s)
        # Choose Down or Right towards (3, 3)
        probs = [0.05, 0.45, 0.45, 0.05]
        guided_policy.append(probs)

    print("\n[1] Computing exact closed-form analytical state-values (I - gamma*P)^(-1) * r...")
    v_uniform_exact = compute_analytical_values(uniform_policy)
    v_guided_exact = compute_analytical_values(guided_policy)

    print("\nAnalytical State Values V(s) for Uniform Policy (4x4 Grid):")
    for r in range(GRID_ROWS):
        row_str = "  ".join(f"{v_uniform_exact[state_to_idx(r, c)]:7.3f}" for c in range(GRID_COLS))
        print(f"  Row {r}: [{row_str}]")

    print("\nAnalytical State Values V(s) for Guided Policy (4x4 Grid):")
    for r in range(GRID_ROWS):
        row_str = "  ".join(f"{v_guided_exact[state_to_idx(r, c)]:7.3f}" for c in range(GRID_COLS))
        print(f"  Row {r}: [{row_str}]")

    # 2. Monte Carlo Empirical Sampling across sample sizes
    print("\n[2] Running Monte Carlo empirical rollouts for state (0, 0)...")
    s0 = state_to_idx(0, 0)
    exact_v0 = v_uniform_exact[s0]

    sample_sizes = [50, 200, 1000, 5000]
    print(f"  Target Analytical V_uniform(0, 0) = {exact_v0:.4f}")
    print(f"  {'Samples (N)':<12} | {'Empirical V_hat':<16} | {'Absolute Error':<16} | {'Std Error'}")
    print("  " + "-" * 60)

    for n_samples in sample_sizes:
        returns = [simulate_trajectory(s0, uniform_policy) for _ in range(n_samples)]
        v_hat = sum(returns) / n_samples
        variance = sum((x - v_hat) ** 2 for x in returns) / (n_samples - 1 if n_samples > 1 else 1)
        std_err = math.sqrt(variance / n_samples)
        abs_err = abs(v_hat - exact_v0)
        print(f"  {n_samples:<12} | {v_hat:<16.4f} | {abs_err:<16.4f} | {std_err:.4f}")

    # Verify statistical convergence
    final_returns = [simulate_trajectory(s0, uniform_policy) for _ in range(10000)]
    final_v_hat = sum(final_returns) / 10000
    final_err = abs(final_v_hat - exact_v0)
    print(f"\nFinal Check with 10,000 rollouts: Error = {final_err:.4f}")
    assert final_err < 0.25, f"Empirical return did not converge to analytical value: {final_err}"

    # Guided policy value must strictly exceed uniform policy value at start state
    assert v_guided_exact[s0] > v_uniform_exact[s0], "Guided policy should outperform uniform policy!"
    print(f"\nPolicy Comparison at Start (0, 0):")
    print(f"  Uniform Policy Value: {v_uniform_exact[s0]:.4f}")
    print(f"  Guided Policy Value:  {v_guided_exact[s0]:.4f} (Outperforms uniform: PASS)")
    print("\n[SUCCESS] Topic 26.1 experiment completed with zero errors.")

if __name__ == "__main__":
    run_experiment()
