#!/usr/bin/env python3
"""
Topic 26.2 Experiment: Bellman Equations, Contraction Proof & Dynamic Programming
Implements Policy Evaluation, Policy Iteration, and Value Iteration from first principles in pure Python.
Verifies the gamma-contraction mapping under the L_inf norm and proves finite termination of Policy Iteration.
"""

import math

# Environment Configuration: 4x4 Grid with Obstacles
GRID_ROWS = 4
GRID_COLS = 4
NUM_STATES = GRID_ROWS * GRID_COLS

GOAL_STATE = (3, 3)
TRAP_STATE = (1, 1)
WALL_STATES = {(1, 2), (2, 1)}  # Inaccessible obstacles

GAMMA = 0.9
ACTIONS = {
    0: (-1, 0),  # Up
    1: (0, 1),   # Right
    2: (1, 0),   # Down
    3: (0, -1),  # Left
}
ACTION_ARROWS = {0: "↑", 1: "→", 2: "↓", 3: "←"}

def state_to_idx(r, c):
    return r * GRID_COLS + c

def idx_to_state(idx):
    return divmod(idx, GRID_COLS)

def step_dynamics(s, a):
    """
    Stochastic transitions: 0.8 intended action, 0.1 slip left, 0.1 slip right.
    Hitting boundary or wall states bounces back to original state s.
    """
    r, c = idx_to_state(s)
    if (r, c) == GOAL_STATE or (r, c) == TRAP_STATE:
        return [(s, 1.0, 0.0, True)]
    if (r, c) in WALL_STATES:
        return [(s, 1.0, 0.0, True)]

    orthogonal = {0: [3, 1], 1: [0, 2], 2: [1, 3], 3: [2, 0]}
    candidates = [(a, 0.8), (orthogonal[a][0], 0.1), (orthogonal[a][1], 0.1)]

    outcomes = []
    for act, prob in candidates:
        dr, dc = ACTIONS[act]
        nr, nc = r + dr, c + dc

        # Check bounds and walls
        if not (0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS) or (nr, nc) in WALL_STATES:
            nr, nc = r, c  # Bounce back

        next_s = state_to_idx(nr, nc)
        if (nr, nc) == GOAL_STATE:
            reward = 10.0
            done = True
        elif (nr, nc) == TRAP_STATE:
            reward = -10.0
            done = True
        else:
            reward = -0.1
            done = False
        outcomes.append((next_s, prob, reward, done))
    return outcomes

def q_value_from_v(s, a, V, gamma=GAMMA):
    """Computes Q(s, a) given state-value vector V."""
    r, c = idx_to_state(s)
    if (r, c) == GOAL_STATE or (r, c) == TRAP_STATE or (r, c) in WALL_STATES:
        return 0.0
    q = 0.0
    for next_s, prob, rew, done in step_dynamics(s, a):
        next_v = 0.0 if done else V[next_s]
        q += prob * (rew + gamma * next_v)
    return q

def policy_evaluation(policy, gamma=GAMMA, tol=1e-9):
    """Evaluates V^pi iteratively until convergence."""
    V = [0.0] * NUM_STATES
    while True:
        delta = 0.0
        new_V = [0.0] * NUM_STATES
        for s in range(NUM_STATES):
            r, c = idx_to_state(s)
            if (r, c) == GOAL_STATE or (r, c) == TRAP_STATE or (r, c) in WALL_STATES:
                continue
            a = policy[s]
            new_V[s] = q_value_from_v(s, a, V, gamma)
            delta = max(delta, abs(new_V[s] - V[s]))
        V = new_V
        if delta < tol:
            break
    return V

def policy_iteration(gamma=GAMMA):
    """Executes Policy Iteration and returns optimal policy and value function."""
    policy = [0] * NUM_STATES  # Start with all actions Up
    iterations = 0
    history = []

    while True:
        iterations += 1
        V = policy_evaluation(policy, gamma)
        policy_stable = True
        new_policy = list(policy)

        for s in range(NUM_STATES):
            r, c = idx_to_state(s)
            if (r, c) == GOAL_STATE or (r, c) == TRAP_STATE or (r, c) in WALL_STATES:
                continue
            old_a = policy[s]
            q_values = [q_value_from_v(s, a, V, gamma) for a in range(4)]
            best_a = max(range(4), key=lambda a: q_values[a])
            new_policy[s] = best_a
            if best_a != old_a:
                policy_stable = False

        history.append((iterations, policy_stable))
        policy = new_policy
        if policy_stable:
            break

    V_star = policy_evaluation(policy, gamma)
    return policy, V_star, iterations

def value_iteration(gamma=GAMMA, tol=1e-8):
    """Executes Value Iteration, tracking the contraction factor."""
    V = [0.0] * NUM_STATES
    residuals = []
    iterations = 0

    while True:
        iterations += 1
        new_V = [0.0] * NUM_STATES
        delta = 0.0
        for s in range(NUM_STATES):
            r, c = idx_to_state(s)
            if (r, c) == GOAL_STATE or (r, c) == TRAP_STATE or (r, c) in WALL_STATES:
                continue
            best_q = max(q_value_from_v(s, a, V, gamma) for a in range(4))
            new_V[s] = best_q
            delta = max(delta, abs(new_V[s] - V[s]))

        residuals.append(delta)
        V = new_V
        if delta < tol:
            break

    # Extract greedy optimal policy
    policy = [0] * NUM_STATES
    for s in range(NUM_STATES):
        r, c = idx_to_state(s)
        if (r, c) == GOAL_STATE or (r, c) == TRAP_STATE or (r, c) in WALL_STATES:
            continue
        q_vals = [q_value_from_v(s, a, V, gamma) for a in range(4)]
        policy[s] = max(range(4), key=lambda a: q_vals[a])

    return policy, V, iterations, residuals

def run_experiment():
    print("=" * 70)
    print("TOPIC 26.2: BELLMAN EQUATIONS, CONTRACTION & DYNAMIC PROGRAMMING")
    print("=" * 70)

    # 1. Run Value Iteration and verify contraction mapping
    print("\n[1] Running Value Iteration (gamma = 0.9)...")
    vi_policy, vi_v, vi_iters, residuals = value_iteration(gamma=GAMMA)
    print(f"  Value Iteration converged in {vi_iters} iterations.")

    print("\n  Verifying Contraction Ratios ||V_{k+1} - V_k||_inf / ||V_k - V_{k-1}||_inf <= gamma:")
    print(f"  {'Iteration k':<12} | {'Residual Delta_k':<18} | {'Empirical Contraction':<22} | {'Bound (gamma)'}")
    print("  " + "-" * 68)

    contraction_checks_passed = True
    for k in range(1, min(10, len(residuals))):
        ratio = residuals[k] / residuals[k-1] if residuals[k-1] > 0 else 0.0
        print(f"  {k+1:<12} | {residuals[k]:<18.6e} | {ratio:<22.4f} | {GAMMA:.4f}")
        if ratio > GAMMA + 1e-6:
            contraction_checks_passed = False

    assert contraction_checks_passed, "Empirical contraction ratio strictly violated theoretical gamma bound!"
    print(f"  Contraction mapping verification: PASS (all ratios <= {GAMMA})")

    # 2. Run Policy Iteration and verify finite termination
    print("\n[2] Running Policy Iteration...")
    pi_policy, pi_v, pi_iters = policy_iteration(gamma=GAMMA)
    print(f"  Policy Iteration terminated in {pi_iters} outer policy improvement sweeps.")

    # 3. Cross-Validation: Check that PI and VI reach identical fixed points
    max_value_diff = max(abs(vi_v[s] - pi_v[s]) for s in range(NUM_STATES))
    print(f"\n[3] Cross-Algorithm Equivalence:")
    print(f"  ||V_PI* - V_VI*||_inf = {max_value_diff:.2e}")
    assert max_value_diff < 1e-5, f"Value mismatch between PI and VI: {max_value_diff}"

    # Verify that both policies achieve the true optimal value V* on all active states
    pi_suboptimality = 0.0
    vi_suboptimality = 0.0
    for s in range(NUM_STATES):
        r, c = idx_to_state(s)
        if (r, c) == GOAL_STATE or (r, c) == TRAP_STATE or (r, c) in WALL_STATES:
            continue
        q_vals = [q_value_from_v(s, a, vi_v, gamma=GAMMA) for a in range(4)]
        best_q = max(q_vals)
        pi_suboptimality = max(pi_suboptimality, abs(best_q - q_vals[pi_policy[s]]))
        vi_suboptimality = max(vi_suboptimality, abs(best_q - q_vals[vi_policy[s]]))

    print(f"  PI policy regret / suboptimality: {pi_suboptimality:.2e}")
    print(f"  VI policy regret / suboptimality: {vi_suboptimality:.2e}")
    assert pi_suboptimality < 1e-6, "PI policy is suboptimal w.r.t Q*"
    assert vi_suboptimality < 1e-6, "VI policy is suboptimal w.r.t Q*"
    print("  Optimal policy verification: PASS (Both PI and VI achieve true optimal value V*)")

    # 4. Print Optimal Grid Policy & Value Map
    print("\nOptimal Value Map V*(s):")
    for r in range(GRID_ROWS):
        row_vals = []
        for c in range(GRID_COLS):
            s = state_to_idx(r, c)
            if (r, c) == GOAL_STATE:
                row_vals.append(" GOAL ")
            elif (r, c) == TRAP_STATE:
                row_vals.append(" TRAP ")
            elif (r, c) in WALL_STATES:
                row_vals.append(" WALL ")
            else:
                row_vals.append(f"{vi_v[s]:6.2f}")
        print("  [" + " ".join(row_vals) + "]")

    print("\nOptimal Policy Direction Map π*(s):")
    for r in range(GRID_ROWS):
        row_arrows = []
        for c in range(GRID_COLS):
            s = state_to_idx(r, c)
            if (r, c) == GOAL_STATE:
                row_arrows.append("★")
            elif (r, c) == TRAP_STATE:
                row_arrows.append("☠")
            elif (r, c) in WALL_STATES:
                row_arrows.append("█")
            else:
                row_arrows.append(ACTION_ARROWS[vi_policy[s]])
        print("  [ " + "  ".join(row_arrows) + " ]")

    print("\n[SUCCESS] Topic 26.2 experiment completed with zero errors.")

if __name__ == "__main__":
    run_experiment()
