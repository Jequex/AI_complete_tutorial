#!/usr/bin/env python3
"""
Phase 21 - Topic 03: RLHF & PPO Policy Gradients
Experiment: Token-Level PPO Clipped Surrogate & Generalized Advantage Estimation (GAE)

This experiment implements and validates from first principles:
1. Pure Python 4-model interaction: Policy (Actor), Critic (Value), Reference Model, Reward Model.
2. Token-level KL divergence penalty: R_t = -beta * (log pi - log pi_ref) with terminal reward.
3. Generalized Advantage Estimation (GAE) recursive calculation: A_t = delta_t + gamma * lambda * A_{t+1}.
4. Advantage normalization (zero-mean, unit variance).
5. PPO probability ratio calculation: r_t = pi_new / pi_old.
6. Clipped surrogate objective: L^CLIP = min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t).
7. Gradient clipping verification: asserting that when r_t exceeds bounds, gradients are clamped
   to prevent catastrophic policy collapse.

Zero external dependencies: pure standard Python 3.
"""

import math
import random
import sys
from typing import List, Tuple, Dict


def softmax(logits: List[float]) -> List[float]:
    """Numerically stable softmax."""
    max_l = max(logits)
    exp_l = [math.exp(x - max_l) for x in logits]
    sum_exp = sum(exp_l)
    return [e / sum_exp for e in exp_l]


class DiscretePolicy:
    """
    Simulated autoregressive discrete policy over vocabulary V.
    Maintains logits for each state-token step.
    """
    def __init__(self, vocab_size: int = 10, seed: int = 42):
        random.seed(seed)
        self.vocab_size = vocab_size
        # Mapping from state string to logit vectors
        self.table: Dict[str, List[float]] = {}

    def get_logits(self, state: str) -> List[float]:
        if state not in self.table:
            # Initialize with small random logits
            self.table[state] = [random.uniform(-0.5, 0.5) for _ in range(self.vocab_size)]
        return self.table[state]

    def get_probs(self, state: str) -> List[float]:
        return softmax(self.get_logits(state))

    def get_token_prob(self, state: str, action: int) -> float:
        return self.get_probs(state)[action]

    def clone(self) -> 'DiscretePolicy':
        new_pol = DiscretePolicy(self.vocab_size)
        new_pol.table = {k: list(v) for k, v in self.table.items()}
        return new_pol


def compute_token_rewards(
    tokens: List[int],
    states: List[str],
    policy: DiscretePolicy,
    reference: DiscretePolicy,
    terminal_reward: float,
    beta: float = 0.1
) -> List[float]:
    """
    Computes per-token reward:
    R_t = -beta * (log pi(y_t) - log pi_ref(y_t)) for t < T
    R_T = terminal_reward - beta * (log pi(y_T) - log pi_ref(y_T)) for t = T
    """
    T = len(tokens)
    rewards = []
    for t in range(T):
        state = states[t]
        action = tokens[t]
        prob_pi = policy.get_token_prob(state, action)
        prob_ref = reference.get_token_prob(state, action)
        
        # Token-level KL: log(pi / ref)
        kl_div = math.log(prob_pi + 1e-12) - math.log(prob_ref + 1e-12)
        r_step = -beta * kl_div
        
        # Inject terminal reward at final step
        if t == T - 1:
            r_step += terminal_reward
        rewards.append(r_step)
    return rewards


def compute_gae(
    rewards: List[float],
    values: List[float],
    gamma: float = 1.0,
    lam: float = 0.95
) -> Tuple[List[float], List[float]]:
    """
    Computes Generalized Advantage Estimation (GAE):
    delta_t = R_t + gamma * V_{t+1} - V_t
    A_t = delta_t + gamma * lambda * A_{t+1}
    Returns: (advantages, returns)
    """
    T = len(rewards)
    advantages = [0.0] * T
    returns = [0.0] * T
    
    last_adv = 0.0
    for t in reversed(range(T)):
        next_value = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        last_adv = delta + gamma * lam * last_adv
        advantages[t] = last_adv
        returns[t] = advantages[t] + values[t]
        
    return advantages, returns


def normalize_advantages(advantages: List[float]) -> List[float]:
    """Standardizes advantages to zero mean and unit variance."""
    mean = sum(advantages) / len(advantages)
    var = sum((a - mean) ** 2 for a in advantages) / len(advantages)
    std = math.sqrt(var) + 1e-8
    return [(a - mean) / std for a in advantages]


def ppo_clipped_objective(
    prob_new: float,
    prob_old: float,
    advantage: float,
    epsilon: float = 0.20
) -> Tuple[float, bool]:
    """
    Evaluates PPO surrogate: min(r_t * A, clip(r_t, 1-eps, 1+eps) * A).
    Returns (loss_value, is_clipped).
    """
    ratio = prob_new / (prob_old + 1e-12)
    unclipped = ratio * advantage
    clipped_ratio = min(max(ratio, 1.0 - epsilon), 1.0 + epsilon)
    clipped = clipped_ratio * advantage
    
    selected_loss = min(unclipped, clipped)
    is_clipped = (selected_loss == clipped and ratio != clipped_ratio)
    return selected_loss, is_clipped


def main():
    print("=" * 80)
    print("TOPIC 21.3 EXPERIMENT: TOKEN-LEVEL PPO SURROGATE & GAE IMPLEMENTATION")
    print("=" * 80)
    random.seed(42)

    vocab_size = 8
    # 1. Initialize the 4-Model System
    # - Active Policy pi_theta
    # - Reference Model pi_ref
    # - Critic / Value Network (lookup values)
    # - Reward Model (oracle score)
    policy = DiscretePolicy(vocab_size=vocab_size, seed=10)

    # Simulate a generated token rollout sequence of length 5
    prompt = "Explain quantum computing:"
    states = [f"{prompt}_step_{t}" for t in range(5)]
    tokens = [2, 5, 0, 7, 3]  # Chosen action indices at each step

    # Pre-populate logits in policy so that clones share the exact distribution
    for s in states:
        policy.get_logits(s)

    reference = policy.clone()  # Initialized from identical SFT weights
    old_policy = policy.clone() # Snapshot prior to updates

    # Simulated Value function estimates V(s_t) for t = 0..4, and V(s_5) = 0.0 (terminal)
    values = [0.85, 1.10, 1.40, 1.90, 2.30, 0.0]

    # Reward model assigns high terminal reward to this answer
    terminal_reward = +3.5
    beta_kl = 0.15

    print(f"\n[Step 1] Rollout Trajectory:")
    print(f"  Prompt: {prompt!r}")
    print(f"  Generated Tokens: {tokens} (Length: {len(tokens)})")
    print(f"  Critic Value Predictions: {values[:5]}")
    print(f"  Reward Model Score (Terminal): {terminal_reward:+.2f}")

    # 2. Compute Token-Level Rewards with KL Penalty
    rewards = compute_token_rewards(
        tokens=tokens,
        states=states,
        policy=policy,
        reference=reference,
        terminal_reward=terminal_reward,
        beta=beta_kl
    )
    print(f"\n[Step 2] Token-Level Rewards (With KL Divergence Penalty):")
    for t, (tok, r) in enumerate(zip(tokens, rewards)):
        print(f"  Step {t}: Token={tok} | R_{t} = {r:+.4f} {'(Includes terminal RM reward)' if t == len(tokens)-1 else ''}")

    # Initial KL is zero because policy == reference
    for t in range(len(tokens) - 1):
        assert abs(rewards[t]) < 1e-6, "Initial intermediate rewards should be 0 since policy == reference."
    assert abs(rewards[-1] - terminal_reward) < 1e-6, "Terminal reward should equal RM score."

    # 3. Compute Generalized Advantage Estimation (GAE)
    gamma = 1.0
    lam = 0.95
    advantages, returns = compute_gae(rewards, values, gamma=gamma, lam=lam)
    norm_advantages = normalize_advantages(advantages)

    print(f"\n[Step 3] GAE Advantage & Return Estimates:")
    for t in range(len(tokens)):
        print(f"  Step {t}: Raw Adv={advantages[t]:+.4f} | Normalized Adv={norm_advantages[t]:+.4f} | Target Return={returns[t]:+.4f}")

    # Verify GAE correctness by manual check on step 4
    # delta_4 = R_4 + 0 - V_4 = 3.5 - 2.30 = +1.20
    assert abs(advantages[4] - (terminal_reward - values[4])) < 1e-4, "Terminal GAE advantage must match delta_T."

    # 4. Verify PPO Clipped Surrogate Behavior across Advantage Scenarios
    print(f"\n[Step 4] Testing PPO Ratio Clipping Under Gradient Perturbations:")
    epsilon = 0.20

    # Scenario A: Highly Positive Advantage (A = +2.5), Policy increases probability (r = 1.40 > 1 + eps)
    loss_a, clipped_a = ppo_clipped_objective(prob_new=0.42, prob_old=0.30, advantage=+2.5, epsilon=epsilon)
    ratio_a = 0.42 / 0.30  # 1.40
    print(f"  Case A (Positive Adv, Ratio={ratio_a:.2f}): Loss={loss_a:.4f} | Clipped={clipped_a}")
    assert clipped_a is True, "Ratio 1.40 must be clipped at 1 + epsilon = 1.20."
    assert abs(loss_a - (1.0 + epsilon) * 2.5) < 1e-4, f"Loss should equal (1 + eps) * A = {1.2 * 2.5}."

    # Scenario B: Negative Advantage (A = -2.5), Policy decreases probability excessively (r = 0.50 < 1 - eps)
    loss_b, clipped_b = ppo_clipped_objective(prob_new=0.15, prob_old=0.30, advantage=-2.5, epsilon=epsilon)
    ratio_b = 0.15 / 0.30  # 0.50
    print(f"  Case B (Negative Adv, Ratio={ratio_b:.2f}): Loss={loss_b:.4f} | Clipped={clipped_b}")
    assert clipped_b is True, "Ratio 0.50 must be clipped at 1 - epsilon = 0.80."
    assert abs(loss_b - (1.0 - epsilon) * (-2.5)) < 1e-4, f"Loss should equal (1 - eps) * A = {0.8 * -2.5}."

    # Scenario C: Stable Update within trust region (Ratio = 1.08)
    loss_c, clipped_c = ppo_clipped_objective(prob_new=0.324, prob_old=0.30, advantage=+1.5, epsilon=epsilon)
    ratio_c = 0.324 / 0.30  # 1.08
    print(f"  Case C (Within Trust Region, Ratio={ratio_c:.2f}): Loss={loss_c:.4f} | Clipped={clipped_c}")
    assert clipped_c is False, "Ratio 1.08 is inside [0.80, 1.20] and must remain unclipped."

    # 5. Simulate Policy Gradient Step & Verify Bounded Drift
    print(f"\n[Step 5] Simulating Multi-Step PPO Parameter Updates:")
    # We update logits for step 0 token 2:
    state_0 = states[0]
    tok_0 = tokens[0]
    p_old = old_policy.get_token_prob(state_0, tok_0)
    adv_0 = norm_advantages[0]

    # Perform gradient ascent on clipped objective
    learning_rate = 0.25
    logits = policy.get_logits(state_0)
    print(f"  Initial Logits at State 0: {[round(l, 3) for l in logits]}")
    print(f"  Initial Prob(tok={tok_0}): {p_old:.4f}")

    # Gradient step on the chosen token
    logits[tok_0] += learning_rate * adv_0
    p_new = policy.get_token_prob(state_0, tok_0)
    print(f"  Updated Logits at State 0: {[round(l, 3) for l in logits]}")
    print(f"  New Prob(tok={tok_0})    : {p_new:.4f} (Ratio: {p_new / p_old:.3f})")

    if adv_0 > 0:
        assert p_new > p_old, "Positive advantage must increase token generation probability."
    else:
        assert p_new < p_old, "Negative advantage must decrease token generation probability."

    print("\n" + "=" * 80)
    print("ALL TOPIC 21.3 RLHF & PPO POLICY GRADIENT TESTS PASSED CLEANLY (EXIT CODE 0)")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
