#!/usr/bin/env python3
"""
Topic 26.5 Experiment: Actor-Critic & Proximal Policy Optimization (PPO)
Implements Generalized Advantage Estimation (GAE) and the PPO Clipped Surrogate Objective
from first principles in pure Python.
Zero external dependencies (no PyTorch, no Gym, no NumPy).
"""

import math
import random

# Fix random seed
random.seed(42)

# -------------------------------------------------------------
# Part 1: Continuous 1D Corridor Navigation Environment
# -------------------------------------------------------------
class CorridorEnv:
    """
    Continuous state space: [position x, velocity v]
    Target is at x >= 0.8
    Actions: 0 (Push Left: -0.1), 1 (Coast: 0.0), 2 (Push Right: +0.1)
    """
    def __init__(self):
        self.min_x = -1.0
        self.max_x = 1.0
        self.goal_x = 0.8
        self.state = None

    def reset(self):
        # Start randomly on the left side
        self.state = [random.uniform(-0.9, -0.6), 0.0]
        return list(self.state)

    def step(self, action):
        x, v = self.state
        accel = [-0.08, 0.0, 0.08][action]
        # Physics with drag/friction
        new_v = 0.9 * v + accel
        new_v = max(-0.25, min(0.25, new_v))
        new_x = x + new_v

        # Boundary checks
        done = False
        reward = -0.05  # Step time penalty

        if new_x >= self.goal_x:
            reward = 10.0
            done = True
            new_x = self.goal_x
        elif new_x <= self.min_x:
            new_x = self.min_x
            new_v = 0.0
            reward = -0.5  # Bump penalty

        self.state = [new_x, new_v]
        return list(self.state), reward, done

# -------------------------------------------------------------
# Part 2: Pure Python Actor and Critic Models
# -------------------------------------------------------------
def softmax(logits):
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    s = sum(exps)
    return [e / s for e in exps]

class PPOActor:
    """Linear Softmax Policy Actor: pi_theta(a | s)."""
    def __init__(self, in_dim=2, num_actions=3, lr=0.03):
        self.in_dim = in_dim
        self.num_actions = num_actions
        self.lr = lr
        # Initialize weights small
        self.W = [[random.uniform(-0.05, 0.05) for _ in range(in_dim)] for _ in range(num_actions)]
        self.b = [0.0] * num_actions

    def get_probs(self, s):
        logits = [sum(self.W[a][i] * s[i] for i in range(self.in_dim)) + self.b[a] for a in range(self.num_actions)]
        return softmax(logits)

    def sample_action(self, s):
        probs = self.get_probs(s)
        r = random.random()
        cum = 0.0
        for a, p in enumerate(probs):
            cum += p
            if r <= cum:
                return a, probs[a]
        return self.num_actions - 1, probs[-1]

    def copy_from(self, other):
        self.W = [row[:] for row in other.W]
        self.b = other.b[:]

class PPOCritic:
    """Linear Value Function Critic: V_phi(s)."""
    def __init__(self, in_dim=2, lr=0.05):
        self.in_dim = in_dim
        self.lr = lr
        self.W = [0.0] * in_dim
        self.b = 0.0

    def predict(self, s):
        return sum(self.W[i] * s[i] for i in range(self.in_dim)) + self.b

# -------------------------------------------------------------
# Part 3: Generalized Advantage Estimation (GAE)
# -------------------------------------------------------------
def compute_gae(rewards, values, next_value, gamma=0.98, lam=0.95):
    """
    Computes GAE advantages backward:
    delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    A_t = delta_t + gamma * lam * A_{t+1}
    """
    T = len(rewards)
    advantages = [0.0] * T
    gae = 0.0

    for t in reversed(range(T)):
        next_v = next_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae

    # Target returns for Critic: V_target = A + V
    returns = [adv + v for adv, v in zip(advantages, values)]
    return advantages, returns

# -------------------------------------------------------------
# Part 4: PPO Training Engine
# -------------------------------------------------------------
def train_ppo(total_iterations=120, steps_per_iter=150, ppo_epochs=4, clip_eps=0.2):
    env = CorridorEnv()
    actor = PPOActor(in_dim=2, num_actions=3, lr=0.04)
    critic = PPOCritic(in_dim=2, lr=0.06)

    # Old policy for importance ratio
    old_actor = PPOActor(in_dim=2, num_actions=3, lr=0.04)

    iteration_rewards = []
    clip_fractions = []

    s = env.reset()

    for it in range(total_iterations):
        # Sync old policy
        old_actor.copy_from(actor)

        # 1. Collect rollout batch
        batch_s = []
        batch_a = []
        batch_r = []
        batch_old_p = []
        batch_v = []

        ep_rewards = []
        current_ep_reward = 0.0

        for _ in range(steps_per_iter):
            a, p_a = old_actor.sample_action(s)
            v = critic.predict(s)

            next_s, r, done = env.step(a)

            batch_s.append(s)
            batch_a.append(a)
            batch_r.append(r)
            batch_old_p.append(p_a)
            batch_v.append(v)

            current_ep_reward += r
            if done:
                ep_rewards.append(current_ep_reward)
                current_ep_reward = 0.0
                s = env.reset()
            else:
                s = next_s

        last_v = critic.predict(s) if not done else 0.0

        # 2. Compute GAE Advantages
        advantages, returns = compute_gae(batch_r, batch_v, last_v)

        # Standardize advantages
        mean_adv = sum(advantages) / len(advantages)
        std_adv = math.sqrt(sum((x - mean_adv) ** 2 for x in advantages) / len(advantages)) + 1e-8
        norm_adv = [(a - mean_adv) / std_adv for a in advantages]

        # 3. PPO Optimization Epochs on collected batch
        total_clips = 0
        total_samples = 0

        for _ in range(ppo_epochs):
            grad_W = [[0.0 for _ in range(2)] for _ in range(3)]
            grad_b = [0.0] * 3

            for i in range(len(batch_s)):
                st = batch_s[i]
                at = batch_a[i]
                old_prob = batch_old_p[i]
                adv = norm_adv[i]

                current_probs = actor.get_probs(st)
                curr_prob = current_probs[at]

                # Importance sampling ratio r_t(theta)
                r_t = curr_prob / (old_prob + 1e-12)

                # Check clipping conditions
                clipped_r = max(1.0 - clip_eps, min(1.0 + clip_eps, r_t))
                is_clipped = False

                if adv > 0 and r_t > 1.0 + clip_eps:
                    is_clipped = True
                elif adv < 0 and r_t < 1.0 - clip_eps:
                    is_clipped = True

                total_samples += 1
                if is_clipped:
                    total_clips += 1
                    effective_adv = 0.0  # Zero gradient when clipped
                else:
                    effective_adv = adv

                # Policy gradient w.r.t logits: s * (I(a==k) - prob_k) * (ratio * adv)
                weight = (effective_adv / (old_prob + 1e-12))
                for a_idx in range(3):
                    indicator = 1.0 if a_idx == at else 0.0
                    logit_grad = (indicator - current_probs[a_idx]) * curr_prob * weight
                    grad_b[a_idx] += logit_grad
                    for dim in range(2):
                        grad_W[a_idx][dim] += logit_grad * st[dim]

            # Apply SGD to Actor
            N = len(batch_s)
            for a_idx in range(3):
                actor.b[a_idx] += actor.lr * (grad_b[a_idx] / N)
                for dim in range(2):
                    actor.W[a_idx][dim] += actor.lr * (grad_W[a_idx][dim] / N)

            # Update Critic via MSE regression
            for i in range(N):
                st = batch_s[i]
                targ = returns[i]
                v_pred = critic.predict(st)
                err = targ - v_pred
                critic.b += critic.lr * (err / N)
                for dim in range(2):
                    critic.W[dim] += critic.lr * (err * st[dim] / N)

        clip_rate = total_clips / (total_samples if total_samples > 0 else 1)
        clip_fractions.append(clip_rate)

        mean_reward = sum(ep_rewards) / len(ep_rewards) if ep_rewards else current_ep_reward
        iteration_rewards.append(mean_reward)

    return actor, critic, iteration_rewards, clip_fractions

def evaluate_agent(actor, num_episodes=15):
    env = CorridorEnv()
    successes = 0
    eval_rewards = []

    for _ in range(num_episodes):
        s = env.reset()
        ep_ret = 0.0
        for _ in range(60):
            probs = actor.get_probs(s)
            a = max(range(3), key=lambda act: probs[act])
            s, r, done = env.step(a)
            ep_ret += r
            if done:
                if s[0] >= 0.8:
                    successes += 1
                break
        eval_rewards.append(ep_ret)

    success_rate = successes / num_episodes * 100
    mean_ret = sum(eval_rewards) / len(eval_rewards)
    return success_rate, mean_ret

def run_experiment():
    print("=" * 70)
    print("TOPIC 26.5: ACTOR-CRITIC & PPO CLIPPED SURROGATE OPTIMIZATION")
    print("=" * 70)

    print("\n[1] Training PPO with GAE on 1D Corridor Navigation (120 iterations)...")
    actor, critic, iter_rewards, clip_fracs = train_ppo(
        total_iterations=120, steps_per_iter=150, ppo_epochs=4, clip_eps=0.2
    )

    initial_perf = sum(iter_rewards[:15]) / 15
    final_perf = sum(iter_rewards[-15:]) / 15
    avg_clip_rate = sum(clip_fracs) / len(clip_fracs) * 100

    print(f"  PPO Initial 15-Iter Avg Reward: {initial_perf:6.2f}")
    print(f"  PPO Final   15-Iter Avg Reward: {final_perf:6.2f}")
    print(f"  Mean Clipping Trigger Rate:     {avg_clip_rate:6.2f}%")

    # Verify performance improvement
    print(f"\n[2] Performance Improvement Verification:")
    improvement = final_perf - initial_perf
    print(f"  Delta Score: +{improvement:.2f} points")
    assert final_perf > initial_perf, "PPO failed to improve expected return!"

    # 3. Test Trained Policy
    print("\n[3] Evaluating Trained Policy across 15 Test Episodes:")
    succ_rate, mean_eval_ret = evaluate_agent(actor, num_episodes=15)
    print(f"  Goal Success Rate: {succ_rate:.1f}%")
    print(f"  Mean Evaluation Score: {mean_eval_ret:.2f}")

    assert succ_rate >= 80.0, f"PPO agent success rate below 80%: {succ_rate}%"
    print("  Goal achievement verification: PASS (>= 80% success rate)")

    print("\n[SUCCESS] Topic 26.5 experiment completed with zero errors.")

if __name__ == "__main__":
    run_experiment()
