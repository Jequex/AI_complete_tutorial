#!/usr/bin/env python3
"""
Topic 26.4 Experiment: Policy Gradient Theorem & REINFORCE with Baseline
Implements an analytical CartPole physical simulation and compares REINFORCE
with and without a state-value baseline from first principles in pure Python.
Zero external dependencies (no Gym, no PyTorch, no NumPy).
"""

import math
import random

# Fix seed
random.seed(42)

# -------------------------------------------------------------
# Part 1: Physical CartPole Simulation from First Principles
# -------------------------------------------------------------
class CartPoleEnv:
    def __init__(self):
        self.gravity = 9.8
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.total_mass = self.mass_cart + self.mass_pole
        self.length = 0.5  # half pole length
        self.polemass_length = self.mass_pole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # time step seconds

        # Failure thresholds
        self.theta_threshold_radians = 12 * 2 * math.pi / 360  # ~0.209 rad
        self.x_threshold = 2.4

        self.state = None

    def reset(self):
        # Small random initialization around 0
        self.state = [
            (random.random() * 0.1 - 0.05),
            (random.random() * 0.1 - 0.05),
            (random.random() * 0.1 - 0.05),
            (random.random() * 0.1 - 0.05)
        ]
        return list(self.state)

    def step(self, action):
        """
        action: 0 (push left, -F), 1 (push right, +F)
        """
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag

        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # Equations of motion
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.mass_pole * costheta * costheta / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # Euler numerical integration
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = [x, x_dot, theta, theta_dot]

        # Check termination condition
        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        reward = 1.0 if not done else 0.0
        return list(self.state), reward, done

# -------------------------------------------------------------
# Part 2: Softmax Policy & Value Baseline
# -------------------------------------------------------------
def softmax(logits):
    max_l = max(logits)
    exp_l = [math.exp(v - max_l) for v in logits]
    sum_exp = sum(exp_l)
    return [v / sum_exp for v in exp_l]

class SoftmaxPolicy:
    def __init__(self, in_dim=4, num_actions=2, lr=0.05):
        self.in_dim = in_dim
        self.num_actions = num_actions
        self.lr = lr
        # Initialize weights
        self.W = [[0.0 for _ in range(in_dim)] for _ in range(num_actions)]
        self.b = [0.0] * num_actions

    def get_probs(self, s):
        logits = [sum(self.W[a][i] * s[i] for i in range(self.in_dim)) + self.b[a] for a in range(self.num_actions)]
        return softmax(logits)

    def sample_action(self, s):
        probs = self.get_probs(s)
        if random.random() < probs[0]:
            return 0, probs
        return 1, probs

    def update(self, trajectory_grads):
        """Applies gradient ascent step: theta <- theta + lr * grad."""
        for grad_W, grad_b in trajectory_grads:
            for a in range(self.num_actions):
                self.b[a] += self.lr * grad_b[a]
                for i in range(self.in_dim):
                    self.W[a][i] += self.lr * grad_W[a][i]

class LinearCritic:
    """State-Value baseline V_phi(s) = w^T s + b."""
    def __init__(self, in_dim=4, lr=0.05):
        self.in_dim = in_dim
        self.lr = lr
        self.w = [0.0] * in_dim
        self.b = 0.0

    def predict(self, s):
        return sum(self.w[i] * s[i] for i in range(self.in_dim)) + self.b

    def update(self, states, returns):
        """Fit baseline to Monte Carlo returns via MSE gradient descent."""
        for s, G in zip(states, returns):
            v_pred = self.predict(s)
            err = G - v_pred
            # Gradient step to minimize 0.5 * (V - G)^2
            self.b += self.lr * err
            for i in range(self.in_dim):
                self.w[i] += self.lr * err * s[i]

# -------------------------------------------------------------
# Part 3: REINFORCE Training Loop
# -------------------------------------------------------------
def train_reinforce(use_baseline=False, episodes=400, gamma=0.99, max_steps=150):
    env = CartPoleEnv()
    policy = SoftmaxPolicy(in_dim=4, num_actions=2, lr=0.02)
    critic = LinearCritic(in_dim=4, lr=0.05) if use_baseline else None

    episode_rewards = []
    gradient_magnitudes = []

    for ep in range(episodes):
        s = env.reset()
        states = []
        actions = []
        rewards = []
        all_probs = []

        for _ in range(max_steps):
            a, probs = policy.sample_action(s)
            next_s, r, done = env.step(a)

            states.append(s)
            actions.append(a)
            rewards.append(r)
            all_probs.append(probs)

            s = next_s
            if done:
                break

        T = len(rewards)
        episode_rewards.append(sum(rewards))

        # 1. Compute discounted Returns-to-go G_t
        returns = [0.0] * T
        running_g = 0.0
        for t in reversed(range(T)):
            running_g = rewards[t] + gamma * running_g
            returns[t] = running_g

        # 2. Update Critic (if baseline enabled)
        if use_baseline:
            critic.update(states, returns)

        # 3. Compute policy gradients
        ep_grads = []
        ep_grad_norm_sq = 0.0

        for t in range(T):
            s_t = states[t]
            a_t = actions[t]
            probs_t = all_probs[t]
            G_t = returns[t]

            # Subtract baseline if enabled
            if use_baseline:
                advantage = G_t - critic.predict(s_t)
            else:
                advantage = G_t

            # Analytical gradient of log pi(a | s): x * (I(a == k) - pi(k))
            grad_W = [[0.0 for _ in range(4)] for _ in range(2)]
            grad_b = [0.0, 0.0]

            for a_idx in range(2):
                indicator = 1.0 if a_idx == a_t else 0.0
                score_coeff = (indicator - probs_t[a_idx]) * advantage
                grad_b[a_idx] = score_coeff
                for i in range(4):
                    grad_W[a_idx][i] = score_coeff * s_t[i]
                    ep_grad_norm_sq += (score_coeff * s_t[i]) ** 2

            ep_grads.append((grad_W, grad_b))

        # Policy parameter update
        policy.update(ep_grads)
        gradient_magnitudes.append(ep_grad_norm_sq / (T if T > 0 else 1))

    return policy, episode_rewards, gradient_magnitudes

def run_experiment():
    print("=" * 70)
    print("TOPIC 26.4: POLICY GRADIENT THEOREM & REINFORCE VARIANCE REDUCTION")
    print("=" * 70)

    # 1. Run REINFORCE without baseline
    print("\n[1] Training REINFORCE without baseline (400 episodes)...")
    pol_no_base, rew_no_base, grad_no_base = train_reinforce(use_baseline=False, episodes=400)
    avg_no_base_initial = sum(rew_no_base[:40]) / 40
    avg_no_base_final = sum(rew_no_base[-40:]) / 40
    var_no_base = sum(g for g in grad_no_base) / len(grad_no_base)
    print(f"  No Baseline: Initial 40-Ep Avg Duration: {avg_no_base_initial:.1f} steps")
    print(f"  No Baseline: Final   40-Ep Avg Duration: {avg_no_base_final:.1f} steps")
    print(f"  No Baseline: Mean Gradient Variance:     {var_no_base:.4f}")

    # 2. Run REINFORCE with baseline
    print("\n[2] Training REINFORCE with state-value baseline (400 episodes)...")
    pol_base, rew_base, grad_base = train_reinforce(use_baseline=True, episodes=400)
    avg_base_initial = sum(rew_base[:40]) / 40
    avg_base_final = sum(rew_base[-40:]) / 40
    var_base = sum(g for g in grad_base) / len(grad_base)
    print(f"  With Baseline: Initial 40-Ep Avg Duration: {avg_base_initial:.1f} steps")
    print(f"  With Baseline: Final   40-Ep Avg Duration: {avg_base_final:.1f} steps")
    print(f"  With Baseline: Mean Gradient Variance:     {var_base:.4f}")

    variance_reduction = (var_no_base - var_base) / var_no_base * 100
    print(f"\n[3] Variance Reduction Verification:")
    print(f"  Empirical Variance Reduction: {variance_reduction:.1f}%")
    assert var_base < var_no_base, "Baseline did not reduce gradient variance!"

    # 3. Test Trained Policy on Evaluation Episodes
    print("\n[4] Evaluating Policy with Baseline on 10 Test Episodes (max 150 steps):")
    eval_env = CartPoleEnv()
    eval_durations = []
    for ep in range(10):
        s = eval_env.reset()
        dur = 0
        for _ in range(150):
            probs = pol_base.get_probs(s)
            a = 0 if probs[0] > probs[1] else 1
            s, r, done = eval_env.step(a)
            dur += 1
            if done:
                break
        eval_durations.append(dur)
    print(f"  Evaluation Durations: {eval_durations}")
    avg_eval = sum(eval_durations) / len(eval_durations)
    print(f"  Mean Evaluation Duration: {avg_eval:.1f} steps (Initial baseline: ~15-20 steps)")
    assert avg_eval > 50, f"Learned policy did not stabilize pole sufficiently: {avg_eval} steps"

    print("\n[SUCCESS] Topic 26.4 experiment completed with zero errors.")

if __name__ == "__main__":
    run_experiment()
