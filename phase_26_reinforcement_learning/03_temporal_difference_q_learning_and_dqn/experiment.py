#!/usr/bin/env python3
"""
Topic 26.3 Experiment: Tabular Q-Learning vs Mini-DQN
Implements Tabular Q-Learning and a pure-Python Multi-Layer Perceptron Deep Q-Network (DQN)
with Experience Replay and a Target Network on a Cliff Walking environment.
Zero external dependencies (no PyTorch, no NumPy).
"""

import math
import random

# Fix random seed
random.seed(42)

# Environment: Cliff Walking Grid (3 rows x 6 cols)
ROWS = 3
COLS = 6
NUM_STATES = ROWS * COLS
START_STATE = (2, 0)
GOAL_STATE = (2, 5)
CLIFF_STATES = {(2, 1), (2, 2), (2, 3), (2, 4)}

ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # 0: Up, 1: Right, 2: Down, 3: Left
NUM_ACTIONS = 4

def state_to_idx(r, c):
    return r * COLS + c

def idx_to_state(idx):
    return divmod(idx, COLS)

def step_env(s, a):
    """Executes action a from state s. Returns next_s, reward, done."""
    r, c = idx_to_state(s)
    dr, dc = ACTIONS[a]
    nr = max(0, min(ROWS - 1, r + dr))
    nc = max(0, min(COLS - 1, c + dc))

    if (nr, nc) == GOAL_STATE:
        return state_to_idx(nr, nc), 10.0, True
    elif (nr, nc) in CLIFF_STATES:
        # Fall off cliff: heavy penalty and reset to start
        return state_to_idx(START_STATE[0], START_STATE[1]), -20.0, False
    else:
        return state_to_idx(nr, nc), -1.0, False

# -------------------------------------------------------------
# Part 1: Tabular Q-Learning
# -------------------------------------------------------------
def train_tabular_q(episodes=300, alpha=0.1, gamma=0.95, eps_start=1.0, eps_end=0.05):
    Q = [[0.0 for _ in range(NUM_ACTIONS)] for _ in range(NUM_STATES)]
    rewards_history = []

    for ep in range(episodes):
        s = state_to_idx(*START_STATE)
        eps = max(eps_end, eps_start - (eps_start - eps_end) * (ep / episodes))
        ep_reward = 0.0

        for _ in range(80):
            if random.random() < eps:
                a = random.randint(0, NUM_ACTIONS - 1)
            else:
                a = max(range(NUM_ACTIONS), key=lambda act: Q[s][act])

            next_s, r, done = step_env(s, a)
            ep_reward += r

            best_next_q = max(Q[next_s]) if not done else 0.0
            target = r + gamma * best_next_q
            Q[s][a] += alpha * (target - Q[s][a])

            s = next_s
            if done:
                break

        rewards_history.append(ep_reward)

    return Q, rewards_history

# -------------------------------------------------------------
# Part 2: Pure Python Mini-DQN (Multi-Layer Perceptron)
# -------------------------------------------------------------
class MiniMLP:
    """A 2-layer Neural Network with ReLU activation in Pure Python."""
    def __init__(self, in_dim, hidden_dim, out_dim, lr=0.02):
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.lr = lr

        scale1 = math.sqrt(2.0 / in_dim)
        self.W1 = [[(random.random() * 2 - 1) * scale1 for _ in range(in_dim)] for _ in range(hidden_dim)]
        self.b1 = [0.0] * hidden_dim

        scale2 = math.sqrt(2.0 / hidden_dim)
        self.W2 = [[(random.random() * 2 - 1) * scale2 for _ in range(hidden_dim)] for _ in range(out_dim)]
        self.b2 = [0.0] * out_dim

    def forward(self, x):
        h_pre = [sum(self.W1[j][i] * x[i] for i in range(self.in_dim)) + self.b1[j] for j in range(self.hidden_dim)]
        h = [max(0.0, val) for val in h_pre]
        q = [sum(self.W2[k][j] * h[j] for j in range(self.hidden_dim)) + self.b2[k] for k in range(self.out_dim)]
        return h_pre, h, q

    def predict(self, x):
        _, _, q = self.forward(x)
        return q

    def train_batch(self, batch_data):
        """
        Computes gradients over mini-batch of (x, a, y_target) and performs SGD update.
        """
        B = len(batch_data)
        if B == 0:
            return 0.0

        dW1 = [[0.0 for _ in range(self.in_dim)] for _ in range(self.hidden_dim)]
        db1 = [0.0] * self.hidden_dim
        dW2 = [[0.0 for _ in range(self.hidden_dim)] for _ in range(self.out_dim)]
        db2 = [0.0] * self.out_dim
        total_loss = 0.0

        for x, a, y in batch_data:
            h_pre, h, q = self.forward(x)
            diff = q[a] - y
            clipped_err = max(-2.0, min(2.0, diff))  # Huber gradient clipping
            total_loss += diff ** 2

            # Output layer gradients
            for j in range(self.hidden_dim):
                dW2[a][j] += clipped_err * h[j]
            db2[a] += clipped_err

            # Hidden layer gradients
            for j in range(self.hidden_dim):
                if h_pre[j] > 0.0:
                    d_h_j = self.W2[a][j] * clipped_err
                    db1[j] += d_h_j
                    for i in range(self.in_dim):
                        dW1[j][i] += d_h_j * x[i]

        # Apply mini-batch averaged SGD
        for k in range(self.out_dim):
            self.b2[k] -= self.lr * db2[k] / B
            for j in range(self.hidden_dim):
                self.W2[k][j] -= self.lr * dW2[k][j] / B

        for j in range(self.hidden_dim):
            self.b1[j] -= self.lr * db1[j] / B
            for i in range(self.in_dim):
                self.W1[j][i] -= self.lr * dW1[j][i] / B

        return total_loss / B

    def copy_weights_from(self, other):
        self.W1 = [row[:] for row in other.W1]
        self.b1 = other.b1[:]
        self.W2 = [row[:] for row in other.W2]
        self.b2 = other.b2[:]

def state_to_features(s):
    vec = [0.0] * NUM_STATES
    vec[s] = 1.0
    return vec

def train_dqn(episodes=300, gamma=0.95, lr=0.03, batch_size=16, target_sync=15):
    online_net = MiniMLP(NUM_STATES, 20, NUM_ACTIONS, lr=lr)
    target_net = MiniMLP(NUM_STATES, 20, NUM_ACTIONS, lr=lr)
    target_net.copy_weights_from(online_net)

    replay_buffer = []
    buffer_capacity = 800
    rewards_history = []
    losses_history = []
    total_steps = 0

    for ep in range(episodes):
        s = state_to_idx(*START_STATE)
        eps = max(0.05, 1.0 - 0.95 * (ep / 220))
        ep_reward = 0.0

        for _ in range(60):
            total_steps += 1
            feat = state_to_features(s)

            if random.random() < eps:
                a = random.randint(0, NUM_ACTIONS - 1)
            else:
                q_vals = online_net.predict(feat)
                a = max(range(NUM_ACTIONS), key=lambda act: q_vals[act])

            next_s, r, done = step_env(s, a)
            ep_reward += r

            replay_buffer.append((s, a, r, next_s, done))
            if len(replay_buffer) > buffer_capacity:
                replay_buffer.pop(0)

            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                batch_training_data = []

                for b_s, b_a, b_r, b_next_s, b_done in batch:
                    b_x = state_to_features(b_s)
                    if b_done:
                        y = b_r
                    else:
                        next_x = state_to_features(b_next_s)
                        next_q = target_net.predict(next_x)
                        y = b_r + gamma * max(next_q)
                    batch_training_data.append((b_x, b_a, y))

                loss = online_net.train_batch(batch_training_data)
                losses_history.append(loss)

            if total_steps % target_sync == 0:
                target_net.copy_weights_from(online_net)

            s = next_s
            if done:
                break

        rewards_history.append(ep_reward)

    return online_net, rewards_history, losses_history

def evaluate_greedy(policy_fn, max_steps=20):
    s = state_to_idx(*START_STATE)
    path = [START_STATE]
    total_reward = 0.0

    for _ in range(max_steps):
        a = policy_fn(s)
        next_s, r, done = step_env(s, a)
        total_reward += r
        path.append(idx_to_state(next_s))
        s = next_s
        if done:
            break
    return path, total_reward

def run_experiment():
    print("=" * 70)
    print("TOPIC 26.3: TABULAR Q-LEARNING VS MINI-DQN ON CLIFF WALKING")
    print("=" * 70)

    # 1. Train Tabular Q-Learning
    print("\n[1] Training Tabular Q-Learning (300 episodes)...")
    Q, tab_rewards = train_tabular_q(episodes=300)
    tab_initial_avg = sum(tab_rewards[:30]) / 30
    tab_final_avg = sum(tab_rewards[-30:]) / 30
    print(f"  Tabular Q Initial 30-Ep Avg Reward: {tab_initial_avg:6.1f}")
    print(f"  Tabular Q Final   30-Ep Avg Reward: {tab_final_avg:6.1f}")

    def tab_policy(s):
        return max(range(NUM_ACTIONS), key=lambda a: Q[s][a])

    tab_path, tab_score = evaluate_greedy(tab_policy)
    print(f"  Greedy Evaluation: Score = {tab_score:.1f}, Steps = {len(tab_path)-1}")
    print(f"  Learned Trajectory: {' -> '.join(str(p) for p in tab_path)}")
    assert tab_path[-1] == GOAL_STATE, "Tabular Q agent failed to reach the goal!"

    # 2. Train Mini-DQN
    print("\n[2] Training Pure-Python Mini-DQN with Replay Buffer & Target Net...")
    dqn_net, dqn_rewards, dqn_losses = train_dqn(episodes=300)
    dqn_initial_avg = sum(dqn_rewards[:30]) / 30
    dqn_final_avg = sum(dqn_rewards[-30:]) / 30
    print(f"  DQN Initial 30-Ep Avg Reward: {dqn_initial_avg:6.1f}")
    print(f"  DQN Final   30-Ep Avg Reward: {dqn_final_avg:6.1f}")

    if dqn_losses:
        init_loss = sum(dqn_losses[:50]) / 50
        final_loss = sum(dqn_losses[-50:]) / 50
        print(f"  DQN Initial Mean TD Loss: {init_loss:6.3f}")
        print(f"  DQN Final   Mean TD Loss: {final_loss:6.3f}")

    def dqn_policy(s):
        q_vals = dqn_net.predict(state_to_features(s))
        return max(range(NUM_ACTIONS), key=lambda a: q_vals[a])

    dqn_path, dqn_score = evaluate_greedy(dqn_policy)
    print(f"  Greedy Evaluation: Score = {dqn_score:.1f}, Steps = {len(dqn_path)-1}")
    print(f"  Learned Trajectory: {' -> '.join(str(p) for p in dqn_path)}")
    assert dqn_path[-1] == GOAL_STATE, "DQN agent failed to reach the goal!"

    for pos in dqn_path:
        assert pos not in CLIFF_STATES, f"DQN agent fell into cliff at {pos}!"

    print("\nBoth Tabular Q-Learning and Mini-DQN successfully solved Cliff Walking.")
    print("[SUCCESS] Topic 26.3 experiment completed with zero errors.")

if __name__ == "__main__":
    run_experiment()
