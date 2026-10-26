#!/usr/bin/env python3
"""
Experiment 0.1: What is Intelligence?
====================================
Demonstration of:
1. Pure Computation (Rule-based agent) vs. Genuine Learning (Adaptive agent)
2. The Agent-Environment Loop: States, Actions, Transitions, Rewards, and Discounted Returns
3. The Generalization Gap: Testing on familiar environments vs. novel environments

Run this script directly:
    python3 experiment.py
"""

import random
import time


class GridWorld:
    """
    A 1D/2D linear grid corridor environment where an agent seeks food and must avoid fire pits.
    Grid cells:
      'S': Start
      '.': Empty path (Reward: -1 per step, encourages speed)
      'F': Fire pit (Reward: -100, ends episode)
      'G': Goal / Food (Reward: +100, ends episode)
    """

    def __init__(self, layout: str):
        self.layout = list(layout)
        self.start_pos = self.layout.index('S')
        self.agent_pos = self.start_pos
        self.done = False

    def reset(self):
        self.agent_pos = self.start_pos
        self.done = False
        return self.agent_pos

    def step(self, action: int):
        """
        Actions:
          0: Move Left
          1: Move Right
        """
        if self.done:
            raise RuntimeError("Episode is finished. Call reset() to play again.")

        # Compute next state
        if action == 0:  # Left
            self.agent_pos = max(0, self.agent_pos - 1)
        elif action == 1:  # Right
            self.agent_pos = min(len(self.layout) - 1, self.agent_pos + 1)

        tile = self.layout[self.agent_pos]

        if tile == 'G':
            reward = 100.0
            self.done = True
        elif tile == 'F':
            reward = -100.0
            self.done = True
        else:
            reward = -1.0  # Step penalty
            self.done = False

        return self.agent_pos, reward, self.done

    def render(self):
        display = list(self.layout)
        display[self.agent_pos] = '🤖' if not self.done else ('💀' if self.layout[self.agent_pos] == 'F' else '🎉')
        return "[" + "".join(display) + "]"


class HardcodedRuleAgent:
    """
    Agent A: Pure Rule-based System (Symbolic / Hardcoded)
    The programmer wrote static rules assuming the goal is always on the right.
    """

    def __init__(self):
        self.name = "Hardcoded Rule Agent (Static Computation)"

    def act(self, state: int) -> int:
        # Static rule: Always walk to the right!
        return 1


class MemorizerAgent:
    """
    Agent B: Memorization Agent (Lookup Table / Overfitter)
    Memorizes exact state-action pairs from previous runs. Fails when world changes.
    """

    def __init__(self):
        self.name = "Memorizer Agent (Lookup Table / Overfitting)"
        self.lookup_table = {}

    def memorize(self, state: int, action: int):
        self.lookup_table[state] = action

    def act(self, state: int) -> int:
        if state in self.lookup_table:
            return self.lookup_table[state]
        # Has never seen this state: picks completely random action
        return random.choice([0, 1])


class SimpleLearningAgent:
    """
    Agent C: Adaptive Learning Agent (Q-Learner from First Principles)
    Maintains a Q-table: Q(state, action) and updates it via temporal difference learning.
    Q(s, a) <- Q(s, a) + alpha * [Reward + gamma * max_a' Q(s', a') - Q(s, a)]
    """

    def __init__(self, num_states: int, alpha: float = 0.2, gamma: float = 0.9):
        self.name = "Adaptive Q-Learning Agent (Genuine Adaptation)"
        self.num_states = num_states
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        # Q-table: rows = states, cols = actions (0: Left, 1: Right)
        self.q_table = {s: [0.0, 0.0] for s in range(num_states)}

    def act(self, state: int, epsilon: float = 0.1) -> int:
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            return random.choice([0, 1])
        q_values = self.q_table[state]
        if q_values[0] == q_values[1]:
            return random.choice([0, 1])
        return 0 if q_values[0] > q_values[1] else 1

    def learn(self, state: int, action: int, reward: float, next_state: int, done: bool):
        old_val = self.q_table[state][action]
        next_max = 0.0 if done else max(self.q_table[next_state])
        # Bellman update formula
        temporal_difference = reward + (self.gamma * next_max) - old_val
        self.q_table[state][action] = old_val + (self.alpha * temporal_difference)


def run_agent_demonstration():
    print("=" * 70)
    print("EXPERIMENT 0.1: THE AGENT-ENVIRONMENT LOOP & GENERALIZATION")
    print("=" * 70)

    # 1. Training Environment: S . . G (Start at 0, Goal at index 3)
    train_layout = "S..G"
    print(f"\n1. Training Environment Layout: '{train_layout}'")
    print("   S = Start, . = Path, G = Goal (+100), F = Fire (-100)\n")

    env = GridWorld(train_layout)

    # Instantiate Agents
    rule_agent = HardcodedRuleAgent()
    learner = SimpleLearningAgent(num_states=len(train_layout))

    # Train the learner for 50 episodes
    print("-> Training the Adaptive Agent over 50 episodes...")
    for ep in range(50):
        s = env.reset()
        done = False
        while not done:
            a = learner.act(s, epsilon=0.2)
            s_next, r, done = env.step(a)
            learner.learn(s, a, r, s_next, done)
            s = s_next

    print("   Training finished! Learned Q-table:")
    for s, q in learner.q_table.items():
        print(f"   State {s} ('{train_layout[s]}'): Q(Left)={q[0]:.2f}, Q(Right)={q[1]:.2f}")

    # Build Memorizer by copying successful actions
    memorizer = MemorizerAgent()
    for s in range(len(train_layout) - 1):
        best_a = 0 if learner.q_table[s][0] > learner.q_table[s][1] else 1
        memorizer.memorize(s, best_a)

    print("\n" + "=" * 70)
    print("TEST 1: KNOWN ENVIRONMENT ('S..G')")
    print("=" * 70)

    for agent in [rule_agent, memorizer, learner]:
        s = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        path = [env.render()]

        while not done and steps < 10:
            a = agent.act(s) if not isinstance(agent, SimpleLearningAgent) else agent.act(s, epsilon=0.0)
            s, r, done = env.step(a)
            total_reward += r
            steps += 1
            path.append(env.render())

        outcome = "SUCCESS (Found Goal)" if total_reward > 0 else "FAILED"
        print(f"\nAgent: {agent.name}")
        print(f"Path: {' -> '.join(path)}")
        print(f"Total Steps: {steps} | Total Return: {total_reward} | Result: {outcome}")

    # TEST 2: THE ENVIRONMENT CHANGES!
    # A fire pit is placed where the goal used to be, and the goal is now on the LEFT!
    # G . S F
    test_layout = "G.SF"
    print("\n" + "=" * 70)
    print(f"TEST 2: UNSEEN ENVIRONMENT WITH A TRAP ('{test_layout}')")
    print("Notice: The goal is now on the LEFT, and walking RIGHT hits FIRE!")
    print("=" * 70)

    test_env = GridWorld(test_layout)
    test_learner = SimpleLearningAgent(num_states=len(test_layout))

    for agent in [rule_agent, memorizer]:
        s = test_env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        path = [test_env.render()]

        while not done and steps < 5:
            a = agent.act(s)
            s, r, done = test_env.step(a)
            total_reward += r
            steps += 1
            path.append(test_env.render())

        outcome = "FAILED (Fell into fire!)" if total_reward < 0 else "SUCCESS"
        print(f"\nAgent: {agent.name}")
        print(f"Path: {' -> '.join(path)}")
        print(f"Total Steps: {steps} | Total Return: {total_reward} | Result: {outcome}")

    # Now let the adaptive agent interact and adapt to the new environment
    print(f"\nAgent: {test_learner.name}")
    print("-> Giving the Adaptive Agent 20 trials to explore and adapt to the new world...")
    for ep in range(20):
        s = test_env.reset()
        done = False
        while not done:
            a = test_learner.act(s, epsilon=0.3)
            s_next, r, done = test_env.step(a)
            test_learner.learn(s, a, r, s_next, done)
            s = s_next

    # Run optimal policy of the newly adapted agent
    s = test_env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    path = [test_env.render()]
    while not done and steps < 10:
        a = test_learner.act(s, epsilon=0.0)
        s, r, done = test_env.step(a)
        total_reward += r
        steps += 1
        path.append(test_env.render())

    outcome = "SUCCESS (Adapted and reached the new goal!)" if total_reward > 0 else "FAILED"
    print(f"Adapted Path: {' -> '.join(path)}")
    print(f"Total Steps: {steps} | Total Return: {total_reward} | Result: {outcome}")

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS:")
    print("1. The Hardcoded Rule Agent blindly walked into the fire pit because its programmer")
    print("   only anticipated one environment configuration.")
    print("2. The Memorizer Agent failed on unseen states because pure memorization has zero generalization.")
    print("3. The Adaptive Agent uses an objective (rewards) and optimization (Bellman updates)")
    print("   to learn the true dynamics of any valid environment.")
    print("=" * 70)


if __name__ == "__main__":
    run_agent_demonstration()

