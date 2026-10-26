# Topic 03: Temporal Difference Learning, Q-Learning & Deep Q-Networks (DQN)

## 1. Introduction & 12-Year-Old Intuition

Imagine you are trying to guess how long your bus ride home from school will take on a rainy Monday afternoon.

There are two distinct ways you could learn:

1. **The Monte Carlo Way**: You write down your guess when you leave the school door (say, 30 minutes). You sit in the bus, get stuck in traffic, wait for all stops, and arrive home 52 minutes later. Only after you are sitting in your living room eating cookies do you update: *"My school-door prediction was off by 22 minutes!"*
2. **The Temporal Difference (TD) Way**: You leave the school door guessing 30 minutes. After 5 minutes, you reach the main highway and see a catastrophic bumper-to-bumper traffic jam. Immediately, without waiting to reach home, your updated forecast jumps to 60 minutes. You update your original prediction *on the spot*: *"Hey, when I stepped out of school, I should have predicted longer because reaching this highway jam is worth 60 minutes!"*

This is **Temporal Difference Learning**: *learning a guess from a guess*, updating predictions step-by-step before the final outcome is even known.

In tabular Q-Learning and Deep Q-Networks (DQN), an agent learns the value of every action from single-step experiences $(S_t, A_t, R_{t+1}, S_{t+1})$. When state spaces grow too immense for lookup tables (like the millions of pixel configurations in Atari games), we replace the table with a neural network.

---

## 2. Mathematical Symbol Decoder

| Symbol | Name | Formal Type | Meaning |
| :--- | :--- | :--- | :--- |
| $\delta_t$ | TD Error | Scalar $\in \mathbb{R}$ | Difference between the 1-step target ($R_{t+1} + \gamma V(S_{t+1})$) and the current estimate $V(S_t)$. |
| $\alpha$ | Learning Rate | Scalar $\alpha \in (0, 1]$ | Step size determining how aggressively estimates update toward target observations. |
| $Q(s, a)$ | Action-Value Function | Table or Network | Estimated return for taking action $a$ in state $s$. |
| $\epsilon$ | Exploration Probability | $\epsilon \in [0, 1]$ | Probability of picking a random exploratory action in $\epsilon$-greedy action selection. |
| $\theta$ | Online Network Parameters | Vector $\mathbb{R}^D$ | Weights and biases of the active Q-network being updated via gradient descent. |
| $\theta^-$ | Target Network Parameters | Vector $\mathbb{R}^D$ | Frozen / periodic clone of weights used to calculate non-moving target labels. |
| $\mathcal{D}$ | Experience Replay Buffer | Collection of tuples | FIFO memory storing past transitions $(s_t, a_t, r_{t+1}, s_{t+1}, d_{t+1})$. |
| $y_i^{\text{DQN}}$ | DQN Regression Target | Scalar $\in \mathbb{R}$ | $r + \gamma (1 - d) \max_{a'} Q(s', a'; \theta^-)$. |

---

## 3. Model-Free Learning: Monte Carlo vs Temporal Difference

In Dynamic Programming, we required complete knowledge of $\mathcal{P}(s' \mid s, a)$ and $\mathcal{R}(s, a)$. In real-world AI, the transition dynamics of the physical world or complex simulator are unknown. We must learn directly from raw observed experience.

### 3.1 Monte Carlo (MC) Policy Evaluation
Monte Carlo updates after an entire episode of length $T$ terminates, using the empirical return $G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}$:
$$V(S_t) \leftarrow V(S_t) + \alpha \left[ G_t - V(S_t) \right]$$
- **Pros**: Unbiased estimate $\mathbb{E}[G_t \mid S_t] = V^\pi(S_t)$; unaffected by non-Markovian dynamics.
- **Cons**: High variance; cannot learn online from continuing (non-episodic) tasks; must wait until termination.

### 3.2 Temporal Difference TD(0)
TD(0) bootstraps immediately using the one-step Bellman expectation:
$$V(S_t) \leftarrow V(S_t) + \alpha \delta_t$$
where the **TD error** $\delta_t$ is:
$$\delta_t \triangleq R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$
- **Pros**: Low variance (one step of stochastic reward vs a sum of stochastic rewards); fully online.
- **Cons**: Biased initially because $V(S_{t+1})$ is an imperfect estimate.

```
       Monte Carlo (Full Trajectory Backup)             TD(0) (One-Step Bootstrap Backup)
                   S_t                                                S_t
                    │                                                  │
                    ▼ (A_t)                                            ▼ (A_t)
                   S_{t+1}                                            S_{t+1}  <-- Bootstrap from V(S_{t+1})!
                    │
                    ▼
                   ...
                    │
                    ▼
                   S_T  <-- Wait for episode end!
```

---

## 4. On-Policy SARSA vs Off-Policy Q-Learning

When learning action-values $Q(s, a)$, the choice of target distinguishes on-policy from off-policy learning.

### 4.1 On-Policy: SARSA
SARSA updates using the transition 5-tuple $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$, where $A_{t+1} \sim \pi(\cdot \mid S_{t+1})$ is the **actual next action chosen by the behavior policy**:
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \right]$$
SARSA learns the value of the policy *inclusive of its exploratory mistakes* (e.g., walking safely far away from the cliff edge because it might randomly jump).

### 4.2 Off-Policy: Q-Learning (Watkins 1989)
Q-Learning uses the transition 4-tuple $(S_t, A_t, R_{t+1}, S_{t+1})$ and maximizes over all possible actions in $S_{t+1}$:
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t) \right]$$
Regardless of what exploratory action the agent actually takes in $S_{t+1}$, the target evaluates the **optimal greedy action**. Q-Learning converges directly to $Q^*(s, a)$ provided all state-action pairs continue to be visited.

---

## 5. Deep Q-Networks (DQN, Mnih et al. 2015)

In continuous or high-dimensional state spaces (e.g. $84 \times 84$ pixel images), tabular storage is impossible. We represent $Q(s, a)$ using a neural network parameterised by weights $\theta$:
$$Q(s, a; \theta) \approx Q^*(s, a)$$

### 5.1 The Deadly Triad
Standard supervised learning assumes data is independent and identically distributed (i.i.d.) with fixed labels. Reinforcement learning violates every single one of these assumptions:
1. **Function Approximation**: Non-linear neural networks introduce generalization errors.
2. **Bootstrapping**: Targets depend on current network predictions.
3. **Off-Policy Learning**: Training on data generated by a different behavior distribution.
When all three are present, naive gradient descent frequently diverges.

### 5.2 DQN Innovation 1: Experience Replay Buffer $\mathcal{D}$
Consecutive experiences $(s_t, a_t, r_t, s_{t+1})$ are strongly correlated over time. Training on sequential transitions causes catastrophic forgetting and feedback loops.
- Store transitions $e_t = (s_t, a_t, r_{t+1}, s_{t+1}, d_{t+1})$ in a large cyclic buffer $\mathcal{D}$.
- Sample uniform random mini-batches $B \sim \mathcal{D}$ to compute stochastic gradient steps.
- Random sampling breaks temporal correlations and restores an i.i.d.-like training distribution.

### 5.3 DQN Innovation 2: Target Network $Q(s, a; \theta^-)$
If the target $y = R + \gamma \max_{a'} Q(S', a'; \theta)$ uses the same weights $\theta$ that are being updated, the target moves every single step. This is like trying to shoot an arrow at a target that moves whenever you release the bow.
- Maintain a separate copy of weights $\theta^-$.
- Compute the regression target using $\theta^-$:
  $$y_i = r_i + \gamma (1 - d_i) \max_{a' \in \mathcal{A}} Q(s_i', a'; \theta^-)$$
- The loss function is:
  $$\mathcal{L}(\theta) = \frac{1}{|B|} \sum_{i \in B} \left( y_i - Q(s_i, a_i; \theta) \right)^2$$
- Periodically update target weights $\theta^- \leftarrow \theta$ every $C$ steps (or via Polyak averaging $\theta^- \leftarrow \tau \theta + (1-\tau)\theta^-$).

```
                     DQN Training Loop
                     
   Environment ──(s, a, r, s')──► Replay Buffer D (FIFO)
                                      │
                                 Sample Batch
                                      ▼
             Target Network:  y = r + γ max_a' Q(s', a'; θ⁻)
                                      │
             Loss: L(θ) = (y - Q(s, a; θ))²
                                      │
             Online Network:  Update θ via ∇_θ L(θ)
                                      │
             Periodic Sync:   θ⁻ ◄── θ (every C steps)
```
