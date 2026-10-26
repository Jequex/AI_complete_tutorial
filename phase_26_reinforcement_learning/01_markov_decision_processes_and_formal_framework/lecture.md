# Topic 01: Markov Decision Processes & The Formal RL Framework

## 1. Introduction & 12-Year-Old Intuition

Imagine you are playing an enchanted board game like *Jumanji*. You stand on square number 14. A magical guardian appears and asks: *"Which direction will you step next?"*

Notice something crucial: the guardian does not care whether you took two turns to reach square 14 or fifty turns. The guardian does not care if you arrived by hopping on one foot, tumbling backwards, or flying on an eagle. The rules of what happens next—your odds of stepping into a trap, finding a treasure chest, or sliding down a snake—depend **entirely and exclusively on where you are right now (square 14) and what action you choose right now**.

This property is called the **Markov Property**: *The future is independent of the past given the present.*

In this enchanted game:
- **State ($s$)**: The square you stand on, plus whatever items you are holding.
- **Action ($a$)**: Moving Up, Down, Left, or Right.
- **Transition Probability ($P$)**: The chance that the dice roll lands you on square 15 rather than slipping on ice into square 10.
- **Reward ($R$)**: Gold nuggets picked up (+10), or damage taken from lava (-100).
- **Discount Factor ($\gamma$)**: A magical decay spell. A gold nugget right now is shiny and worth 100 points. A gold nugget promised 10 turns in the future is covered in cobwebs and worth only $\gamma^{10} \times 100$ points.

Reinforcement Learning (RL) is the mathematical science of training an autonomous agent to play this enchanted game, discovering through trial and error the exact sequence of decisions that maximizes total discounted gold over its entire lifetime.

---

## 2. Mathematical Symbol Decoder

| Symbol | Name | Formal Type | Intuition / Meaning |
| :--- | :--- | :--- | :--- |
| $\mathcal{S}$ | State Space | Set $\{s_1, s_2, \dots\}$ or $\mathbb{R}^d$ | The set of all possible environmental situations or configurations. |
| $\mathcal{A}$ | Action Space | Set $\{a_1, a_2, \dots\}$ or $\mathbb{R}^k$ | The set of all allowable decisions or moves available to the agent. |
| $\mathcal{P}$ (or $T$) | Transition Kernel | $\mathcal{S} \times \mathcal{A} \times \mathcal{S} \to [0, 1]$ | Conditional probability distribution $P(s' \mid s, a)$ of landing in state $s'$ after action $a$ in $s$. |
| $\mathcal{R}$ | Reward Function | $\mathcal{S} \times \mathcal{A} \to \mathbb{R}$ | Scalar feedback signal received immediately after taking action $a$ in state $s$. |
| $\gamma$ | Discount Factor | Scalar $\gamma \in [0, 1)$ | Temporal preference parameter measuring how much current rewards outweigh distant future rewards. |
| $\tau$ | Trajectory / Episode | Sequence $(s_0, a_0, r_1, s_1, \dots)$ | An ordered history of states, actions, and rewards experienced across time. |
| $\pi(a \mid s)$ | Policy | $\mathcal{S} \times \mathcal{A} \to [0, 1]$ | The agent's decision rule or brain: probability distribution over actions given the current state. |
| $G_t$ | Discounted Return | Scalar $\in \mathbb{R}$ | Cumulative sum of future rewards from step $t$ onward, discounted geometrically by powers of $\gamma$. |
| $V^\pi(s)$ | State-Value Function | $\mathcal{S} \to \mathbb{R}$ | Expected return starting from state $s$ and following policy $\pi$ thereafter. |
| $Q^\pi(s, a)$ | Action-Value Function | $\mathcal{S} \times \mathcal{A} \to \mathbb{R}$ | Expected return starting from state $s$, taking action $a$, and following policy $\pi$ thereafter. |

---

## 3. The Markov Decision Process (MDP) 5-Tuple

Formally, a discrete-time Markov Decision Process is defined by the 5-tuple:
$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$$

### 3.1 The Markov Property
A stochastic process possesses the **Markov property** if the conditional probability distribution of future states depends only upon the current state and action, not upon the sequence of events that preceded it:

$$\mathbb{P}(S_{t+1} = s_{t+1}, R_{t+1} = r_{t+1} \mid S_t = s_t, A_t = a_t, S_{t-1} = s_{t-1}, A_{t-1} = a_{t-1}, \dots, S_0 = s_0, A_0 = a_0)$$
$$= \mathbb{P}(S_{t+1} = s_{t+1}, R_{t+1} = r_{t+1} \mid S_t = s_t, A_t = a_t)$$

If a system violates this (e.g., in a partially observable setting like poker where opponent cards are hidden, or a video game where velocity is not captured in a single frame), the state representation is *non-Markovian*. In practice, we restore the Markov property by concatenating historical observations (e.g., stacking the last 4 video frames in Atari DQN) into a belief state or augmented state space.

### 3.2 The Transition Kernel
The transition probability function $\mathcal{P}: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \to [0, 1]$ satisfies:
$$\mathcal{P}(s' \mid s, a) = \mathbb{P}(S_{t+1} = s' \mid S_t = s, A_t = a)$$
Subject to the probability simplex constraint:
$$\sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) = 1, \quad \forall s \in \mathcal{S}, \, \forall a \in \mathcal{A}$$

### 3.3 The Reward Function
The reward function quantifies task performance. It can be formalized as:
- Transition-dependent: $\mathcal{R}(s, a, s') = \mathbb{E}[R_{t+1} \mid S_t = s, A_t = a, S_{t+1} = s']$
- State-action expected: $\mathcal{R}(s, a) = \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) \mathcal{R}(s, a, s')$

---

## 4. Trajectory Distribution and Discounted Return

### 4.1 Trajectory Distribution
Let an agent interact with an MDP starting from an initial state distribution $\rho_0(s_0) = \mathbb{P}(S_0 = s_0)$. A trajectory $\tau$ of horizon $H$ is a sequence:
$$\tau = (s_0, a_0, s_1, a_1, s_2, a_2, \dots, s_H)$$

Under policy $\pi(a \mid s)$, the joint probability of generating trajectory $\tau$ is:
$$\mathbb{P}(\tau \mid \pi) = \rho_0(s_0) \prod_{t=0}^{H-1} \pi(a_t \mid s_t) \mathcal{P}(s_{t+1} \mid s_t, a_t)$$

### 4.2 The Discounted Return $G_t$
The discounted return $G_t$ at time step $t$ is the sum of discounted rewards accumulated into the infinite horizon:
$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^\infty \gamma^k R_{t+k+1}$$

#### Why Geometric Discounting $\gamma^k$?
1. **Mathematical Convergence**: If rewards are bounded by $|R_t| \le R_{\max}$, the infinite geometric series converges absolutely for $\gamma \in [0, 1)$:
   $$|G_t| \le \sum_{k=0}^\infty \gamma^k |R_{t+k+1}| \le R_{\max} \sum_{k=0}^\infty \gamma^k = \frac{R_{\max}}{1 - \gamma} < \infty$$
2. **Time Consistency (Stationarity)**: Geometric discounting is the unique discount function where the relative preference between receiving reward at time $t_1$ vs $t_2$ remains invariant under constant time shifts:
   $$\frac{\gamma^{t_2}}{\gamma^{t_1}} = \gamma^{t_2 - t_1}$$
   Hyperbolic discounting (common in human psychology) causes time-inconsistent behavior (dynamic inconsistency).
3. **Uncertainty of Termination**: A discount factor $\gamma$ is mathematically equivalent to assuming that at every step, the episode terminates with constant probability $1 - \gamma$.

---

## 5. Policies, State-Value, and Action-Value Functions

### 5.1 The Policy $\pi$
A policy $\pi$ dictates the agent's behavior:
- **Deterministic**: $a = \mu(s), \quad \mu: \mathcal{S} \to \mathcal{A}$
- **Stochastic**: $\pi(a \mid s) = \mathbb{P}(A_t = a \mid S_t = s)$, where $\sum_{a \in \mathcal{A}} \pi(a \mid s) = 1$ and $\pi(a \mid s) \ge 0$.

### 5.2 The State-Value Function $V^\pi(s)$
The value of a state $s$ under policy $\pi$ is the expected return starting from $s$:
$$V^\pi(s) \triangleq \mathbb{E}_{\pi} \left[ G_t \mid S_t = s \right] = \mathbb{E}_{\pi} \left[ \sum_{k=0}^\infty \gamma^k R_{t+k+1} \, \middle|\, S_t = s \right]$$

### 5.3 The Action-Value Function $Q^\pi(s, a)$
The value of taking action $a$ in state $s$ under policy $\pi$ is:
$$Q^\pi(s, a) \triangleq \mathbb{E}_{\pi} \left[ G_t \mid S_t = s, A_t = a \right] = \mathbb{E}_{\pi} \left[ \sum_{k=0}^\infty \gamma^k R_{t+k+1} \, \middle|\, S_t = s, A_t = a \right]$$

### 5.4 The Fundamental Relationship Between $V^\pi$ and $Q^\pi$
By the Law of Total Expectation, taking the expectation over all possible actions under policy $\pi$:
$$V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) Q^\pi(s, a)$$

Conversely, taking an action $a$ yields immediate reward $R(s, a)$ plus the discounted expected value of the next state:
$$Q^\pi(s, a) = \mathcal{R}(s, a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) V^\pi(s')$$

Substituting $Q^\pi$ into $V^\pi$ produces the recursive structure that forms the foundation of all Dynamic Programming in RL:
$$V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \left[ \mathcal{R}(s, a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) V^\pi(s') \right]$$

---

## 6. Analytical Matrix Solution for Finite MDPs

In a finite MDP with $|\mathcal{S}| = N$ states, under a fixed policy $\pi$, we define:
- State-value vector: $\mathbf{v}^\pi = [V^\pi(s_1), \dots, V^\pi(s_N)]^T \in \mathbb{R}^N$
- Induced reward vector: $\mathbf{r}^\pi = [r^\pi(s_1), \dots, r^\pi(s_N)]^T \in \mathbb{R}^N$, where $r^\pi(s) = \sum_a \pi(a \mid s) \mathcal{R}(s, a)$
- Induced state-transition matrix: $\mathbf{P}^\pi \in \mathbb{R}^{N \times N}$, where $\mathbf{P}^\pi_{i, j} = \sum_a \pi(a \mid s_i) \mathcal{P}(s_j \mid s_i, a)$

The Bellman equation in vector-matrix notation is:
$$\mathbf{v}^\pi = \mathbf{r}^\pi + \gamma \mathbf{P}^\pi \mathbf{v}^\pi$$

Rearranging:
$$(\mathbf{I} - \gamma \mathbf{P}^\pi) \mathbf{v}^\pi = \mathbf{r}^\pi$$

Since $\mathbf{P}^\pi$ is a row-stochastic matrix (eigenvalues $|\lambda| \le 1$) and $\gamma < 1$, the spectral radius $\rho(\gamma \mathbf{P}^\pi) < 1$. Thus, the matrix $(\mathbf{I} - \gamma \mathbf{P}^\pi)$ is strictly invertible via the Neumann series:
$$(\mathbf{I} - \gamma \mathbf{P}^\pi)^{-1} = \sum_{k=0}^\infty (\gamma \mathbf{P}^\pi)^k$$

The exact closed-form analytical solution is:
$$\mathbf{v}^\pi = (\mathbf{I} - \gamma \mathbf{P}^\pi)^{-1} \mathbf{r}^\pi$$

This analytical result provides a benchmark against which Monte Carlo rollouts and iterative algorithms can be evaluated.
