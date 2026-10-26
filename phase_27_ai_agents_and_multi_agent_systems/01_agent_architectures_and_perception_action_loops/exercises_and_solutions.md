# Topic 01: Exercises & Solutions — Agent Architectures & Perception-Action Loops

---

### Exercise 1: Analytical Bayesian Belief State Update in a 3-State POMDP

Consider an autonomous cleaning robot navigating a 1D corridor with three discrete cells: $\mathcal{S} = \{s_1, s_2, s_3\}$.
At time $t=0$, the robot has uniform uncertainty over its location:
$$b_0 = [b(s_1), b(s_2), b(s_3)] = \left[\frac{1}{3}, \frac{1}{3}, \frac{1}{3}\right]$$

The robot takes action $a = \text{RIGHT}$. The transition matrix $\mathcal{T}(s' \mid s, \text{RIGHT})$ is given by:
- From $s_1$: transitions to $s_2$ with probability 0.8, remains at $s_1$ with probability 0.2.
- From $s_2$: transitions to $s_3$ with probability 0.8, remains at $s_2$ with probability 0.2.
- From $s_3$: bumper hits wall, remains at $s_3$ with probability 1.0.

The robot then takes an acoustic sonar reading $o \in \{\text{OPEN}, \text{WALL}\}$.
State $s_3$ is adjacent to the end wall. The sensor model $\mathcal{O}(o \mid s')$ satisfies:
$$\mathcal{O}(\text{WALL} \mid s_1) = 0.05, \quad \mathcal{O}(\text{WALL} \mid s_2) = 0.10, \quad \mathcal{O}(\text{WALL} \mid s_3) = 0.90$$

**Question:** If the robot takes action $a = \text{RIGHT}$ and subsequently perceives observation $o = \text{WALL}$, compute the exact posterior belief distribution $b_1 = [b_1(s_1), b_1(s_2), b_1(s_3)]$.

---

#### Solution to Exercise 1

**Step 1: Compute the prior predictive distribution (prediction step):**
$$\bar{b}_1(s') = \sum_{s \in \mathcal{S}} \mathcal{T}(s' \mid s, \text{RIGHT}) \cdot b_0(s)$$

For each state $s'$:
- $\bar{b}_1(s_1) = \mathcal{T}(s_1 \mid s_1) b_0(s_1) + \mathcal{T}(s_1 \mid s_2) b_0(s_2) + \mathcal{T}(s_1 \mid s_3) b_0(s_3) = (0.2)\left(\frac{1}{3}\right) + (0)\left(\frac{1}{3}\right) + (0)\left(\frac{1}{3}\right) = \frac{0.2}{3} \approx 0.0667$
- $\bar{b}_1(s_2) = \mathcal{T}(s_2 \mid s_1) b_0(s_1) + \mathcal{T}(s_2 \mid s_2) b_0(s_2) + \mathcal{T}(s_2 \mid s_3) b_0(s_3) = (0.8)\left(\frac{1}{3}\right) + (0.2)\left(\frac{1}{3}\right) + (0)\left(\frac{1}{3}\right) = \frac{1.0}{3} \approx 0.3333$
- $\bar{b}_1(s_3) = \mathcal{T}(s_3 \mid s_1) b_0(s_1) + \mathcal{T}(s_3 \mid s_2) b_0(s_2) + \mathcal{T}(s_3 \mid s_3) b_0(s_3) = (0)\left(\frac{1}{3}\right) + (0.8)\left(\frac{1}{3}\right) + (1.0)\left(\frac{1}{3}\right) = \frac{1.8}{3} = 0.6000$

Check: $0.0667 + 0.3333 + 0.6000 = 1.0000$.

**Step 2: Measurement update with observation $o = \text{WALL}$:**
Unnormalized likelihoods $\tilde{b}_1(s') = \mathcal{O}(\text{WALL} \mid s') \cdot \bar{b}_1(s')$:
- $\tilde{b}_1(s_1) = 0.05 \times \frac{0.2}{3} = \frac{0.010}{3} \approx 0.003333$
- $\tilde{b}_1(s_2) = 0.10 \times \frac{1.0}{3} = \frac{0.100}{3} \approx 0.033333$
- $\tilde{b}_1(s_3) = 0.90 \times \frac{1.8}{3} = \frac{1.620}{3} = 0.540000$

**Step 3: Normalizing constant $Z = \mathbb{P}(\text{WALL} \mid b_0, \text{RIGHT})$:**
$$Z = \tilde{b}_1(s_1) + \tilde{b}_1(s_2) + \tilde{b}_1(s_3) = \frac{0.010 + 0.100 + 1.620}{3} = \frac{1.730}{3} \approx 0.57667$$

**Step 4: Normalized posterior distribution:**
- $b_1(s_1) = \frac{0.010}{1.730} \approx \mathbf{0.00578}$ ($0.58\%$)
- $b_1(s_2) = \frac{0.100}{1.730} \approx \mathbf{0.05780}$ ($5.78\%$)
- $b_1(s_3) = \frac{1.620}{1.730} \approx \mathbf{0.93642}$ ($93.64\%$)

The robot's belief shifts from complete uncertainty ($33.3\%$ each) to overwhelming certainty ($93.6\%$) that it has reached the wall at $s_3$.

---

### Exercise 2: Proof that a POMDP is an MDP over Continuous Belief Space $\Delta(\mathcal{S})$

Let $\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \Omega, \mathcal{O}, \gamma \rangle$ be a discrete POMDP.
Define the continuous belief space $\mathcal{B} \triangleq \Delta(\mathcal{S}) = \left\{ b \in \mathbb{R}^{|\mathcal{S}|} : \sum_s b(s) = 1, \, b(s) \ge 0 \right\}$.

**Question:** Prove that the belief state dynamics satisfy the Markov property:
$$\mathbb{P}(b_{t+1} \mid b_t, a_t, b_{t-1}, a_{t-1}, \dots, b_0, a_0) = \mathbb{P}(b_{t+1} \mid b_t, a_t)$$
and define the transition density $\mathcal{T}_{\mathcal{B}}(b' \mid b, a)$ and reward function $\rho(b, a)$ for this continuous "Belief MDP".

---

#### Solution to Exercise 2

**Step 1: Deterministic update given observation:**
The next belief $b_{t+1}$ is a deterministic function $\psi(b_t, a_t, o_{t+1})$ of the current belief $b_t$, current action $a_t$, and the new observation $o_{t+1}$:
$$b_{t+1} = \psi(b_t, a_t, o_{t+1}) = \frac{\mathcal{O}(o_{t+1} \mid \cdot, a_t) \odot (\mathcal{T}(\cdot \mid \cdot, a_t) b_t)}{\mathbb{P}(o_{t+1} \mid b_t, a_t)}$$

**Step 2: Probability distribution over next observations:**
The observation $o_{t+1}$ is drawn from:
$$\mathbb{P}(o_{t+1} = o \mid b_t, a_t) = \sum_{s' \in \mathcal{S}} \mathcal{O}(o \mid s', a_t) \sum_{s \in \mathcal{S}} \mathcal{T}(s' \mid s, a_t) b_t(s)$$
Notice that this probability depends **only on $b_t$ and $a_t$**, and is conditionally independent of all preceding beliefs $b_{t-1}, \dots, b_0$ and actions $a_{t-1}, \dots, a_0$.

**Step 3: Markov Property Proof:**
Because $b_{t+1} = \psi(b_t, a_t, o_{t+1})$ and the probability distribution of $o_{t+1}$ depends strictly on $(b_t, a_t)$, the transition distribution to the next belief $b_{t+1}$ is:
$$\mathbb{P}(b_{t+1} = b' \mid b_t, a_t, \dots, b_0, a_0) = \sum_{o \in \Omega} \mathbb{I}(\psi(b_t, a_t, o) = b') \cdot \mathbb{P}(o \mid b_t, a_t) = \mathbb{P}(b_{t+1} = b' \mid b_t, a_t)$$
Thus, the Markov property holds strictly over $\mathcal{B}$.

**Step 4: Continuous Belief MDP Reward Function:**
The expected reward in belief state $b$ under action $a$ is the linear expectation over states:
$$\rho(b, a) \triangleq \sum_{s \in \mathcal{S}} b(s) \mathcal{R}(s, a)$$
Hence, any discrete POMDP is equivalent to a fully observable MDP with continuous state space $\mathcal{B} = \Delta(\mathcal{S})$, action space $\mathcal{A}$, transition kernel $\mathcal{T}_{\mathcal{B}}$, and reward function $\rho(b, a)$. $\blacksquare$

---

### Exercise 3: Context Window Truncation and Epistemic Drift in LLM Agents

In an LLM-based autonomous coding agent, the context window length is bounded by $L_{\max} = 8192$ tokens. When an execution trajectory exceeds $L_{\max}$, a naive FIFO (First-In, First-Out) sliding window discards the earliest tokens $(o_1, a_1, \dots)$.

**Question:** Using the POMDP formalism, explain why FIFO context truncation breaks the sufficient statistic property of the agent's belief state, and describe two concrete failure modes (loops or contradictions) that arise in practice.

---

#### Solution to Exercise 3

**1. Violation of Sufficient Statistic Property:**
In a POMDP, $b_t = \mathbb{P}(S_t \mid h_t)$ is a sufficient statistic only if $h_t$ accounts for the entire sequence from $t=0$, or if the truncated summary preserves all mutual information $I(S_t; h_{0:t-k})$.
When early tokens $h_{0:k} = (o_1, a_1, \dots, o_k)$ are dropped without state consolidation:
$$h_t^{\text{FIFO}} = (a_k, o_{k+1}, \dots, a_{t-1}, o_t) \neq h_t$$
The agent's implicit belief state $\hat{b}_t \approx \text{LLM}(h_t^{\text{FIFO}})$ loses conditioning on early events. Consequently:
$$I(S_t; h_t^{\text{FIFO}}) < I(S_t; h_t)$$
The agent enters an **epistemic amnesia** regime where the Markov property is violated in prompt space.

**2. Real-World Failure Modes:**
- **Action Cycling (Infinite Oscillations)**: Suppose at $t=2$ the agent edited file `config.py` and learned that port 8080 was blocked. At $t=25$, that observation falls outside the sliding window. At $t=26$, encountering a connection failure, the agent re-executes the exact same failed action (binding port 8080), triggering an infinite repeating loop.
- **Goal Drift & Hypothesis Inversion**: The user's original objective and initial constraints (e.g., *"Do not modify the database schema"*) were defined at $t=0$. After 30 tool calls, this prompt prefix is truncated. The agent, seeing only error traces mentioning database column mismatches, proceeds to drop and alter tables, violating hard user constraints.
