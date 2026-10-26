# Topic 02: Exercises & Solutions — Bellman Equations & Dynamic Programming

---

### Exercise 1: Monotonicity of the Bellman Operator
**Problem:**
Let $U, V \in \mathbb{R}^{|\mathcal{S}|}$ be two state-value functions satisfying $U(s) \le V(s)$ for all $s \in \mathcal{S}$ (denoted $U \le V$).
Prove that both the Bellman expectation operator $\mathcal{T}^\pi$ and the Bellman optimality operator $\mathcal{T}^*$ are monotonic:
1. $\mathcal{T}^\pi U \le \mathcal{T}^\pi V$
2. $\mathcal{T}^* U \le \mathcal{T}^* V$

<details>
<summary><strong>Solution</strong></summary>

**Part 1: Monotonicity of $\mathcal{T}^\pi$**
For any state $s$:
$$(\mathcal{T}^\pi U)(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \left[ \mathcal{R}(s, a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) U(s') \right]$$
$$(\mathcal{T}^\pi V)(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \left[ \mathcal{R}(s, a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) V(s') \right]$$
Subtracting the two expressions:
$$(\mathcal{T}^\pi V)(s) - (\mathcal{T}^\pi U)(s) = \gamma \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) \big( V(s') - U(s') \big)$$
Since $\pi(a \mid s) \ge 0$, $\mathcal{P}(s' \mid s, a) \ge 0$, $\gamma \ge 0$, and $V(s') - U(s') \ge 0$ for all $s'$, the entire sum is a non-negative linear combination of non-negative terms.
Therefore, $(\mathcal{T}^\pi V)(s) - (\mathcal{T}^\pi U)(s) \ge 0 \implies \mathcal{T}^\pi U \le \mathcal{T}^\pi V$.

**Part 2: Monotonicity of $\mathcal{T}^*$**
For any action $a$:
$$\mathcal{R}(s, a) + \gamma \sum_{s'} \mathcal{P}(s' \mid s, a) U(s') \le \mathcal{R}(s, a) + \gamma \sum_{s'} \mathcal{P}(s' \mid s, a) V(s')$$
Because this inequality holds for every individual action $a \in \mathcal{A}$, taking the maximum over all $a \in \mathcal{A}$ preserves the inequality:
$$\max_{a \in \mathcal{A}} \left[ \mathcal{R}(s, a) + \gamma \sum_{s'} \mathcal{P}(s' \mid s, a) U(s') \right] \le \max_{a \in \mathcal{A}} \left[ \mathcal{R}(s, a) + \gamma \sum_{s'} \mathcal{P}(s' \mid s, a) V(s') \right]$$
$$(\mathcal{T}^* U)(s) \le (\mathcal{T}^* V)(s)$$
Thus $\mathcal{T}^*$ is monotonic.
</details>

---

### Exercise 2: Proof of the Policy Improvement Theorem
**Problem:**
Let $\pi$ and $\pi'$ be two deterministic policies such that for all states $s \in \mathcal{S}$:
$$Q^\pi(s, \pi'(s)) \ge V^\pi(s)$$
Prove that $V^{\pi'}(s) \ge V^\pi(s)$ for all $s \in \mathcal{S}$.

<details>
<summary><strong>Solution</strong></summary>

By assumption, $V^\pi(s) \le Q^\pi(s, \pi'(s))$ for all $s$.
Expanding $Q^\pi(s, \pi'(s))$ using the Bellman expectation definition:
$$V^\pi(s) \le \mathcal{R}(s, \pi'(s)) + \gamma \sum_{s'} \mathcal{P}(s' \mid s, \pi'(s)) V^\pi(s')$$
In operator notation, let $\mathcal{T}^{\pi'}$ be the Bellman expectation operator under policy $\pi'$. The inequality states:
$$V^\pi \le \mathcal{T}^{\pi'} V^\pi$$
Now, apply $\mathcal{T}^{\pi'}$ to both sides. By the monotonicity property proven in Exercise 1:
$$\mathcal{T}^{\pi'} V^\pi \le \mathcal{T}^{\pi'} (\mathcal{T}^{\pi'} V^\pi) = (\mathcal{T}^{\pi'})^2 V^\pi$$
Combining with $V^\pi \le \mathcal{T}^{\pi'} V^\pi$:
$$V^\pi \le \mathcal{T}^{\pi'} V^\pi \le (\mathcal{T}^{\pi'})^2 V^\pi \le \dots \le (\mathcal{T}^{\pi'})^k V^\pi$$
Taking the limit as $k \to \infty$:
$$V^\pi \le \lim_{k \to \infty} (\mathcal{T}^{\pi'})^k V^\pi = V^{\pi'}$$
Thus, $V^{\pi'}(s) \ge V^\pi(s)$ for all $s \in \mathcal{S}$. Furthermore, if there is strict inequality in at least one state, the new policy is strictly superior.
</details>

---

### Exercise 3: Value Iteration Stopping Criterion Derivation
**Problem:**
In Value Iteration, we terminate when $\|V_{k+1} - V_k\|_\infty < \theta$.
Prove that:
$$\|V_{k+1} - V^*\|_\infty \le \frac{\gamma}{1 - \gamma} \|V_{k+1} - V_k\|_\infty$$
And deduce that if $\theta = \epsilon \frac{1 - \gamma}{2\gamma}$, the policy $\pi$ greedy with respect to $V_{k+1}$ satisfies $\|V^\pi - V^*\|_\infty \le \epsilon$.

<details>
<summary><strong>Solution</strong></summary>

**Step 1:** By the triangle inequality:
$$\|V_{k+1} - V^*\|_\infty \le \sum_{j=1}^\infty \|V_{k+j+1} - V_{k+j}\|_\infty$$
Because $\mathcal{T}^*$ is a $\gamma$-contraction mapping under $\ell_\infty$:
$$\|V_{k+j+1} - V_{k+j}\|_\infty \le \gamma^j \|V_{k+1} - V_k\|_\infty$$
Summing the geometric series:
$$\|V_{k+1} - V^*\|_\infty \le \left( \sum_{j=1}^\infty \gamma^j \right) \|V_{k+1} - V_k\|_\infty = \frac{\gamma}{1 - \gamma} \|V_{k+1} - V_k\|_\infty$$

**Step 2:**
Let $\pi$ be greedy with respect to $V_{k+1}$, so $\mathcal{T}^\pi V_{k+1} = \mathcal{T}^* V_{k+1} = V_{k+2}$.
The value difference between $V^\pi$ and $V^*$ satisfies:
$$\|V^\pi - V^*\|_\infty \le \|V^\pi - V_{k+1}\|_\infty + \|V_{k+1} - V^*\|_\infty \le \frac{2\gamma}{1 - \gamma} \|V_{k+1} - V_k\|_\infty$$
Setting $\|V_{k+1} - V_k\|_\infty < \theta = \epsilon \frac{1 - \gamma}{2\gamma}$ immediately implies:
$$\|V^\pi - V^*\|_\infty < \frac{2\gamma}{1 - \gamma} \left( \epsilon \frac{1 - \gamma}{2\gamma} \right) = \epsilon$$
This provides the mathematically guaranteed stopping criterion for Value Iteration.
</details>

---

### Exercise 4: Manual Two-Sweep Value Iteration Trace
**Problem:**
Consider a 2-state MDP with states $\{A, B\}$ and actions $\{1, 2\}$:
- State A:
  - Action 1: transitions to A with prob 1.0, reward +1.
  - Action 2: transitions to B with prob 1.0, reward +3.
- State B:
  - Action 1: transitions to B with prob 0.5, transitions to A with prob 0.5, reward +2.
  - Action 2: transitions to A with prob 1.0, reward 0.
Let $\gamma = 0.5$, and initialize $V_0(A) = 0, V_0(B) = 0$.
Perform exactly 2 synchronous sweeps of Value Iteration and compute $V_1$ and $V_2$.

<details>
<summary><strong>Solution</strong></summary>

**Initialization:**
$$V_0(A) = 0, \quad V_0(B) = 0$$

**Iteration 1:**
- State A:
  - $Q_1(A, 1) = 1 + 0.5 \times V_0(A) = 1 + 0 = 1$
  - $Q_1(A, 2) = 3 + 0.5 \times V_0(B) = 3 + 0 = 3$
  - $V_1(A) = \max(1, 3) = 3$ (best action: 2)
- State B:
  - $Q_1(B, 1) = 2 + 0.5 [ 0.5 V_0(B) + 0.5 V_0(A) ] = 2 + 0 = 2$
  - $Q_1(B, 2) = 0 + 0.5 \times V_0(A) = 0$
  - $V_1(B) = \max(2, 0) = 2$ (best action: 1)

**Iteration 2 (using $V_1(A) = 3, V_1(B) = 2$):**
- State A:
  - $Q_2(A, 1) = 1 + 0.5 \times V_1(A) = 1 + 0.5(3) = 2.5$
  - $Q_2(A, 2) = 3 + 0.5 \times V_1(B) = 3 + 0.5(2) = 4.0$
  - $V_2(A) = \max(2.5, 4.0) = 4.0$ (best action: 2)
- State B:
  - $Q_2(B, 1) = 2 + 0.5 [ 0.5(2) + 0.5(3) ] = 2 + 0.5(2.5) = 3.25$
  - $Q_2(B, 2) = 0 + 0.5 \times V_1(A) = 0 + 0.5(3) = 1.5$
  - $V_2(B) = \max(3.25, 1.5) = 3.25$ (best action: 1)

**Result:**
- $V_1 = [3.0, 2.0]$
- $V_2 = [4.0, 3.25]$
</details>
