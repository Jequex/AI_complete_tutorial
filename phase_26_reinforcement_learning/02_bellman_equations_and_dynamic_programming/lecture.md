# Topic 02: Bellman Equations & Dynamic Programming

## 1. Introduction & 12-Year-Old Intuition

Imagine you are standing in front of a giant pyramid with thousands of stone steps. At the top of the pyramid is a diamond treasure chest. Every stone step has a number carved on it telling you how much gold you gain or lose by stepping there.

You want to find the best route to the top. Do you need to test all 10 billion possible paths from the bottom to the top? **No!**

Think backwards from the top:
- If you are on a step right next to the diamond, your choice is trivial: step directly onto the diamond!
- If you are one step further back, the value of your step is just: *the gold on your next move plus the value of wherever you land.*

This simple, beautiful insight was discovered by mathematician Richard Bellman in 1957. It is called **The Principle of Optimality**:
> *"An optimal policy has the property that whatever the initial state and initial decision are, the remaining decisions must constitute an optimal policy with regard to the state resulting from the first decision."*

Instead of solving an infinite chain of future events all at once, Bellman broke the problem into a recursive equation:
$$\text{Value today} = \text{Reward right now} + \gamma \times \text{Expected Value tomorrow}$$

---

## 2. Mathematical Symbol Decoder

| Symbol | Name | Formal Type | Meaning |
| :--- | :--- | :--- | :--- |
| $\mathcal{T}^\pi$ | Bellman Expectation Operator | $(\mathbb{R}^{|\mathcal{S}|} \to \mathbb{R}^{|\mathcal{S}|})$ | Maps a value function to its one-step expectation under policy $\pi$. |
| $\mathcal{T}^*$ | Bellman Optimality Operator | $(\mathbb{R}^{|\mathcal{S}|} \to \mathbb{R}^{|\mathcal{S}|})$ | Maps a value function to its maximum achievable one-step lookahead. |
| $\|\cdot\|_\infty$ | Supremum / Infinity Norm | $\mathbb{R}^{|\mathcal{S}|} \to \mathbb{R}_{\ge 0}$ | Maximum absolute error across all states: $\|V\|_\infty = \max_{s \in \mathcal{S}} |V(s)|$. |
| $V^*$ | Optimal State-Value Function | $\mathcal{S} \to \mathbb{R}$ | Maximum expected return achievable from state $s$ under any possible policy. |
| $Q^*$ | Optimal Action-Value Function | $\mathcal{S} \times \mathcal{A} \to \mathbb{R}$ | Maximum expected return achievable by taking action $a$ in $s$ and behaving optimally thereafter. |
| $\pi^*$ | Optimal Policy | $\mathcal{S} \to \mathcal{A}$ | Any policy satisfying $V^{\pi^*}(s) = V^*(s)$ for all $s \in \mathcal{S}$. |
| $\delta_k$ | Bellman Residual / Error | $\mathbb{R}_{\ge 0}$ | $\ell_\infty$ distance between consecutive iterates: $\|V_{k+1} - V_k\|_\infty$. |

---

## 3. Bellman Expectation Equations

Let $\pi$ be a stationary policy. Unrolling the return $G_t = R_{t+1} + \gamma G_{t+1}$ inside the expectation definition of $V^\pi(s)$:

$$V^\pi(s) = \mathbb{E}_\pi [G_t \mid S_t = s]$$
$$= \mathbb{E}_\pi [R_{t+1} + \gamma G_{t+1} \mid S_t = s]$$
$$= \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) \left[ \mathcal{R}(s, a, s') + \gamma \mathbb{E}_\pi [G_{t+1} \mid S_{t+1} = s'] \right]$$

Thus, we obtain the **Bellman Expectation Equation for $V^\pi$**:
$$V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \left[ \mathcal{R}(s, a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) V^\pi(s') \right]$$

Similarly, for the action-value function $Q^\pi(s, a)$:
$$Q^\pi(s, a) = \mathcal{R}(s, a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) V^\pi(s')$$
Substituting $V^\pi(s') = \sum_{a' \in \mathcal{A}} \pi(a' \mid s') Q^\pi(s', a')$ yields the **Bellman Expectation Equation for $Q^\pi$**:
$$Q^\pi(s, a) = \mathcal{R}(s, a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) \sum_{a' \in \mathcal{A}} \pi(a' \mid s') Q^\pi(s', a')$$

---

## 4. Bellman Optimality Equations

Define the optimal value functions:
$$V^*(s) \triangleq \max_{\pi} V^\pi(s), \quad \forall s \in \mathcal{S}$$
$$Q^*(s, a) \triangleq \max_{\pi} Q^\pi(s, a), \quad \forall s \in \mathcal{S}, a \in \mathcal{A}$$

Under an optimal policy, the value of state $s$ must equal the expected return from the best action available in $s$:
$$V^*(s) = \max_{a \in \mathcal{A}} Q^*(s, a)$$

Substituting $Q^*(s, a)$ gives the **Bellman Optimality Equation for $V^*$**:
$$V^*(s) = \max_{a \in \mathcal{A}} \left[ \mathcal{R}(s, a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) V^*(s') \right]$$

And the **Bellman Optimality Equation for $Q^*$**:
$$Q^*(s, a) = \mathcal{R}(s, a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) \max_{a' \in \mathcal{A}} Q^*(s', a')$$

Notice that unlike the expectation equation (which is linear), the optimality equation is **non-linear** due to the $\max$ operator. It cannot be solved directly via standard matrix inversion $(I - \gamma P)^{-1} r$. We must use iterative Dynamic Programming.

---

## 5. The Banach Fixed-Point Theorem & Contraction Proof

Define the Bellman optimality operator $\mathcal{T}^*: \mathbb{R}^{|\mathcal{S}|} \to \mathbb{R}^{|\mathcal{S}|}$ as:
$$(\mathcal{T}^* V)(s) \triangleq \max_{a \in \mathcal{A}} \left[ \mathcal{R}(s, a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) V(s') \right]$$

### 5.1 Theorem: $\mathcal{T}^*$ is a $\gamma$-Contraction Mapping under the $\ell_\infty$ Norm
For any two value functions $U, V \in \mathbb{R}^{|\mathcal{S}|}$:
$$\|\mathcal{T}^* U - \mathcal{T}^* V\|_\infty \le \gamma \|U - V\|_\infty$$

#### Formal Step-by-Step Proof:
1. For any state $s \in \mathcal{S}$:
   $$|(\mathcal{T}^* U)(s) - (\mathcal{T}^* V)(s)| = \left| \max_{a \in \mathcal{A}} \left( \mathcal{R}(s, a) + \gamma \sum_{s'} \mathcal{P}(s' \mid s, a) U(s') \right) - \max_{b \in \mathcal{A}} \left( \mathcal{R}(s, b) + \gamma \sum_{s'} \mathcal{P}(s' \mid s, b) V(s') \right) \right|$$

2. We invoke the mathematical lemma for maximums: for any real-valued functions $f$ and $g$,
   $$|\max_{x} f(x) - \max_{x} g(x)| \le \max_{x} |f(x) - g(x)|$$
   *(Proof of lemma: Let $x_1 = \arg\max f(x)$ and $x_2 = \arg\max g(x)$. If $f(x_1) \ge g(x_2)$, then $0 \le f(x_1) - g(x_2) \le f(x_1) - g(x_1) \le \max_x |f(x) - g(x)|$. Symmetrically if $g(x_2) \ge f(x_1)$).*

3. Applying the lemma to our operator difference:
   $$|(\mathcal{T}^* U)(s) - (\mathcal{T}^* V)(s)| \le \max_{a \in \mathcal{A}} \left| \left( \mathcal{R}(s, a) + \gamma \sum_{s'} \mathcal{P}(s' \mid s, a) U(s') \right) - \left( \mathcal{R}(s, a) + \gamma \sum_{s'} \mathcal{P}(s' \mid s, a) V(s') \right) \right|$$

4. The immediate reward $\mathcal{R}(s, a)$ cancels out algebraically:
   $$= \max_{a \in \mathcal{A}} \gamma \left| \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) (U(s') - V(s')) \right|$$

5. By the triangle inequality and non-negativity of probabilities:
   $$\le \max_{a \in \mathcal{A}} \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) |U(s') - V(s')|$$

6. Since $|U(s') - V(s')| \le \|U - V\|_\infty = \max_{s''} |U(s'') - V(s'')|$:
   $$\le \max_{a \in \mathcal{A}} \gamma \left( \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) \right) \|U - V\|_\infty$$

7. Since $\mathcal{P}$ is a valid transition probability distribution, $\sum_{s'} \mathcal{P}(s' \mid s, a) = 1$:
   $$|(\mathcal{T}^* U)(s) - (\mathcal{T}^* V)(s)| \le \gamma \|U - V\|_\infty, \quad \forall s \in \mathcal{S}$$

8. Taking the maximum over all states $s \in \mathcal{S}$ on the left-hand side:
   $$\|\mathcal{T}^* U - \mathcal{T}^* V\|_\infty \le \gamma \|U - V\|_\infty \quad \blacksquare$$

### 5.2 Consequences of Banach's Fixed-Point Theorem
Because $\mathbb{R}^{|\mathcal{S}|}$ equipped with the $\ell_\infty$ norm is a complete metric space and $\gamma \in [0, 1)$:
1. **Unique Fixed Point**: There exists one and only one optimal value function $V^* \in \mathbb{R}^{|\mathcal{S}|}$ such that $\mathcal{T}^* V^* = V^*$.
2. **Global Linear Convergence**: Starting from any arbitrary initial vector $V_0$, repeated application $V_{k+1} = \mathcal{T}^* V_k$ converges to $V^*$ with linear rate $\gamma$:
   $$\|V_k - V^*\|_\infty \le \frac{\gamma^k}{1 - \gamma} \|V_1 - V_0\|_\infty$$

---

## 6. Dynamic Programming Algorithms

```
                          Dynamic Programming Suite
                   ┌───────────────────┴───────────────────┐
                   ▼                                       ▼
           Policy Iteration                         Value Iteration
    (Alternating GPI loop)                     (Direct Bellman Operator)
    1. Policy Evaluation (V^π)                 V_{k+1}(s) = max_a [R + γ Σ P V_k]
    2. Policy Improvement (greedy w.r.t Q)     Until ||V_{k+1} - V_k||_∞ < θ
    Until π_{k+1} == π_k                       Then π*(s) = argmax_a Q*(s, a)
```

### 6.1 Policy Evaluation
Computes $V^\pi$ for an arbitrary policy $\pi$ by repeatedly applying the linear Bellman expectation operator:
$$V_{k+1}(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \left[ \mathcal{R}(s, a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) V_k(s') \right]$$
Stops when $\max_s |V_{k+1}(s) - V_k(s)| < \theta$.

### 6.2 Policy Iteration
1. **Evaluation**: Compute $V^{\pi_k}$ using policy evaluation.
2. **Improvement**: Form the updated policy greedily:
   $$\pi_{k+1}(s) = \arg\max_{a \in \mathcal{A}} \left[ \mathcal{R}(s, a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) V^{\pi_k}(s') \right]$$
3. **Termination**: If $\pi_{k+1}(s) = \pi_k(s)$ for all $s \in \mathcal{S}$, stop and output $\pi^* = \pi_k$.

#### Policy Improvement Theorem:
By construction, $Q^{\pi_k}(s, \pi_{k+1}(s)) \ge V^{\pi_k}(s)$. By mathematical induction over time steps:
$$V^{\pi_{k+1}}(s) \ge V^{\pi_k}(s), \quad \forall s \in \mathcal{S}$$
Because a finite MDP has only $|\mathcal{A}|^{|\mathcal{S}|}$ distinct deterministic policies, Policy Iteration **must terminate in a finite number of iterations**.

### 6.3 Value Iteration
Combines evaluation and improvement into a single sweep per iteration:
$$V_{k+1}(s) = \max_{a \in \mathcal{A}} \left[ \mathcal{R}(s, a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) V_k(s') \right]$$
Once $\|V_{k+1} - V_k\|_\infty < \theta \frac{1 - \gamma}{2\gamma}$, the greedy policy extracted w.r.t $V_{k+1}$ is guaranteed to be within $\theta$ of the true optimal policy $\pi^*$.
