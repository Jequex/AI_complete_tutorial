# Topic 01: Exercises & Solutions — Markov Decision Processes & Formal Framework

---

### Exercise 1: Geometric Discounting as a Constant Termination Hazard
**Problem:**
Suppose an environment terminates at every discrete time step with a fixed probability $p \in (0, 1)$, independent of history. If the process does not terminate, the agent receives reward $R_t$. If the process terminates, it transitions to an absorbing terminal state with reward 0 forever.
Show that evaluating the undiscounted expected return in this terminating environment is mathematically equivalent to evaluating the discounted return with discount factor $\gamma = 1 - p$ in an infinite-horizon non-terminating environment.

<details>
<summary><strong>Solution</strong></summary>

Let $T \in \{1, 2, 3, \dots\}$ be the random time step at which the process terminates. At each step, survival occurs with probability $1 - p$.
The probability that the agent survives for at least $k$ steps is:
$$\mathbb{P}(T \ge k) = (1 - p)^{k-1}$$
The expected undiscounted return across the stochastic lifetime $T$ is:
$$\mathbb{E}\left[ \sum_{t=1}^T R_t \right] = \mathbb{E}\left[ \sum_{t=1}^\infty R_t \cdot \mathbb{I}(T \ge t) \right]$$
By the linearity of expectation and independence of termination:
$$= \sum_{t=1}^\infty R_t \cdot \mathbb{P}(T \ge t) = \sum_{t=1}^\infty R_t (1 - p)^{t-1}$$
Letting $k = t - 1$ and defining $\gamma \triangleq 1 - p \in (0, 1)$:
$$= \sum_{k=0}^\infty \gamma^k R_{k+1}$$
This is precisely the definition of the discounted return $G_0$ with discount factor $\gamma = 1 - p$. Hence, geometric discounting is isomorphic to an ongoing survival process with constant failure rate $p = 1 - \gamma$.
</details>

---

### Exercise 2: Strict Invertibility of $(\mathbf{I} - \gamma \mathbf{P}^\pi)$ via Gershgorin Circle Theorem
**Problem:**
Let $\mathbf{P}^\pi \in \mathbb{R}^{N \times N}$ be a valid row-stochastic transition matrix (i.e. $\sum_{j=1}^N P_{ij} = 1$ and $P_{ij} \ge 0$ for all $i, j$). Let $\gamma \in [0, 1)$.
Prove that the matrix $\mathbf{A} = \mathbf{I} - \gamma \mathbf{P}^\pi$ is strictly diagonally dominant and therefore invertible.

<details>
<summary><strong>Solution</strong></summary>

A square matrix $\mathbf{A}$ is strictly diagonally dominant if for every row $i$:
$$|A_{ii}| > \sum_{j \neq i} |A_{ij}|$$
Let us compute the elements of $\mathbf{A} = \mathbf{I} - \gamma \mathbf{P}^\pi$:
- Diagonal elements: $A_{ii} = 1 - \gamma P_{ii}$
- Off-diagonal elements: $A_{ij} = -\gamma P_{ij}$ for $j \neq i$

Since $P_{ii} \in [0, 1]$ and $\gamma \in [0, 1)$:
$$|A_{ii}| = |1 - \gamma P_{ii}| = 1 - \gamma P_{ii}$$
Now compute the sum of absolute values of off-diagonal elements in row $i$:
$$\sum_{j \neq i} |A_{ij}| = \sum_{j \neq i} |-\gamma P_{ij}| = \gamma \sum_{j \neq i} P_{ij}$$
Since $\mathbf{P}^\pi$ is row-stochastic, $\sum_{j \neq i} P_{ij} = 1 - P_{ii}$. Therefore:
$$\sum_{j \neq i} |A_{ij}| = \gamma (1 - P_{ii})$$
Comparing the diagonal term with the off-diagonal sum:
$$|A_{ii}| - \sum_{j \neq i} |A_{ij}| = (1 - \gamma P_{ii}) - \gamma (1 - P_{ii}) = 1 - \gamma > 0$$
Since $\gamma < 1$, we have $1 - \gamma > 0$ strictly for every row $i$.
By the **Levy-Desplanques Theorem** (a direct corollary of the **Gershgorin Circle Theorem**), every strictly diagonally dominant matrix is non-singular.
Thus, $\mathbf{I} - \gamma \mathbf{P}^\pi$ is strictly non-singular and invertible, guaranteeing that the analytical state-value vector $\mathbf{v}^\pi = (\mathbf{I} - \gamma \mathbf{P}^\pi)^{-1} \mathbf{r}^\pi$ exists and is unique.
</details>

---

### Exercise 3: Analytical Calculation on a 3-State Line World
**Problem:**
Consider a 3-state environment $\mathcal{S} = \{s_1, s_2, s_3\}$.
- In state $s_1$, the agent can only move right to $s_2$ with reward $R(s_1) = 0$.
- In state $s_2$, the agent moves right to $s_3$ with probability $0.8$, or left back to $s_1$ with probability $0.2$. Reward $R(s_2) = +1$.
- State $s_3$ is a terminal absorbing state with reward $0$ forever ($P(s_3 \mid s_3) = 1, R(s_3) = 0$).
Let $\gamma = 0.9$. Compute the exact state values $V(s_1), V(s_2), V(s_3)$.

<details>
<summary><strong>Solution</strong></summary>

State $s_3$ is terminal with reward 0:
$$V(s_3) = 0$$
For state $s_1$:
$$V(s_1) = R(s_1) + \gamma V(s_2) = 0 + 0.9 V(s_2) = 0.9 V(s_2)$$
For state $s_2$:
$$V(s_2) = R(s_2) + \gamma [ 0.2 V(s_1) + 0.8 V(s_3) ]$$
$$V(s_2) = 1 + 0.9 [ 0.2 V(s_1) + 0 ] = 1 + 0.18 V(s_1)$$
Substitute $V(s_1) = 0.9 V(s_2)$ into the equation for $V(s_2)$:
$$V(s_2) = 1 + 0.18 (0.9 V(s_2)) = 1 + 0.162 V(s_2)$$
$$V(s_2) (1 - 0.162) = 1 \implies 0.838 V(s_2) = 1 \implies V(s_2) = \frac{1}{0.838} \approx 1.1933$$
Now compute $V(s_1)$:
$$V(s_1) = 0.9 \times 1.193317 \approx 1.0740$$
Final values:
- $V(s_1) \approx 1.0740$
- $V(s_2) \approx 1.1933$
- $V(s_3) = 0.0$
</details>

---

### Exercise 4: Markovian State Augmentation
**Problem:**
A robot velocity controller observes only position $x_t \in \mathbb{R}$. Acceleration $a_t$ causes new positions following Newtonian mechanics: $x_{t+1} = 2x_t - x_{t-1} + \Delta t^2 a_t + \epsilon_t$.
Explain why the scalar state $S_t = x_t$ is non-Markovian, and show how to construct an augmented state that restores the Markov property.

<details>
<summary><strong>Solution</strong></summary>

1. **Why $S_t = x_t$ is non-Markovian:**
   The conditional distribution of $x_{t+1}$ given $S_t = x_t$ and action $a_t$ depends explicitly on $x_{t-1}$:
   $$P(x_{t+1} \mid x_t, a_t, x_{t-1}) \neq P(x_{t+1} \mid x_t, a_t)$$
   Knowing only the current position $x_t$ gives no information about current velocity $v_t \approx \frac{x_t - x_{t-1}}{\Delta t}$.
2. **State Augmentation:**
   Define the augmented state vector $\tilde{S}_t \in \mathbb{R}^2$:
   $$\tilde{S}_t = \begin{bmatrix} x_t \\ x_{t-1} \end{bmatrix} \quad \text{or equivalently} \quad \tilde{S}_t = \begin{bmatrix} x_t \\ v_t \end{bmatrix} \text{ where } v_t = \frac{x_t - x_{t-1}}{\Delta t}$$
   Then the transition for $\tilde{S}_{t+1}$ is:
   $$\tilde{S}_{t+1} = \begin{bmatrix} x_{t+1} \\ x_t \end{bmatrix} = \begin{bmatrix} 2 & -1 \\ 1 & 0 \end{bmatrix} \begin{bmatrix} x_t \\ x_{t-1} \end{bmatrix} + \begin{bmatrix} \Delta t^2 \\ 0 \end{bmatrix} a_t + \begin{bmatrix} \epsilon_t \\ 0 \end{bmatrix}$$
   This transition depends exclusively on $\tilde{S}_t$ and $a_t$, satisfying the Markov property.
</details>
