# Topic 04: Exercises & Solutions — Policy Gradient Theorem & REINFORCE

---

### Exercise 1: Causality Proof — Why Past Rewards Drop Out
**Problem:**
Let a trajectory be $\tau = (s_0, a_0, r_1, s_1, \dots, s_T)$.
Prove that for any reward $r_{t'}$ received prior to or at time step $t$ ($t' \le t$):
$$\mathbb{E}_{\tau \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) r_{t'} \right] = \mathbf{0}$$
thereby formally justifying replacing the total trajectory return $R(\tau)$ with the reward-to-go $G_t = \sum_{k=t}^{T-1} \gamma^{k-t} r_{k+1}$.

<details>
<summary><strong>Solution</strong></summary>

Using the law of iterated expectations (tower property), condition on all random variables up to step $t$, including state $s_t$:
$$\mathbb{E}_{\tau} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) r_{t'} \right] = \mathbb{E}_{\tau_{0:t}} \left[ \mathbb{E}_{a_t \sim \pi_\theta(\cdot \mid s_t)} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) r_{t'} \, \middle|\, \tau_{0:t} \right] \right]$$
Because $t' \le t$, the reward $r_{t'}$ occurred in the past and is completely fixed and deterministic given the trajectory prefix $\tau_{0:t}$.
We factor $r_{t'}$ out of the inner expectation over action $a_t$:
$$= \mathbb{E}_{\tau_{0:t}} \left[ r_{t'} \, \mathbb{E}_{a_t \sim \pi_\theta(\cdot \mid s_t)} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \, \middle|\, s_t \right] \right]$$
Now evaluate the inner expectation:
$$\mathbb{E}_{a_t \sim \pi_\theta(\cdot \mid s_t)} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right] = \sum_{a \in \mathcal{A}} \pi_\theta(a \mid s_t) \frac{\nabla_\theta \pi_\theta(a \mid s_t)}{\pi_\theta(a \mid s_t)} = \sum_{a \in \mathcal{A}} \nabla_\theta \pi_\theta(a \mid s_t)$$
$$= \nabla_\theta \left( \sum_{a \in \mathcal{A}} \pi_\theta(a \mid s_t) \right) = \nabla_\theta (1) = \mathbf{0}$$
Therefore:
$$\mathbb{E}_{\tau} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) r_{t'} \right] = \mathbb{E}_{\tau_{0:t}} [ r_{t'} \cdot \mathbf{0} ] = \mathbf{0} \quad \blacksquare$$
Hence past rewards have zero expected correlation with the policy score function and can be dropped, reducing estimator variance without introducing bias.
</details>

---

### Exercise 2: Derivation of the Softmax Score Function
**Problem:**
Let an agent have discrete actions with policy parameterized by linear weights $\boldsymbol{\theta}_a \in \mathbb{R}^d$:
$$\pi_\theta(a \mid s) = \frac{\exp(\boldsymbol{\theta}_a^T \mathbf{x}(s))}{\sum_{b \in \mathcal{A}} \exp(\boldsymbol{\theta}_b^T \mathbf{x}(s))}$$
Derive the analytical gradient of the log-policy $\nabla_{\boldsymbol{\theta}_k} \log \pi_\theta(a \mid s)$ for any action $k \in \mathcal{A}$.

<details>
<summary><strong>Solution</strong></summary>

Take the natural logarithm of $\pi_\theta(a \mid s)$:
$$\log \pi_\theta(a \mid s) = \boldsymbol{\theta}_a^T \mathbf{x}(s) - \log \left( \sum_{b \in \mathcal{A}} \exp(\boldsymbol{\theta}_b^T \mathbf{x}(s)) \right)$$
Differentiate with respect to the parameter vector $\boldsymbol{\theta}_k$:
$$\nabla_{\boldsymbol{\theta}_k} \log \pi_\theta(a \mid s) = \nabla_{\boldsymbol{\theta}_k} \left[ \boldsymbol{\theta}_a^T \mathbf{x}(s) \right] - \nabla_{\boldsymbol{\theta}_k} \log \left( \sum_{b \in \mathcal{A}} \exp(\boldsymbol{\theta}_b^T \mathbf{x}(s)) \right)$$
For the first term:
$$\nabla_{\boldsymbol{\theta}_k} \left[ \boldsymbol{\theta}_a^T \mathbf{x}(s) \right] = \mathbb{I}(a = k) \mathbf{x}(s)$$
where $\mathbb{I}$ is the indicator function ($\mathbb{I}(a=k) = 1$ if $a=k$ and $0$ otherwise).
For the second term:
$$\nabla_{\boldsymbol{\theta}_k} \log \left( \sum_{b} \exp(\boldsymbol{\theta}_b^T \mathbf{x}(s)) \right) = \frac{1}{\sum_b \exp(\boldsymbol{\theta}_b^T \mathbf{x}(s))} \nabla_{\boldsymbol{\theta}_k} \left( \sum_b \exp(\boldsymbol{\theta}_b^T \mathbf{x}(s)) \right)$$
$$= \frac{\exp(\boldsymbol{\theta}_k^T \mathbf{x}(s)) \mathbf{x}(s)}{\sum_b \exp(\boldsymbol{\theta}_b^T \mathbf{x}(s))} = \pi_\theta(k \mid s) \mathbf{x}(s)$$
Subtracting the two terms:
$$\nabla_{\boldsymbol{\theta}_k} \log \pi_\theta(a \mid s) = \mathbf{x}(s) \left[ \mathbb{I}(a = k) - \pi_\theta(k \mid s) \right]$$
**Intuition:**
If action $k$ was selected ($a = k$), the gradient is $\mathbf{x}(s) (1 - \pi_\theta(k \mid s)) > 0$, increasing its probability.
If action $k$ was not selected ($a \neq k$), the gradient is $-\mathbf{x}(s) \pi_\theta(k \mid s) < 0$, decreasing its probability by an amount proportional to its current probability mass.
</details>

---

### Exercise 3: Analytical Variance-Minimizing Baseline $b^*$
**Problem:**
Let the policy gradient estimator be $\hat{g}(b) = \sum_i g_i (R_i - b)$ where $g_i = \nabla_\theta \log \pi_\theta(a_i \mid s_i)$.
Find the scalar constant $b^*$ that minimizes the variance $\text{Var}(\hat{g}(b))$.

<details>
<summary><strong>Solution</strong></summary>

Because $\mathbb{E}[g b] = b \mathbb{E}[g] = 0$, the expectation $\mathbb{E}[\hat{g}(b)] = \mathbb{E}[g R]$ is constant for all $b$.
Therefore, minimizing the variance is equivalent to minimizing the second moment:
$$\text{Var}(\hat{g}(b)) = \mathbb{E}\left[ (g(R - b))^2 \right] - (\mathbb{E}[\hat{g}(b)])^2$$
Define $f(b) \triangleq \mathbb{E}\left[ g^2 (R - b)^2 \right] = \mathbb{E}\left[ g^2 (R^2 - 2 b R + b^2) \right] = \mathbb{E}[g^2 R^2] - 2 b \mathbb{E}[g^2 R] + b^2 \mathbb{E}[g^2]$.
Differentiating with respect to $b$ and setting to zero:
$$\frac{df}{db} = -2 \mathbb{E}[g^2 R] + 2 b \mathbb{E}[g^2] = 0$$
Solving for $b$:
$$b^* = \frac{\mathbb{E}[g^2 R]}{\mathbb{E}[g^2]} = \frac{\mathbb{E}[\|\nabla_\theta \log \pi_\theta(a \mid s)\|^2 R]}{\mathbb{E}[\|\nabla_\theta \log \pi_\theta(a \mid s)\|^2]}$$
This proves that the optimal baseline is a weighted average of returns, weighted by the squared norm of the score function. In practice, the state-value function $V(s) \approx \mathbb{E}[R \mid s]$ serves as an exceptionally close and computationally tractable approximation.
</details>
