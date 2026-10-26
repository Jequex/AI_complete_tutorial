# Topic 05: Exercises & Solutions — Actor-Critic & PPO

---

### Exercise 1: Algebraic Equivalence of GAE
**Problem:**
Let the $k$-step advantage estimator be $\hat{A}_t^{(k)} = \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}^V$.
Define the Generalized Advantage Estimator as the geometrically weighted average:
$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} \triangleq (1 - \lambda) \sum_{k=1}^\infty \lambda^{k-1} \hat{A}_t^{(k)}$$
Prove that this definition is identically equal to:
$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}^V$$

<details>
<summary><strong>Solution</strong></summary>

Substitute the expansion of $\hat{A}_t^{(k)}$ into the geometric mixture:
$$\hat{A}_t^{\text{GAE}} = (1 - \lambda) \sum_{k=1}^\infty \lambda^{k-1} \left( \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}^V \right)$$
Now interchange the order of summation. The condition $0 \le l \le k-1$ is equivalent to $k \ge l + 1$:
$$= (1 - \lambda) \sum_{l=0}^\infty \gamma^l \delta_{t+l}^V \left( \sum_{k=l+1}^\infty \lambda^{k-1} \right)$$
Shift the index of the inner sum by setting $m = k - 1$:
$$\sum_{k=l+1}^\infty \lambda^{k-1} = \sum_{m=l}^\infty \lambda^m$$
Using the geometric series formula for $|\lambda| < 1$:
$$\sum_{m=l}^\infty \lambda^m = \lambda^l \sum_{j=0}^\infty \lambda^j = \frac{\lambda^l}{1 - \lambda}$$
Substitute this back into the expression:
$$\hat{A}_t^{\text{GAE}} = (1 - \lambda) \sum_{l=0}^\infty \gamma^l \delta_{t+l}^V \left( \frac{\lambda^l}{1 - \lambda} \right)$$
The factor $(1 - \lambda)$ cancels out completely:
$$= \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}^V \quad \blacksquare$$
This proves that GAE is an exponential discount of TD residuals with effective decay parameter $\gamma \lambda$.
</details>

---

### Exercise 2: Subgradient Analysis of the PPO Clipped Objective
**Problem:**
Let the PPO clipped surrogate objective for a single sample with advantage $\hat{A}$ and probability ratio $r \triangleq \frac{\pi_\theta(a \mid s)}{\pi_{\theta_{\text{old}}}(a \mid s)}$ be:
$$L(r) = \min\left( r \hat{A}, \, \text{clip}(r, 1 - \epsilon, 1 + \epsilon) \hat{A} \right)$$
Compute the derivative $\frac{\partial L}{\partial r}$ across all possible values of $\hat{A}$ and $r$, and show where the gradient vanishes.

<details>
<summary><strong>Solution</strong></summary>

We analyze the two distinct cases based on the sign of the advantage $\hat{A}$:

**Case 1: Positive Advantage ($\hat{A} > 0$)**
Here, $\text{clip}(r, 1-\epsilon, 1+\epsilon) \hat{A}$ equals:
- $(1 - \epsilon) \hat{A}$ if $r < 1 - \epsilon$
- $r \hat{A}$ if $1 - \epsilon \le r \le 1 + \epsilon$
- $(1 + \epsilon) \hat{A}$ if $r > 1 + \epsilon$

Taking the minimum with $r \hat{A}$:
- If $r < 1 - \epsilon$: $\min(r \hat{A}, (1 - \epsilon) \hat{A}) = r \hat{A}$ (since $r < 1 - \epsilon$).
- If $1 - \epsilon \le r \le 1 + \epsilon$: $\min(r \hat{A}, r \hat{A}) = r \hat{A}$.
- If $r > 1 + \epsilon$: $\min(r \hat{A}, (1 + \epsilon) \hat{A}) = (1 + \epsilon) \hat{A}$.

Therefore:
$$L(r) = \begin{cases} r \hat{A} & \text{if } r \le 1 + \epsilon \\ (1 + \epsilon) \hat{A} & \text{if } r > 1 + \epsilon \end{cases}$$
The derivative is:
$$\frac{\partial L}{\partial r} = \begin{cases} \hat{A} & \text{if } r < 1 + \epsilon \\ 0 & \text{if } r > 1 + \epsilon \end{cases}$$

**Case 2: Negative Advantage ($\hat{A} < 0$)**
Because $\hat{A} < 0$, multiplying by $\hat{A}$ reverses inequalities.
The clipped product is:
- $(1 - \epsilon) \hat{A}$ if $r < 1 - \epsilon$
- $r \hat{A}$ if $1 - \epsilon \le r \le 1 + \epsilon$
- $(1 + \epsilon) \hat{A}$ if $r > 1 + \epsilon$

Taking the minimum with $r \hat{A}$:
- If $r < 1 - \epsilon$: $(1 - \epsilon) \hat{A} < r \hat{A}$ (since $r < 1 - \epsilon$ and $\hat{A} < 0$). So $\min = (1 - \epsilon) \hat{A}$.
- If $1 - \epsilon \le r \le 1 + \epsilon$: $\min = r \hat{A}$.
- If $r > 1 + \epsilon$: $r \hat{A} < (1 + \epsilon) \hat{A}$. So $\min = r \hat{A}$.

Therefore:
$$L(r) = \begin{cases} (1 - \epsilon) \hat{A} & \text{if } r < 1 - \epsilon \\ r \hat{A} & \text{if } r \ge 1 - \epsilon \end{cases}$$
The derivative is:
$$\frac{\partial L}{\partial r} = \begin{cases} 0 & \text{if } r < 1 - \epsilon \\ \hat{A} & \text{if } r > 1 - \epsilon \end{cases}$$

**Summary:**
The gradient vanishes whenever:
1. $\hat{A} > 0$ and $r > 1 + \epsilon$ (preventing over-updating good actions).
2. $\hat{A} < 0$ and $r < 1 - \epsilon$ (preventing destructive collapse on bad actions).
</details>

---

### Exercise 3: Policy Entropy Gradient
**Problem:**
Let $\pi_\theta(a \mid s)$ be a categorical policy with Shannon entropy $\mathcal{H}(\pi_\theta) = -\sum_{a \in \mathcal{A}} \pi_\theta(a \mid s) \log \pi_\theta(a \mid s)$.
Derive the analytical gradient $\nabla_\theta \mathcal{H}(\pi_\theta)$.

<details>
<summary><strong>Solution</strong></summary>

Differentiating using the product rule:
$$\nabla_\theta \mathcal{H}(\pi_\theta) = - \sum_{a \in \mathcal{A}} \left[ \nabla_\theta \pi_\theta(a \mid s) \log \pi_\theta(a \mid s) + \pi_\theta(a \mid s) \nabla_\theta \log \pi_\theta(a \mid s) \right]$$
Using $\nabla_\theta \log \pi = \frac{\nabla_\theta \pi}{\pi}$:
$$\pi_\theta(a \mid s) \nabla_\theta \log \pi_\theta(a \mid s) = \nabla_\theta \pi_\theta(a \mid s)$$
Summing this second term across all actions:
$$\sum_{a \in \mathcal{A}} \nabla_\theta \pi_\theta(a \mid s) = \nabla_\theta \left( \sum_{a \in \mathcal{A}} \pi_\theta(a \mid s) \right) = \nabla_\theta (1) = \mathbf{0}$$
The second term vanishes completely!
Thus:
$$\nabla_\theta \mathcal{H}(\pi_\theta) = - \sum_{a \in \mathcal{A}} \nabla_\theta \pi_\theta(a \mid s) \log \pi_\theta(a \mid s)$$
Or equivalently, using the log-derivative trick $\nabla_\theta \pi = \pi \nabla_\theta \log \pi$:
$$\nabla_\theta \mathcal{H}(\pi_\theta) = - \mathbb{E}_{a \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a \mid s) \log \pi_\theta(a \mid s) \right]$$
Maximizing entropy discourages probabilities from collapsing to 0 or 1 prematurely, sustaining exploration.
</details>
