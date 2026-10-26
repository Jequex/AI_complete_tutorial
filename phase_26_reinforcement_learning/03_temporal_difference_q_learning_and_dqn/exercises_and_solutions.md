# Topic 03: Exercises & Solutions — TD Learning, Q-Learning & DQN

---

### Exercise 1: Overestimation Bias in Q-Learning via Jensen's Inequality
**Problem:**
Let $\hat{Q}(s, a) = Q^*(s, a) + \epsilon_a$ be noisy estimates of the true action-values $Q^*(s, a)$, where errors $\epsilon_a$ are independent mean-zero random variables ($\mathbb{E}[\epsilon_a] = 0$).
Prove that:
$$\mathbb{E}\left[ \max_{a \in \mathcal{A}} \hat{Q}(s, a) \right] \ge \max_{a \in \mathcal{A}} Q^*(s, a)$$
and explain why this creates a systematic positive maximization bias in Q-Learning and DQN (the motivation for Double DQN).

<details>
<summary><strong>Solution</strong></summary>

The maximum function $f(\mathbf{x}) = \max_i x_i$ is a convex function on $\mathbb{R}^k$.
By **Jensen's Inequality**, for any random vector $\mathbf{X}$ and convex function $f$:
$$\mathbb{E}[f(\mathbf{X})] \ge f(\mathbb{E}[\mathbf{X}])$$
Applying this directly:
$$\mathbb{E}\left[ \max_{a \in \mathcal{A}} \hat{Q}(s, a) \right] \ge \max_{a \in \mathcal{A}} \mathbb{E}[\hat{Q}(s, a)]$$
Since $\mathbb{E}[\hat{Q}(s, a)] = \mathbb{E}[Q^*(s, a) + \epsilon_a] = Q^*(s, a) + 0 = Q^*(s, a)$, we have:
$$\mathbb{E}\left[ \max_{a \in \mathcal{A}} \hat{Q}(s, a) \right] \ge \max_{a \in \mathcal{A}} Q^*(s, a) \quad \blacksquare$$
Furthermore, if the noise distribution has positive support and variance $\sigma^2 > 0$, the inequality is strict ($>$).
**Consequence:**
At every Bellman optimality update, the $\max_{a'}$ operator selects the action with the largest positive noise fluctuation. This causes Q-values to be systematically overestimated, leading to sub-optimal policies and divergence.
**Double DQN Solution (van Hasselt et al. 2015):**
Decouple action selection from action evaluation using the online network to pick the action and the target network to evaluate its value:
$$y^{\text{Double}} = R + \gamma Q\left(S', \arg\max_{a'} Q(S', a'; \theta); \, \theta^-\right)$$
</details>

---

### Exercise 2: Semi-Gradient Nature of DQN Loss
**Problem:**
Consider the mean squared Bellman error loss for a transition $(s, a, r, s', d)$:
$$\mathcal{L}(\theta) = \frac{1}{2} \left( r + \gamma(1 - d) \max_{a'} Q(s', a'; \theta) - Q(s, a; \theta) \right)^2$$
Explain why taking the true full gradient $\nabla_\theta \mathcal{L}(\theta)$ (including differentiation through the target) does NOT converge to $Q^*$, and why DQN uses a **semi-gradient** where the target is treated as constant.

<details>
<summary><strong>Solution</strong></summary>

Let $y(\theta) = r + \gamma (1 - d) \max_{a'} Q(s', a'; \theta)$.
If we take the full gradient with respect to $\theta$:
$$\nabla_\theta \mathcal{L}(\theta) = - \left( y(\theta) - Q(s, a; \theta) \right) \cdot \left( \nabla_\theta Q(s, a; \theta) - \nabla_\theta y(\theta) \right)$$
$$= - (y(\theta) - Q(s, a; \theta)) \nabla_\theta Q(s, a; \theta) + \gamma (y(\theta) - Q(s, a; \theta)) \nabla_\theta \max_{a'} Q(s', a'; \theta)$$
Notice the second term $+ \gamma (y(\theta) - Q(s, a; \theta)) \nabla_\theta \max_{a'} Q(s', a'; \theta)$.
This second term pushes the parameters so that the target $y(\theta)$ *moves toward the current prediction*, rather than forcing the current prediction toward the true Bellman backup!
In fact, the true minimum of the Mean Squared Bellman Error objective $\mathbb{E}[( \mathcal{T} Q_\theta - Q_\theta )^2]$ can be heavily distorted or fail to coincide with the fixed point of $\mathcal{T}$ (Baird's counterexample).
**The Semi-Gradient Approach:**
Dynamic programming view: the target $y = r + \gamma \max_{a'} Q(s', a'; \theta^-)$ is treated as an independent fixed supervisory regression label.
Taking gradient only with respect to the prediction $Q(s, a; \theta)$:
$$\nabla_\theta^{\text{semi}} \mathcal{L}(\theta) = - \left( y - Q(s, a; \theta) \right) \nabla_\theta Q(s, a; \theta)$$
This guarantees that updates perform contractive Bellman projection steps.
</details>

---

### Exercise 3: Variance Comparison: Monte Carlo vs TD(0)
**Problem:**
Consider a Markov reward process where transitions from state $s$ yield independent rewards $R_1, R_2, \dots, R_T$ each with mean $\mu$ and variance $\sigma^2$.
Compute the variance of the Monte Carlo return $G_0 = \sum_{t=0}^{T-1} \gamma^t R_{t+1}$ as a function of horizon $T$.
Contrast this with the one-step variance of the TD(0) target $R_1 + \gamma V(S_1)$.

<details>
<summary><strong>Solution</strong></summary>

**Monte Carlo Return Variance:**
Since the rewards are independent:
$$\text{Var}(G_0) = \text{Var}\left( \sum_{t=0}^{T-1} \gamma^t R_{t+1} \right) = \sum_{t=0}^{T-1} \text{Var}(\gamma^t R_{t+1}) = \sum_{t=0}^{T-1} (\gamma^{2})^t \text{Var}(R_{t+1}) = \sigma^2 \sum_{t=0}^{T-1} (\gamma^2)^t$$
Using the geometric series formula:
$$\text{Var}(G_0) = \sigma^2 \frac{1 - \gamma^{2T}}{1 - \gamma^2}$$
As the horizon $T \to \infty$:
$$\lim_{T \to \infty} \text{Var}(G_0) = \frac{\sigma^2}{1 - \gamma^2}$$
For $\gamma = 0.99$, $\frac{1}{1 - \gamma^2} = \frac{1}{1 - 0.9801} \approx 50.25 \sigma^2$. The variance is 50 times larger than the single-step reward variance!

**TD(0) Target Variance:**
Assuming the estimated value function $V(S_1)$ is relatively stable across updates:
$$\text{Var}(R_1 + \gamma V(S_1)) \approx \text{Var}(R_1) = \sigma^2$$
The variance of the TD target is on the order of $\sigma^2$, whereas the Monte Carlo target accumulates variance across all future steps $\approx 50 \sigma^2$. This demonstrates why TD methods exhibit significantly faster sample efficiency and lower variance.
</details>
