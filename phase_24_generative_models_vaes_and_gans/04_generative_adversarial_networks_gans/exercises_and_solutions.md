# Topic 24.4: Exercises & Solutions — Generative Adversarial Networks (GANs)

---

### Problem 1: Optimal Discriminator at Distributional Equilibrium (Easy)

Assume the generator has converged perfectly to the true data distribution:
$$p_g(x) = p_{\text{data}}(x) \quad \forall x \in \mathcal{X}$$

1. Compute the exact numerical value of the optimal discriminator $D^*(x)$ for any input $x$.
2. Compute the exact value of the minimax objective $V(D^*, G)$ under this equilibrium:
   $$V(D^*, G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D^*(x)] + \mathbb{E}_{x \sim p_g}[\log(1 - D^*(x))]$$
3. Verify that this result matches $-\log 4 + 2 \cdot \text{JSD}(p_{\text{data}} \parallel p_g)$.

<details>
<summary>View Complete Solution</summary>

**Step 1: Compute $D^*(x)$:**
$$D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)} = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_{\text{data}}(x)} = \frac{p_{\text{data}}(x)}{2 p_{\text{data}}(x)} = \frac{1}{2} = 0.5$$

**Step 2: Evaluate Minimax Objective:**
Substituting $D^*(x) = 0.5$ into the expectations:
$$\mathbb{E}_{x \sim p_{\text{data}}}[\log(0.5)] + \mathbb{E}_{x \sim p_g}[\log(1 - 0.5)] = \log\left(\frac{1}{2}\right) + \log\left(\frac{1}{2}\right) = 2 \log\left(\frac{1}{2}\right) = -2 \log 2 = -\log 4$$
In numerical terms:
$$V(D^*, G) = -\log(4) \approx -1.386294$$

**Step 3: Verification with Jensen-Shannon Divergence:**
When $p_g = p_{\text{data}}$, the mixture distribution is $M = \frac{p_{\text{data}} + p_{\text{data}}}{2} = p_{\text{data}}$.
$$D_{\text{KL}}(p_{\text{data}} \parallel M) = D_{\text{KL}}(p_{\text{data}} \parallel p_{\text{data}}) = 0$$
Thus, $\text{JSD}(p_{\text{data}} \parallel p_g) = 0$.
The formula gives:
$$V(D^*, G) = -\log 4 + 2(0) = -\log 4 \approx -1.386294$$
The two expressions match identically.
</details>

---

### Problem 2: Saturated vs. Non-Saturating Gradient Dynamics (Medium)

Let $s$ denote the raw scalar logit output by the discriminator, such that the predicted probability is $D = \sigma(s) = \frac{1}{1 + e^{-s}}$.
Consider two candidate loss functions for training the generator:
- **Minimax Objective:** $\mathcal{L}_G^{\text{minimax}} = \log(1 - \sigma(s))$
- **Non-Saturating Objective:** $\mathcal{L}_G^{\text{non-sat}} = -\log \sigma(s)$

1. Derive the analytical derivatives $\frac{d\mathcal{L}_G^{\text{minimax}}}{ds}$ and $\frac{d\mathcal{L}_G^{\text{non-sat}}}{ds}$ as explicit functions of $D = \sigma(s)$.
2. Calculate the numerical gradient values when the discriminator is overwhelmingly winning ($D = 0.01$).
3. Calculate the numerical gradient values near convergence ($D = 0.80$).
4. Explain why the minimax objective stalls training in early epochs.

<details>
<summary>View Complete Solution</summary>

**Step 1: Analytical Derivatives with Respect to Logit $s$:**
Recall the derivative of the sigmoid function: $\frac{d\sigma(s)}{ds} = \sigma(s)(1 - \sigma(s)) = D(1 - D)$.

- **Minimax Derivative:**
  $$\frac{d\mathcal{L}_G^{\text{minimax}}}{ds} = \frac{d}{ds} \log(1 - \sigma(s)) = \frac{1}{1 - \sigma(s)} \cdot \left(-\frac{d\sigma(s)}{ds}\right) = -\frac{\sigma(s)(1 - \sigma(s))}{1 - \sigma(s)} = -\sigma(s) = -D$$

- **Non-Saturating Derivative:**
  $$\frac{d\mathcal{L}_G^{\text{non-sat}}}{ds} = \frac{d}{ds} [-\log \sigma(s)] = -\frac{1}{\sigma(s)} \cdot \frac{d\sigma(s)}{ds} = -\frac{\sigma(s)(1 - \sigma(s))}{\sigma(s)} = -(1 - \sigma(s)) = -(1 - D)$$

**Step 2: Numerical Gradients at $D = 0.01$ (Early Training):**
- Minimax: $\frac{d\mathcal{L}_G^{\text{minimax}}}{ds} = -0.01$
- Non-Saturating: $\frac{d\mathcal{L}_G^{\text{non-sat}}}{ds} = -(1 - 0.01) = -0.99$
- **Comparison:** The non-saturating gradient is $99\times$ stronger!

**Step 3: Numerical Gradients at $D = 0.80$ (Near Convergence):**
- Minimax: $\frac{d\mathcal{L}_G^{\text{minimax}}}{ds} = -0.80$
- Non-Saturating: $\frac{d\mathcal{L}_G^{\text{non-sat}}}{ds} = -(1 - 0.80) = -0.20$

**Step 4: Why Minimax Stalls Early Training:**
- Early in training, the generator produces poor images easily recognized by the discriminator ($D \to 0$).
- Under the minimax objective, the gradient magnitude $|-D| \to 0$. The generator receives vanishingly small updates precisely when its weights require the strongest corrections.
- Under the non-saturating objective, the gradient magnitude $|-(1 - D)| \to 1.0$, providing massive, stable updates to quickly escape poor local regions.
</details>

---

### Problem 3: Mode Collapse in a 1D Discrete Zero-Sum Game (Hard)

Consider a toy 1D discrete problem where the true data distribution consists of two equally weighted spikes:
$$p_{\text{data}}(x) = \frac{1}{2}\delta(x + 3) + \frac{1}{2}\delta(x - 3)$$
Suppose the generator has capacity to output only a single scalar value $\theta \in \mathbb{R}$: $p_g(x) = \delta(x - \theta)$.
Assume the discriminator is parameterized as a continuous scoring function $D(x) = \sigma(w \cdot x + b)$.

1. If $\theta$ currently sits at $\theta = +3$ (capturing the positive mode), what is the optimal response of the discriminator?
2. What gradient $\frac{\partial V}{\partial \theta}$ does the generator receive at $\theta = +3$?
3. Show how simultaneous gradient updates cause $\theta$ to oscillate back and forth between $-3$ and $+3$ rather than discovering a probability distribution that covers both modes.

<details>
<summary>View Complete Solution</summary>

**Step 1: Optimal Discriminator Response when $\theta = +3$:**
At $x = +3$, both real and fake data exist: $p_{\text{data}}(3) = 0.5$, $p_g(3) = 1.0$.
$$D^*(+3) = \frac{0.5}{0.5 + 1.0} = \frac{0.5}{1.5} = \frac{1}{3} \approx 0.333$$
At $x = -3$, only real data exists: $p_{\text{data}}(-3) = 0.5$, $p_g(-3) = 0$.
$$D^*(-3) = \frac{0.5}{0.5 + 0} = 1.0$$
The discriminator learns to assign high score to $x = -3$ and lower score to $x = +3$.

**Step 2: Generator Gradient:**
The generator wants to maximize the discriminator's score $D(\theta)$.
Since $D(-3) = 1.0 > D(+3) = 0.333$, the spatial derivative of the discriminator at $\theta = +3$ points strongly in the negative direction toward $-3$:
$$\nabla_\theta D(\theta) < 0$$
The generator takes a large gradient step moving $\theta$ leftward toward $-3$.

**Step 3: Limit Cycle Oscillation (Mode Collapse):**
1. Once $\theta$ moves to $-3$, all generated samples are at $-3$.
2. The discriminator immediately adapts: now $x = -3$ has fake samples, so $D^*(-3) \to 1/3$, while $x = +3$ has only real samples, so $D^*(+3) \to 1.0$.
3. The gradient now forces $\theta$ to jump back to $+3$.
4. Instead of learning a mixed distribution with $50\%$ probability at each mode, the deterministic generator endlessly hops between modes. This is the canonical signature of **Mode Collapse** caused by non-convex minimax dynamics.
</details>
