# Topic 25.1: Exercises & Solutions — Score-Based Models and Langevin Dynamics

---

### Problem 1: Partition Function Vanishing Property (Easy)

Suppose an unnormalized probability distribution over $x \in \mathbb{R}$ is defined by the energy function $E(x) = \frac{1}{4} x^4 - \frac{1}{2} x^2$:
$$\tilde{p}(x) = \exp(-E(x)) = \exp\left(-\frac{1}{4}x^4 + \frac{1}{2}x^2\right), \quad p(x) = \frac{\tilde{p}(x)}{Z}$$
where $Z = \int_{-\infty}^{\infty} \exp(-\frac{1}{4}x^4 + \frac{1}{2}x^2) dx$.

1. Compute the analytical score function $s(x) = \frac{d}{dx} \log p(x)$.
2. Identify the roots (zeros) of the score function $s(x) = 0$.
3. Classify each root as a local maximum (mode), local minimum (anti-mode), or inflection point of $p(x)$.

<details>
<summary>View Complete Solution</summary>

**Step 1: Compute Score Function:**
$$p(x) = \frac{1}{Z} \exp\left(-\frac{1}{4}x^4 + \frac{1}{2}x^2\right)$$
$$\log p(x) = -\log Z - \frac{1}{4}x^4 + \frac{1}{2}x^2$$
Differentiating with respect to $x$:
$$s(x) = \frac{d}{dx} \log p(x) = 0 - x^3 + x = x(1 - x^2) = x(1 - x)(1 + x)$$
Notice that the normalizer $Z$ vanished identically without needing to evaluate the non-elementary integral!

**Step 2: Find Zeros:**
$$s(x) = 0 \iff x(1 - x^2) = 0 \implies x_1 = -1, \quad x_2 = 0, \quad x_3 = +1$$

**Step 3: Classify Critical Points:**
We inspect the second derivative of the log-density, $s'(x) = 1 - 3x^2$:
- At $x = -1$: $s'(-1) = 1 - 3(1) = -2 < 0 \implies$ Local maximum of $p(x)$ (Stable Mode).
- At $x = 0$: $s'(0) = 1 - 0 = +1 > 0 \implies$ Local minimum of $p(x)$ (Unstable Anti-mode / Energy Barrier).
- At $x = +1$: $s'(1) = 1 - 3(1) = -2 < 0 \implies$ Local maximum of $p(x)$ (Stable Mode).

Langevin particles will be attracted to the two symmetric modes at $x = \pm 1$ and repelled away from $x = 0$.
</details>

---

### Problem 2: 1D Integration by Parts Derivation of Hyvärinen Score Matching (Medium)

Let $p(x)$ be a continuously differentiable probability density function on $\mathbb{R}$ with support $(-\infty, \infty)$ satisfying $\lim_{|x| \to \infty} p(x) = 0$. Let $s_\theta(x)$ be a differentiable score model satisfying $\lim_{|x| \to \infty} p(x) s_\theta(x) = 0$.

Prove from first principles that:
$$\mathbb{E}_{x \sim p(x)} \left[ s_\theta(x) \frac{d}{dx} \log p(x) \right] = -\mathbb{E}_{x \sim p(x)} [s_\theta'(x)]$$
and conclude why this enables estimating the score function without knowing $p(x)$.

<details>
<summary>View Complete Solution</summary>

**Step 1: Express the expectation in integral form:**
$$\mathbb{E}_{x \sim p(x)} \left[ s_\theta(x) \frac{d}{dx} \log p(x) \right] = \int_{-\infty}^{\infty} p(x) s_\theta(x) \left( \frac{1}{p(x)} \frac{dp(x)}{dx} \right) dx = \int_{-\infty}^{\infty} s_\theta(x) \frac{dp(x)}{dx} dx$$

**Step 2: Integration by Parts:**
Recall the integration by parts formula: $\int u \, dv = [u v] - \int v \, du$.
Let:
$$u = s_\theta(x) \implies du = s_\theta'(x) dx$$
$$dv = \frac{dp(x)}{dx} dx \implies v = p(x)$$

Applying the formula:
$$\int_{-\infty}^{\infty} s_\theta(x) \frac{dp(x)}{dx} dx = \left[ s_\theta(x) p(x) \right]_{-\infty}^{\infty} - \int_{-\infty}^{\infty} p(x) s_\theta'(x) dx$$

**Step 3: Boundary Term Evaluation:**
Given the regularity condition $\lim_{|x| \to \infty} p(x) s_\theta(x) = 0$, the boundary term evaluates to:
$$\left[ s_\theta(x) p(x) \right]_{-\infty}^{\infty} = 0 - 0 = 0$$

Therefore:
$$\int_{-\infty}^{\infty} s_\theta(x) \frac{dp(x)}{dx} dx = -\int_{-\infty}^{\infty} p(x) s_\theta'(x) dx = -\mathbb{E}_{x \sim p(x)} [s_\theta'(x)]$$

**Conclusion:**
Expanding the Explicit Score Matching objective $\frac{1}{2}\mathbb{E}[(s_\theta(x) - \nabla_x \log p(x))^2]$ yields:
$$\frac{1}{2}\mathbb{E}[s_\theta(x)^2] - \mathbb{E}\left[ s_\theta(x) \frac{d \log p(x)}{dx} \right] + C = \mathbb{E}\left[ \frac{1}{2} s_\theta(x)^2 + s_\theta'(x) \right] + C$$
The unknown density $p(x)$ only appears as the sampling distribution for the expectation, which is approximated by empirical data samples $\{x_i\}_{i=1}^N \sim p(x)$.
</details>

---

### Problem 3: Analytical Equivalence in Denoising Score Matching (Hard)

Let the true data distribution be a 1D Gaussian: $x \sim \mathcal{N}(\mu, \sigma_0^2)$.
We add Gaussian noise $\tilde{x} = x + \sigma \epsilon$, where $\epsilon \sim \mathcal{N}(0, 1)$, so the perturbation kernel is $q_\sigma(\tilde{x} \mid x) = \mathcal{N}(\tilde{x}; x, \sigma^2)$.

1. Find the true marginal distribution of the noisy variable $\tilde{x}$, and compute its exact score $s_{\text{marginal}}(\tilde{x}) = \frac{d}{d\tilde{x}} \log q_\sigma(\tilde{x})$.
2. The optimal Denoising Score Matching estimator minimizes:
   $$\mathcal{L}(s) = \mathbb{E}_{x, \tilde{x}} \left[ \left( s(\tilde{x}) - \nabla_{\tilde{x}} \log q_\sigma(\tilde{x} \mid x) \right)^2 \right]$$
   Find the analytical minimizer $s^*(\tilde{x}) = \mathbb{E}_{x \sim q(x \mid \tilde{x})} [\nabla_{\tilde{x}} \log q_\sigma(\tilde{x} \mid x)]$.
3. Prove that $s^*(\tilde{x}) \equiv s_{\text{marginal}}(\tilde{x})$.

<details>
<summary>View Complete Solution</summary>

**Step 1: Marginal Distribution and Score:**
Since the sum of two independent Gaussians is Gaussian:
$$\tilde{x} = x + \sigma \epsilon \sim \mathcal{N}(\mu, \sigma_0^2 + \sigma^2)$$
Let $\sigma_T^2 = \sigma_0^2 + \sigma^2$. The log marginal density is:
$$\log q_\sigma(\tilde{x}) = -\frac{1}{2}\log(2\pi \sigma_T^2) - \frac{(\tilde{x} - \mu)^2}{2\sigma_T^2}$$
Differentiating gives the marginal score:
$$s_{\text{marginal}}(\tilde{x}) = -\frac{\tilde{x} - \mu}{\sigma_0^2 + \sigma^2}$$

**Step 2: Optimal DSM Estimator:**
The perturbation conditional score is:
$$\nabla_{\tilde{x}} \log q_\sigma(\tilde{x} \mid x) = -\frac{\tilde{x} - x}{\sigma^2}$$
The Bayes-optimal estimator under mean squared error is the conditional expectation:
$$s^*(\tilde{x}) = \mathbb{E}_{x \mid \tilde{x}} \left[ -\frac{\tilde{x} - x}{\sigma^2} \right] = -\frac{\tilde{x} - \mathbb{E}[x \mid \tilde{x}]}{\sigma^2}$$

By Gaussian conditioning (Bayes' rule for linear Gaussian models):
$$\mathbb{E}[x \mid \tilde{x}] = \mu + \frac{\text{Cov}(x, \tilde{x})}{\text{Var}(\tilde{x})} (\tilde{x} - \mu)$$
Here $\text{Cov}(x, \tilde{x}) = \text{Cov}(x, x + \sigma \epsilon) = \text{Var}(x) = \sigma_0^2$, and $\text{Var}(\tilde{x}) = \sigma_0^2 + \sigma^2$.
Therefore:
$$\mathbb{E}[x \mid \tilde{x}] = \mu + \frac{\sigma_0^2}{\sigma_0^2 + \sigma^2}(\tilde{x} - \mu) = \frac{\sigma^2 \mu + \sigma_0^2 \tilde{x}}{\sigma_0^2 + \sigma^2}$$

**Step 3: Substitution and Equivalence Proof:**
Substitute $\mathbb{E}[x \mid \tilde{x}]$ into $s^*(\tilde{x})$:
$$\tilde{x} - \mathbb{E}[x \mid \tilde{x}] = \tilde{x} - \frac{\sigma^2 \mu + \sigma_0^2 \tilde{x}}{\sigma_0^2 + \sigma^2} = \frac{(\sigma_0^2 + \sigma^2)\tilde{x} - \sigma^2 \mu - \sigma_0^2 \tilde{x}}{\sigma_0^2 + \sigma^2} = \frac{\sigma^2(\tilde{x} - \mu)}{\sigma_0^2 + \sigma^2}$$

Now divide by $-\sigma^2$:
$$s^*(\tilde{x}) = -\frac{1}{\sigma^2} \left( \frac{\sigma^2(\tilde{x} - \mu)}{\sigma_0^2 + \sigma^2} \right) = -\frac{\tilde{x} - \mu}{\sigma_0^2 + \sigma^2} \equiv s_{\text{marginal}}(\tilde{x})$$

The optimal denoising score matching model perfectly converges to the ground-truth score of the perturbed distribution.
</details>
