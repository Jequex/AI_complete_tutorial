# Topic 25.2: Exercises & Solutions — DDPM Forward & Reverse Processes

---

### Problem 1: Closed-Form Marginal Calculation (Easy)

Consider a 3-step diffusion schedule with noise variances:
$$\beta_1 = 0.10, \quad \beta_2 = 0.20, \quad \beta_3 = 0.30$$
A clean 1D data sample is $x_0 = 5.0$.

1. Compute the retention factors $\alpha_1, \alpha_2, \alpha_3$ and cumulative products $\bar{\alpha}_1, \bar{\alpha}_2, \bar{\alpha}_3$.
2. Write down the closed-form distribution $q(x_3 \mid x_0)$.
3. If an injected noise sample is $\epsilon = -1.2$, compute the exact value of $x_3$.

<details>
<summary>View Complete Solution</summary>

**Step 1: Compute $\alpha_t$ and $\bar{\alpha}_t$:**
- $\alpha_1 = 1 - \beta_1 = 1 - 0.10 = 0.90 \implies \bar{\alpha}_1 = \alpha_1 = 0.900$
- $\alpha_2 = 1 - \beta_2 = 1 - 0.20 = 0.80 \implies \bar{\alpha}_2 = \bar{\alpha}_1 \cdot \alpha_2 = 0.900 \times 0.80 = 0.720$
- $\alpha_3 = 1 - \beta_3 = 1 - 0.30 = 0.70 \implies \bar{\alpha}_3 = \bar{\alpha}_2 \cdot \alpha_3 = 0.720 \times 0.70 = 0.504$

**Step 2: Distribution $q(x_3 \mid x_0)$:**
The closed-form distribution is:
$$q(x_3 \mid x_0) = \mathcal{N}\left(x_3; \sqrt{\bar{\alpha}_3} x_0, (1 - \bar{\alpha}_3)\right)$$
- Mean: $\mu_3 = \sqrt{0.504} \times 5.0 \approx 0.70993 \times 5.0 \approx 3.5496$
- Variance: $\sigma_3^2 = 1 - 0.504 = 0.496$
- Standard deviation: $\sigma_3 = \sqrt{0.496} \approx 0.70427$

Thus, $q(x_3 \mid x_0) = \mathcal{N}(3.5496, 0.496)$.

**Step 3: Sample Calculation:**
Using the reparameterization formula:
$$x_3 = \sqrt{\bar{\alpha}_3} x_0 + \sqrt{1 - \bar{\alpha}_3} \epsilon = 3.5496 + 0.70427 \times (-1.2) = 3.5496 - 0.8451 = 2.7045$$
</details>

---

### Problem 2: Posterior Mean Completing the Square (Medium)

In deriving $q(x_{t-1} \mid x_t, x_0)$, the Gaussian exponent contains:
$$-\frac{1}{2} \left[ \frac{\|x_t - \sqrt{\alpha_t} x_{t-1}\|^2}{\beta_t} + \frac{\|x_{t-1} - \sqrt{\bar{\alpha}_{t-1}} x_0\|^2}{1 - \bar{\alpha}_{t-1}} - \frac{\|x_t - \sqrt{\bar{\alpha}_t} x_0\|^2}{1 - \bar{\alpha}_t} \right]$$

Prove algebraically that the linear coefficient of $x_{t-1}$ combined with the posterior variance $\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$ gives the posterior mean:
$$\tilde{\boldsymbol{\mu}}_t(x_t, x_0) = \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} x_0$$

<details>
<summary>View Complete Solution</summary>

**Step 1: Collect Linear Terms in $x_{t-1}$:**
The terms inside the bracket linear in $x_{t-1}$ are:
$$-\frac{2\sqrt{\alpha_t} x_t x_{t-1}}{\beta_t} - \frac{2\sqrt{\bar{\alpha}_{t-1}} x_0 x_{t-1}}{1 - \bar{\alpha}_{t-1}} = -2 \left( \frac{\sqrt{\alpha_t}}{\beta_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}} x_0 \right) x_{t-1}$$

In a univariate Gaussian $\mathcal{N}(\mu, \sigma^2)$, the exponent is:
$$-\frac{1}{2\sigma^2}(x - \mu)^2 = -\frac{1}{2\sigma^2} x^2 + \frac{\mu}{\sigma^2} x - \frac{\mu^2}{2\sigma^2}$$
Matching the linear coefficient of $x_{t-1}$:
$$\frac{\tilde{\boldsymbol{\mu}}_t}{\tilde{\beta}_t} = \frac{\sqrt{\alpha_t}}{\beta_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}} x_0$$

**Step 2: Multiply by $\tilde{\beta}_t$:**
$$\tilde{\boldsymbol{\mu}}_t = \tilde{\beta}_t \left( \frac{\sqrt{\alpha_t}}{\beta_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}} x_0 \right)$$
Substitute $\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$:
$$\tilde{\boldsymbol{\mu}}_t = \left( \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t \right) \frac{\sqrt{\alpha_t}}{\beta_t} x_t + \left( \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t \right) \frac{\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}} x_0$$

Notice the exact cancellations:
- In the $x_t$ term, $\beta_t$ cancels with $\beta_t$:
  $$\frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t$$
- In the $x_0$ term, $(1 - \bar{\alpha}_{t-1})$ cancels with $(1 - \bar{\alpha}_{t-1})$:
  $$\frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} x_0$$

Summing both terms:
$$\tilde{\boldsymbol{\mu}}_t(x_t, x_0) = \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} x_0$$
This completes the derivation.
</details>

---

### Problem 3: Behavior at the Final Step $t = 1$ (Medium)

Show that at the very final step of the reverse process ($t = 1$, transitioning from $x_1$ to $x_0$):
1. The posterior variance $\tilde{\beta}_1$ is identically zero.
2. The posterior mean $\tilde{\boldsymbol{\mu}}_1(x_1, x_0)$ equals $x_0$ exactly.
3. Explain why this guarantees deterministic recovery of clean data when $x_0$ is known.

<details>
<summary>View Complete Solution</summary>

**Step 1: Evaluate $\tilde{\beta}_1$:**
By definition, $\bar{\alpha}_0 = 1$ (since at $t=0$ no noise has been added).
$$\tilde{\beta}_1 = \frac{1 - \bar{\alpha}_0}{1 - \bar{\alpha}_1} \beta_1 = \frac{1 - 1}{1 - \bar{\alpha}_1} \beta_1 = \frac{0}{1 - \bar{\alpha}_1} \beta_1 = 0$$

**Step 2: Evaluate $\tilde{\boldsymbol{\mu}}_1(x_1, x_0)$:**
Substitute $\bar{\alpha}_0 = 1$ and $\bar{\alpha}_1 = \alpha_1 = 1 - \beta_1$:
$$\tilde{\boldsymbol{\mu}}_1(x_1, x_0) = \frac{\sqrt{\alpha_1}(1 - \bar{\alpha}_0)}{1 - \bar{\alpha}_1} x_1 + \frac{\sqrt{\bar{\alpha}_0}\beta_1}{1 - \bar{\alpha}_1} x_0$$
$$= \frac{\sqrt{\alpha_1}(0)}{1 - \bar{\alpha}_1} x_1 + \frac{(1)\beta_1}{1 - (1 - \beta_1)} x_0 = 0 + \frac{\beta_1}{\beta_1} x_0 = x_0$$

**Step 3: Physical Meaning:**
At $t=1$, the transition distribution collapses to a Dirac delta function:
$$q(x_0 \mid x_1, x_0) = \mathcal{N}(x_0, 0 \cdot \mathbf{I}) = \delta(x_0)$$
Conditioned on knowing the ground-truth target $x_0$, there is zero residual uncertainty in the final reconstruction.
</details>

---

### Problem 4: Tweedie's Formula and Score Connection (Hard)

Herbert Robbins and Maurice Tweedie proved that for an observation $y \sim \mathcal{N}(x, \sigma^2 \mathbf{I})$ with prior $p(x)$, the posterior mean is:
$$\mathbb{E}[x \mid y] = y + \sigma^2 \nabla_y \log p(y)$$

In DDPM, we have $x_t \sim \mathcal{N}(\sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t)\mathbf{I})$.
1. Express $x_t / \sqrt{\bar{\alpha}_t}$ in Tweedie's form and derive the optimal Bayes estimator $\hat{x}_0(x_t)$ in terms of the score $\nabla_{x_t} \log q(x_t)$.
2. Relate the predicted noise $\boldsymbol{\epsilon}_\theta(x_t, t)$ to the score function $s_\theta(x_t, t) = \nabla_{x_t} \log q(x_t)$.

<details>
<summary>View Complete Solution</summary>

**Step 1: Apply Tweedie's Formula:**
Divide the closed-form forward equation by $\sqrt{\bar{\alpha}_t}$:
$$\frac{x_t}{\sqrt{\bar{\alpha}_t}} = x_0 + \sqrt{\frac{1 - \bar{\alpha}_t}{\bar{\alpha}_t}} \boldsymbol{\epsilon}$$
Here the observation is $y = \frac{x_t}{\sqrt{\bar{\alpha}_t}}$ with additive noise variance $\sigma_t^2 = \frac{1 - \bar{\alpha}_t}{\bar{\alpha}_t}$.
By Tweedie's formula:
$$\hat{x}_0(x_t) = \mathbb{E}[x_0 \mid x_t] = \frac{x_t}{\sqrt{\bar{\alpha}_t}} + \sigma_t^2 \nabla_{x_t / \sqrt{\bar{\alpha}_t}} \log q(x_t)$$
Using the chain rule $\nabla_{x_t / \sqrt{\bar{\alpha}_t}} = \sqrt{\bar{\alpha}_t} \nabla_{x_t}$:
$$\hat{x}_0(x_t) = \frac{x_t}{\sqrt{\bar{\alpha}_t}} + \left(\frac{1 - \bar{\alpha}_t}{\bar{\alpha}_t}\right) \sqrt{\bar{\alpha}_t} \nabla_{x_t} \log q(x_t) = \frac{x_t + (1 - \bar{\alpha}_t) \nabla_{x_t} \log q(x_t)}{\sqrt{\bar{\alpha}_t}}$$

**Step 2: Relation to Noise Prediction:**
In DDPM, the denoised estimate is:
$$\hat{x}_0(x_t) = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}$$
Equating the two expressions:
$$\frac{x_t + (1 - \bar{\alpha}_t) \nabla_{x_t} \log q(x_t)}{\sqrt{\bar{\alpha}_t}} = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}$$
$$(1 - \bar{\alpha}_t) \nabla_{x_t} \log q(x_t) = -\sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}_\theta(x_t, t)$$
$$\nabla_{x_t} \log q(x_t) = -\frac{\boldsymbol{\epsilon}_\theta(x_t, t)}{\sqrt{1 - \bar{\alpha}_t}}$$

**Fundamental Unification:**
DDPM noise prediction and score matching are mathematically identical up to the scaling constant $-\frac{1}{\sqrt{1 - \bar{\alpha}_t}}$! Predicting noise is literally predicting the score vector field pointing uphill toward high data density.
</details>
