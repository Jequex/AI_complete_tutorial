# Topic 25.3: Exercises & Solutions — Noise Schedules & Denoising Networks

---

### Problem 1: Cosine Schedule Boundary Values (Easy)

The cosine schedule defines:
$$f(t) = \cos^2\left( \frac{t/T + s}{1 + s} \cdot \frac{\pi}{2} \right), \quad \bar{\alpha}_t = \frac{f(t)}{f(0)}$$
with $T = 1000$ and offset $s = 0.008$.

1. Compute the analytical value of $f(0)$ to four decimal places.
2. Compute the value of $f(T)$ and the terminal retention factor $\bar{\alpha}_T$.
3. Why is the offset $s > 0$ strictly necessary instead of simply using $s = 0$?

<details>
<summary>View Complete Solution</summary>

**Step 1: Evaluate $f(0)$:**
At $t = 0$:
$$\theta_0 = \frac{0 + 0.008}{1 + 0.008} \cdot \frac{\pi}{2} = \frac{0.008}{1.008} \cdot \frac{\pi}{2} \approx 0.0079365 \times 1.570796 \approx 0.012467 \text{ radians}$$
$$f(0) = \cos^2(0.012467) \approx (0.999922)^2 \approx 0.999844$$
By construction:
$$\bar{\alpha}_0 = \frac{f(0)}{f(0)} = 1.0000$$

**Step 2: Evaluate $f(T)$ and $\bar{\alpha}_T$:**
At $t = T$:
$$\theta_T = \frac{1 + 0.008}{1 + 0.008} \cdot \frac{\pi}{2} = 1.0 \cdot \frac{\pi}{2} = \frac{\pi}{2}$$
$$f(T) = \cos^2\left(\frac{\pi}{2}\right) = 0.0000$$
Thus:
$$\bar{\alpha}_T = \frac{f(T)}{f(0)} = \frac{0}{0.999844} = 0.0000$$
This guarantees that at the final timestep $T$, exactly 100% of the initial data signal is destroyed, leaving pure standard isotropic Gaussian noise $x_T \sim \mathcal{N}(0, \mathbf{I})$.

**Step 3: Purpose of $s > 0$:**
If $s = 0$, then at $t = 0$:
$$\theta_0 = 0 \implies \left. \frac{d}{dt} \cos^2\left(\frac{t}{T} \frac{\pi}{2}\right) \right|_{t=0} = -2\cos(0)\sin(0)\frac{\pi}{2T} = 0$$
The derivative of the cosine squared at 0 is identically zero! This causes $\beta_1 = 1 - \frac{\bar{\alpha}_1}{\bar{\alpha}_0} \approx 0$. If $\beta_1$ were too close to zero, virtually zero noise would be added at step 1, wasting steps and causing numerical precision issues. Setting $s = 0.008$ introduces a slight initial slope, ensuring $\beta_1 > 0$.
</details>

---

### Problem 2: Analysis of the VLB Weight vs. Simplified Loss (Medium)

In the exact Variational Lower Bound (ELBO), the loss term at step $t$ is:
$$L_{t-1} = w(t) \|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(x_t, t)\|^2, \quad \text{where} \quad w(t) = \frac{\beta_t^2}{2 \sigma_t^2 \alpha_t (1 - \bar{\alpha}_t)}$$

Suppose we choose $\sigma_t^2 = \tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$.
1. Simplify the weighting factor $w(t)$ under this choice of $\sigma_t^2$.
2. Analyze the behavior of $w(t)$ as $t \to 1$ (near clean data, where $\bar{\alpha}_t \approx 1$ and $\bar{\alpha}_{t-1} \approx 1$).
3. Explain why discarding $w(t)$ and using $L_{\text{simple}} = \mathbb{E}[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta\|^2]$ leads to dramatically superior perceptual image generation quality.

<details>
<summary>View Complete Solution</summary>

**Step 1: Simplify $w(t)$:**
Substitute $\sigma_t^2 = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$:
$$w(t) = \frac{\beta_t^2}{2 \left( \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t \right) \alpha_t (1 - \bar{\alpha}_t)}$$
Notice that $(1 - \bar{\alpha}_t)$ in the denominator cancels with $(1 - \bar{\alpha}_t)$ from $\sigma_t^2$:
$$w(t) = \frac{\beta_t}{2 \alpha_t (1 - \bar{\alpha}_{t-1})}$$

**Step 2: Behavior as $t \to 1$:**
At $t = 1$, by definition $\bar{\alpha}_0 = 1$.
Therefore, as $t \to 1$:
$$1 - \bar{\alpha}_{t-1} = 1 - \bar{\alpha}_0 = 1 - 1 = 0$$
The denominator $(1 - \bar{\alpha}_{t-1})$ approaches zero, which causes:
$$w(1) \to \infty$$
The theoretical VLB places near-infinite weight on predicting noise at step $t = 1$!

**Step 3: Engineering Consequence:**
- In exact Maximum Likelihood (optimizing VLB), the network devotes almost all its capacity to minimizing errors at tiny timesteps ($t \approx 1$), where noise is imperceptible to the human eye.
- Conversely, at large timesteps ($t \to T$), where the global layout, subject identity, and composition are established, $w(t)$ is small.
- By setting $w(t) = 1.0$ everywhere ($L_{\text{simple}}$), Ho et al. effectively down-weighted the trivial high-frequency noise removal at small $t$, and elevated the importance of large-scale semantic generation at large $t$. This results in sharper, more semantically coherent images.
</details>

---

### Problem 3: Gradient Calculation under $L_{\text{simple}}$ (Hard)

Consider a toy 1D diffusion model with a single scalar weight $w \in \mathbb{R}$:
$$\epsilon_w(x_t, t) = w \cdot x_t$$
Clean data $x_0 \sim p_{\text{data}}$ has mean $\mu_0$ and variance $\sigma_0^2$.
At timestep $t$, $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$, where $\epsilon \sim \mathcal{N}(0, 1)$ is independent of $x_0$.

1. Write the expected loss $\mathcal{L}(w; t) = \mathbb{E}_{x_0, \epsilon} \left[ \frac{1}{2} (\epsilon - \epsilon_w(x_t, t))^2 \right]$.
2. Derive the analytical gradient $\frac{d\mathcal{L}(w; t)}{dw}$.
3. Solve for the optimal weight $w^*(t)$ that minimizes the loss at timestep $t$.
4. Interpret how $w^*(t)$ changes from $t = 1$ (near clean data) to $t = T$ (pure noise).

<details>
<summary>View Complete Solution</summary>

**Step 1: Expected Loss Expansion:**
$$\mathcal{L}(w; t) = \frac{1}{2} \mathbb{E} [ (\epsilon - w x_t)^2 ] = \frac{1}{2} \left( \mathbb{E}[\epsilon^2] - 2w \mathbb{E}[\epsilon x_t] + w^2 \mathbb{E}[x_t^2] \right)$$

We evaluate the expectations:
1. $\mathbb{E}[\epsilon^2] = 1$.
2. $\mathbb{E}[\epsilon x_t] = \mathbb{E}[\epsilon (\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon)] = \sqrt{\bar{\alpha}_t} \mathbb{E}[\epsilon x_0] + \sqrt{1 - \bar{\alpha}_t} \mathbb{E}[\epsilon^2]$.
   Since $\epsilon$ and $x_0$ are independent, $\mathbb{E}[\epsilon x_0] = 0$, so:
   $$\mathbb{E}[\epsilon x_t] = \sqrt{1 - \bar{\alpha}_t}$$
3. $\mathbb{E}[x_t^2] = \bar{\alpha}_t \mathbb{E}[x_0^2] + (1 - \bar{\alpha}_t) \mathbb{E}[\epsilon^2] + 2\sqrt{\bar{\alpha}_t(1 - \bar{\alpha}_t)}\mathbb{E}[x_0 \epsilon]$:
   $$\mathbb{E}[x_t^2] = \bar{\alpha}_t (\mu_0^2 + \sigma_0^2) + (1 - \bar{\alpha}_t)$$

Substitute these into $\mathcal{L}(w; t)$:
$$\mathcal{L}(w; t) = \frac{1}{2} \left[ 1 - 2w \sqrt{1 - \bar{\alpha}_t} + w^2 \left( \bar{\alpha}_t(\mu_0^2 + \sigma_0^2) + 1 - \bar{\alpha}_t \right) \right]$$

**Step 2: Analytical Gradient:**
$$\frac{d\mathcal{L}(w; t)}{dw} = - \sqrt{1 - \bar{\alpha}_t} + w \left[ \bar{\alpha}_t(\mu_0^2 + \sigma_0^2) + 1 - \bar{\alpha}_t \right]$$

**Step 3: Optimal Weight $w^*(t)$:**
Setting the derivative to zero:
$$w^*(t) = \frac{\sqrt{1 - \bar{\alpha}_t}}{\bar{\alpha}_t(\mu_0^2 + \sigma_0^2) + 1 - \bar{\alpha}_t}$$

**Step 4: Interpretation Across Timesteps:**
- At $t \to 0$ ($\bar{\alpha}_t \approx 1$):
  $$w^*(0) \approx \frac{\sqrt{1 - 1}}{\mu_0^2 + \sigma_0^2 + 0} = 0$$
  When the signal is pristine, the observation $x_t \approx x_0$ contains virtually no noise, so the best linear predictor outputs zero noise!
- At $t \to T$ ($\bar{\alpha}_t \approx 0$):
  $$w^*(T) \approx \frac{\sqrt{1 - 0}}{0 + 1 - 0} = 1.0$$
  When all signal is erased, $x_t \approx \epsilon$, so the best predictor simply outputs the observation itself: $\epsilon_w(x_t) = 1.0 \cdot x_t = x_t$!
</details>
