# Topic 24.3: Exercises & Solutions — Reparameterization Trick & Latent Space

---

### Problem 1: Gradient Calculation through the Reparameterization Node (Easy)

In a 1D latent bottleneck VAE, an encoder outputs mean $\mu = 1.5$ and log-variance $\gamma = \log(\sigma^2) = -0.69315$ (so $\sigma^2 = 0.5$, $\sigma = \sqrt{0.5} \approx 0.7071$).
A standard normal sample $\epsilon = -0.4$ is drawn.
1. Compute the sampled latent coordinate $z = \mu + \sigma \epsilon$.
2. Suppose the downstream decoder evaluates reconstruction error, producing upstream loss gradient $\frac{\partial \mathcal{L}_{\text{recon}}}{\partial z} = 2.4$.
   Compute the analytical gradients propagated to the encoder outputs:
   $$\frac{\partial \mathcal{L}_{\text{recon}}}{\partial \mu} \quad \text{and} \quad \frac{\partial \mathcal{L}_{\text{recon}}}{\partial \gamma}$$
3. Compute the KL divergence gradients with respect to $\mu$ and $\gamma$ given:
   $$\mathcal{L}_{\text{KL}} = -\frac{1}{2} (1 + \gamma - \mu^2 - \exp(\gamma))$$
4. State the final combined gradient $\nabla_{[\mu, \gamma]} \mathcal{L}_{\text{total}}$ where $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{KL}}$.

<details>
<summary>View Complete Solution</summary>

**Step 1: Compute Latent Value $z$:**
$$z = \mu + \sigma \epsilon = 1.5 + (0.7071) \times (-0.4) = 1.5 - 0.28284 = 1.21716$$

**Step 2: Reconstruction Gradients via Pathwise Chain Rule:**
- With respect to $\mu$:
  $$\frac{\partial \mathcal{L}_{\text{recon}}}{\partial \mu} = \frac{\partial \mathcal{L}_{\text{recon}}}{\partial z} \cdot \frac{\partial z}{\partial \mu} = 2.4 \times 1.0 = 2.4$$
- With respect to $\gamma = \log(\sigma^2)$:
  Recall $z = \mu + \exp(0.5 \gamma) \epsilon$.
  $$\frac{\partial z}{\partial \gamma} = 0.5 \exp(0.5 \gamma) \epsilon = 0.5 \sigma \epsilon = 0.5 \times (0.7071) \times (-0.4) = -0.14142$$
  $$\frac{\partial \mathcal{L}_{\text{recon}}}{\partial \gamma} = \frac{\partial \mathcal{L}_{\text{recon}}}{\partial z} \cdot \frac{\partial z}{\partial \gamma} = 2.4 \times (-0.14142) \approx -0.33941$$

**Step 3: Analytical KL Divergence Gradients:**
$$\frac{\partial \mathcal{L}_{\text{KL}}}{\partial \mu} = -\frac{1}{2}(-2\mu) = \mu = 1.5$$
$$\frac{\partial \mathcal{L}_{\text{KL}}}{\partial \gamma} = -\frac{1}{2}(1 - \exp(\gamma)) = 0.5(\exp(\gamma) - 1) = 0.5(0.5 - 1.0) = -0.25$$

**Step 4: Combined Gradients:**
$$\frac{\partial \mathcal{L}_{\text{total}}}{\partial \mu} = 2.4 + 1.5 = 3.9$$
$$\frac{\partial \mathcal{L}_{\text{total}}}{\partial \gamma} = -0.33941 - 0.25 = -0.58941$$
</details>

---

### Problem 2: Pathwise Derivative vs. REINFORCE Variance Explosion (Medium)

Let $z \sim q_\mu(z) = \mathcal{N}(\mu, 1)$ where we wish to optimize $\mu$ to minimize $\mathbb{E}_{q_\mu}[f(z)]$ for the cost function $f(z) = z^2$.
1. Compute the exact true expectation $\mathcal{J}(\mu) = \mathbb{E}_{q_\mu}[z^2]$ and its analytical gradient $\frac{d\mathcal{J}}{d\mu}$.
2. Under the **Reparameterization Trick**, express $z = \mu + \epsilon$ with $\epsilon \sim \mathcal{N}(0, 1)$. Compute the estimator $\hat{g}_{\text{path}} = \frac{\partial f(\mu + \epsilon)}{\partial \mu}$ and find its variance $\text{Var}(\hat{g}_{\text{path}})$.
3. Under the **Score Function (REINFORCE) Estimator**, $\hat{g}_{\text{score}} = f(z) \nabla_\mu \log q_\mu(z) = z^2 (z - \mu)$. Compute $\mathbb{E}[\hat{g}_{\text{score}}]$ and compute its variance $\text{Var}(\hat{g}_{\text{score}})$.
4. Evaluate both variances when $\mu = 10$. What is the variance ratio?

<details>
<summary>View Complete Solution</summary>

**Step 1: True Expectation and Gradient:**
$$\mathcal{J}(\mu) = \mathbb{E}[z^2] = \text{Var}(z) + (\mathbb{E}[z])^2 = 1 + \mu^2$$
$$\frac{d\mathcal{J}}{d\mu} = 2\mu$$

**Step 2: Pathwise Gradient Estimator & Variance:**
$$z = \mu + \epsilon \implies f(z) = (\mu + \epsilon)^2$$
$$\hat{g}_{\text{path}} = \frac{d}{d\mu}(\mu + \epsilon)^2 = 2(\mu + \epsilon) = 2\mu + 2\epsilon$$
- Expected value: $\mathbb{E}[\hat{g}_{\text{path}}] = 2\mu + 2\mathbb{E}[\epsilon] = 2\mu$ (Unbiased).
- Variance:
  $$\text{Var}(\hat{g}_{\text{path}}) = \text{Var}(2\mu + 2\epsilon) = 4\text{Var}(\epsilon) = 4 \times 1 = 4$$
  *(Remarkably constant and independent of $\mu$!)*

**Step 3: Score Function (REINFORCE) Estimator & Variance:**
Recall $z = \mu + \epsilon$.
$$\hat{g}_{\text{score}} = z^2 (z - \mu) = (\mu + \epsilon)^2 \epsilon = (\mu^2 + 2\mu \epsilon + \epsilon^2)\epsilon = \mu^2 \epsilon + 2\mu \epsilon^2 + \epsilon^3$$
- Expected value:
  $$\mathbb{E}[\hat{g}_{\text{score}}] = \mu^2 \mathbb{E}[\epsilon] + 2\mu \mathbb{E}[\epsilon^2] + \mathbb{E}[\epsilon^3] = 0 + 2\mu(1) + 0 = 2\mu \quad \text{(Unbiased)}$$
- Variance:
  $$\mathbb{E}[\hat{g}_{\text{score}}^2] = \mathbb{E}[(\mu^2 \epsilon + 2\mu \epsilon^2 + \epsilon^3)^2]$$
  Using standard Gaussian higher moments ($\mathbb{E}[\epsilon^2]=1, \mathbb{E}[\epsilon^4]=3, \mathbb{E}[\epsilon^6]=15$, odd moments $= 0$):
  $$= \mu^4 \mathbb{E}[\epsilon^2] + 4\mu^2 \mathbb{E}[\epsilon^4] + \mathbb{E}[\epsilon^6] + 2\mu^2 \mathbb{E}[\epsilon^4]$$
  $$= \mu^4(1) + 6\mu^2(3) + 15 = \mu^4 + 18\mu^2 + 15$$
  Subtracting squared mean $(2\mu)^2 = 4\mu^2$:
  $$\text{Var}(\hat{g}_{\text{score}}) = \mu^4 + 18\mu^2 + 15 - 4\mu^2 = \mu^4 + 14\mu^2 + 15$$

**Step 4: Comparison at $\mu = 10$:**
- Pathwise Variance: $\text{Var}(\hat{g}_{\text{path}}) = 4$.
- REINFORCE Variance: $\text{Var}(\hat{g}_{\text{score}}) = 10^4 + 14(10^2) + 15 = 10,000 + 1,400 + 15 = 11,415$.
- Variance Ratio:
  $$\frac{\text{Var}(\hat{g}_{\text{score}})}{\text{Var}(\hat{g}_{\text{path}})} = \frac{11,415}{4} = 2,853.75\times$$
  The score function estimator has nearly $3,000\times$ higher variance! This proves why the reparameterization trick is essential for training deep generative models.
</details>

---

### Problem 3: The Midpoint Norm Collapse in High-Dimensional LERP (Hard)

Consider a latent space of dimension $d = 100$.
Let $z_A, z_B \in \mathbb{R}^{100}$ be two independent samples drawn from the standard Gaussian prior $p(z) = \mathcal{N}(0, \mathbf{I}_{100})$.
1. By the law of large numbers, what is the expected squared norm $\mathbb{E}[\|z_A\|_2^2]$ and approximate length $\|z_A\|_2$?
2. What is the expected dot product $\mathbb{E}[z_A^\top z_B]$ and the expected angle $\Omega$ between them?
3. Calculate the expected length of the midpoint latent vector $z_{\text{mid}} = \frac{z_A + z_B}{2}$ under Linear Interpolation (LERP).
4. Compute the percentage norm drop between the training endpoints and the LERP midpoint. Explain why Spherical Linear Interpolation (SLERP) completely prevents this collapse.

<details>
<summary>View Complete Solution</summary>

**Step 1: Expected Length of Endpoints:**
$$\mathbb{E}[\|z_A\|_2^2] = \sum_{j=1}^{100} \mathbb{E}[z_{A, j}^2] = 100 \times 1 = 100$$
$$\|z_A\|_2 \approx \sqrt{100} = 10$$
Similarly, $\|z_B\|_2 \approx 10$. Both points lie on the Gaussian shell of radius $R = 10$.

**Step 2: Expected Dot Product and Angle:**
Because $z_A$ and $z_B$ are independent zero-mean vectors:
$$\mathbb{E}[z_A^\top z_B] = \sum_{j=1}^{100} \mathbb{E}[z_{A, j}] \mathbb{E}[z_{B, j}] = 0$$
$$\cos \Omega = \frac{z_A^\top z_B}{\|z_A\|_2 \|z_B\|_2} \approx 0 \implies \Omega \approx \frac{\pi}{2} \ (90^\circ)$$
In high dimensions, any two random Gaussian vectors are virtually orthogonal!

**Step 3: Expected Length of LERP Midpoint:**
Let $z_{\text{mid}} = \frac{1}{2}(z_A + z_B)$.
$$\|z_{\text{mid}}\|_2^2 = \frac{1}{4} \|z_A + z_B\|_2^2 = \frac{1}{4} (\|z_A\|_2^2 + \|z_B\|_2^2 + 2 z_A^\top z_B)$$
Taking expectation:
$$\mathbb{E}[\|z_{\text{mid}}\|_2^2] = \frac{1}{4}(100 + 100 + 0) = \frac{200}{4} = 50$$
$$\|z_{\text{mid}}\|_2 \approx \sqrt{50} \approx 7.071$$

**Step 4: Percentage Norm Drop & SLERP Solution:**
$$\text{Percentage Drop} = \frac{10 - 7.071}{10} \times 100\% \approx 29.29\%$$
- Under LERP, the trajectory dips towards the center of the sphere, losing nearly $30\%$ of its radial distance. In 1000 dimensions, the norm drops by $1 - 1/\sqrt{2} \approx 29.3\%$ regardless of $d$, placing the midpoint deep inside the low-density interior of the Gaussian ball where the decoder has zero training data.
- **SLERP:** By explicitly interpolating along the geodesic arc $z_{\text{SLERP}}(\alpha) = \frac{\sin((1-\alpha)\Omega)}{\sin \Omega} z_A + \frac{\sin(\alpha \Omega)}{\sin \Omega} z_B$, the norm at $\alpha = 0.5$ remains exactly $10.0$ for orthogonal vectors, ensuring the decoder is always evaluated on the authentic data manifold shell.
</details>
