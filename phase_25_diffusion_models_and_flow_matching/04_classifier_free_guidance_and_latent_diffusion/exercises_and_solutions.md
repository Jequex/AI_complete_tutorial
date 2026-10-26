# Topic 25.4: Exercises & Solutions — Classifier-Free Guidance & Latent Diffusion

---

### Problem 1: Bayes' Rule Score Separation (Easy)

Consider a conditional probability distribution $p(x_t \mid c)$ where $x_t \in \mathbb{R}^D$ is a noisy image vector and $c \in \{1, \dots, K\}$ is a discrete class label.

1. State Bayes' rule expressing $p(x_t \mid c)$ in terms of the prior $p(x_t)$, likelihood $p(c \mid x_t)$, and evidence $p(c)$.
2. Take the gradient with respect to $x_t$ of the logarithm of Bayes' rule.
3. Explicitly explain why the term $\nabla_{x_t} \log p(c)$ evaluates to identically $\mathbf{0}$.

<details>
<summary>View Complete Solution</summary>

**Step 1: Bayes' Rule:**
$$p(x_t \mid c) = \frac{p(x_t) p(c \mid x_t)}{p(c)}$$

**Step 2: Logarithm and Gradient:**
Taking the natural logarithm:
$$\log p(x_t \mid c) = \log p(x_t) + \log p(c \mid x_t) - \log p(c)$$
Differentiating with respect to the vector $x_t$:
$$\nabla_{x_t} \log p(x_t \mid c) = \nabla_{x_t} \log p(x_t) + \nabla_{x_t} \log p(c \mid x_t) - \nabla_{x_t} \log p(c)$$

**Step 3: Evaluation of $\nabla_{x_t} \log p(c)$:**
The prior probability distribution $p(c)$ (the overall frequency of class $c$ in the dataset) is a function strictly of the categorical variable $c$. It contains zero mathematical dependency on the continuous state variable $x_t$:
$$\frac{\partial p(c)}{\partial x_{t, i}} = 0 \quad \forall i \in \{1, \dots, D\} \implies \nabla_{x_t} \log p(c) = \mathbf{0}$$
Therefore:
$$\nabla_{x_t} \log p(x_t \mid c) = \nabla_{x_t} \log p(x_t) + \nabla_{x_t} \log p(c \mid x_t)$$
The conditional score is simply the sum of the unconditional score and the classifier score gradient.
</details>

---

### Problem 2: Vector Extrapolation under Classifier-Free Guidance (Medium)

At a particular reverse diffusion step $t$, the unconditional noise estimate and conditional noise estimate are evaluated as 2D vectors:
$$\boldsymbol{\epsilon}_\theta(x_t, \emptyset) = \begin{bmatrix} 0.20 \\ -0.50 \end{bmatrix}, \quad \boldsymbol{\epsilon}_\theta(x_t, c) = \begin{bmatrix} 0.80 \\ 0.10 \end{bmatrix}$$

1. Compute the guidance difference vector $\Delta \boldsymbol{\epsilon} = \boldsymbol{\epsilon}_\theta(x_t, c) - \boldsymbol{\epsilon}_\theta(x_t, \emptyset)$.
2. Calculate the resulting guided noise prediction $\tilde{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_\theta(x_t, \emptyset) + s \cdot \Delta \boldsymbol{\epsilon}$ for guidance scales $s = 0.0, 1.0, 2.5, 5.0$.
3. Compute the Euclidean norm $\|\tilde{\boldsymbol{\epsilon}}\|_2$ for each scale and explain why high guidance scales cause image over-saturation and contrast artifacts.

<details>
<summary>View Complete Solution</summary>

**Step 1: Compute Difference Vector:**
$$\Delta \boldsymbol{\epsilon} = \begin{bmatrix} 0.80 - 0.20 \\ 0.10 - (-0.50) \end{bmatrix} = \begin{bmatrix} 0.60 \\ 0.60 \end{bmatrix}$$

**Step 2: Guided Vectors Across Scales:**
- **$s = 0.0$ (Unconditional):**
  $$\tilde{\boldsymbol{\epsilon}}_{0.0} = \begin{bmatrix} 0.20 \\ -0.50 \end{bmatrix} + 0.0 \begin{bmatrix} 0.60 \\ 0.60 \end{bmatrix} = \begin{bmatrix} 0.20 \\ -0.50 \end{bmatrix}$$
- **$s = 1.0$ (Standard Conditional):**
  $$\tilde{\boldsymbol{\epsilon}}_{1.0} = \begin{bmatrix} 0.20 \\ -0.50 \end{bmatrix} + 1.0 \begin{bmatrix} 0.60 \\ 0.60 \end{bmatrix} = \begin{bmatrix} 0.80 \\ 0.10 \end{bmatrix}$$
- **$s = 2.5$ (Moderate CFG):**
  $$\tilde{\boldsymbol{\epsilon}}_{2.5} = \begin{bmatrix} 0.20 \\ -0.50 \end{bmatrix} + 2.5 \begin{bmatrix} 0.60 \\ 0.60 \end{bmatrix} = \begin{bmatrix} 0.20 + 1.50 \\ -0.50 + 1.50 \end{bmatrix} = \begin{bmatrix} 1.70 \\ 1.00 \end{bmatrix}$$
- **$s = 5.0$ (High CFG):**
  $$\tilde{\boldsymbol{\epsilon}}_{5.0} = \begin{bmatrix} 0.20 \\ -0.50 \end{bmatrix} + 5.0 \begin{bmatrix} 0.60 \\ 0.60 \end{bmatrix} = \begin{bmatrix} 0.20 + 3.00 \\ -0.50 + 3.00 \end{bmatrix} = \begin{bmatrix} 3.20 \\ 2.50 \end{bmatrix}$$

**Step 3: Norms and Artifacts:**
- $\|\tilde{\boldsymbol{\epsilon}}_{0.0}\| = \sqrt{0.2^2 + (-0.5)^2} = \sqrt{0.29} \approx 0.5385$
- $\|\tilde{\boldsymbol{\epsilon}}_{1.0}\| = \sqrt{0.8^2 + 0.1^2} = \sqrt{0.65} \approx 0.8062$
- $\|\tilde{\boldsymbol{\epsilon}}_{2.5}\| = \sqrt{1.7^2 + 1.0^2} = \sqrt{3.89} \approx 1.9723$
- $\|\tilde{\boldsymbol{\epsilon}}_{5.0}\| = \sqrt{3.2^2 + 2.5^2} = \sqrt{16.49} \approx 4.0608$

**Physical Explanation:**
As $s$ increases, the magnitude of the predicted noise vector blows up linearly with $s$. In DDPM sampling, we subtract $\frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \tilde{\boldsymbol{\epsilon}}$ from $x_t$. If $\|\tilde{\boldsymbol{\epsilon}}\|$ is artificially large, the reverse step subtracts massive values, driving pixel values beyond $[-1, 1]$ into hard clamp limits, causing burned contrast, edge halos, and blown-out highlights!
</details>

---

### Problem 3: Latent Space Dimensionality & Attention FLOP Reductions (Medium)

Consider training a diffusion model on high-resolution $1024 \times 1024$ RGB images ($H = W = 1024$, $C = 3$).
We compare standard Pixel Diffusion against Latent Diffusion using an autoencoder with downsampling factor $f = 8$ and latent channel depth $c_z = 4$.

1. Compute the total number of input dimensions in Pixel Space vs Latent Space.
2. Calculate the exact dimensionality reduction factor.
3. In a single self-attention layer, the attention map $A = \text{softmax}(QK^\top / \sqrt{d})$ operates across $N$ spatial tokens. Calculate $N_{\text{pixel}}$ vs $N_{\text{latent}}$.
4. Compute the theoretical speedup in computing $QK^\top$ (which scales as $\mathcal{O}(N^2)$).

<details>
<summary>View Complete Solution</summary>

**Step 1: Input Dimensions:**
- Pixel Space: $D_{\text{pixel}} = 1024 \times 1024 \times 3 = 3,145,728$ values.
- Latent Space:
  $$h = \frac{1024}{8} = 128, \quad w = \frac{1024}{8} = 128, \quad c_z = 4$$
  $$D_{\text{latent}} = 128 \times 128 \times 4 = 65,536 \text{ values.}$$

**Step 2: Dimensionality Reduction Ratio:**
$$\text{Ratio} = \frac{D_{\text{pixel}}}{D_{\text{latent}}} = \frac{3,145,728}{65,536} = 48.0\times$$

**Step 3: Number of Spatial Tokens $N$:**
- In pixel space: $N_{\text{pixel}} = 1024 \times 1024 = 1,048,576$ tokens.
- In latent space: $N_{\text{latent}} = 128 \times 128 = 16,384$ tokens.

**Step 4: Self-Attention Speedup:**
The complexity of full self-attention scales quadratically with sequence length: $\mathcal{O}(N^2)$.
$$\text{Speedup} = \left( \frac{N_{\text{pixel}}}{N_{\text{latent}}} \right)^2 = \left( \frac{1,048,576}{16,384} \right)^2 = (64)^2 = 4,096\times$$
Computing self-attention in latent space is **over 4,000 times faster** per layer than in pixel space!
</details>

---

### Problem 4: Mathematical Mode Sharpening under CFG (Hard)

The guided score can be integrated back to express the effective unnormalized probability density $\tilde{p}_s(x \mid c)$:
$$\nabla_x \log \tilde{p}_s(x \mid c) = \nabla_x \log p(x) + s \nabla_x \log p(c \mid x)$$

1. Integrate this gradient equation with respect to $x$ to find the formula for $\tilde{p}_s(x \mid c)$.
2. Express $\tilde{p}_s(x \mid c)$ in terms of the conditional density $p(x \mid c)$ and unconditional prior $p(x)$.
3. For a 1D Gaussian model where $p(x) = \mathcal{N}(0, \sigma_0^2)$ and likelihood is $p(c \mid x) = \mathcal{N}(c; x, \sigma_L^2)$, compute the variance of $\tilde{p}_s(x \mid c)$ as a function of $s$.
4. Prove what happens to the variance and sample diversity as $s \to \infty$.

<details>
<summary>View Complete Solution</summary>

**Step 1: Integrate the Score:**
$$\int \nabla_x \log \tilde{p}_s(x \mid c) \, dx = \int \nabla_x \left[ \log p(x) + s \log p(c \mid x) \right] \, dx$$
$$\log \tilde{p}_s(x \mid c) = \log p(x) + s \log p(c \mid x) + C$$
Exponentiating both sides:
$$\tilde{p}_s(x \mid c) \propto p(x) \cdot \left[ p(c \mid x) \right]^s$$

**Step 2: Express in terms of $p(x \mid c)$:**
Since $p(c \mid x) = \frac{p(x \mid c) p(c)}{p(x)} \propto \frac{p(x \mid c)}{p(x)}$:
$$\tilde{p}_s(x \mid c) \propto p(x) \cdot \left( \frac{p(x \mid c)}{p(x)} \right)^s = p(x)^{1 - s} \cdot p(x \mid c)^s$$

**Step 3: Variance for Gaussian Case:**
For $p(x) \propto \exp\left(-\frac{x^2}{2\sigma_0^2}\right)$ and $p(c \mid x) \propto \exp\left(-\frac{(x - c)^2}{2\sigma_L^2}\right)$:
$$\log \tilde{p}_s(x \mid c) = -\frac{x^2}{2\sigma_0^2} - s \frac{(x - c)^2}{2\sigma_L^2} + \text{const}$$
Collecting the quadratic coefficient of $x^2$:
$$\text{Coeff}(x^2) = -\frac{1}{2} \left( \frac{1}{\sigma_0^2} + \frac{s}{\sigma_L^2} \right)$$
In any 1D Gaussian $\mathcal{N}(\mu_s, \sigma_s^2)$, the quadratic exponent coefficient is $-\frac{1}{2\sigma_s^2}$.
Therefore, the guided variance is:
$$\sigma_s^2 = \frac{1}{\frac{1}{\sigma_0^2} + \frac{s}{\sigma_L^2}} = \frac{\sigma_0^2 \sigma_L^2}{\sigma_L^2 + s \sigma_0^2}$$

**Step 4: Limit as $s \to \infty$:**
Taking the limit as $s$ approaches infinity:
$$\lim_{s \to \infty} \sigma_s^2 = \lim_{s \to \infty} \frac{\sigma_0^2 \sigma_L^2}{\sigma_L^2 + s \sigma_0^2} = 0$$
$$\lim_{s \to \infty} \tilde{p}_s(x \mid c) = \delta(x - c)$$
As $s \to \infty$, the guided distribution collapses into a Dirac delta function centered exactly on the class condition $c$!
This proves the fundamental trade-off of Classifier-Free Guidance:
- Higher $s$ increases prompt alignment / fidelity (samples match condition $c$ perfectly).
- Higher $s$ destroys sample diversity by shrinking distribution variance toward zero.
</details>
