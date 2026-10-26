# Topic 24.2: Variational Autoencoders & The Evidence Lower Bound (ELBO)

---

## 1. Executive Summary & The 12-Year-Old Analogy

### The Quest for Tractable Latent Variable Models
In generative modeling, we hypothesize that observed high-dimensional data $x$ (e.g. an image of a handwritten digit) is produced by underlying, unobserved factors of variation $z$ (e.g. digit identity, stroke thickness, slant angle, scale). 

The generative narrative is a two-step probabilistic process:
1. Nature samples a latent vector from a prior distribution: $z \sim p(z)$ (typically $p(z) = \mathcal{N}(0, \mathbf{I})$).
2. Nature generates the observable instance from a conditional likelihood: $x \sim p_\theta(x \mid z)$.

To evaluate the probability of observing $x$, we must integrate over all possible configurations of latent causes $z$:
$$p_\theta(x) = \int p_\theta(x, z) \, dz = \int p_\theta(x \mid z) p(z) \, dz$$

**The Intractability Catastrophe:** This integral has no closed-form analytical solution when $p_\theta(x \mid z)$ is parameterized by a non-linear deep neural network. Furthermore, numerical quadrature requires evaluating an exponential grid over $\mathbb{R}^d$ ($\mathcal{O}(K^d)$ points), which becomes physically impossible even for $d = 10$.

To invert this process and find the latent code that generated an observed image $x$, Bayes' theorem requires:
$$p_\theta(z \mid x) = \frac{p_\theta(x \mid z) p(z)}{p_\theta(x)} = \frac{p_\theta(x \mid z) p(z)}{\int p_\theta(x \mid z') p(z') \, dz'}$$
Because the denominator $p_\theta(x)$ is intractable, the true posterior $p_\theta(z \mid x)$ is **strictly intractable**.

**Variational Autoencoders (Kingma & Welling, 2013; Rezende et al., 2014)** solve this profound impasse by introducing **Variational Inference**: approximating the intractable true posterior $p_\theta(z \mid x)$ with a tractable parametric family $q_\phi(z \mid x)$ (the encoder), and optimizing a mathematically rigorous surrogate objective: the **Evidence Lower Bound (ELBO)**.

---

### The 12-Year-Old Analogy: The Detective's Flashlight and the Treasure Map

1. **The Mystery of the Buried Chest:**
   - Imagine a treasure hunter finds thousands of mysterious artifacts washed up on a beach ($x$). 
   - Deep beneath the ocean, hidden currents and tectonic faults ($z$) determine what washes ashore.
   - You want to figure out the exact underwater formula ($p(z \mid x)$), but exploring the entire ocean would take a million lifetimes (intractable integral).

2. **The Guessing Flashlight (Variational Encoder $q_\phi(z \mid x)$):**
   - Instead of checking every cubic meter of ocean water, a detective turns on a focused flashlight beam ($q_\phi(z \mid x)$) pointing right at the most likely underwater coordinates $(\mu, \sigma)$ that could have created that artifact.

3. **The Rubber Band (The KL Divergence Regularizer):**
   - If the detective only cared about matching the artifact, they might point the flashlight into bizarre, isolated corners of the ocean.
   - To keep the ocean map organized, a giant rubber band pulls the flashlight beam back toward the center of the world ($\mathcal{N}(0, \mathbf{I})$).
   - This ensures that every coordinate in the ocean is smooth, walkable, and connected—never leaving empty black holes where you get lost.

---

## 2. Mathematical Symbol Decoder

| Symbol | Mathematical Domain | Description & Meaning |
| :--- | :--- | :--- |
| $x \in \mathbb{R}^D$ | Observable Tensor | Observed high-dimensional data point (evidence). |
| $z \in \mathbb{R}^J$ | Continuous Latent Vector | Unobserved latent factors of variation ($J \ll D$). |
| $p(z) = \mathcal{N}(0, \mathbf{I})$ | Standard Gaussian Prior | Prior distribution over latent space before observing data. |
| $p_\theta(x \mid z)$ | Likelihood / Decoder | Generative decoder network parameterized by weights $\theta$. |
| $p_\theta(x) = \int p_\theta(x, z) dz$ | Marginal Likelihood (Evidence) | The marginal probability of observing $x$; strictly intractable. |
| $p_\theta(z \mid x)$ | True Posterior | The exact, intractable distribution over latents given observation $x$. |
| $q_\phi(z \mid x)$ | Variational Recognition Model | Encoder neural network parameterized by weights $\phi$ approximating $p(z \mid x)$. |
| $\boldsymbol{\mu}_\phi(x) \in \mathbb{R}^J$ | Mean Vector | Variational posterior mean predicted by encoder. |
| $\boldsymbol{\sigma}_\phi^2(x) \in \mathbb{R}^J_+$ | Variance Vector | Variational posterior diagonal variance predicted by encoder. |
| $\text{ELBO}(\theta, \phi; x)$ | Scalar ($\mathbb{R}$) | Evidence Lower Bound; the objective maximized during VAE training. |
| $D_{\text{KL}}(q \parallel p)$ | Non-negative Scalar ($\ge 0$) | Kullback-Leibler divergence measuring informational discrepancy between $q$ and $p$. |

---

## 3. Derivation 1: ELBO via Jensen's Inequality

The natural objective in probabilistic modeling is to maximize the log-marginal likelihood of observed data $\log p_\theta(x)$ (termed the **evidence**).

Recall **Jensen's Inequality**: For any random variable $X$ and concave function $f$:
$$f(\mathbb{E}[X]) \ge \mathbb{E}[f(X)]$$
Because the natural logarithm $f(u) = \log u$ is strictly concave on $(0, \infty)$ (its second derivative $f''(u) = -1/u^2 < 0$), we have:
$$\log \mathbb{E}[u] \ge \mathbb{E}[\log u]$$

Now consider the log-marginal likelihood:
$$\log p_\theta(x) = \log \int p_\theta(x, z) \, dz$$

We introduce an arbitrary proposal distribution $q_\phi(z \mid x)$ having common support with $p_\theta(x, z)$ by multiplying and dividing by $q_\phi(z \mid x)$:
$$\log p_\theta(x) = \log \int q_\phi(z \mid x) \frac{p_\theta(x, z)}{q_\phi(z \mid x)} \, dz$$

The integral is simply the mathematical expectation of the ratio with respect to $q_\phi(z \mid x)$:
$$\log p_\theta(x) = \log \mathbb{E}_{z \sim q_\phi(z \mid x)} \left[ \frac{p_\theta(x, z)}{q_\phi(z \mid x)} \right]$$

Applying Jensen's Inequality to pull the expectation outside the logarithm:
$$\log p_\theta(x) \ge \mathbb{E}_{z \sim q_\phi(z \mid x)} \left[ \log \left( \frac{p_\theta(x, z)}{q_\phi(z \mid x)} \right) \right]$$

Expanding the joint distribution $p_\theta(x, z) = p_\theta(x \mid z) p(z)$:
$$\mathbb{E}_{q_\phi(z \mid x)} \left[ \log \left( \frac{p_\theta(x \mid z) p(z)}{q_\phi(z \mid x)} \right) \right] = \mathbb{E}_{q_\phi(z \mid x)} \left[ \log p_\theta(x \mid z) + \log \left(\frac{p(z)}{q_\phi(z \mid x)}\right) \right]$$

By linearity of expectation:
$$= \mathbb{E}_{q_\phi(z \mid x)} [\log p_\theta(x \mid z)] - \mathbb{E}_{q_\phi(z \mid x)} \left[ \log \left(\frac{q_\phi(z \mid x)}{p(z)}\right) \right]$$

Recognizing the second term as the Kullback-Leibler divergence $D_{\text{KL}}(q_\phi(z \mid x) \parallel p(z))$:
$$\boxed{\text{ELBO}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z \mid x)} [\log p_\theta(x \mid z)] - D_{\text{KL}}(q_\phi(z \mid x) \parallel p(z))}$$
$$\log p_\theta(x) \ge \text{ELBO}(\theta, \phi; x)$$

---

## 4. Derivation 2: ELBO via KL Divergence Decomposition

The second derivation reveals the exact relationship between the ELBO, the true evidence, and the approximation quality.

Consider the Kullback-Leibler divergence between the variational approximation $q_\phi(z \mid x)$ and the true posterior $p_\theta(z \mid x)$:
$$D_{\text{KL}}(q_\phi(z \mid x) \parallel p_\theta(z \mid x)) = \int q_\phi(z \mid x) \log \left( \frac{q_\phi(z \mid x)}{p_\theta(z \mid x)} \right) \, dz$$

By Bayes' theorem, $p_\theta(z \mid x) = \frac{p_\theta(x, z)}{p_\theta(x)}$. Substituting this identity:
$$D_{\text{KL}}(q_\phi(z \mid x) \parallel p_\theta(z \mid x)) = \int q_\phi(z \mid x) \log \left( \frac{q_\phi(z \mid x) \cdot p_\theta(x)}{p_\theta(x, z)} \right) \, dz$$

Applying properties of logarithms ($\log(ab/c) = \log a - \log c + \log b$):
$$= \int q_\phi(z \mid x) \left[ \log \left( \frac{q_\phi(z \mid x)}{p_\theta(x, z)} \right) + \log p_\theta(x) \right] \, dz$$

Splitting into two integrals:
$$= \int q_\phi(z \mid x) \log \left( \frac{q_\phi(z \mid x)}{p_\theta(x, z)} \right) \, dz + \log p_\theta(x) \underbrace{\int q_\phi(z \mid x) \, dz}_{= 1}$$

Note that the first integral is the negative of our previously defined ELBO:
$$-\int q_\phi(z \mid x) \log \left( \frac{p_\theta(x, z)}{q_\phi(z \mid x)} \right) \, dz = -\text{ELBO}(\theta, \phi; x)$$

Thus:
$$D_{\text{KL}}(q_\phi(z \mid x) \parallel p_\theta(z \mid x)) = -\text{ELBO}(\theta, \phi; x) + \log p_\theta(x)$$

Rearranging terms:
$$\boxed{\log p_\theta(x) = \text{ELBO}(\theta, \phi; x) + D_{\text{KL}}(q_\phi(z \mid x) \parallel p_\theta(z \mid x))}$$

### The Fundamental Implications:
1. **Gibbs' Inequality:** Because the KL divergence is strictly non-negative ($D_{\text{KL}} \ge 0$), the ELBO is a rigorous mathematical lower bound on the true log-evidence:
   $$\text{ELBO}(\theta, \phi; x) \le \log p_\theta(x)$$
2. **Tightness of the Bound:** The gap between the evidence $\log p_\theta(x)$ and the lower bound is **identically equal to $D_{\text{KL}}(q_\phi(z \mid x) \parallel p_\theta(z \mid x))$**.
3. **Dual Optimization:** Maximizing the ELBO with respect to variational parameters $\phi$:
   - Pushes the lower bound upward toward the true log-marginal $\log p_\theta(x)$.
   - Directly minimizes the KL divergence between the variational distribution and the true posterior, forcing $q_\phi(z \mid x) \to p_\theta(z \mid x)$.
   - When $q_\phi(z \mid x) = p_\theta(z \mid x)$, the divergence vanishes ($D_{\text{KL}} = 0$) and the bound becomes **exact**.

---

## 5. Analytical Closed-Form Derivation of Gaussian Prior KL Divergence

In standard VAE architectures, both the prior and the variational posterior are parameterized as multivariate Gaussians with diagonal covariance matrices:
- **Prior:** $p(z) = \mathcal{N}(0, \mathbf{I}) = \prod_{j=1}^J \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{z_j^2}{2}\right)$
- **Variational Posterior:** $q_\phi(z \mid x) = \mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2)) = \prod_{j=1}^J \frac{1}{\sqrt{2\pi \sigma_j^2}} \exp\left(-\frac{(z_j - \mu_j)^2}{2\sigma_j^2}\right)$

Because dimensions are mutually independent, the total KL divergence decomposes into a sum over the $J$ latent dimensions:
$$D_{\text{KL}}(q_\phi(z \mid x) \parallel p(z)) = \sum_{j=1}^J D_{\text{KL}}(q_j(z_j) \parallel p_j(z_j))$$

### Step-by-Step 1D Integration:
Consider a single dimension with $q(z) = \mathcal{N}(\mu, \sigma^2)$ and $p(z) = \mathcal{N}(0, 1)$:
$$D_{\text{KL}}(q \parallel p) = \int_{-\infty}^\infty q(z) \log \frac{q(z)}{p(z)} \, dz = \int_{-\infty}^\infty q(z) \log q(z) \, dz - \int_{-\infty}^\infty q(z) \log p(z) \, dz$$

#### Term 1: Differential Entropy of Gaussian $q(z)$
$$\log q(z) = -\frac{1}{2} \log(2\pi) - \frac{1}{2}\log(\sigma^2) - \frac{(z - \mu)^2}{2\sigma^2}$$
Taking expectation with respect to $q(z)$:
$$\mathbb{E}_q[\log q(z)] = -\frac{1}{2}\log(2\pi) - \frac{1}{2}\log(\sigma^2) - \frac{1}{2\sigma^2} \underbrace{\mathbb{E}_q[(z - \mu)^2]}_{= \sigma^2}$$
$$\mathbb{E}_q[\log q(z)] = -\frac{1}{2}\log(2\pi) - \frac{1}{2}\log(\sigma^2) - \frac{1}{2}$$

#### Term 2: Cross-Entropy with Prior $p(z) = \mathcal{N}(0, 1)$
$$\log p(z) = -\frac{1}{2}\log(2\pi) - \frac{z^2}{2}$$
Taking expectation with respect to $q(z)$:
$$\mathbb{E}_q[\log p(z)] = -\frac{1}{2}\log(2\pi) - \frac{1}{2}\mathbb{E}_q[z^2]$$
Recall that $\mathbb{E}[z^2] = \text{Var}(z) + (\mathbb{E}[z])^2 = \sigma^2 + \mu^2$:
$$\mathbb{E}_q[\log p(z)] = -\frac{1}{2}\log(2\pi) - \frac{1}{2}(\sigma^2 + \mu^2)$$

#### Combining Terms:
$$D_{\text{KL}}(q \parallel p) = \mathbb{E}_q[\log q(z)] - \mathbb{E}_q[\log p(z)]$$
$$= \left( -\frac{1}{2}\log(2\pi) - \frac{1}{2}\log(\sigma^2) - \frac{1}{2} \right) - \left( -\frac{1}{2}\log(2\pi) - \frac{1}{2}(\sigma^2 + \mu^2) \right)$$
The constant $-\frac{1}{2}\log(2\pi)$ cancels out completely:
$$= -\frac{1}{2}\log(\sigma^2) - \frac{1}{2} + \frac{1}{2}(\sigma^2 + \mu^2)$$
Factoring out $-\frac{1}{2}$:
$$\boxed{D_{\text{KL}}(\mathcal{N}(\mu, \sigma^2) \parallel \mathcal{N}(0, 1)) = -\frac{1}{2} \left( 1 + \log(\sigma^2) - \mu^2 - \sigma^2 \right)}$$

Summing over all $J$ independent latent dimensions:
$$\boxed{D_{\text{KL}}(q_\phi(z \mid x) \parallel p(z)) = -\frac{1}{2} \sum_{j=1}^J \left( 1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2 \right)}$$

In practice, the neural network encoder predicts the mean $\mu_j$ and the log-variance $\gamma_j = \log(\sigma_j^2)$ to guarantee numerical stability:
$$\mathcal{L}_{\text{KL}} = -\frac{1}{2} \sum_{j=1}^J \left( 1 + \gamma_j - \mu_j^2 - \exp(\gamma_j) \right)$$

---

## 6. The Dual Mechanics of the ELBO Objective

The complete VAE loss minimized during gradient descent is the negative ELBO:
$$\mathcal{L}_{\text{VAE}}(\theta, \phi) = -\text{ELBO} = \underbrace{-\mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)]}_{\text{Reconstruction Loss (Fidelity)}} + \underbrace{D_{\text{KL}}(q_\phi(z \mid x) \parallel p(z))}_{\text{Regularization Loss (Continuity)}}$$

```
+---------------------------------------------------------------------------------------------------+
|                                  THE ELBO TENSION ARBITRATION                                     |
|                                                                                                   |
|       Reconstruction Term (-E_q[log p(x|z)])            KL Regularization (D_KL(q || p))          |
|                                                                                                   |
|  * Pulls mu away from origin to encode unique   * Pulls all mu -> 0 and all sigma^2 -> 1.         |
|    features for each distinct training class.   * Prevents overfitting; ensures latent space is   |
|  * Shrinks sigma -> 0 to minimize sampling        a smooth, continuous, hole-free Gaussian cloud. |
|    blur and maximize crisp reconstruction.                                                        |
|                                                                                                   |
|              RESULTING EQUILIBRIUM: A smooth, continuous latent manifold where                    |
|              nearby coordinates decode into semantically interpolated data points.                |
+---------------------------------------------------------------------------------------------------+
```
