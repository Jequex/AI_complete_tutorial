# Topic 24.2: Exercises & Solutions — Variational Autoencoders & ELBO

---

### Problem 1: Manual Calculation of Gaussian Prior KL Divergence (Easy)

An encoder network processes an input image and predicts the following 3-dimensional latent distribution parameters:
$$\boldsymbol{\mu} = [0.8, -1.2, 0.0]^\top$$
$$\log \boldsymbol{\sigma}^2 = \boldsymbol{\gamma} = [-0.5, 0.4, 0.0]^\top$$

The latent prior is the standard multivariate normal: $p(z) = \mathcal{N}(0, \mathbf{I}_3)$.
1. Compute the variance $\sigma_j^2$ for each of the three dimensions.
2. Compute the KL divergence contribution for each dimension individually:
   $$\text{KL}_j = -\frac{1}{2} \left( 1 + \gamma_j - \mu_j^2 - \exp(\gamma_j) \right)$$
3. Compute the total analytical KL divergence loss $\mathcal{L}_{\text{KL}} = \sum_{j=1}^3 \text{KL}_j$.

<details>
<summary>View Complete Solution</summary>

**Step 1: Compute Variances:**
- Dimension 1: $\sigma_1^2 = \exp(-0.5) \approx 0.60653$
- Dimension 2: $\sigma_2^2 = \exp(0.4) \approx 1.49182$
- Dimension 3: $\sigma_3^2 = \exp(0.0) = 1.0$

**Step 2: Component-wise KL Calculation:**
Recall the formula: $\text{KL}_j = -\frac{1}{2}(1 + \gamma_j - \mu_j^2 - \sigma_j^2)$.

- **Dimension 1:**
  $$\mu_1^2 = (0.8)^2 = 0.64, \quad \gamma_1 = -0.5, \quad \sigma_1^2 \approx 0.60653$$
  $$1 + \gamma_1 - \mu_1^2 - \sigma_1^2 = 1 - 0.5 - 0.64 - 0.60653 = -0.74653$$
  $$\text{KL}_1 = -\frac{1}{2}(-0.74653) \approx 0.37327$$

- **Dimension 2:**
  $$\mu_2^2 = (-1.2)^2 = 1.44, \quad \gamma_2 = 0.4, \quad \sigma_2^2 \approx 1.49182$$
  $$1 + \gamma_2 - \mu_2^2 - \sigma_2^2 = 1 + 0.4 - 1.44 - 1.49182 = -1.53182$$
  $$\text{KL}_2 = -\frac{1}{2}(-1.53182) \approx 0.76591$$

- **Dimension 3:**
  $$\mu_3^2 = (0.0)^2 = 0.0, \quad \gamma_3 = 0.0, \quad \sigma_3^2 = 1.0$$
  $$1 + \gamma_3 - \mu_3^2 - \sigma_3^2 = 1 + 0.0 - 0.0 - 1.0 = 0.0$$
  $$\text{KL}_3 = -\frac{1}{2}(0.0) = 0.0$$
  *(Dimension 3 matches the prior perfectly, incurring zero penalty!)*

**Step 3: Total KL Divergence:**
$$\mathcal{L}_{\text{KL}} = \text{KL}_1 + \text{KL}_2 + \text{KL}_3 \approx 0.37327 + 0.76591 + 0.0 \approx 1.13918$$
</details>

---

### Problem 2: Proving the Exactness of ELBO at True Posterior (Medium)

Prove that the Evidence Lower Bound satisfies:
$$\text{ELBO}(\theta, \phi; x) = \log p_\theta(x) \iff q_\phi(z \mid x) = p_\theta(z \mid x) \text{ almost everywhere}$$
Start from the Kullback-Leibler divergence identity:
$$D_{\text{KL}}(q \parallel p) = \int q(z) \log \left(\frac{q(z)}{p(z)}\right) dz$$
and use the strict convexity of $-\log(u)$.

<details>
<summary>View Complete Solution</summary>

**Step 1: Express KL Divergence as an Expectation:**
$$D_{\text{KL}}(q_\phi(z \mid x) \parallel p_\theta(z \mid x)) = -\int q_\phi(z \mid x) \log \left( \frac{p_\theta(z \mid x)}{q_\phi(z \mid x)} \right) dz = -\mathbb{E}_{q_\phi}\left[ \log \left( \frac{p_\theta(z \mid x)}{q_\phi(z \mid x)} \right) \right]$$

**Step 2: Apply Jensen's Inequality to Strictly Convex Function:**
The function $g(u) = -\log u$ is strictly convex on $(0, \infty)$ (its second derivative $g''(u) = 1/u^2 > 0$).
By Jensen's Inequality:
$$\mathbb{E}[g(U)] \ge g(\mathbb{E}[U])$$
with equality if and only if the random variable $U$ is degenerate (constant almost everywhere).

Let $U = \frac{p_\theta(z \mid x)}{q_\phi(z \mid x)}$. Then:
$$D_{\text{KL}}(q_\phi \parallel p_\theta) = \mathbb{E}_{q_\phi}[g(U)] \ge g(\mathbb{E}_{q_\phi}[U]) = -\log \left( \mathbb{E}_{q_\phi}\left[ \frac{p_\theta(z \mid x)}{q_\phi(z \mid x)} \right] \right)$$

Evaluating the expectation inside the logarithm:
$$\mathbb{E}_{q_\phi}\left[ \frac{p_\theta(z \mid x)}{q_\phi(z \mid x)} \right] = \int q_\phi(z \mid x) \frac{p_\theta(z \mid x)}{q_\phi(z \mid x)} \, dz = \int p_\theta(z \mid x) \, dz = 1$$

Therefore:
$$D_{\text{KL}}(q_\phi(z \mid x) \parallel p_\theta(z \mid x)) \ge -\log(1) = 0$$

**Step 3: Condition for Equality:**
Equality $D_{\text{KL}} = 0$ holds if and only if $U = \frac{p_\theta(z \mid x)}{q_\phi(z \mid x)}$ is constant almost everywhere on the support of $q_\phi$.
Since $\int q_\phi(z \mid x) dz = \int p_\theta(z \mid x) dz = 1$, the constant must be identically $1$.
Thus, $q_\phi(z \mid x) = p_\theta(z \mid x)$ almost everywhere.

Since $\log p_\theta(x) - \text{ELBO} = D_{\text{KL}}(q_\phi \parallel p_\theta)$, we conclude:
$$\text{ELBO} = \log p_\theta(x) \iff D_{\text{KL}} = 0 \iff q_\phi(z \mid x) = p_\theta(z \mid x) \text{ a.e.}$$
</details>

---

### Problem 3: The $\beta$-VAE Formulation & Posterior Collapse (Hard)

Higgins et al. (2017) introduced the $\beta$-VAE objective to encourage disentangled latent factor discovery:
$$\mathcal{L}_{\beta}(\theta, \phi; x) = -\mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)] + \beta \cdot D_{\text{KL}}(q_\phi(z \mid x) \parallel p(z)), \quad \beta > 0$$

1. Interpret the physical meaning of setting $\beta > 1$ versus $\beta < 1$ from an Information Bottleneck perspective.
2. Define **Posterior Collapse** (latent over-pruning). Show mathematically how posterior collapse occurs when the decoder network is an overly powerful autoregressive model (such as a PixelCNN or multi-layer Causal Transformer).
3. Why does posterior collapse yield an optimal KL loss of $0.0$, and why is this devastating for representation learning?

<details>
<summary>View Complete Solution</summary>

**Step 1: Information Bottleneck Interpretation of $\beta$:**
The objective can be framed as minimizing the mutual information $I(X; Z)$ between input $X$ and latent code $Z$ while maximizing reconstruction accuracy:
- $\beta > 1$: Imposes a heavy compression penalty. Forces latent dimensions to become mutually independent and encode only the most dominant global semantic factors (e.g. azimuth, scale, color). However, overly high $\beta$ causes blurry, low-fidelity reconstructions.
- $\beta < 1$: Relaxes latent regularization, allowing the encoder to store fine-grained pixel details, but risks creating latent gaps similar to deterministic autoencoders.

**Step 2: Mathematical Mechanism of Posterior Collapse:**
Suppose the decoder $p_\theta(x \mid z)$ is an autoregressive model capable of modeling the full conditional distribution $p(x_t \mid x_{<t})$ without needing $z$:
$$p_\theta(x \mid z) = \prod_{t=1}^T p_\theta(x_t \mid x_{<t})$$
Notice that $z$ does not even appear on the right side! The decoder simply ignores $z$.

Under this condition:
$$\mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)] = \log \prod_{t=1}^T p_\theta(x_t \mid x_{<t}) = \text{Constant with respect to } \phi$$
The encoder parameters $\phi$ have zero gradient flow from the reconstruction term:
$$\nabla_\phi \mathbb{E}_{q_\phi}[\log p_\theta(x \mid z)] = 0$$

The only term driving $\phi$ is the KL regularizer:
$$\min_\phi D_{\text{KL}}(q_\phi(z \mid x) \parallel p(z))$$
The analytical minimum of this non-negative term is achieved when:
$$q_\phi(z \mid x) = p(z) = \mathcal{N}(0, \mathbf{I}) \quad \forall x$$
Meaning $\boldsymbol{\mu}_\phi(x) = \mathbf{0}$ and $\boldsymbol{\sigma}_\phi^2(x) = \mathbf{1}$ for every single input $x$!

**Step 3: Why Posterior Collapse is Devastating:**
- When $q_\phi(z \mid x) = p(z)$, the latent vector $z$ contains **zero mutual information** with input $x$:
  $$I(X; Z) = 0$$
- The latent codes for an airplane, a dog, and a digit are identical standard Gaussians.
- The encoder fails entirely to learn any meaningful representation, and the VAE degrades into a standard language model or autoregressive density estimator without latent controls.
</details>
