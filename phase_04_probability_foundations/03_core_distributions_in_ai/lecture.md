# Topic 4.3: Core Probability Distributions in AI (Bernoulli, Categorical, Gaussian & Multivariate Gaussian)

Welcome to **Topic 4.3: Core Probability Distributions in AI**. In machine learning, almost every model prediction, latent embedding, and noise schedule is governed by a canonical probability distribution. Understanding the mathematical properties, shapes, and formulas of these distributions allows you to choose appropriate loss functions, design generative architectures, and understand model outputs.

---

## 1. Symbol Decoder Table

| Symbol | Mathematical Name | Plain-English Meaning | AI Architecture Role |
| :---: | :--- | :--- | :--- |
| $\text{Bernoulli}(p)$ | Bernoulli Distribution | Single coin flip with success probability $p$ | Binary classification, Sigmoid neuron, Binary Cross-Entropy |
| $\text{Cat}(p_1, \dots, p_K)$ | Categorical Distribution | Choice among $K$ distinct categories | Next-token prediction in LLMs, Softmax output layer |
| $\mathcal{N}(\mu, \sigma^2)$ | Univariate Gaussian (Normal) | Symmetric bell-shaped continuous distribution | Weight initialization (He / Xavier), noise addition |
| $\mathcal{N}(\mu, \Sigma)$ | Multivariate Gaussian | Multi-dimensional bell curve in $\mathbb{R}^D$ | Latent space in VAEs, DDPM Diffusion models |
| $\Sigma$ | Covariance Matrix | $D \times D$ matrix capturing variances & correlations | Controls the orientation & stretch of multi-D contour ellipses |
| $|\Sigma|$ | Determinant of Covariance | Hyper-volume scaling factor of the Gaussian | Normalizing denominator constant in multivariate PDF |
| $(\mathbf{x} - \mu)^T \Sigma^{-1} (\mathbf{x} - \mu)$ | Mahalanobis Distance Squared | Standardized multi-dimensional distance from mean | Replaces standard Euclidean distance $(x - \mu)^2 / \sigma^2$ |

---

## 2. Discrete Distributions in AI

### 2.1 The Bernoulli Distribution (Binary Decisions)
A Bernoulli random variable $X \in \{0, 1\}$ represents a single trial with success probability $p \in [0, 1]$:
$$P(X = 1) = p, \quad P(X = 0) = 1 - p$$

We write the PMF compactly as:
$$P(X = k) = p^k (1 - p)^{1 - k} \quad \text{for } k \in \{0, 1\}$$

- **Mean:** $\mathbb{E}[X] = p$
- **Variance:** $\text{Var}(X) = p(1 - p)$

#### Connection to Deep Learning:
In binary classification (e.g., Cat vs Dog), a neural network outputs a single scalar logit $z$. The **Sigmoid function** converts $z$ into Bernoulli parameter $p$:
$$p = \sigma(z) = \frac{1}{1 + e^{-z}}$$
The negative log-likelihood of a Bernoulli sample gives the celebrated **Binary Cross-Entropy (BCE) Loss**:
$$\mathcal{L}_{\text{BCE}} = - \left[ y \log(p) + (1 - y) \log(1 - p) \right]$$

---

### 2.2 The Categorical Distribution (Next-Token Prediction in LLMs)
A Categorical random variable $X \in \{1, 2, \dots, K\}$ generalizes the Bernoulli distribution to $K$ mutually exclusive categories:
$$P(X = k) = p_k \quad \text{where } p_k \ge 0 \text{ and } \sum_{k=1}^K p_k = 1.0$$

#### Connection to Large Language Models:
When a Transformer language model generates text, it outputs a vector of unnormalized logits $\mathbf{z} \in \mathbb{R}^V$ over a vocabulary of $V$ tokens (e.g. $V = 50{,}257$ in GPT-2).
The **Softmax function** parameterizes the Categorical distribution:
$$p_k = \text{Softmax}(\mathbf{z})_k = \frac{e^{z_k}}{\sum_{j=1}^V e^{z_j}}$$
The training loss is the negative log-likelihood of the true target token $y$:
$$\mathcal{L}_{\text{CE}} = - \log(p_y)$$

---

## 3. The Univariate Gaussian (Normal) Distribution $\mathcal{N}(\mu, \sigma^2)$

The Gaussian distribution is the most ubiquitous distribution in mathematics and nature due to the Central Limit Theorem.

### The Probability Density Function (PDF)
For mean $\mu \in \mathbb{R}$ and variance $\sigma^2 > 0$ (standard deviation $\sigma > 0$):

$$f(x) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)$$

```
                     f(x)
                      ▲
                    ╭─┴─╮      Peak height = 1 / (σ * √(2π))
                   ╱  │  ╲
                 ╭╯   │   ╰╮
               ╭─╯    │    ╰─╮
        ──────┴───────┼───────┴──────▶ x
                     μ (Mean)
              [←──── 2σ ────→] (68.27% of all area)
```

### The 12-Year-Old Breakdown of the Gaussian Formula:
1. **The Bell Shape $\exp(-(x - \mu)^2)$:** The squared difference $(x - \mu)^2$ measures how far $x$ is from the center $\mu$. The negative exponential turns large distances into tiny numbers that decay smoothly to zero on both sides.
2. **The Spread Divider $2\sigma^2$:** Dividing by $\sigma^2$ stretches or squashes the bell. A larger $\sigma$ makes the bell wider and flatter.
3. **The Normalizing Constant $\frac{1}{\sigma \sqrt{2\pi}}$:** Ensures the total area under the curve integrates to exactly $1.0$ regardless of $\sigma$.

### The 68–95–99.7 Empirical Rule:
- $68.27\%$ of all values lie within $1$ standard deviation: $[\mu - \sigma, \mu + \sigma]$
- $95.45\%$ of all values lie within $2$ standard deviations: $[\mu - 2\sigma, \mu + 2\sigma]$
- $99.73\%$ of all values lie within $3$ standard deviations: $[\mu - 3\sigma, \mu + 3\sigma]$

---

## 4. The Multivariate Gaussian Distribution $\mathcal{N}(\mu, \Sigma)$

In deep learning, data points and latent representations are vectors in multi-dimensional space $\mathbf{x} \in \mathbb{R}^D$. A **Multivariate Gaussian** describes uncertainty across multiple interconnected dimensions simultaneously.

### The Mathematical Formula
Given mean vector $\mu \in \mathbb{R}^D$ and a $D \times D$ symmetric, positive-definite **Covariance Matrix $\Sigma$**:

$$f(\mathbf{x}) = \frac{1}{(2\pi)^{D/2} |\Sigma|^{1/2}} \exp\left( -\frac{1}{2} (\mathbf{x} - \mu)^T \Sigma^{-1} (\mathbf{x} - \mu) \right)$$

### Unpacking the Multivariate Terms:
1. **The Covariance Matrix $\Sigma$:**
   $$\Sigma = \begin{bmatrix} \text{Var}(x_1) & \text{Cov}(x_1, x_2) \\ \text{Cov}(x_2, x_1) & \text{Var}(x_2) \end{bmatrix}$$
   - Diagonal elements $\Sigma_{ii}$ define the variance (spread) along axis $i$.
   - Off-diagonal elements $\Sigma_{ij}$ define the correlation between variable $i$ and variable $j$.
2. **The Mahalanobis Distance $(\mathbf{x} - \mu)^T \Sigma^{-1} (\mathbf{x} - \mu)$:**
   This replaces $(x - \mu)^2 / \sigma^2$. It measures distance in units of standard deviations, taking into account correlations and rotations in space.
3. **The Determinant $|\Sigma|^{1/2}$:**
   The determinant $|\Sigma|$ represents the multi-dimensional volume of the covariance ellipsoid.

### Three Types of Covariance Matrices:
```
1. Spherical (Isotropic)         2. Diagonal                      3. Full Covariance
   Σ = σ² I                         Σ = diag(σ₁², σ₂²)               Σ has off-diagonal entries
   (Circular contours)              (Axis-aligned ellipse)           (Tilted / rotated ellipse)

         ▲ y                              ▲ y                              ▲ y
      ╭─────╮                          ╭───────╮                           .---.
     │   •   │                        │    •    │                         /  •  /
      ╰─────╯                          ╰───────╯                         '---'
     ─────────▶ x                     ───────────▶ x                    ────────▶ x
```

---

## 5. Why Multivariate Gaussians Dominate Generative AI

### 5.1 Variational Autoencoders (VAEs) and the Reparameterization Trick
In a VAE, an encoder network takes an image $x$ and outputs the parameters of a multivariate Gaussian in latent space: a mean vector $\mu(x)$ and log-variance $\log \sigma^2(x)$.
To sample a latent code $z \sim \mathcal{N}(\mu, \Sigma)$ while allowing backpropagation, VAEs use the **Reparameterization Trick**:
$$\mathbf{z} = \mu + \sigma \odot \epsilon \quad \text{where } \epsilon \sim \mathcal{N}(0, I)$$
This moves the stochastic randomness to an external parameter-free variable $\epsilon$, allowing gradients to flow back into $\mu$ and $\sigma$!

### 5.2 Generative Diffusion Models (DDPMs)
In diffusion models (Stable Diffusion, Midjourney, Sora), the forward process gradually destroys an image over $T = 1000$ timesteps by repeatedly adding small amounts of isotropic Gaussian noise:
$$q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}\left( \mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t I \right)$$
At timestep $T$, the image is indistinguishable from pure Gaussian static $\mathbf{x}_T \sim \mathcal{N}(0, I)$. The neural network is trained to reverse this process, predicting the Gaussian noise to sculpt clean photorealistic images out of randomness!

---

## 6. Poisson & Exponential Distributions

1. **Poisson Distribution:** Models the count of rare events occurring in a fixed time interval (e.g. server request traffic):
   $$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!} \quad (k \in \{0, 1, 2, \dots\})$$
   - $\mathbb{E}[X] = \text{Var}(X) = \lambda$.
2. **Exponential Distribution:** Models the continuous waiting time between Poisson arrivals:
   $$f(x) = \lambda e^{-\lambda x} \quad (x \ge 0)$$
   - $\mathbb{E}[X] = \frac{1}{\lambda}$, $\text{Var}(X) = \frac{1}{\lambda^2}$.
   - **Memoryless Property:** $P(X > s + t \mid X > s) = P(X > t)$.

