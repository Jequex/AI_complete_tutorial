# Topic 4.3: Core Probability Distributions in AI (Bernoulli, Categorical, Gaussian & Multivariate Gaussian)

Welcome to **Topic 4.3: Core Probability Distributions in AI**. 

In machine learning and generative artificial intelligence, models do not simply output static numbers; they output parameters for **probability distributions**. 

When a vision model predicts whether an image is a cat, when an LLM selects the next word, when a diffusion model adds noise to pixels, and when a VAE compresses an image into latent space, they are all working with the core distributions explored in this lecture.

---

## 1. 🧒 The 12-Year-Old Analogy: Maya & Barnaby at the Golden Retriever Kennel

Imagine 12-year-old Maya and Timmy visiting a puppy rescue center filled with 100 golden retriever puppies, including Maya's energetic puppy Barnaby.

```text
               THE GOLDEN RETRIEVER PUPPY BELL CURVE
                               
                                ▲  
                           ╭────┴────╮  ◀ Average Height: 20 inches
                         ╭─╯         ╰─╮
                       ╭─╯             ╰─╮
                     ╭─╯                 ╰─╮
               ──────┴─────────┬─────────┴──────▶ Puppy Height (inches)
                              20
                     68% of puppies between 18 and 22 inches (±1σ)
                     95% of puppies between 16 and 24 inches (±2σ)
```

1. **The Single Puppy Question (Bernoulli Distribution):**
   - Maya looks at a puppy: *"Is this puppy wearing a red bandana or not?"*
   - There are only two answers: Yes ($1$) or No ($0$).
   - In AI, this is **Binary Classification** parameterized by the Sigmoid function!
2. **The Feeding Bowl Choice (Categorical Distribution):**
   - The kennel has $5$ colored food bowls: Red, Blue, Green, Yellow, and Purple.
   - When let loose, a puppy chooses exactly one bowl.
   - In AI, an LLM picking the next word from $50{,}000$ possible words is a **Categorical Distribution** parameterized by Softmax!
3. **Puppy Heights (The Univariate Gaussian / Normal Distribution):**
   - Maya measures the height of all $100$ puppies.
   - Most puppies are close to average: $20$ inches. A few are shorter ($18$ inches) and a few taller ($22$ inches). Almost none are tiny ($12$ inches) or giant ($28$ inches).
   - The heights form a smooth, symmetric **Bell Curve** $\mathcal{N}(\mu, \sigma^2)$!
4. **Height AND Weight Together (The Multivariate Gaussian $\mathcal{N}(\mu, \Sigma)$):**
   - Now Maya measures both height AND weight for each puppy.
   - Taller puppies naturally tend to weigh more! The data forms a 2D tilted oval cloud of points.
   - In AI, **Multivariate Gaussians** represent multi-dimensional latent vectors in Variational Autoencoders (VAEs) and Stable Diffusion models!

---

## 2. 📋 Mathematical Symbol Decoder Table

| Symbol | Pronunciation | Plain English Meaning | Concrete Real-World Analogy / Flashcard Example |
| :---: | :---: | :--- | :--- |
| $\text{Bernoulli}(p)$ | "Bernoulli of p" | Single coin flip with success probability $p$ | Asking whether Barnaby is wearing a red bandana (Yes/No) |
| $\text{Cat}(p_1, \dots, p_K)$ | "Categorical of p" | Choice among $K$ distinct categories ($\sum p_i = 1$) | A puppy choosing one of 5 colored food bowls |
| $\mathcal{N}(\mu, \sigma^2)$ | "Normal of mu, sigma squared" | 1D symmetric bell curve centered at $\mu$ with spread $\sigma^2$ | The height distribution of 1,000 golden retriever puppies |
| $\mu$ | "mu" | Mean / Center point of the distribution | The average puppy height ($20$ inches) |
| $\sigma$ | "sigma" | Standard deviation (spread / width of bell curve) | How much individual puppies typically deviate from average ($2$ inches) |
| $\sigma^2$ | "sigma squared" | Variance (the square of standard deviation) | The variance of heights ($4\text{ inches}^2$) |
| $\mathcal{N}(\mu, \Sigma)$ | "Multivariate normal of mu, Sigma" | Multi-dimensional bell curve in $\mathbb{R}^D$ | Measuring puppy height, weight, and ear length simultaneously |
| $\Sigma$ | "capital Sigma" | $D \times D$ Covariance matrix | Grid showing variances along axes and correlations between features |
| $|\Sigma|$ | "determinant of Sigma" | Hyper-volume scaling factor of the multi-D cloud | How much 3D volume the puppy data cloud occupies |
| $d_M^2$ | "Mahalanobis distance squared" | Normalized distance accounting for correlation | How unusual a puppy's build is, adjusted for height-weight correlation |

---

## 3. Discrete Distributions in AI

### 3.1 The Bernoulli Distribution (Binary Classification)
For a single trial $X \in \{0, 1\}$ with probability of success $p$:
$$P(X = k) = p^k (1 - p)^{1 - k} \quad \text{for } k \in \{0, 1\}$$

- **Mean:** $\mathbb{E}[X] = p$
- **Variance:** $\text{Var}(X) = p(1 - p)$

#### Connection to Deep Learning: Binary Cross-Entropy (BCE)
A neural network outputs logit $z$. The Sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$ outputs $p$.
The negative log-likelihood of a sample $(x, y)$ gives **Binary Cross-Entropy Loss**:
$$\mathcal{L}_{\text{BCE}} = -\left[ y \ln(p) + (1 - y) \ln(1 - p) \right]$$

---

### 3.2 The Categorical Distribution (Next-Token Prediction in LLMs)
For $K$ categories where $P(X = k) = p_k$ with $\sum_{k=1}^K p_k = 1.0$:

#### Connection to Large Language Models:
Transformers generate raw logits $\mathbf{z} \in \mathbb{R}^V$ across a vocabulary of $V$ tokens.
Softmax parameterizes the Categorical distribution:
$$p_k = \frac{e^{z_k}}{\sum_{j=1}^V e^{z_j}}$$
The training loss is the negative log-likelihood of the true token $y$:
$$\mathcal{L}_{\text{CE}} = -\ln(p_y)$$

---

## 4. The Univariate Gaussian (Normal) Distribution $\mathcal{N}(\mu, \sigma^2)$

### Probability Density Function (PDF):
$$f(x) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)$$

#### Line-by-Line Formula Breakdown:
$$f(x) = \underbrace{\frac{1}{\sigma \sqrt{2\pi}}}_{\text{Normalizing height so total area } = 1.0} \cdot \exp\left( -\underbrace{\frac{(x - \mu)^2}{2\sigma^2}}_{\text{Squared distance from mean in units of variance}} \right)$$

- 🧒 **What the math is actually doing:**
  If $x = \mu$, distance is zero and $e^0 = 1$ (the highest point of the bell curve).
  As $x$ moves away from $\mu$, $(x - \mu)^2$ grows, pushing $e^{-\text{huge}}$ down toward zero symmetrically on both sides!

### The 68–95–99.7 Empirical Rule:
- $68.27\%$ of all samples fall within $\pm 1\sigma$ of the mean.
- $95.45\%$ of all samples fall within $\pm 2\sigma$ of the mean.
- $99.73\%$ of all samples fall within $\pm 3\sigma$ of the mean.

---

## 5. The Multivariate Gaussian Distribution $\mathcal{N}(\mu, \Sigma)$

In $D$-dimensional space, points form a continuous cloud centered at vector $\mu \in \mathbb{R}^D$ with shape governed by the $D \times D$ covariance matrix $\Sigma$:

$$f(\mathbf{x}) = \frac{1}{(2\pi)^{D/2} |\Sigma|^{1/2}} \exp\left( -\frac{1}{2} (\mathbf{x} - \mu)^T \Sigma^{-1} (\mathbf{x} - \mu) \right)$$

#### Line-by-Line Breakdown:
$$f(\mathbf{x}) = \underbrace{\frac{1}{(2\pi)^{D/2} |\Sigma|^{1/2}}}_{\text{Hyper-volume normalizer}} \cdot \exp\left( -\frac{1}{2} \underbrace{(\mathbf{x} - \mu)^T \Sigma^{-1} (\mathbf{x} - \mu)}_{\text{Mahalanobis squared distance } d_M^2} \right)$$

- 🧒 **The Mahalanobis Distance ($d_M^2$):**
  Instead of measuring raw Euclidean distance, Mahalanobis measures distance **divided by the variance along each correlated direction**! An outlier puppy that is 30 inches tall and weighs only 10 pounds has a massive Mahalanobis distance because that combination violates the height-weight correlation ellipse.

---

## 6. The VAE Reparameterization Trick: Gradients Through Randomness

In Variational Autoencoders (VAEs), an encoder network predicts mean $\mu$ and variance $\sigma^2$ for a latent code $\mathbf{z}$.
If we sample directly:
$$\mathbf{z} \sim \mathcal{N}(\mu, \sigma^2)$$
backpropagation crashes because **you cannot compute derivatives through random sampling nodes**!

### The Solution: Kingma & Welling's Reparameterization Trick
Pull the randomness out of the computational graph by sampling independent standard noise $\epsilon \sim \mathcal{N}(0, I)$:

$$\mathbf{\mathbf{z} = \mu + \sigma \odot \epsilon}$$

```text
       DIRECT SAMPLING (Broken Backprop)         REPARAMETERIZATION (Gradients Flow!)
       
         [Encoder: μ, σ]                           [Encoder: μ, σ]      [Noise ε ~ N(0, I)]
                │                                         │                     │
          (Random Node) ❌                                └─────────┬───────────┘
                │                                                   ▼
         [Latent Code z]                                     [ z = μ + σ ⊙ ε ] ✅
       (Cannot backprop!)                             (Gradients flow back into μ and σ!)
```

Now $\mathbf{z}$ is a completely deterministic, differentiable function of $\mu$ and $\sigma$, allowing backpropagation to train the entire autoencoder end-to-end!

---

## 7. Concrete Numerical Walkthrough

Let us evaluate a 2D Gaussian density at point $\mathbf{x} = [3.0, 4.0]^T$ with:
$$\mu = \begin{bmatrix} 2.0 \\ 2.0 \end{bmatrix}, \quad \Sigma = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 4.0 \end{bmatrix}$$

1. **Difference Vector:**
   $$\mathbf{x} - \mu = \begin{bmatrix} 3.0 - 2.0 \\ 4.0 - 2.0 \end{bmatrix} = \begin{bmatrix} 1.0 \\ 2.0 \end{bmatrix}$$
2. **Invert Covariance Matrix:**
   $$\Sigma^{-1} = \begin{bmatrix} 1/1.0 & 0 \\ 0 & 1/4.0 \end{bmatrix} = \begin{bmatrix} 1.0 & 0 \\ 0 & 0.25 \end{bmatrix}$$
3. **Mahalanobis Distance Squared ($d_M^2$):**
   $$d_M^2 = \begin{bmatrix} 1.0 & 2.0 \end{bmatrix} \begin{bmatrix} 1.0 & 0 \\ 0 & 0.25 \end{bmatrix} \begin{bmatrix} 1.0 \\ 2.0 \end{bmatrix} = \begin{bmatrix} 1.0 & 2.0 \end{bmatrix} \begin{bmatrix} 1.0 \\ 0.5 \end{bmatrix} = (1.0)(1.0) + (2.0)(0.5) = 1.0 + 1.0 = \mathbf{2.0}$$
4. **Compute Determinant and Normalizer:**
   $$|\Sigma| = (1.0)(4.0) - (0)(0) = 4.0 \implies |\Sigma|^{1/2} = 2.0$$
   $$\text{Normalizer} = \frac{1}{(2\pi)^{2/2} |\Sigma|^{1/2}} = \frac{1}{2\pi \times 2.0} = \frac{1}{4\pi} \approx \mathbf{0.07958}$$
5. **Evaluate PDF:**
   $$f(\mathbf{x}) = 0.07958 \times \exp\left( -\frac{1}{2}(2.0) \right) = 0.07958 \times e^{-1} \approx 0.07958 \times 0.36788 = \mathbf{0.02927}$$
