# Topic 24.1: Taxonomy of Generative Models & Autoregressive Foundations

---

## 1. Executive Summary & The 12-Year-Old Analogy

### The Fundamental Objective of Generative Modeling
In discriminative machine learning, our objective is to compute a conditional probability distribution $P(y \mid x)$—predicting a categorical label or continuous target $y$ given observed input features $x$. Generative modeling addresses an infinitely more ambitious problem: **learning the true underlying data distribution $P(x)$** (or joint distribution $P(x, y)$) over high-dimensional input spaces (such as images $\mathbb{R}^{H \times W \times C}$, audio sequences $\mathbb{R}^T$, or textual token sequences $\mathcal{V}^T$).

A fully trained generative model allows us to:
1. **Sample:** Synthesize novel, previously unobserved instances $\tilde{x} \sim P_\theta(x)$ that look indistinguishable from real data.
2. **Evaluate Density (when tractable):** Calculate the exact or approximate likelihood $P_\theta(x)$ of an observation, enabling anomaly detection, hypothesis testing, and compression.
3. **Infer Latent Representations:** Uncover the hidden factors of variation $z$ that generated the observed world.

```
+---------------------------------------------------------------------------------------------------+
|                               TAXONOMY OF GENERATIVE MODELS                                       |
|                                                                                                   |
|                                    Generative Models                                              |
|                                            |                                                      |
|                   +------------------------+------------------------+                             |
|                   |                                                 |                             |
|           Explicit Likelihood                              Implicit Likelihood                    |
|         (Optimizes log P(x))                          (No explicit density function)              |
|                   |                                                 |                             |
|         +---------+---------+                             +---------+---------+                   |
|         |                   |                             |                   |                   |
|   Tractable Density   Approximate Density               GANs              GFlowNets               |
|         |                   |                   (Adversarial game)   (Flow trajectories)          |
|   +-----+-----+       +-----+-----+                                                               |
|   |           |       |           |                                                               |
| Autoregressive Normalizing Variational  Diffusion / Score                                         |
|   Models       Flows   Autoencoders      Models                                                   |
| (PixelCNN,    (RealNVP)   (VAEs)     (DDPM, Score SDE)                                            |
|  GPT, WaveNet)                                                                                    |
+---------------------------------------------------------------------------------------------------+
```

---

### The 12-Year-Old Analogy: The Blueprint Architect vs. The Swiss Cheese Map

Imagine you are tasked with building a magnificent Lego castle:

1. **The Blueprint Architect (Autoregressive Models):**
   - The Architect builds the castle **one single Lego brick at a time**, in strict chronological order.
   - Before placing Brick #100, the architect glances down at the exact positions of Bricks #1 through #99.
   - Because you can check the exact probability of every single brick given the ones placed before it, you have an exact, error-free recipe.
   - *The Catch:* Placing a million bricks takes a million sequential steps. It cannot be done in parallel!

2. **The Magic Illusionist (Implicit Models / GANs):**
   - The Illusionist doesn't have a blueprint. Instead, an Apprentice sculpts a complete castle in a single flash, and an ultra-strict Art Critic inspects it and yells: *"Fake! Castles don't have windows on the floor!"*
   - Through millions of critiques, the Apprentice learns to fool the Critic without ever calculating an exact blueprint formula.

3. **The Swiss Cheese Island (Why Regular Autoencoders Fail to Generate):**
   - Suppose an explorer visits 5 cities on an ocean planet and draws a map. A regular (deterministic) autoencoder compresses each city into a tiny pin on the map.
   - City A is at $(1, 2)$ and City B is at $(9, 8)$.
   - What happens if you drop a parachute at $(5, 5)$, halfway between them in the open ocean?
   - In a regular autoencoder, the decoder has *never* seen $(5, 5)$. The open ocean is an **empty void of Swiss cheese**. When the decoder tries to build a city from $(5, 5)$, it spits out an abomination: half an Eiffel tower glued to a submarine inside a toaster!
   - A regular autoencoder is a compressor, **not a generator**. To make it generate, we need **Variational Autoencoders (VAEs)** that turn the map into a smooth, continuous fog where every single coordinate makes sense.

---

## 2. Mathematical Symbol Decoder

| Symbol | Mathematical Domain | Description & Meaning |
| :--- | :--- | :--- |
| $\mathcal{X}$ | $\mathbb{R}^D$ or $\mathcal{V}^T$ | High-dimensional data space (e.g. images, audio waveforms, token sequences). |
| $x \in \mathcal{X}$ | Vector / Sequence | An observed data point sampled from the true distribution: $x \sim p_{\text{data}}(x)$. |
| $p_{\text{data}}(x)$ | Probability Density | The true, unknown data-generating distribution. |
| $p_\theta(x)$ | Parametric Density | The generative model's parameterized probability distribution with weights $\theta$. |
| $x_{<t}$ | Sequence Prefix | The history of tokens or dimensions preceding step $t$: $(x_1, x_2, \dots, x_{t-1})$. |
| $z \in \mathcal{Z} \subseteq \mathbb{R}^d$ | Continuous Vector | Latent variable representation in low-dimensional space ($d \ll D$). |
| $f_\theta: \mathcal{X} \to \mathcal{Z}$ | Deterministic Function | Encoder neural network parameterizing the latent projection $z = f_\theta(x)$. |
| $g_\phi: \mathcal{Z} \to \mathcal{X}$ | Deterministic Function | Decoder neural network parameterizing the reconstruction $\hat{x} = g_\phi(z)$. |
| $\mathcal{L}_{\text{recon}}$ | Scalar Loss ($\mathbb{R}^+$) | Mean squared error or cross-entropy reconstruction loss $\|x - \hat{x}\|^2$. |
| $p(z)$ | Prior Density | The assumed distribution over latent factors (typically standard isotropic normal $\mathcal{N}(0, \mathbf{I})$). |
| $q_\phi(z \mid x)$ | Variational Posterior | The recognition model approximating the intractable true posterior $p(z \mid x)$. |
| $\text{JSD}(p \parallel q)$ | $[0, \log 2]$ | Jensen-Shannon Divergence measuring symmetric distance between distributions. |

---

## 3. Explicit Likelihood vs. Implicit Generative Models

Generative models fall into two primary philosophical paradigms based on how they treat the likelihood function $p_\theta(x)$:

### 3.1 Explicit Likelihood Models
In an explicit likelihood framework, the model parameters $\theta$ are trained directly by maximizing the log-likelihood of the observed training dataset $\mathcal{D} = \{x^{(1)}, x^{(2)}, \dots, x^{(N)}\}$:
$$\max_\theta \sum_{i=1}^N \log p_\theta(x^{(i)}) \iff \min_\theta D_{\text{KL}}(p_{\text{data}}(x) \parallel p_\theta(x))$$

Explicit models diverge based on how they evaluate $p_\theta(x)$:
1. **Tractable Likelihood:**
   - The model architecture is restricted such that computing $p_\theta(x)$ and evaluating its exact gradient $\nabla_\theta \log p_\theta(x)$ requires no approximations.
   - *Autoregressive Models:* Factorize $p(x) = \prod_{t=1}^T p(x_t \mid x_{<t})$.
   - *Normalizing Flows:* Transform a simple base density $p(z)$ via invertible, bijective functions $x = f(z)$ using the change-of-variables formula:
     $$p_X(x) = p_Z(f^{-1}(x)) \left| \det \left( \frac{\partial f^{-1}(x)}{\partial x} \right) \right|$$
2. **Approximate Likelihood:**
   - The true density involves an intractable marginalization over latent variables $z$:
     $$p_\theta(x) = \int p_\theta(x, z) \, dz = \int p_\theta(x \mid z) p(z) \, dz$$
   - *Variational Autoencoders (VAEs):* Maximize the Evidence Lower Bound (ELBO):
     $$\log p_\theta(x) \ge \mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)] - D_{\text{KL}}(q_\phi(z \mid x) \parallel p(z))$$
   - *Diffusion Models:* Invert a forward Brownian noise process using a reverse denoising score matching objective.

### 3.2 Implicit Likelihood Models
Implicit models abandon density estimation entirely. They do not define an analytical expression for $p_\theta(x)$, meaning you cannot evaluate the likelihood of an input image or query if a sample is an anomaly. Instead, they learn a generator function $G_\theta(z): \mathcal{Z} \to \mathcal{X}$ that maps random noise $z \sim p(z)$ into high-fidelity data samples:
- *Generative Adversarial Networks (GANs):* Pit the generator against an adversarial discriminator $D_\phi(x)$ in a zero-sum minimax game.
- *Advantage:* Highly sharp sample synthesis without blurriness caused by maximum likelihood density covering.
- *Disadvantage:* Susceptible to mode collapse, unstable non-convex minimax dynamics, and unable to evaluate densities.

---

## 4. Autoregressive Models: Exact Factorization via the Chain Rule

### 4.1 The Probability Chain Rule
Any arbitrary joint distribution of a $T$-dimensional random vector $x = (x_1, x_2, \dots, x_T)$ can be factorized without loss of generality into a product of 1D conditional distributions:
$$\boxed{P(x_1, x_2, \dots, x_T) = \prod_{t=1}^T P(x_t \mid x_1, x_2, \dots, x_{t-1}) = \prod_{t=1}^T P(x_t \mid x_{<t})}$$

This identity follows strictly from the definition of conditional probability:
$$P(x_1, x_2) = P(x_1) P(x_2 \mid x_1)$$
$$P(x_1, x_2, x_3) = P(x_1, x_2) P(x_3 \mid x_1, x_2) = P(x_1) P(x_2 \mid x_1) P(x_3 \mid x_1, x_2)$$
Iterating this expansion yields the general product formula.

### 4.2 Training via Teacher Forcing
During training, the ground truth sequence $x = (x_1, x_2, \dots, x_T)$ is known in advance. We minimize the Negative Log-Likelihood (NLL):
$$\mathcal{L}_{\text{AR}}(\theta) = -\sum_{t=1}^T \log P_\theta(x_t \mid x_{<t})$$

Under **Teacher Forcing**, each conditional distribution $P_\theta(x_t \mid x_{<t})$ is conditioned on the ground truth prefix $x_{<t}$, not the model's own past predictions. This enables massive parallelization during training across all time steps $t \in [1, T]$ using causal attention masks.

### 4.3 Sequential Generation & Exposure Bias
At test time, the model must sample auto-regressively:
1. Sample $\tilde{x}_1 \sim P_\theta(x_1)$
2. Sample $\tilde{x}_2 \sim P_\theta(x_2 \mid \tilde{x}_1)$
3. Sample $\tilde{x}_3 \sim P_\theta(x_3 \mid \tilde{x}_1, \tilde{x}_2)$
4. Repeat sequentially until termination token $\langle\text{EOS}\rangle$.

**Exposure Bias:** Because the model was trained exclusively on flawless gold-standard prefixes $x_{<t}$, an early mistake during sampling can push the state into an unseen region of prefix space, compounding errors down the sequence.

---

## 5. Vanilla Autoencoders: The Deterministic Bottleneck

### 5.1 Architecture and Reconstruction Loss
A standard deterministic autoencoder consists of two functions:
1. **Encoder:** $z = f_\theta(x) \in \mathbb{R}^d$, compressing input $x \in \mathbb{R}^D$ into a low-dimensional bottleneck ($d \ll D$).
2. **Decoder:** $\hat{x} = g_\phi(z) \in \mathbb{R}^D$, attempting to invert the compression.

The parameters $\theta, \phi$ are optimized to minimize reconstruction error:
$$\mathcal{L}_{\text{AE}}(\theta, \phi) = \frac{1}{N} \sum_{i=1}^N \|x^{(i)} - g_\phi(f_\theta(x^{(i)}))\|_2^2$$

```
+---------------------------------------------------------------------------------------------------+
|                            VANILLA DETERMINISTIC AUTOENCODER                                      |
|                                                                                                   |
|    Input x           Encoder             Latent Space            Decoder           Reconstruction |
|   (Dim D)          f_theta(x)                (Dim d)             g_phi(z)              x_hat      |
|                                                                                                   |
|  [x_1, x_2, ...]  ───> [ MLP / CNN ] ───>   z = [z_1, z_2]   ───> [ MLP / CNN ] ───> [x_hat_1, ...] |
|                                                    |                                              |
|                                        Unconstrained isolated pins                                |
|                                           (Empty space = garbage)                                 |
+---------------------------------------------------------------------------------------------------+
```

### 5.2 Why Vanilla Autoencoders Cannot Generate
A deterministic autoencoder achieves low training reconstruction error by mapping training data points to isolated spikes or manifolds in $\mathbb{R}^d$. However, it suffers from two critical flaws that make it incapable of generative sampling:

1. **Unconstrained Latent Space Geometry:**
   The objective contains zero terms penalizing the shape, density, or bounds of $z$. One cluster of points might live near $[+100, -50]$ while another lives near $[-0.01, +0.02]$.
2. **The "Holes" in Latent Space (The Support Void Problem):**
   The aggregate posterior distribution is defined as:
   $$q(z) = \int q(z \mid x) p_{\text{data}}(x) \, dx$$
   In a deterministic autoencoder, $q(z \mid x) = \delta(z - f_\theta(x))$ (a Dirac delta function). Consequently, $q(z)$ is supported only on a disjoint, measure-zero set of points or thin sheets.
   
   If you sample a random vector from an isotropic Gaussian prior $z_{\text{rand}} \sim \mathcal{N}(0, \mathbf{I})$, with probability nearly 1:
   $$z_{\text{rand}} \notin \text{supp}(q(z))$$
   The decoder $g_\phi(z_{\text{rand}})$ is evaluated on inputs it was never trained on. Because neural networks do not extrapolate smoothly outside their training manifold, the decoder outputs chaotic, blurry, or nonsensical noise.

```
       DETERMINISTIC AUTOENCODER                       VARIATIONAL AUTOENCODER (VAE)
      
       z_2 ^                                           z_2 ^
           |    [City B]                                   |       +---+
           |    (8, 7)                                     |     /       \
           |      *                                        |    |  N(0,I) |
           |                                               |    |  Prior  |
           |   ??? EMPTY VOID ???                          |     \       /
           |   (Decoder fails here!)                       |       +---+
           |                                               |
           |  *                                            |
           | (1, 1) [City A]                               |
       ----+--------------------> z_1                  ----+--------------------> z_1
           |                                               |
           Disjoint, measure-zero points                   Smooth, continuous, regularized density
```

This fundamental gap directly motivates the invention of **Variational Autoencoders (VAEs)**: forcing the latent representation $q_\phi(z \mid x)$ to match a well-behaved continuous prior $p(z)$ via a Kullback-Leibler divergence penalty.

---

## 6. Summary Comparison Table

| Attribute | Autoregressive Models | Deterministic Autoencoders | Variational Autoencoders (VAEs) | GANs |
| :--- | :--- | :--- | :--- | :--- |
| **Likelihood Formulation** | Explicit (Exact) | None | Explicit (ELBO Lower Bound) | Implicit (No density) |
| **Generation Speed** | Slow (Sequential $\mathcal{O}(T)$) | Not Generative | Fast (One forward pass) | Fast (One forward pass) |
| **Training Stability** | Extremely Stable (Convex MLE) | Extremely Stable (MSE/BCE) | Stable (Variational Bound) | Unstable (Minimax Oscillation) |
| **Latent Space** | No explicit latent variable | Unregularized, measure-zero | Regularized smooth Gaussian | Continuous noise prior $\mathcal{Z}$ |
| **Primary Failure Mode** | Exposure bias, $\mathcal{O}(T)$ latency | Empty latent voids (cannot sample) | Blurry reconstructions | Mode collapse, vanishing gradients |
