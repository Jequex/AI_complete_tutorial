# Topic 25.3: Noise Schedules and Denoising U-Net Architectures

---

## 1. Executive Summary & The 12-Year-Old Analogy

### Moving from Theory to Scalable Deep Generative Models
While the forward and reverse Markov chains of DDPM provide a rigorous mathematical framework, naive implementations fail in practice due to two critical engineering dimensions:
1. **The Noise Schedule $\beta_1, \dots, \beta_T$:** How rapidly noise is injected across the diffusion process determines whether network capacity is distributed effectively across coarse and fine structural scales.
2. **The Loss Objective & Denoising Architecture:** Translating the raw variational bound into an unweighted regression objective, combined with multi-scale U-Net architectures equipped with sinusoidal time embeddings and residual skip connections.

```
+----------------------------------------------------------------------------------------------------+
|                               THE DENOISING U-NET ARCHITECTURE                                     |
|                                                                                                    |
|    Input: Noisy Image x_t                                      Output: Predicted Noise eps_theta   |
|          │                                                                    ▲                    |
|          ▼                                                                    │                    |
|   ┌──────────────┐                                                     ┌──────────────┐            |
|   │ Down Block 1 │ ──────────────── Skip Connection ──────────────────> │  Up Block 1  │            |
|   └──────────────┘                                                     └──────────────┘            |
|          │                                                                    ▲                    |
|          ▼                                                                    │                    |
|   ┌──────────────┐                                                     ┌──────────────┐            |
|   │ Down Block 2 │ ──────────────── Skip Connection ──────────────────> │  Up Block 2  │            |
|   └──────────────┘                                                     └──────────────┘            |
|          │                                                                    ▲                    |
|          ▼                                                                    │                    |
|          └─────────────────> ┌───────────────────┐ ───────────────────────────┘                    |
|                              │  Bottleneck Block │                                                 |
|                              │   + Self-Attention│                                                 |
|                              └───────────────────┘                                                 |
|                                        ▲                                                           |
|                                        │ (Injected at every block)                                 |
|                         [Sinusoidal Time Embedding PE(t)]                                          |
+----------------------------------------------------------------------------------------------------+
```

---

### The 12-Year-Old Analogy: The Fog Dimmer Switch & The High-Rise Elevator

Imagine you are filming a painting through an electronic glass window that can fog up:
1. **The Faulty Linear Dimmer:**
   - With an old-fashioned linear knob (Linear Schedule), you turn the dial just 20% of the way, and suddenly the window is pitch-black thick fog! The remaining 80% of the dial does nothing except make black fog blacker. The model learns nothing for 80% of its training time.
2. **The Smart Cosine Dimmer:**
   - Alex Nichol and Prafulla Dhariwal (2021) engineered a smooth cosine dimmer (Cosine Schedule). Now, turning the dial gently softens fine pencil lines first, then medium brush strokes, and only at 95% turns into complete white noise. Every click of the knob now has meaningful educational content!
3. **The U-Net with Skip Elevators:**
   - Imagine a 5-story art studio. As an image travels down to the basement (the encoder), it gets squashed into tiny abstract conceptual thoughts (e.g., *"this is a cat"*).
   - But when building the image back up in the penthouse (the decoder), the painters need to know where the cat's whiskers were located down to the exact pixel.
   - **Skip connections** are express elevators that carry the sharp blueprints directly across from the 5th-floor entrance to the 5th-floor exit, allowing the network to merge high-level semantics with microscopic edge precision!

---

## 2. Mathematical Symbol Decoder

| Symbol | Mathematical Domain | Description & Meaning |
| :--- | :--- | :--- |
| $\beta_t \in (0, 1)$ | Discrete Sequence | Forward noise variance at diffusion timestep $t$. |
| $\bar{\alpha}_t \in (0, 1)$ | Cumulative Signal | Cumulative signal retention factor: $\bar{\alpha}_t = \prod_{s=1}^t (1 - \beta_s)$. |
| $s \in \mathbb{R}^+$ | Hyperparameter | Cosine schedule offset parameter (typically $s = 0.008$) to prevent tiny $\beta_t$ at $t=0$. |
| $L_{VLB}$ | Variational Bound | Complete Variational Lower Bound (negative ELBO) in bits/nats. |
| $L_t$ | KL Divergence | Timestep-specific KL divergence term $D_{KL}(q(x_{t-1} \mid x_t, x_0) \parallel p_\theta(x_{t-1} \mid x_t))$. |
| $L_{\text{simple}}(\theta)$ | Unweighted MSE Loss | Jonathan Ho's simplified denoising loss $\mathbb{E}_{t, x_0, \boldsymbol{\epsilon}} [\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(x_t, t)\|^2]$. |
| $\text{PE}_{(t, 2i)}$ | Harmonic Embedding | Transformer-style sinusoidal positional embedding at frequency channel $i$. |
| $\boldsymbol{\epsilon}_\theta(x_t, t)$ | Neural Network | Denoising network (typically U-Net or DiT) with time-conditioned residual blocks. |

---

## 3. Mathematical Foundations & Derivations

### 3.1 Linear vs. Cosine Noise Schedule

In Ho et al. (2020), the variance schedule was chosen to increase linearly from $\beta_1 = 10^{-4}$ to $\beta_T = 0.02$ with $T = 1000$.
Under this linear schedule, $\bar{\alpha}_t = \prod_{s=1}^t (1 - \beta_s)$ drops rapidly:
$$\bar{\alpha}_t \approx \exp\left(-\sum_{s=1}^t \beta_s\right)$$

For small image sizes ($32 \times 32$ or $64 \times 64$), Nichol & Dhariwal (2021) proved that $\bar{\alpha}_t$ drops nearly to zero by $t \approx 0.2 T$. As a result:
- The network encounters almost pure Gaussian noise for over 70% of the forward diffusion steps.
- The forward process destroys information too abruptly in the first few steps, preventing the network from learning subtle high-frequency details.

#### The Improved Cosine Schedule Formulation
To ensure a linear decay of information rather than an exponential drop, Nichol & Dhariwal set:
$$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad \text{where} \quad f(t) = \cos^2\left( \frac{t/T + s}{1 + s} \cdot \frac{\pi}{2} \right)$$
The small constant offset $s = 0.008$ ensures that:
1. At $t = 0$: $\beta_1 = 1 - \frac{\bar{\alpha}_1}{\bar{\alpha}_0}$ is small and non-zero (preventing singularities).
2. Near $t = T$: $\bar{\alpha}_T \to 0$ smoothly without abrupt terminal cliffs.
3. The resulting $\beta_t$ is extracted via $\beta_t = 1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}$, clipped to $\beta_t \le 0.999$ to avoid numerical instability.

```
Signal Retention (alpha_bar_t) vs Timestep t:
1.0 ┬───────╮ (Cosine Schedule: Smooth, gradual decay of information)
    │        \
    │         \
    │          \        Linear Schedule (Drops too fast!)
0.5 │           \      /
    │            \    /
    │             \  /
    │              ▼
0.0 ┴─────────────────────────────────────────────> Timestep t (0 -> T)
```

---

### 3.2 Full Derivation of the Variational Bound & Simplified Objective

The log-likelihood of clean data $\log p_\theta(x_0)$ is bounded by the Evidence Lower Bound (ELBO):
$$\log p_\theta(x_0) \ge \mathbb{E}_{q(x_{1:T} \mid x_0)} \left[ \log \frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)} \right] \triangleq -L_{VLB}$$

Using Bayes' rule $q(x_t \mid x_{t-1}) = \frac{q(x_{t-1} \mid x_t, x_0) q(x_t \mid x_0)}{q(x_{t-1} \mid x_0)}$, the terms inside the logarithm telescope:
$$L_{VLB} = \mathbb{E}_q \left[ \underbrace{D_{KL}(q(x_T \mid x_0) \parallel p(x_T))}_{L_T} + \sum_{t=2}^T \underbrace{D_{KL}(q(x_{t-1} \mid x_t, x_0) \parallel p_\theta(x_{t-1} \mid x_t))}_{L_{t-1}} \underbrace{- \log p_\theta(x_0 \mid x_1)}_{L_0} \right]$$

- $L_T$: Has no trainable parameters $\theta$ (since $q(x_T \mid x_0) \approx \mathcal{N}(0, \mathbf{I})$ and $p(x_T) = \mathcal{N}(0, \mathbf{I})$), so $L_T \approx 0$.
- $L_0$: Reconstruction log-likelihood modeled by a discrete Gaussian decoder.
- $L_{t-1}$: KL divergence between two multivariate Gaussian distributions:
  $$q(x_{t-1} \mid x_t, x_0) = \mathcal{N}\left(x_{t-1}; \tilde{\boldsymbol{\mu}}_t(x_t, x_0), \sigma_t^2 \mathbf{I}\right)$$
  $$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\left(x_{t-1}; \boldsymbol{\mu}_\theta(x_t, t), \sigma_t^2 \mathbf{I}\right)$$

For two Gaussians with identical covariance $\Sigma = \sigma_t^2 \mathbf{I}$, the KL divergence simplifies to the squared Euclidean distance between their means:
$$D_{KL}(q \parallel p_\theta) = \frac{1}{2\sigma_t^2} \|\tilde{\boldsymbol{\mu}}_t(x_t, x_0) - \boldsymbol{\mu}_\theta(x_t, t)\|^2$$

Recall from Topic 25.2 that the ground-truth posterior mean is:
$$\tilde{\boldsymbol{\mu}}_t(x_t, x_0) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon} \right)$$

If we parameterize the neural network to output the noise vector $\boldsymbol{\epsilon}_\theta(x_t, t)$:
$$\boldsymbol{\mu}_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(x_t, t) \right)$$

Substituting both means into the KL divergence:
$$L_{t-1} = \frac{1}{2\sigma_t^2} \left\| \frac{1}{\sqrt{\alpha_t}}\left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\boldsymbol{\epsilon} \right) - \frac{1}{\sqrt{\alpha_t}}\left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\boldsymbol{\epsilon}_\theta(x_t, t) \right) \right\|^2$$
$$L_{t-1} = \frac{\beta_t^2}{2 \sigma_t^2 \alpha_t (1 - \bar{\alpha}_t)} \|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(x_t, t)\|^2$$

#### The Simplified Loss Formulation $L_{\text{simple}}(\theta)$
Ho et al. (2020) discovered that setting the weighting coefficient $\frac{\beta_t^2}{2 \sigma_t^2 \alpha_t (1 - \bar{\alpha}_t)}$ to **$1.0$** dramatically improves visual sample quality:
$$L_{\text{simple}}(\theta) \triangleq \mathbb{E}_{t \sim \mathcal{U}(1, T), x_0, \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})} \left[ \|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(x_t, t)\|^2 \right]$$

**Why does dropping the weight improve sample quality?**
- The theoretical weight $\frac{\beta_t^2}{2 \sigma_t^2 \alpha_t (1 - \bar{\alpha}_t)}$ is large for very small $t$ (where noise is imperceptible) and tiny for large $t$.
- Optimizing unweighted MSE down-weights the loss at small $t$, focusing network capacity on intermediate and large $t$ where large-scale semantic structure and composition are decided!

---

### 3.3 U-Net Architectural Motifs

1. **Sinusoidal Positional Embeddings:**
   Timestep $t \in [1, T]$ is transformed into a continuous embedding vector using multi-frequency harmonic sinusoids:
   $$\text{PE}_{(t, 2i)} = \sin\left( \frac{t}{10000^{2i/d_{\text{emb}}}} \right), \quad \text{PE}_{(t, 2i+1)} = \cos\left( \frac{t}{10000^{2i/d_{\text{emb}}}} \right)$$
   This continuous vector is passed through an MLP and added/modulated into every residual block.

2. **Residual Blocks & Skip Connections:**
   - Spatial downsampling (via strided convolutions or average pooling) extracts semantic context in deep feature maps.
   - Symmetric skip connections concatenate the high-resolution feature maps from the encoder directly to the decoder inputs.
   - This allows fine pixel boundaries and textures to bypass the latent bottleneck.
