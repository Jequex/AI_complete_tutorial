# Topic 25.4: Classifier-Free Guidance and Latent Diffusion Models

---

## 1. Executive Summary & The 12-Year-Old Analogy

### Directing the Generative Trajectory
Unconditional diffusion models generate diverse samples from the overall data distribution $p(x)$. However, modern generative applications require precise steering: generating an image of a *"golden retriever wearing sunglasses on a beach"* or a protein binding a specific receptor.

Controlling diffusion models evolved through two revolutionary breakthroughs:
1. **Classifier-Free Guidance (CFG, Jonathan Ho & Tim Salimans 2021):** Extrapolating the score vector field in the direction of the conditional prompt without requiring a separate, fragile classifier network.
2. **Latent Diffusion Models (LDM / Stable Diffusion, Robin Rombach et al. 2022):** Shifting the diffusion process from raw pixel space into the compact, perceptually compressed latent space of a pretrained VAE, reducing computational complexity by orders of magnitude.

```
+----------------------------------------------------------------------------------------------------+
|                               CLASSIFIER-FREE GUIDANCE VECTOR FIELD                                |
|                                                                                                    |
|                       eps_theta(x_t, prompt)                                                       |
|                                ↗                                                                   |
|                               /                                                                    |
|                              /  Difference Vector: (eps_cond - eps_null)                           |
|                             / ──────────────────────────────────────────────>                      |
|                            /                                                \                      |
|                           /                                                  ▼                     |
|     eps_theta(x_t, null) ─                                          EXTRAPOLATED VECTOR            |
|                                                           eps_tilde = eps_null + s*(eps_c - eps_0) |
|                                                                                                    |
|   Scale s = 0: Pure random unconditional sample (no prompt adherence)                              |
|   Scale s = 1: Standard conditional sample (moderate fidelity)                                     |
|   Scale s = 3-7: Hyper-conditioned generation (sharp contrast, extreme prompt alignment)           |
+----------------------------------------------------------------------------------------------------+
```

---

### The 12-Year-Old Analogy: The Movie Director & The Compact Flash Drive

1. **The Movie Director's Megaphone (CFG):**
   - Imagine an actor is sketching a random landscape on a canvas.
   - If the director whispers *"draw a dragon"* quietly (scale $s = 1$), the actor might draw a mountain and hide a tiny dragon behind a tree.
   - If the director picks up a giant electric megaphone and shouts *"DRAW A DRAGON! ONLY A DRAGON!"* (scale $s = 5$), the actor ignores the background trees and fills every square inch of canvas with giant fiery dragon scales!
   - That megaphone volume knob is the **Guidance Scale $s$**. Turning it up forces the artwork to obey your instructions with absolute certainty.
2. **The Compact Flash Drive (Latent Diffusion):**
   - If an IMAX movie camera recorded every single microscopic atom on every leaf, the movie file would be 50 terabytes per second. Editing it would crash every computer on Earth.
   - Instead, a smart camera sensor compresses the scene onto a compact flash drive (the VAE encoder). It discards invisible imperceptible noise while preserving all shapes, colors, and textures ($48\times$ smaller!).
   - The special-effects team adds monsters and explosions directly onto the compact file (diffusion in latent space). Once finished, a projector unzips it back onto the giant IMAX screen (the VAE decoder). This is **Latent Diffusion**.

---

## 2. Mathematical Symbol Decoder

| Symbol | Mathematical Domain | Description & Meaning |
| :--- | :--- | :--- |
| $c$ | Discrete or Vector | Conditioning signal (class label, CLIP text token embeddings, depth map). |
| $\emptyset$ | Null Token | The unconditional / empty condition representation. |
| $p_{\text{uncond}} \in (0, 1)$ | Probability | Dropout rate of conditioning during training (typically $0.10 \le p_{\text{uncond}} \le 0.20$). |
| $s \in \mathbb{R}^+$ | Hyperparameter | Classifier-Free Guidance scale ($s = 1$ is standard conditional; $s > 1$ is guided). |
| $\boldsymbol{\epsilon}_\theta(x_t, c)$ | Neural Output | Noise predicted conditioned on prompt or label $c$. |
| $\boldsymbol{\epsilon}_\theta(x_t, \emptyset)$ | Neural Output | Noise predicted with conditioning set to null token. |
| $\tilde{\boldsymbol{\epsilon}}_\theta(x_t, c)$ | Guided Vector | Extrapolated CFG noise vector: $\boldsymbol{\epsilon}_\emptyset + s (\boldsymbol{\epsilon}_c - \boldsymbol{\epsilon}_\emptyset)$. |
| $\mathcal{E}(x) = z$ | VAE Encoder | Encoder mapping pixel image $x \in \mathbb{R}^{H \times W \times 3}$ to latent $z \in \mathbb{R}^{h \times w \times d_z}$. |
| $\mathcal{D}(z) = \hat{x}$ | VAE Decoder | Decoder reconstructing high-resolution pixel image from latent $z$. |
| $f = H/h = W/w$ | Downsampling Factor | Spatial compression ratio (typically $f = 8$). |

---

## 3. Mathematical Foundations & Derivations

### 3.1 Classifier Guidance: Differentiating Bayes' Rule
In conditional diffusion generation, the goal is to sample from $p(x \mid c)$.
By Bayes' rule, the conditional probability distribution at noisy timestep $t$ satisfies:
$$p(x_t \mid c) = \frac{p(x_t) p(c \mid x_t)}{p(c)}$$

Taking the natural logarithm of both sides:
$$\log p(x_t \mid c) = \log p(x_t) + \log p(c \mid x_t) - \log p(c)$$

Differentiating with respect to the continuous state $x_t$:
$$\nabla_{x_t} \log p(x_t \mid c) = \nabla_{x_t} \log p(x_t) + \nabla_{x_t} \log p(c \mid x_t)$$
where $\nabla_{x_t} \log p(c) = \mathbf{0}$ because the prompt prior $p(c)$ does not depend on the noisy image $x_t$.

Dhariwal & Nichol (2021) introduced an auxiliary scalar multiplier $\gamma > 1$ on the classifier gradient to artificially boost conditioning strength:
$$\nabla_{x_t} \log p_\gamma(x_t \mid c) \triangleq \nabla_{x_t} \log p(x_t) + \gamma \nabla_{x_t} \log p(c \mid x_t)$$

#### Why Classifier Guidance Failed in Practice:
1. **Need for a Separate Noisy Classifier:** An auxiliary neural network $p_\phi(c \mid x_t, t)$ must be trained to classify heavily corrupted images across all noise levels $t \in [1, T]$.
2. **Adversarial Exploitation:** Gradient descent along $\nabla_{x_t} \log p_\phi(c \mid x_t, t)$ exploits vulnerabilities in the classifier, pushing samples toward bizarre, unnatural adversarial patterns that trick the classifier rather than generating realistic imagery.
3. **Modal Inflexibility:** Training noisy classifiers for complex multimodal inputs (such as free-form natural language prompts) is practically intractable.

---

### 3.2 Classifier-Free Guidance (CFG): Ho & Salimans (2021)
Jonathan Ho and Tim Salimans realized that the classifier score gradient $\nabla_{x_t} \log p(c \mid x_t)$ can be rewritten purely in terms of generative score functions:
$$\nabla_{x_t} \log p(c \mid x_t) = \nabla_{x_t} \log p(x_t \mid c) - \nabla_{x_t} \log p(x_t)$$

Substitute this identity directly into the guided score formulation:
$$\nabla_{x_t} \log \tilde{p}(x_t \mid c) = \nabla_{x_t} \log p(x_t) + \gamma \left( \nabla_{x_t} \log p(x_t \mid c) - \nabla_{x_t} \log p(x_t) \right)$$
$$= (1 - \gamma) \nabla_{x_t} \log p(x_t) + \gamma \nabla_{x_t} \log p(x_t \mid c)$$

Recalling the fundamental score-to-noise correspondence $\nabla_{x_t} \log p(x_t) = -\frac{\boldsymbol{\epsilon}_\theta(x_t, t)}{\sqrt{1 - \bar{\alpha}_t}}$, multiplying by $-\sqrt{1 - \bar{\alpha}_t}$ converts this equation into noise-prediction space:
$$\tilde{\boldsymbol{\epsilon}}_\theta(x_t, c) = (1 - \gamma) \boldsymbol{\epsilon}_\theta(x_t, \emptyset) + \gamma \boldsymbol{\epsilon}_\theta(x_t, c)$$
Rewriting with guidance scale $s \triangleq \gamma$:
$$\tilde{\boldsymbol{\epsilon}}_\theta(x_t, c) = \boldsymbol{\epsilon}_\theta(x_t, \emptyset) + s \cdot \left( \boldsymbol{\epsilon}_\theta(x_t, c) - \boldsymbol{\epsilon}_\theta(x_t, \emptyset) \right)$$

```
CFG Extrapolation Spectrum:
s = 0: ──> eps_null (Unconditional base model)
s = 1: ──> eps_cond (Standard conditional model)
s > 1: ──> Extrapolates beyond standard conditional along the direction (eps_cond - eps_null)
```

#### Joint Training with Conditioning Dropout
To train a single unified network capable of evaluating both $\boldsymbol{\epsilon}_\theta(x_t, c)$ and $\boldsymbol{\epsilon}_\theta(x_t, \emptyset)$:
- During training, randomly replace the conditioning vector $c$ with a learned null token $\emptyset$ with probability $p_{\text{uncond}} \in [0.10, 0.20]$.
- At inference time, two forward passes are computed: one with prompt $c$ and one with $\emptyset$.

---

### 3.3 Latent Diffusion Models (LDM / Stable Diffusion)

In standard DDPM, diffusion operates in raw pixel space $\mathcal{X} = \mathbb{R}^{H \times W \times 3}$.
This has severe computational consequences:
- For a $512 \times 512$ RGB image, $D = 512 \times 512 \times 3 = 786,432$ dimensions.
- The computational cost of 2D convolutional layers scales linearly with $H \times W$, while self-attention blocks scale **quadratically** with spatial tokens: $\mathcal{O}((H \cdot W)^2) = \mathcal{O}((512 \cdot 512)^2) \approx \mathcal{O}(6.8 \times 10^{10})$ operations per layer!
- High-frequency pixel variations (such as imperceptible digital sensor noise) consume massive network capacity without improving semantic content.

```
+----------------------------------------------------------------------------------------------------+
|                               LATENT DIFFUSION ARCHITECTURE (LDM)                                  |
|                                                                                                    |
|    Pixel Space Image x in R^{512x512x3}                                                            |
|          │                                                                                         |
|          ▼ [Pretrained VAE Encoder E]  (Downsampling Factor f = 8)                                 |
|    Latent Space Vector z in R^{64x64x4}   (48x Dimensionality Reduction!)                          |
|          │                                                                                         |
|          ▼                                                                                         |
|    [Latent Diffusion Process]  <═══════ Conditioning: Text Prompt y via Cross-Attention            |
|    z_t = sqrt(alpha_bar_t)*z_0 + sqrt(1 - alpha_bar_t)*eps                                         |
|          │                                                                                         |
|          ▼ [Iterative Denoising in Latent Space]                                                   |
|    Denoised Latent z_0                                                                             |
|          │                                                                                         |
|          ▼ [Pretrained VAE Decoder D]                                                              |
|    Reconstructed Pixel Image x_hat in R^{512x512x3}                                                |
+----------------------------------------------------------------------------------------------------+
```

#### The Two-Stage Paradigm (Rombach et al., 2022)
1. **Perceptual Compression (Autoencoder Stage):**
   A variational autoencoder is pretrained with perceptual losses and adversarial patch discriminator objectives:
   $$z = \mathcal{E}(x), \quad \hat{x} = \mathcal{D}(z)$$
   For downsampling factor $f = 8$, an input $512 \times 512 \times 3$ is encoded into $64 \times 64 \times 4$.
   The number of latent dimensions drops from $786,432$ to $16,384$ ($48\times$ reduction).
   Self-attention across $64 \times 64 = 4096$ tokens is $\mathbf{4096\times}$ faster than across $512 \times 512 = 262,144$ tokens!

2. **Semantic Compression (Latent Diffusion Stage):**
   The diffusion model operates entirely in latent space $\mathcal{Z}$:
   $$L_{\text{LDM}}(\theta) \triangleq \mathbb{E}_{\mathcal{E}(x), y, \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I}), t} \left[ \left\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(z_t, t, \tau_\theta(y)) \right\|^2 \right]$$
   where $\tau_\theta(y)$ is a pretrained text encoder (such as CLIP ViT-L/14 or T5-XXL) projecting natural language prompts into a rich semantic token sequence.

3. **Cross-Attention Conditioning Mechanism:**
   Intermediate features $\phi_i(z_t)$ of the U-Net query the text embeddings via multi-head cross-attention:
   $$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^\top}{\sqrt{d_k}} \right) V$$
   $$Q = W_Q^{(i)} \phi_i(z_t), \quad K = W_K^{(i)} \tau_\theta(y), \quad V = W_V^{(i)} \tau_\theta(y)$$
