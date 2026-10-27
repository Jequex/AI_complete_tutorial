# Topic 04: Multimodal Alignment & Representation Binding

## 1. Introduction & 12-Year-Old Intuition

Imagine you have two friends:
1. One friend only speaks in **photographs** and paintings. They have never heard a spoken word or read a book in their life.
2. The other friend is a **radio announcer** who has been blind from birth. They only speak in eloquent verbal sentences, paragraphs, and poetry.

If you place them in the same room, how could they ever understand each other?

You could build a **Universal Rosetta Stone** in the center of the room. You create a physical map of ideas:
- When the photographer shows a photo of a golden puppy chasing a tennis ball, you place a pin at coordinate $(X=14, Y=42)$.
- When the radio announcer speaks the sentence *"A joyful young dog running across the grass"*, you guide their hand to place a pin at the exact same coordinate $(X=14, Y=42)$.

Even though visual photons and acoustic audio waveforms are completely different physical signals, they point to the **same underlying concept in reality**.

In research-level artificial intelligence, this is called **Multimodal Alignment & Representation Binding**. 

Instead of training isolated models for images and text, architectures like **CLIP** (Contrastive Language-Image Pre-training) and **SigLIP** (Sigmoid Language-Image Pre-training) project raw visual pixels and raw linguistic tokens into a **shared unit hypersphere**. Once aligned, an AI can classify images without ever seeing training labels (Zero-Shot Learning), search through millions of videos using natural language sentences, and ground language models in physical sensory reality (as in modern Vision-Language Models like LLaVA and Flamingo).

```
Image Frame ──────► Vision Encoder f(v) ──► Latent Vector z_v ──┐
                                                                ▼ Normalize to Unit Sphere
                                                          [ z_v . z_t ] Cosine Similarity
                                                                ▲
Text Caption ─────► Text Encoder g(t) ────► Latent Vector z_t ──┘
```

---

## 2. Mathematical Symbol Decoder

| Symbol | Mathematical Entity | Dimensionality / Domain | Operational Meaning & Research Role |
| :--- | :--- | :--- | :--- |
| $\mathbf{v}_i$ | Visual Input Vector / Image | $\mathbb{R}^{D_v}$ | Raw pixel tensor or visual feature representation of sample $i$. |
| $\mathbf{t}_j$ | Textual Input Vector / Prompt | $\mathbb{R}^{D_t}$ | Token sequence or text transformer representation of prompt $j$. |
| $f_\phi(\mathbf{v})$ | Vision Encoder Network | $\mathbb{R}^{D_v} \to \mathbb{R}^d$ | Vision Transformer (ViT) projecting images into latent space. |
| $g_\psi(\mathbf{t})$ | Text Encoder Network | $\mathbb{R}^{D_t} \to \mathbb{R}^d$ | Autoregressive / Causal Transformer projecting text into latent space. |
| $\hat{\mathbf{z}}_{v, i}$ | Normalized Vision Embedding | $\mathbb{S}^{d-1} \subset \mathbb{R}^d$ | $\hat{\mathbf{z}}_{v, i} = \frac{f_\phi(\mathbf{v}_i)}{\|f_\phi(\mathbf{v}_i)\|_2}$; unit-norm image representation. |
| $\hat{\mathbf{z}}_{t, j}$ | Normalized Text Embedding | $\mathbb{S}^{d-1} \subset \mathbb{R}^d$ | $\hat{\mathbf{z}}_{t, j} = \frac{g_\psi(\mathbf{t}_j)}{\|g_\psi(\mathbf{t}_j)\|_2}$; unit-norm caption representation. |
| $S_{ij}$ | Logit Similarity Matrix | $\mathbb{R}^{B \times B}$ | $S_{ij} = \frac{\hat{\mathbf{z}}_{v, i}^\top \hat{\mathbf{z}}_{t, j}}{\tau}$; temperature-scaled cosine similarities. |
| $\tau$ | Temperature Parameter | Positive scalar $> 0$ | Governs the sharpness / entropy of the contrastive softmax distribution. |
| $B$ | Batch Size | Positive integer | Number of paired image-text examples per contrastive mini-batch. |
| $I(V; T)$ | Mutual Information | Non-negative scalar | Statistical information shared between visual and textual modalities. |

---

## 3. Contrastive Representation Learning: CLIP & InfoNCE

### 3.1 The Symmetric InfoNCE Objective
Given a mini-batch of $B$ paired samples $\{(\mathbf{v}_i, \mathbf{t}_i)\}_{i=1}^B$, the $B$ diagonal pairs $(\mathbf{v}_i, \mathbf{t}_i)$ are **positive matches**, while the $B(B - 1)$ off-diagonal pairs $(\mathbf{v}_i, \mathbf{t}_j)_{j \neq i}$ are treated as **negative distractors**.

The contrastive objective consists of two symmetric cross-entropy losses:

#### 1. Image-to-Text Loss:
$$\mathcal{L}_{v2t} = -\frac{1}{B} \sum_{i=1}^B \ln \left( \frac{\exp\left(\hat{\mathbf{z}}_{v, i}^\top \hat{\mathbf{z}}_{t, i} / \tau\right)}{\sum_{j=1}^B \exp\left(\hat{\mathbf{z}}_{v, i}^\top \hat{\mathbf{z}}_{t, j} / \tau\right)} \right)$$

#### 2. Text-to-Image Loss:
$$\mathcal{L}_{t2v} = -\frac{1}{B} \sum_{j=1}^B \ln \left( \frac{\exp\left(\hat{\mathbf{z}}_{v, j}^\top \hat{\mathbf{z}}_{t, j} / \tau\right)}{\sum_{i=1}^B \exp\left(\hat{\mathbf{z}}_{v, i}^\top \hat{\mathbf{z}}_{t, j} / \tau\right)} \right)$$

#### Total CLIP Loss:
$$\mathcal{L}_{\text{CLIP}} = \frac{1}{2} \left( \mathcal{L}_{v2t} + \mathcal{L}_{t2v} \right)$$

```
                    Text Captions (1 ... B)
                   t_1      t_2      t_3 ...  t_B
               ┌────────┬────────┬────────┬────────┐
         v_1   │  +1.0  │  -0.3  │  -0.5  │  -0.4  │  ──► Softmax across rows (I2T)
               ├────────┼────────┼────────┼────────┤
Image    v_2   │  -0.2  │  +0.9  │  -0.1  │  -0.6  │
Frames   v_3   │  -0.4  │  -0.2  │  +1.0  │  -0.3  │
 (1...B) ...   ├────────┼────────┼────────┼────────┤
         v_B   │  -0.5  │  -0.4  │  -0.2  │  +0.9  │
               └────────┴────────┴────────┴────────┘
                   ▲
                   │ Softmax down columns (T2I)
```

---

## 4. Mathematical Proof: InfoNCE Maximizes Mutual Information

Why does minimizing InfoNCE lead to rich, semantics-preserving representations?

Let $X$ and $Y$ be random variables representing the two modalities with joint distribution $p(x, y)$ and marginals $p(x), p(y)$.

The Mutual Information is:

$$I(X; Y) = \mathbb{E}_{p(x, y)} \left[ \ln \frac{p(x, y)}{p(x)p(y)} \right]$$

### Theorem (van den Oord et al., 2018):
For a batch of size $B$, the InfoNCE loss provides a rigorous lower bound on mutual information:

$$I(X; Y) \ge \ln(B) - \mathcal{L}_{\text{InfoNCE}}$$

#### Proof:
Consider the optimal critic function $f^*(x, y) \propto \frac{p(y \mid x)}{p(y)}$.
The probability that candidate $x_1$ was drawn from the joint distribution $p(x, y_1)$ rather than the independent product of marginals among $B$ candidates is:

$$P(D=1 \mid x_{1:B}, y_1) = \frac{\frac{p(y_1 \mid x_1)}{p(y_1)}}{\frac{p(y_1 \mid x_1)}{p(y_1)} + \sum_{j=2}^B \frac{p(y_1 \mid x_j)}{p(y_1)}} = \frac{\exp(f^*(x_1, y_1))}{\sum_{j=1}^B \exp(f^*(x_j, y_1))}$$

Evaluating the expectation of $\ln P(D=1 \mid \cdot)$ over all draws:

$$\mathbb{E} \left[ \ln \frac{\frac{p(y_1 \mid x_1)}{p(y_1)}}{\frac{p(y_1 \mid x_1)}{p(y_1)} + \sum_{j=2}^B \frac{p(y_1 \mid x_j)}{p(y_1)}} \right] = -\mathcal{L}_{\text{InfoNCE}}$$

Expanding the denominator using Jensen's inequality:

$$\mathbb{E} \left[ \ln \left( 1 + \frac{p(y_1)}{p(y_1 \mid x_1)} \sum_{j=2}^B \frac{p(y_1 \mid x_j)}{p(y_1)} \right) \right] \approx \ln(1 + (B-1) e^{-I(X; Y)}) \le \ln(B) - I(X; Y)$$

Rearranging terms:

$$I(X; Y) \ge \ln(B) - \mathcal{L}_{\text{InfoNCE}}$$

**Core Insight**: Larger batch sizes $B$ mathematically tighten the lower bound, explaining why frontier models like CLIP scale batch sizes to $B = 32,768$ across thousands of GPUs.

---

## 5. SigLIP: Overcoming the Softmax Communication Bottleneck

In standard CLIP, computing the denominator $\sum_{j=1}^B \exp(S_{ij})$ requires an all-to-all communication primitive across all GPU workers. When batch size reaches $B = 65,536$, network synchronization latency dominates training.

**SigLIP (Sigmoid Loss for Language Image Pre-training)** (Zhai et al., 2023) eliminates the global softmax entirely, replacing it with pairwise binary logistic regression:

$$\mathcal{L}_{\text{SigLIP}} = -\sum_{i=1}^B \sum_{j=1}^B \left[ y_{ij} \ln \sigma\left(t \cdot \hat{\mathbf{z}}_{v, i}^\top \hat{\mathbf{z}}_{t, j} + b\right) + (1 - y_{ij}) \ln \left(1 - \sigma\left(t \cdot \hat{\mathbf{z}}_{v, i}^\top \hat{\mathbf{z}}_{t, j} + b\right)\right) \right]$$

where $y_{ij} = 1$ if $i = j$ else $-1$ (or $0$).

### Advantages of SigLIP:
1. **Decoupled Batches**: Each image-text pair $(i, j)$ is an independent binary classification problem.
2. **Zero Global Reduction**: Gradients can be computed asynchronously without gathering the full $B \times B$ matrix across the entire computing cluster.
3. **State-of-the-Art Scaling**: Enabled Google's PaliGemma and Gemini vision encoders to train efficiently on billions of image-text pairs.

---

## 6. Zero-Shot Classification & Vision-Language Adapters

Once vision and text vectors are bound to the same hypersphere, zero-shot classification is trivial:

```
Candidate Text Prompts:
t_1 = "a photo of a red sports car"          ──► Text Encoder ──► \hat{z}_{t, 1}
t_2 = "a photo of a golden retriever dog"   ──► Text Encoder ──► \hat{z}_{t, 2}
t_3 = "a photo of a green apple"            ──► Text Encoder ──► \hat{z}_{t, 3}

Unseen Query Image:
v   = [ Novel Image ]                       ──► Vision Encoder ──► \hat{z}_v

Prediction:
c* = argmax_k ( \hat{z}_v . \hat{z}_{t, k} )
```

### Vision-Language Adapters for Multimodal LLMs:
How do modern Vision-Language Models (VLMs) feed visual representations into Large Language Models?
1. **Linear / MLP Projection (LLaVA)**: Visual patch tokens $\mathbf{H}_v \in \mathbb{R}^{N_v \times D_v}$ pass through a 2-layer MLP $\mathbf{W}_{\text{proj}} \mathbf{H}_v \in \mathbb{R}^{N_v \times D_{\text{llm}}}$, prepending visual tokens directly into the LLM's input context.
2. **Perceiver Resampler (Flamingo)**: A fixed set of learned latent queries $\mathbf{Q} \in \mathbb{R}^{K \times d}$ attend to variable-length visual tokens via cross-attention, producing a fixed count of $K = 64$ visual tokens regardless of image resolution.
