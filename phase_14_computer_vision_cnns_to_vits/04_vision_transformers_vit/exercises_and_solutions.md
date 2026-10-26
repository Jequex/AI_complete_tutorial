# Topic 14.4: Exercises & Solutions — Vision Transformers (ViT)

---

### Problem 1: Manual Patch Extraction, Linear Projection & Sequence Assembly (Easy)
Consider a small synthetic grayscale image $X \in \mathbb{R}^{1 \times 4 \times 4}$ ($C = 1, H = 4, W = 4$):
$$X = \begin{bmatrix}
1.0 & 2.0 & 3.0 & 4.0 \\
2.0 & 1.0 & 4.0 & 3.0 \\
0.0 & 1.0 & 5.0 & 6.0 \\
1.0 & 0.0 & 6.0 & 5.0
\end{bmatrix}$$

We configure a Vision Transformer with patch size $P = 2$ and hidden embedding dimension $D = 2$.
1. Compute the number of patches $N$, and write down each flattened patch vector $\mathbf{x}_p^i \in \mathbb{R}^4$ for $i \in \{1, 2, 3, 4\}$.
2. The projection matrix $E \in \mathbb{R}^{4 \times 2}$ is given by:
   $$E = \begin{bmatrix} 0.5 & 0.0 \\ 0.5 & 0.0 \\ 0.0 & 0.5 \\ 0.0 & 0.5 \end{bmatrix}$$
   Compute the projected patch embeddings $\mathbf{x}_p^i E \in \mathbb{R}^2$ for all 4 patches.
3. Prepend the learnable `[CLS]` token $\mathbf{x}_{\text{cls}} = [1.0, 1.0]$.
4. Given the 1D positional embedding matrix $E_{\text{pos}} \in \mathbb{R}^{5 \times 2}$:
   $$E_{\text{pos}} = \begin{bmatrix} 0.1 & 0.1 \\ 0.2 & 0.2 \\ 0.3 & 0.3 \\ 0.4 & 0.4 \\ 0.5 & 0.5 \end{bmatrix}$$
   Compute the initial sequence tensor $\mathbf{z}_0 \in \mathbb{R}^{5 \times 2}$ fed into the Transformer encoder.

<details>
<summary>View Complete Solution</summary>

**Part 1: Patch Extraction ($P = 2$):**
$$N = \frac{H \cdot W}{P^2} = \frac{4 \times 4}{2 \times 2} = \frac{16}{4} = 4 \text{ patches}$$
Extracting $2 \times 2$ spatial blocks:
- **Patch 1 (Top-Left):** $\begin{bmatrix} 1.0 & 2.0 \\ 2.0 & 1.0 \end{bmatrix} \implies \mathbf{x}_p^1 = [1.0, 2.0, 2.0, 1.0]$
- **Patch 2 (Top-Right):** $\begin{bmatrix} 3.0 & 4.0 \\ 4.0 & 3.0 \end{bmatrix} \implies \mathbf{x}_p^2 = [3.0, 4.0, 4.0, 3.0]$
- **Patch 3 (Bottom-Left):** $\begin{bmatrix} 0.0 & 1.0 \\ 1.0 & 0.0 \end{bmatrix} \implies \mathbf{x}_p^3 = [0.0, 1.0, 1.0, 0.0]$
- **Patch 4 (Bottom-Right):** $\begin{bmatrix} 5.0 & 6.0 \\ 6.0 & 5.0 \end{bmatrix} \implies \mathbf{x}_p^4 = [5.0, 6.0, 6.0, 5.0]$

**Part 2: Linear Projection ($\mathbf{e}_i = \mathbf{x}_p^i E$):**
Notice that multiplying by $E$ computes $e_{i, 1} = 0.5(x_1 + x_2)$ and $e_{i, 2} = 0.5(x_3 + x_4)$:
- $\mathbf{e}_1 = [0.5(1+2), 0.5(2+1)] = [1.5, 1.5]$
- $\mathbf{e}_2 = [0.5(3+4), 0.5(4+3)] = [3.5, 3.5]$
- $\mathbf{e}_3 = [0.5(0+1), 0.5(1+0)] = [0.5, 0.5]$
- $\mathbf{e}_4 = [0.5(5+6), 0.5(6+5)] = [5.5, 5.5]$

**Part 3 & 4: Sequence Construction with `[CLS]` and Positional Embeddings:**
Prepend `[CLS]` at index 0 and add $E_{\text{pos}}$ row-wise:
- Row 0 (`[CLS]`): $[1.0, 1.0] + [0.1, 0.1] = [1.1, 1.1]$
- Row 1 (Patch 1): $[1.5, 1.5] + [0.2, 0.2] = [1.7, 1.7]$
- Row 2 (Patch 2): $[3.5, 3.5] + [0.3, 0.3] = [3.8, 3.8]$
- Row 3 (Patch 3): $[0.5, 0.5] + [0.4, 0.4] = [0.9, 0.9]$
- Row 4 (Patch 4): $[5.5, 5.5] + [0.5, 0.5] = [6.0, 6.0]$

$$\mathbf{z}_0 = \begin{bmatrix}
1.1 & 1.1 \\
1.7 & 1.7 \\
3.8 & 3.8 \\
0.9 & 0.9 \\
6.0 & 6.0
\end{bmatrix} \in \mathbb{R}^{5 \times 2}$$
</details>

---

### Problem 2: Self-Attention Computational Complexity & Quadratic Resolution Scaling (Medium)
In standard Vision Transformers, self-attention computes an attention affinity matrix of shape $(N+1) \times (N+1)$ for each head.
Consider a ViT model with patch size $P = 16$ and hidden dimension $D = 768$:

1. Compute the sequence length $N$ (excluding `[CLS]`) for three standard image input resolutions:
   - Resolution 1: $H = 224, W = 224$
   - Resolution 2: $H = 448, W = 448$
   - Resolution 3: $H = 896, W = 896$
2. For each resolution, compute the number of elements in the self-attention matrix ($N \times N$).
3. By what factor does the attention matrix memory increase when image resolution doubles from $224 \times 224$ to $448 \times 448$?
4. Why does standard ViT struggle with dense pixel-level vision tasks like semantic segmentation and object detection at high resolutions ($1024 \times 1024$ or larger)?

<details>
<summary>View Complete Solution</summary>

**Part 1: Sequence Length $N = \frac{HW}{P^2}$ ($P = 16, P^2 = 256$):**
- $224 \times 224$: $N = \frac{50,176}{256} = \mathbf{196 \text{ tokens}}$
- $448 \times 448$: $N = \frac{200,704}{256} = \mathbf{784 \text{ tokens}}$
- $896 \times 896$: $N = \frac{802,816}{256} = \mathbf{3,136 \text{ tokens}}$

**Part 2: Attention Matrix Elements ($N \times N$):**
- $224 \times 224$: $196 \times 196 = \mathbf{38,416 \text{ entries}}$
- $448 \times 448$: $784 \times 784 = \mathbf{614,656 \text{ entries}}$
- $896 \times 896$: $3,136 \times 3,136 = \mathbf{9,834,496 \text{ entries}}$

**Part 3: Growth Factor on Doubling Resolution:**
$$\text{Scaling Factor} = \frac{614,656}{38,416} = \mathbf{16\times}$$
When the spatial image dimensions double ($2\times$ in height and $2\times$ in width), the number of pixels quadruples ($4\times$).
Because self-attention complexity scales as $\mathcal{O}(N^2)$, the attention compute and memory scale as:
$$(4\times)^2 = \mathbf{16\times \text{ increase!}}$$

**Part 4: Dense Vision Implications:**
Dense tasks (e.g., object detection, instance segmentation) require high input resolutions ($1024 \times 1024$ or $2048 \times 1024$). At $1024 \times 1024$, $N = 4,096$ and $N^2 = 16,777,216$ elements per attention head per layer. This quadratic memory footprint causes GPU Out-Of-Memory (OOM) errors, motivating hierarchical architectures like **Swin Transformer** (which restricts attention to local shifted windows) and **Dilated Attention**.
</details>

---

### Problem 3: Hybrid ViT Architectures & The Inductive Bias Trade-Off (Hard)
In many practical production settings, engineers combine CNN backbones with Transformer encoders in a **Hybrid ViT** architecture (e.g. ResNet stem + ViT encoder).

1. In a Hybrid ViT, an image of size $224 \times 224 \times 3$ is first passed through a shallow CNN stem (3 residual stages) that outputs a feature volume of shape $[C=512, H=14, W=14]$. How many tokens $N$ does this produce when flattened directly without patch extraction?
2. Explain two key training advantages that the CNN stem provides over raw $16 \times 16$ linear patch projection when training from scratch on moderate datasets like ImageNet-1k (1.3 million images).
3. Contrast how early convolutional layers extract features compared to early self-attention layers in a standard ViT.
4. Why did pure ViTs ultimately prevail over hybrid architectures at foundation-model scale (e.g. CLIP, LLaVA, DINOv2)?

<details>
<summary>View Complete Solution</summary>

**Part 1: Hybrid Token Count:**
The feature map shape is $[512, 14, 14]$. Slicing along the spatial dimensions yields:
$$N = 14 \times 14 = \mathbf{196 \text{ tokens}}$$
Each token has feature dimension $D_{\text{in}} = 512$, which is then linearly projected to the Transformer dimension $D = 768$.

**Part 2: Training Advantages of CNN Stem:**
- **Sample Efficiency on Moderate Data:** The CNN stem applies hardcoded 2D translation equivariance and local spatial priors, preventing the network from wasting gradient updates learning basic pixel adjacencies and edge grouping.
- **Optimization Stability:** Early convolutions act as stable continuous feature extractors, mitigating the training instability and sensitivity to learning rate warmup often encountered with raw patch projection on smaller datasets.

**Part 3: Early Conv vs. Early Self-Attention Behavior:**
- Early CNN layers are mathematically restricted to rigid local windows (e.g. $3 \times 3$ or $7 \times 7$). They are forced to compute low-level derivatives (edges, corners, color gradients).
- Early ViT layers have global receptive fields: a query at patch $(0, 0)$ can attend to patch $(13, 13)$ in Layer 1. ViT empirically learns both local filters and broad global contextual interactions concurrently from the earliest blocks.

**Part 4: Foundation Scale Dominance of Pure ViT:**
At foundation-model scale (training on billions of multimodal image-text pairs, e.g. LAION-5B, JFT-4B):
- **Homogeneous Computation:** Pure Transformers use uniform GEMM operations ($Q K^T$, $A V$, MLP linear projections) across both vision and text modalities, unlocking extreme hardware optimization on TPUs and Hopper/Blackwell GPUs.
- **Architectural Bottleneck Removal:** When trillions of tokens of visual data are available, inductive biases become an artificial ceiling. Pure ViTs continue scaling monotonically with parameter count and compute without saturation.
</details>
