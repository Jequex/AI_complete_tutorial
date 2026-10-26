# Topic 14.4: Vision Transformers (ViT)

---

## 1. Executive Summary & The 12-Year-Old Analogy

For nearly a decade following AlexNet, convolutional neural networks reigned supreme as the undisputed architecture of computer vision. Convolutions were believed to be biologically indispensable for vision because of their hardcoded **inductive biases**: local connectivity and translation equivariance.

In 2020, Alexey Dosovitskiy and the Google Brain team challenged this dogma with a groundbreaking paper: *"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"*. They proved that by treating an image not as a 2D pixel grid, but as a **sequence of flattened patches** (analogous to words in NLP), a standard Transformer encoder with **Multi-Head Self-Attention** could match and surpass state-of-the-art CNNs.

The Vision Transformer (ViT) represents a fundamental paradigm shift: moving away from handcrafted convolutional operators toward a **general-purpose, scale-hungry architecture** with minimal visual inductive bias.

```
                    THE VISION TRANSFORMER (ViT) ARCHITECTURE
                    
     Input Image                     Patch Extraction & Linear Projection
    ┌──────────┐                     ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐
    │          │  P=16x16            │x_p^1 │  │x_p^2 │  │x_p^3 │  │x_p^4 │ ... [N patches]
    │ 224x224  │ ───────────>        └──────┘  └──────┘  └──────┘  └──────┘
    │  RGB     │                        │         │         │         │   Linear E
    └──────────┘                        ▼         ▼         ▼         ▼
                                     ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐
                                     │ x₁ E │  │ x₂ E │  │ x₃ E │  │ x₄ E │ ...
                                     └──────┘  └──────┘  └──────┘  └──────┘
                                        ▲         ▲         ▲         ▲
                     ┌──────┐           │         │         │         │
    Prepend [CLS] ──>│x_cls │───────────┴─────────┴─────────┴─────────┴─── (+) Pos Embeddings
                     └──────┘
                                                  │
                                                  ▼
                                     ┌─────────────────────────┐
                                     │  Transformer Encoder    │  x L blocks
                                     │   (Pre-LN, MSA, MLP)    │
                                     └─────────────────────────┘
                                                  │
                                                  ▼
                                     Extract z_L^0 ([CLS] representation)
                                                  │
                                                  ▼
                                          [ MLP Head / Logits ]
```

---

### The 12-Year-Old Analogy: Turning a Picture Puzzle into a Book of Sentences

Imagine you have a giant picture of a dragon, but your friend only knows how to read books, not look at pictures:

1. **Cutting the Picture into Puzzle Pieces (Patch Extraction):**
   - You take a pair of scissors and slice the dragon picture into a grid of 16 square cardboard tiles.
   - Tile 1 is the tip of the tail, Tile 7 is the dragon's wing, and Tile 16 is its breathing fire.

2. **Describing Each Tile in a Secret Code (Linear Projection):**
   - For each tile, you write down an 8-number secret code that describes all the colors and shapes inside that tile. Now each tile looks like a "word".

3. **Numbering the Tiles (Positional Embeddings):**
   - If you shook the puzzle pieces in a bag, you would lose track of where they go!
   - So you stamp a number on the back of each piece: piece #1 belongs in the top-left, piece #16 belongs in the bottom-right.

4. **The Special Master Detective Tile (The `[CLS]` Token):**
   - Before handing the puzzle pieces to your friend, you add a blank golden tile at the very front (#0).
   - The golden tile is the **Master Detective**. Its only job is to talk to all the other tiles, ask them what they are showing, and synthesize the final answer: *"This is a fire-breathing dragon!"*

5. **Global Attention (Every Tile Talks to Every Other Tile):**
   - In a CNN, Tile 1 could only whisper to its immediate neighbor Tile 2.
   - In a Vision Transformer, the eye tile can instantly look straight across the entire table at the claw tile on the opposite side! That is **Global Self-Attention**.

---

## 2. Mathematical Symbol Decoder

| Symbol | Type / Domain | Mathematical Meaning |
| :--- | :--- | :--- |
| $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$ | 3D Tensor | Original input image with height $H$, width $W$, and $C$ channels |
| $P \in \mathbb{Z}_{\ge 1}$ | Integer | Spatial size of square patch ($P \times P$ pixels, typically $16$ or $14$) |
| $N \in \mathbb{Z}_{\ge 1}$ | Integer | Number of non-overlapping patches: $N = \frac{HW}{P^2}$ |
| $\mathbf{x}_p^i \in \mathbb{R}^{P^2 C}$ | 1D Vector | Flattened pixel vector of the $i$-th spatial patch |
| $D \in \mathbb{Z}_{\ge 1}$ | Integer | Hidden representation dimension of the Transformer (e.g. 768 for ViT-Base) |
| $E \in \mathbb{R}^{(P^2 C) \times D}$ | Matrix | Learnable patch projection matrix mapping flattened patches to dimension $D$ |
| $\mathbf{x}_{\text{cls}} \in \mathbb{R}^D$ | 1D Vector | Learnable classification token prepended at index 0 |
| $E_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$ | 2D Matrix | 1D learnable spatial positional embedding matrix |
| $\mathbf{z}_0 \in \mathbb{R}^{(N+1) \times D}$ | 2D Matrix | Initial token sequence fed into the first Transformer encoder layer |
| $\text{MSA}(\cdot)$ | Operator | Multi-Head Self-Attention block |
| $\text{LN}(\cdot)$ | Operator | Layer Normalization applied across hidden dimension $D$ |
| $L \in \mathbb{Z}_{\ge 1}$ | Integer | Total number of stacked Transformer encoder blocks |
| $\mathbf{z}_L^0 \in \mathbb{R}^D$ | 1D Vector | Final state of the `[CLS]` token after layer $L$ |

---

## 3. Patch Extraction and Linear Projection

Standard Transformers were designed for 1D sequences of discrete language tokens $\mathbf{x} \in \mathbb{R}^{T \times D}$. To handle 2D images, ViT flattens spatial patches into a sequence of continuous vector embeddings.

### 3.1 The Patch Slicing Equation
An image $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$ is partitioned into $N$ non-overlapping patches $\mathbf{x}_p \in \mathbb{R}^{N \times (P^2 C)}$, where:
$$N = \frac{H \cdot W}{P^2}$$

For standard ImageNet input ($H = 224, W = 224, C = 3$) with patch size $P = 16$:
$$N = \frac{224 \times 224}{16 \times 16} = \frac{50,176}{256} = 196 \text{ patches}$$
Each flattened patch vector has dimension:
$$P^2 C = 16 \times 16 \times 3 = 768 \text{ values}$$

### 3.2 Linear Projection
Each flattened patch $\mathbf{x}_p^i \in \mathbb{R}^{1 \times (P^2 C)}$ is multiplied by a learnable projection matrix $E \in \mathbb{R}^{(P^2 C) \times D}$:
$$\mathbf{e}_i = \mathbf{x}_p^i E \in \mathbb{R}^{1 \times D}$$

*Implementation Note:* In PyTorch, patch extraction and projection are implemented in a single step using a 2D convolution:
```python
# Conv2d with kernel_size=P and stride=P is mathematically identical to patch projection!
self.patch_embed = nn.Conv2d(in_channels=3, out_channels=D, kernel_size=P, stride=P)
```

---

## 4. The `[CLS]` Token and Positional Embeddings

### 4.1 The Learnable `[CLS]` Token
Following BERT, ViT prepends a learnable parameter vector $\mathbf{x}_{\text{cls}} \in \mathbb{R}^{1 \times D}$ to the sequence of projected patch embeddings:
$$\mathbf{z}_0 = \left[ \mathbf{x}_{\text{cls}}; \, \mathbf{x}_p^1 E; \, \mathbf{x}_p^2 E; \, \dots; \, \mathbf{x}_p^N E \right] \in \mathbb{R}^{(N + 1) \times D}$$

**Why use a `[CLS]` token instead of averaging all patch representations?**
- In self-attention, all tokens attend to all other tokens symmetrically. If we used global average pooling across patch tokens, the network might bias its representation toward high-contrast background patches.
- The `[CLS]` token has no spatial patch origin; it acts as an unbiased, dedicated global accumulator whose final representation $\mathbf{z}_L^0$ captures the global semantic label of the entire image.

### 4.2 1D Learnable Positional Embeddings
Because self-attention is **permutation invariant** (it operates on a set of tokens with no inherent knowledge of order), the model would have no way of knowing whether a patch came from the top-left or bottom-right corner!

To inject spatial layout, a 1D learnable positional embedding matrix $E_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$ is added element-wise:
$$\mathbf{z}_0 = \left[ \mathbf{x}_{\text{cls}}; \, \mathbf{x}_p^1 E; \, \mathbf{x}_p^2 E; \, \dots; \, \mathbf{x}_p^N E \right] + E_{\text{pos}}$$

*Surprising Empirical Result:* Dosovitskiy et al. compared 1D learnable embeddings, 2D sinusoidal embeddings, and relative positional embeddings. They discovered that simple 1D learnable embeddings performed equally well: during pretraining, the 1D embeddings naturally learn 2D spatial relationships (nearby patches have higher cosine similarity in embedding space).

---

## 5. Transformer Encoder Block: Pre-LN Architecture

ViT employs the standard **Pre-LayerNorm** Transformer encoder block (Wang et al., 2019):

```
                     PRE-LN TRANSFORMER ENCODER BLOCK
                     
                                 Input z_{l-1}
                                       │
                      ┌────────────────┼────────────────┐
                      │                │ Shortcut       │
                      │                ▼                │
                      │            LayerNorm            │
                      │                ▼                │
                      │      Multi-Head Attention       │
                      │                ▼                │
                      └──────────────>(+)               │
                                       │                │
                      ┌────────────────┼────────────────┐
                      │                │ Shortcut       │
                      │                ▼                │
                      │            LayerNorm            │
                      │                ▼                │
                      │           MLP (GELU)            │
                      │                ▼                │
                      └──────────────>(+)               │
                                       │
                                       ▼
                                  Output z_l
```

### 5.1 Mathematical Equations for Block $l$
$$\mathbf{z}'_l = \text{MSA}(\text{LN}(\mathbf{z}_{l-1})) + \mathbf{z}_{l-1}, \qquad l = 1, \dots, L$$
$$\mathbf{z}_l = \text{MLP}(\text{LN}(\mathbf{z}'_l)) + \mathbf{z}'_l, \qquad l = 1, \dots, L$$

Where:
1. **Multi-Head Self-Attention (MSA):**
   $$Q = Z W_Q, \quad K = Z W_K, \quad V = Z W_V$$
   $$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) V$$
   Where $d_k = D / h$ for $h$ attention heads.
2. **MLP Block:**
   Consists of two linear layers with an expansion ratio of 4 ($D \to 4D \to D$) and GELU activation:
   $$\text{MLP}(\mathbf{u}) = \text{GELU}(\mathbf{u} W_1 + \mathbf{b}_1) W_2 + \mathbf{b}_2$$
3. **Classification Head:**
   The final prediction is extracted exclusively from the `[CLS]` token:
   $$\mathbf{y} = \text{Linear}(\text{LN}(\mathbf{z}_L^0))$$

---

## 6. Inductive Bias: CNNs vs. Vision Transformers

The debate between CNNs and ViTs centers on the concept of **inductive bias**: the assumptions an algorithm makes to predict outputs on unseen data.

| Property | Convolutional Neural Network (CNN) | Vision Transformer (ViT) |
| :--- | :--- | :--- |
| **Locality** | Hardcoded by small local receptive fields ($3 \times 3$). | Non-existent. Any token can attend to any other token from layer 1. |
| **Translation Equivariance** | Strictly enforced by spatial weight sharing: $f(T(x)) = T(f(x))$. | Must be learned from data through positional embeddings and attention. |
| **Receptive Field Growth** | Grows linearly with depth: $RF \sim \mathcal{O}(L \cdot K)$. | Global from the very first layer: $RF = \text{Full Image}$ at Layer 1. |
| **Data Efficiency** | High on small datasets (ImageNet-1k) due to strong architectural prior. | Lower on small datasets; prone to overfitting without massive pretraining. |
| **Capacity & Scaling Ceiling** | Often saturates at ultra-large scale due to rigid locality constraints. | Scales monotonically with compute and dataset size (JFT-300M, ImageNet-21k). |

```
                 PERFORMANCE VS. PRETRAINING DATASET SIZE
                 
       Top-1 Accuracy (%)
          │                             ViT-Huge (Scales effortlessly)
       88 │                                      ▲
          │                                     ╱
       84 │                 ResNet-152         ╱
          │               ───────────────     ╱
       80 │              ╱               ───-
          │             ╱
       76 │  ViT-Base  ╱
          │  (Struggles on small data)
          └──────────────────────────────────────────────
               ImageNet-1k         ImageNet-21k      JFT-300M
               (1.3M images)       (14M images)     (300M images)
```

### Key Takeaway on Inductive Bias
When data is scarce (e.g. 50,000 images), the strong inductive bias of a CNN acts as a valuable guide rail. But when training on tens of millions of images, rigid inductive biases become a handicap that limits model capacity. ViT removes these guide rails, allowing the model to learn flexible, global visual representations.

---

## 7. Model Configurations: ViT-Base, Large, and Huge

| Model | Layers ($L$) | Hidden Dimension ($D$) | MLP Size ($4D$) | Heads ($h$) | Parameters |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ViT-Base (ViT-B/16)** | 12 | 768 | 3072 | 12 | 86 Million |
| **ViT-Large (ViT-L/16)** | 24 | 1024 | 4096 | 16 | 307 Million |
| **ViT-Huge (ViT-H/14)** | 32 | 1280 | 5120 | 16 | 632 Million |

*Notation:* `ViT-B/16` denotes ViT-Base architecture using patch size $P = 16 \times 16$.
