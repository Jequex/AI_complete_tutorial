# Topic 17.2: Causal Self-Attention & Pre-LN Transformer Blocks

---

## 1. Executive Summary & The 12-Year-Old Analogy

A standard bidirectional Transformer (such as BERT or Vision Transformers) allows every token in an input sequence to look forward and backward at every other token simultaneously. However, generative language models (such as GPT-2, GPT-3, LLaMA, and Claude) are **autoregressive**: their sole objective during pre-training and inference is to predict the next token given only the past tokens:
$$P(x_1, x_2, \dots, x_T) = \prod_{t=1}^T P(x_t \mid x_1, x_2, \dots, x_{t-1})$$

If token $t$ were allowed to attend to token $t+1$, the model would "cheat" by looking into the future rather than learning to predict it. The mechanism that strictly enforces this temporal arrow of time is **Causal Masking**.

```
                   CAUSAL SELF-ATTENTION MASKING
                   
   Raw Logits: Q K^T / sqrt(d_k)         Causal Additive Mask M         Softmax Attention Weights A
   ┌───────────────────────────┐         ┌────────────────────┐         ┌──────────────────────────┐
   │ S₀₀   S₀₁   S₀₂   S₀₃   S₀₄│         │  0   -∞   -∞   -∞   -∞│         │ 1.00  0.00  0.00  0.00  0.00 │
   │ S₁₀   S₁₁   S₁₂   S₁₃   S₁₄│    +    │  0    0   -∞   -∞   -∞│   ───>  │ 0.55  0.45  0.00  0.00  0.00 │
   │ S₂₀   S₂₁   S₂₂   S₂₃   S₂₄│         │  0    0    0   -∞   -∞│         │ 0.43  0.34  0.23  0.00  0.00 │
   │ S₃₀   S₃₁   S₃₂   S₃₃   S₃₄│         │  0    0    0    0   -∞│         │ 0.14  0.51  0.07  0.28  0.00 │
   │ S₄₀   S₄₁   S₄₂   S₄₃   S₄₄│         │  0    0    0    0    0│         │ 0.19  0.06  0.08  0.65  0.02 │
   └───────────────────────────┘         └────────────────────┘         └──────────────────────────┘
```

Surrounding causal self-attention is the modern **Pre-LayerNorm Transformer Block** (Radford et al., 2019):
1. **Pre-LayerNorm:** Normalizing inputs before entering the multi-head attention and MLP sub-layers, ensuring stable gradient backpropagation through hundreds of layers without warmup sensitivity.
2. **Residual Stream:** A wide direct highway where activations are updated additively: $\mathbf{x} \leftarrow \mathbf{x} + \Delta\mathbf{x}$.
3. **MLP Expansion:** A position-wise feed-forward network expanding hidden dimensions $4\times$ ($d_{\text{model}} \to 4 d_{\text{model}} \to d_{\text{model}}$) using smooth **GELU** non-linearities.

---

### The 12-Year-Old Analogy: The Time-Travel Blocking Shield & The Assembly Line

Imagine you and four classmates are playing an improvised storytelling game around a camp fire:
- Token 0: *"Once"*
- Token 1: *"upon"*
- Token 2: *"a"*
- Token 3: *"time"*
- Token 4: *"there"*

1. **The Time-Travel Blocking Shield (Causal Masking):**
   - When Person 1 is guessing what word comes after *"Once"*, they must only look at *"Once"*. They cannot look at the later words *"time"* or *"there"*.
   - We place a one-way mirrored glass shield in front of everyone. You can look at what has already been said behind you, but any word spoken in the future is covered by pitch-black tint ($-\infty$). When you compute who to listen to, you give zero attention to the future!

2. **The Residual Stream (The Conveyor Belt Highway):**
   - Imagine a conveyor belt moving through a car factory. The bare car chassis rides on the conveyor belt.
   - At Station 1 (Attention), robot arms look at what parts have been added so far, construct a side-mirror, and weld it onto the chassis ($\mathbf{x} + \text{Attention}(\text{LN}(\mathbf{x}))$).
   - At Station 2 (The MLP Engine Room), a four-times larger robotic workstation examines the car, shapes an engine piece, and welds it onto the chassis ($\mathbf{x} + \text{MLP}(\text{LN}(\mathbf{x}))$).
   - The original car chassis is never destroyed or completely replaced; every station just contributes a helpful additive tweak to the conveyor belt!

---

## 2. Mathematical Symbol Decoder

| Symbol | Mathematical Domain | Description & Meaning |
| :--- | :--- | :--- |
| $\mathbf{X}$ | Tensor $\mathbb{R}^{B \times T \times d_{\text{model}}}$ | Hidden activation tensor along the residual stream. |
| $B, T, d_{\text{model}}$ | Dimensions ($\mathbb{N}$) | Batch size, sequence length, and residual model feature dimension. |
| $H, d_k$ | Integers ($\mathbb{N}$) | Number of attention heads and per-head query/key dimension ($d_k = d_{\text{model}} / H$). |
| $Q, K, V$ | Projections $\mathbb{R}^{B \times H \times T \times d_k}$ | Query, Key, and Value tensors for each head. |
| $M$ | Matrix $\mathbb{R}^{T \times T}$ | Causal mask matrix where $M_{ij} = 0$ for $j \le i$, and $M_{ij} = -\infty$ for $j > i$. |
| $A$ | Matrix $\mathbb{R}^{T \times T}$ | Softmax attention weight matrix with strictly lower-triangular non-zero entries. |
| $\text{LN}(\mathbf{x})$ | Function $\mathbb{R}^D \to \mathbb{R}^D$ | Layer Normalization: $\frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \boldsymbol{\gamma} + \boldsymbol{\beta}$. |
| $\text{GELU}(x)$ | Activation Function | Gaussian Error Linear Unit: $x \, \Phi(x) = x \, P(X \le x), \; X \sim \mathcal{N}(0, 1)$. |
| $W_1, W_2$ | Weight Matrices | MLP projection weights: $W_1 \in \mathbb{R}^{d_{\text{model}} \times 4 d_{\text{model}}}$, $W_2 \in \mathbb{R}^{4 d_{\text{model}} \times d_{\text{model}}}$. |

---

## 3. Mathematical Formulation of Causal Self-Attention

### 3.1 Multi-Head Projection and Dot Products

Given input sequence $\mathbf{X} \in \mathbb{R}^{T \times d_{\text{model}}}$, linear projections produce query, key, and value vectors for head $h \in \{1, \dots, H\}$:
$$Q_h = \mathbf{X} W_q^{(h)}, \quad K_h = \mathbf{X} W_k^{(h)}, \quad V_h = \mathbf{X} W_v^{(h)}$$
where $W_q^{(h)}, W_k^{(h)}, W_v^{(h)} \in \mathbb{R}^{d_{\text{model}} \times d_k}$.

The raw attention logit between query token $i$ and key token $j$ is the scaled dot product:
$$S_{ij}^{(h)} = \frac{\mathbf{q}_i^{(h)} \cdot \mathbf{k}_j^{(h)}}{\sqrt{d_k}}$$

---

### 3.2 The Causal Mask Formulation

In autoregressive modeling, query $i$ must not access key $j$ whenever $j > i$. We enforce this by adding an upper-triangular negative infinity mask $M \in \mathbb{R}^{T \times T}$:
$$M_{ij} = \begin{cases} 0 & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases}$$

When added to the raw logits before softmax:
$$A_{ij}^{(h)} = \text{softmax}_j\left( S_{ij}^{(h)} + M_{ij} \right) = \frac{\exp(S_{ij}^{(h)} + M_{ij})}{\sum_{m=1}^T \exp(S_{im}^{(h)} + M_{im})}$$

Because $\exp(-\infty) = 0$:
$$\forall j > i: \quad A_{ij}^{(h)} = 0$$
$$\sum_{j=1}^i A_{ij}^{(h)} = 1$$

Each token's output context vector is a strictly past-dependent linear combination of values:
$$\mathbf{c}_i^{(h)} = \sum_{j=1}^i A_{ij}^{(h)} \mathbf{v}_j^{(h)}$$

Multi-head outputs are concatenated and linearly projected:
$$\text{Attention}(\mathbf{X}) = \left[ \mathbf{c}^{(1)} \mathbin{\Vert} \mathbf{c}^{(2)} \mathbin{\Vert} \dots \mathbin{\Vert} \mathbf{c}^{(H)} \right] W_o$$

---

## 4. Pre-LayerNorm Transformer Block Architecture

Original Vaswani et al. (2017) Transformers used **Post-LN**:
$$\mathbf{x}_{l} = \text{LN}(\mathbf{x}_{l-1} + \text{SubLayer}(\mathbf{x}_{l-1}))$$
In deep networks ($L > 12$), Post-LN places LayerNorm directly on the residual path, degrading gradient flow and requiring careful warmups.

GPT-2, GPT-3, and modern LLMs use **Pre-LN** (Radford et al., 2019):
$$\mathbf{x}^{(1)} = \mathbf{x} + \text{CausalSelfAttention}(\text{LN}_1(\mathbf{x}))$$
$$\mathbf{x}_{\text{out}} = \mathbf{x}^{(1)} + \text{MLP}(\text{LN}_2(\mathbf{x}^{(1)}))$$

```
                         PRE-LN TRANSFORMER BLOCK
                         
                    Input x  [B, T, d_model]
                       │
             ┌─────────┴─────────┐
             │                   │
             │              LayerNorm 1
             │                   │
             │           Causal Self-Attention
             │                   │
             └─────────(+)───────┘  Residual Stream 1
                        │
             ┌──────────┴────────┐
             │                   │
             │              LayerNorm 2
             │                   │
             │              MLP (4x Expansion + GELU)
             │                   │
             └──────────(+)──────┘  Residual Stream 2
                        │
                   Output x_out [B, T, d_model]
```

---

## 5. The Position-Wise Feed-Forward Network (MLP)

The MLP acts on each token vector independently:
$$\text{MLP}(\mathbf{z}) = \text{GELU}(\mathbf{z} W_1 + \mathbf{b}_1) W_2 + \mathbf{b}_2$$
where:
- $W_1 \in \mathbb{R}^{d_{\text{model}} \times 4 d_{\text{model}}}$ expands feature dimensionality by $4\times$.
- $W_2 \in \mathbb{R}^{4 d_{\text{model}} \times d_{\text{model}}}$ projects back to the residual stream dimension.

### The GELU Non-Linearity
Unlike ReLU ($\max(0, x)$), which has a sharp derivative discontinuity at $x=0$, GELU (Hendrycks & Gimpel, 2016) smoothly gates activations by their probability under a Gaussian distribution:
$$\text{GELU}(x) = x \, \Phi(x) = x \, P(Z \le x), \quad Z \sim \mathcal{N}(0, 1)$$
Approximated via tanh:
$$\text{GELU}(x) \approx 0.5 x \left( 1 + \tanh\left(\sqrt{\frac{2}{\pi}} \left(x + 0.044715 x^3\right)\right) \right)$$

---

## 6. Summary & Key Takeaways

1. **Causality is Mandatory for Generation:** An autoregressive model must not attend to future tokens.
2. **Masking via Additive Infinity:** Adding $-\infty$ before softmax sets future attention weights strictly to 0.0.
3. **Pre-LN Protects the Gradient Highway:** Residual updates add smoothly to $\mathbf{x}$, while LayerNorm conditions each sub-layer.
4. **4x MLP Expansion Stores Knowledge:** The wide intermediate hidden layer acts as associative memory for factual information.
