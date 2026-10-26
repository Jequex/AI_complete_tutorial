# Topic 16.4: Transformer Architecture & Causal Masking

---

## 1. Executive Summary & The 12-Year-Old Analogy

With Scaled Dot-Product Attention, Multi-Head Attention, and Positional Encodings in our mathematical toolkit, we can now assemble the complete **Transformer Architecture**.

Originally designed as an **Encoder-Decoder** sequence-to-sequence model for machine translation (Vaswani et al., 2017), the architecture has branched into three dominant families:
1. **Encoder-Only (BERT, RoBERTa)**: Bidirectional attention across all tokens; excels at classification, embeddings, and feature extraction.
2. **Encoder-Decoder (T5, BART)**: Bidirectional encoder coupled with an autoregressive decoder; excels at translation, summarization, and conditional generation.
3. **Decoder-Only (GPT-4, LLaMA 3, Claude, Gemini)**: Causal autoregressive attention; the universal foundation of modern Generative AI and Large Language Models.

The core challenge of generative autoregressive decoding is simple: **Tokens cannot look into the future!**
During generation, token $t$ is predicted solely based on prefix context $x_1, \dots, x_{t-1}$.
To train this efficiently in parallel on GPUs, we introduce **Causal Masking**.

```
                       THE MODERN PRE-LN TRANSFORMER DECODER BLOCK
                       
                             Input Tokens: x_t  [B, T, d_model]
                                          │
                        ┌─────────────────┴─────────────────┐ (Residual Stream)
                        │                                   │
                        ▼                                   │
                 Layer Normalization                        │
                        │                                   │
                        ▼                                   │
           Causal Masked Multi-Head Attention               │
            (Softmax with Upper-Triangle M_ij = -∞)         │
                        │                                   │
                        ▼                                   │
                 Dropout / Linear                           │
                        │                                   │
                        ▼                                   │
                        └───────────────> ( + ) <───────────┘ (First Residual Add)
                                           │
                        ┌──────────────────┴────────────────┐ (Residual Stream)
                        │                                   │
                        ▼                                   │
                 Layer Normalization                        │
                        │                                   │
                        ▼                                   │
              Position-wise Feed-Forward Network            │
             (MLP: d_model -> 4*d_model -> d_model)         │
                        │                                   │
                        ▼                                   │
                        └───────────────> ( + ) <───────────┘ (Second Residual Add)
                                           │
                                           ▼
                             Output Tensor [B, T, d_model]
```

---

### The 12-Year-Old Analogy: The Blindfolded Reader & The Time-Machine Curtain

Imagine you are reading a detective mystery novel, trying to guess what happens on page 50:

1. **The Time-Machine Curtain (Causal Masking):**
   - If someone lets you flip ahead to page 100, you will immediately see who the murderer is! That's cheating (information leakage).
   - A **Causal Mask** is a magic curtain that follows your eyes. When you are reading word 3, the curtain drops over words 4, 5, 6... making them completely invisible ($-\infty$).
   - You are only allowed to look backwards at what has already been said!

2. **The Residual Stream (The Super-Fast Express Subway):**
   - Deep inside the Transformer, there are 80 layers of detectives. If the original words had to travel through every single detective's complex notes, the original message would get distorted and forgotten.
   - The **Residual Connection** is an express subway track running right beside the detectives ($x + f(x)$).
   - If a detective has something smart to add, they hop onto the train and whisper it. If they have nothing useful to say, the train just speeds right past them completely unharmed!

3. **The Feed-Forward Brain (The Fact Memorizer):**
   - Attention is good at looking around and asking: *"How does 'she' connect to 'Alice'?"*
   - But the Feed-Forward Network (MLP) is the library of facts stored in memory: *"Paris is the capital of France"*, *"Water boils at 100°C"*.

---

## 2. Mathematical Symbol Decoder

| Symbol | Tensor Shape | Mathematical Domain | Description & Meaning |
| :--- | :--- | :--- | :--- |
| $T$ | Scalar $\in \mathbb{N}$ | Hyperparameter | Sequence length (context window) |
| $d_{\text{model}}$ | Scalar $\in \mathbb{N}$ | Hyperparameter | Token hidden dimension |
| $d_{\text{ff}}$ | Scalar $\in \mathbb{N}$ | $4 \times d_{\text{model}}$ | Hidden dimension of the feed-forward MLP layer |
| $M \in \mathbb{R}^{T \times T}$ | Square Matrix | Additive Mask | Causal attention mask matrix with $0$ and $-\infty$ entries |
| $\text{LN}(x)$ | Operator $\mathbb{R}^d \to \mathbb{R}^d$ | Normalization | Layer Normalization: $(x - \mu) / \sqrt{\sigma^2 + \epsilon} \odot \gamma + \beta$ |
| $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$ | Parameter Matrix | MLP Layer 1 | Expansion projection of the Feed-Forward Network |
| $W_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$ | Parameter Matrix | MLP Layer 2 | Contraction projection of the Feed-Forward Network |
| $S_{ij}$ | Scalar Real | Attention Logit | Scaled dot product score between position $i$ and position $j$ |
| $A_{ij}$ | Scalar $\in [0, 1]$ | Attention Probability | Attention weight: $A_{ij} \equiv 0$ for all $j > i$ under causal masking |

---

## 3. Causal Self-Attention Masking

In autoregressive language modeling, the joint probability of a sequence of tokens $(x_1, \dots, x_T)$ is factorized using the chain rule of probability:
$$P(x_1, x_2, \dots, x_T) = \prod_{t=1}^T P(x_t \mid x_1, x_2, \dots, x_{t-1})$$

To evaluate $P(x_t \mid x_{<t})$ for all $t \in \{1, \dots, T\}$ in a **single parallel forward pass** during training, we cannot allow the query at position $t$ to access keys or values at positions $t+1, \dots, T$.

### The Additive Mask Formulation
We define the causal mask matrix $M \in \mathbb{R}^{T \times T}$ as an upper-triangular matrix:

$$M_{ij} = \begin{cases} 0 & \text{if } j \le i \quad (\text{past or present token}) \\ -\infty & \text{if } j > i \quad (\text{future token}) \end{cases}$$

For a sequence of length $T = 4$:
$$M = \begin{bmatrix} 
0 & -\infty & -\infty & -\infty \\ 
0 & 0 & -\infty & -\infty \\ 
0 & 0 & 0 & -\infty \\ 
0 & 0 & 0 & 0 
\end{bmatrix}$$

We add $M$ directly to the scaled attention logits before applying softmax:
$$\boxed{\text{MaskedAttention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right) V}$$

### Softmax Behavior Under the Causal Mask
Consider row $i$ of the masked logits matrix:
$$Z_{ij} = \frac{\mathbf{q}_i^T \mathbf{k}_j}{\sqrt{d_k}} + M_{ij}$$

For any future token $j > i$:
$$Z_{ij} = \frac{\mathbf{q}_i^T \mathbf{k}_j}{\sqrt{d_k}} + (-\infty) = -\infty$$

Evaluating the softmax probability for $j > i$:
$$A_{ij} = \frac{\exp(Z_{ij})}{\sum_{k=1}^T \exp(Z_{ik})} = \frac{\exp(-\infty)}{\sum_{k=1}^i \exp(Z_{ik}) + 0} = \frac{0}{\sum_{k=1}^i \exp(Z_{ik})} \equiv 0.0$$

The attention weight placed on every future token is **identically zero**!
Furthermore, for all valid past tokens $j \le i$, the softmax probabilities automatically re-normalize so that they sum to exactly $1.0$:
$$\sum_{j=1}^i A_{ij} = 1.0$$

---

## 4. Full Block Anatomy: Pre-LN vs Post-LN

The original 2017 Transformer used **Post-LayerNorm**:
$$x^{(l+1)} = \text{LayerNorm}(x^{(l)} + \text{SubLayer}(x^{(l)}))$$

However, Post-LN suffered from severe gradient instability in deep networks, requiring fragile learning rate warmups. Modern architectures (GPT-3/4, LLaMA, PaLM, Mistral) universally use **Pre-LayerNorm**:

$$\boxed{x^{(l+1)} = x^{(l)} + \text{MHA}(\text{LayerNorm}_1(x^{(l)}))}$$
$$\boxed{x^{(l+2)} = x^{(l+1)} + \text{FFN}(\text{LayerNorm}_2(x^{(l+1)}))}$$

### Why Pre-LN Enables Gradient Preservation:
In Pre-LN, the identity path across $L$ layers is clean and unencumbered:
$$x^{(L)} = x^{(0)} + \sum_{l=0}^{L-1} \mathcal{F}_l(\text{LN}(x^{(l)}))$$

By the chain rule, the gradient of the loss $\mathcal{L}$ with respect to the input representation $x^{(0)}$ is:
$$\frac{\partial \mathcal{L}}{\partial x^{(0)}} = \frac{\partial \mathcal{L}}{\partial x^{(L)}} \left( I + \sum_{l=0}^{L-1} \frac{\partial \mathcal{F}_l}{\partial x^{(0)}} \right)$$

The identity matrix $I$ acts as an unobstructed highway that channels gradients directly back to the very first layer without any vanishing or exploding decay!

---

## 5. The Position-Wise Feed-Forward Network (FFN / MLP)

Between attention layers, every Transformer block includes a position-wise two-layer Multi-Layer Perceptron. It operates independently on each token position:

$$\boxed{\text{FFN}(\mathbf{x}) = \text{GELU}(\mathbf{x} W_1 + \mathbf{b}_1) W_2 + \mathbf{b}_2}$$

In standard Transformers:
- $W_1$ expands the feature dimension: $d_{\text{model}} \to d_{\text{ff}} = 4 \times d_{\text{model}}$.
- A non-linear activation (ReLU, GELU, or SwiGLU in modern LLMs) injects representational depth.
- $W_2$ projects back down: $d_{\text{ff}} \to d_{\text{model}}$.

**Functional Role**: While Attention routes and aggregates contextual information across tokens, the FFN stores factual associative memory (key-value factual pairs stored directly in the weight parameters).

---

## 6. Computational Complexity & Memory Bottlenecks

Let $T$ be sequence length and $d$ be $d_{\text{model}}$:

| Operation | Time Complexity (FLOPs) | Peak Memory Complexity | Bottleneck Nature |
| :--- | :--- | :--- | :--- |
| **Q, K, V Projections** | $\mathcal{O}(T \cdot d^2)$ | $\mathcal{O}(T \cdot d)$ | Compute-bound |
| **Attention Scores ($Q K^T$)** | $\mathcal{O}(T^2 \cdot d)$ | $\mathcal{O}(T^2)$ | Memory bandwidth bound |
| **Softmax & Masking** | $\mathcal{O}(T^2)$ | $\mathcal{O}(T^2)$ | Memory bandwidth bound |
| **Attention Context ($A V$)** | $\mathcal{O}(T^2 \cdot d)$ | $\mathcal{O}(T \cdot d)$ | Memory bandwidth bound |
| **Output Projection ($W^O$)** | $\mathcal{O}(T \cdot d^2)$ | $\mathcal{O}(T \cdot d)$ | Compute-bound |
| **Feed-Forward MLP** | $\mathcal{O}(T \cdot d \cdot d_{\text{ff}}) \approx \mathcal{O}(8 T d^2)$ | $\mathcal{O}(T \cdot d_{\text{ff}})$ | Compute-bound |

### The Quadratic Bottleneck:
The attention score matrix $A$ scales as $\mathcal{O}(T^2)$.
- For $T = 2,048$ tokens: $A$ requires $2048^2 \times 4\text{ bytes} \approx 16.7\text{ MB}$ per head.
- For $T = 128,000$ tokens (LLaMA 3 128k context): $A$ requires $128000^2 \times 2\text{ bytes} \approx 32.7\text{ GB}$ **per head**! Storing $A$ in GPU HBM is physically impossible.

This quadratic memory explosion is why modern LLMs use:
1. **FlashAttention (Dao et al.)**: Computes softmax tiling in fast SRAM without ever materializing the $T \times T$ matrix in GPU HBM.
2. **KV-Cache**: In autoregressive generation, caches previous keys and values $[K, V]$ so step $t$ computes attention in $\mathcal{O}(T)$ instead of recomputing all past tokens $\mathcal{O}(T^2)$.
