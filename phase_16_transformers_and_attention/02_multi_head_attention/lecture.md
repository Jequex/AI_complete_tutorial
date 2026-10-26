# Topic 16.2: Multi-Head Attention

---

## 1. Executive Summary & The 12-Year-Old Analogy

Single-head scaled dot-product attention has a fundamental limitation: **it can only focus on one pattern of relationships at any given layer**.

Consider the sentence:
> *"The animal didn't cross the street because it was too tired."*

To resolve what *"it"* refers to, the model needs to attend to *"animal"* (syntactic coreference).
Simultaneously, to know why, it must connect *"tired"* with *"animal"* (semantic predicate).
At the same time, it might need to track positional adjacency (*"was"* precedes *"too tired"*).

If we only have one attention distribution, the softmax must compute a single compromise:
$$A_{i, :} = \text{softmax}\left(\frac{\mathbf{q}_i K^T}{\sqrt{d_k}}\right)$$
By averaging distinct semantic, syntactic, and positional signals together into a single weighted sum, the distinct representations become a **blurry, diluted compromise**.

**Multi-Head Attention (MHA)** solves this by projecting Queries, Keys, and Values into $h$ distinct, lower-dimensional subspace representations, running $h$ attention calculations in parallel, concatenating the results, and projecting them back into the model dimension:

$$\boxed{\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O}$$
$$\text{where } \text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

```
                         MULTI-HEAD ATTENTION ARCHITECTURE
                         
                 Input Vectors: Q, K, V  [B, T, d_model]
                                    │
       ┌────────────────────────────┼────────────────────────────┐
       ▼                            ▼                            ▼
  Linear Projection            Linear Projection            Linear Projection
       W_1^Q, W_1^K, W_1^V          W_2^Q, W_2^K, W_2^V          W_h^Q, W_h^K, W_h^V
       │                            │                            │
       ▼                            ▼                            ▼
  Scaled Dot-Product           Scaled Dot-Product           Scaled Dot-Product
       Attention                    Attention                    Attention
       (Head 1)                     (Head 2)                     (Head h)
  [B, T, d_k]                  [B, T, d_k]                  [B, T, d_k]
       │                            │                            │
       └────────────────────────────┼────────────────────────────┘
                                    │
                                    ▼
                         Concatenate Heads
                         [B, T, h * d_k] = [B, T, d_model]
                                    │
                                    ▼
                         Linear Output Projection
                         W^O  [d_model, d_model]
                                    │
                                    ▼
                         Output Tensor [B, T, d_model]
```

---

### The 12-Year-Old Analogy: The Team of Specialized Detectives

Imagine a crime scene investigation where a mysterious event occurred at school:

1. **The Lone Detective (Single Head):**
   - If only one detective inspects the room, they have to look at footprints, fingerprints, broken windows, and listen to witness statements all at once.
   - Their brain gets scrambled trying to pay attention to everything simultaneously. They end up with a blurry, confusing report that misses critical clues.

2. **The Specialized Detective Squad (Multi-Head Attention):**
   - Instead, the police chief sends in a team of 4 specialized detectives ($h = 4$), each equipped with a different tool:
     - **Detective 1 (Grammar & Structure):** Has a magnifying glass looking only at who did what (subject-verb agreement).
     - **Detective 2 (Coreference & Identity):** Tracks pronouns (*"it"*, *"they"*, *"who"*) back to the character names.
     - **Detective 3 (Emotion & Tone):** Reads between the lines to detect anger, tiredness, or excitement.
     - **Detective 4 (Time & Sequence):** Constructs a strict chronological timeline of events.

3. **The Debriefing Conference (Head Concatenation):**
   - After investigating, all 4 detectives stand side-by-side at the whiteboard and tape their 4 separate reports together:
     $$\text{Concat}(\text{head}_1, \text{head}_2, \text{head}_3, \text{head}_4)$$

4. **The Chief's Final Synthesis (Output Projection $W^O$):**
   - The Police Chief ($W^O$) reads the 4 side-by-side reports, synthesizes the cross-correlations, and issues one master briefing that combines all four specialized perspectives into a unified understanding!

---

## 2. Mathematical Symbol Decoder

| Symbol | Tensor Dimension | Domain | Meaning & Description |
| :--- | :--- | :--- | :--- |
| $d_{\text{model}}$ | Scalar $\in \mathbb{N}$ | Hyperparameter | Token embedding dimension (e.g., 512 in original paper, 4096 in LLaMA) |
| $h$ | Scalar $\in \mathbb{N}$ | Hyperparameter | Number of parallel attention heads (e.g., 8, 32, 64) |
| $d_k$ | Scalar $\in \mathbb{N}$ | $d_{\text{model}} / h$ | Dimension of Keys and Queries per head (e.g., $512 / 8 = 64$) |
| $d_v$ | Scalar $\in \mathbb{N}$ | $d_{\text{model}} / h$ | Dimension of Values per head (typically equal to $d_k$) |
| $W_i^Q$ | $\mathbb{R}^{d_{\text{model}} \times d_k}$ | Parameter Matrix | Query projection weight matrix for head $i \in \{1, \dots, h\}$ |
| $W_i^K$ | $\mathbb{R}^{d_{\text{model}} \times d_k}$ | Parameter Matrix | Key projection weight matrix for head $i \in \{1, \dots, h\}$ |
| $W_i^V$ | $\mathbb{R}^{d_{\text{model}} \times d_v}$ | Parameter Matrix | Value projection weight matrix for head $i \in \{1, \dots, h\}$ |
| $W^O$ | $\mathbb{R}^{h d_v \times d_{\text{model}}}$ | Parameter Matrix | Output projection matrix merging all heads back to $d_{\text{model}}$ |
| $\text{head}_i$ | Tensor $\in \mathbb{R}^{B \times T \times d_v}$ | Activation Tensor | Output of scaled dot-product attention for head $i$ |
| $A^{(i)}$ | Tensor $\in [0, 1]^{B \times T \times T}$ | Attention Matrix | Distinct attention pattern learned by head $i$ |

---

## 3. Mathematical Formulation

### 3.1 Linear Projections per Head
Given an input sequence representation $X \in \mathbb{R}^{B \times T \times d_{\text{model}}}$ (in self-attention, $Q = K = V = X$):

For each head $i \in \{1, \dots, h\}$:
$$Q_i = Q W_i^Q \in \mathbb{R}^{B \times T \times d_k}$$
$$K_i = K W_i^K \in \mathbb{R}^{B \times T \times d_k}$$
$$V_i = V W_i^V \in \mathbb{R}^{B \times T \times d_v}$$

Notice that:
$$d_k = d_v = \frac{d_{\text{model}}}{h}$$

By setting $d_k = d_{\text{model}} / h$, the total computational cost of multi-head attention is **identical** to single-head attention with full dimensionality $d_{\text{model}}$! We gain multi-subspace diversity for free without increasing FLOPs:
$$\mathcal{O}(h \cdot T^2 \cdot d_k) = \mathcal{O}\left(h \cdot T^2 \cdot \frac{d_{\text{model}}}{h}\right) = \mathcal{O}(T^2 \cdot d_{\text{model}})$$

### 3.2 Subspace Attention
Each head computes its own scaled dot-product attention independently:
$$\text{head}_i = \text{Attention}(Q_i, K_i, V_i) = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i \in \mathbb{R}^{B \times T \times d_v}$$

### 3.3 Head Concatenation & Final Output Projection
The $h$ heads are concatenated along the feature dimension:
$$\mathbf{H} = \text{Concat}(\text{head}_1, \text{head}_2, \dots, \text{head}_h) \in \mathbb{R}^{B \times T \times (h \cdot d_v)}$$

Since $h \cdot d_v = h \cdot (d_{\text{model}} / h) = d_{\text{model}}$, $\mathbf{H}$ has shape $[B, T, d_{\text{model}}]$.

Finally, the concatenated tensor is linearly transformed by $W^O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$:
$$\text{MHA}(Q, K, V) = \mathbf{H} W^O \in \mathbb{R}^{B \times T \times d_{\text{model}}}$$

This final linear projection is critical: it allows the features discovered across the different specialized subspaces to interact and mix linearly.

---

## 4. Tensor Reshaping: Parallel Head Execution Without Loops

In production implementations (PyTorch, JAX, CUDA), we never execute a Python `for head in range(h):` loop. Looping introduces sequential kernel launch overhead and underutilizes GPU tensor cores.

Instead, we pack all $h$ projection matrices into a single monolithic weight matrix:
$$W^Q = \begin{bmatrix} W_1^Q & W_2^Q & \dots & W_h^Q \end{bmatrix} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$$
$$W^K = \begin{bmatrix} W_1^K & W_2^K & \dots & W_h^K \end{bmatrix} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$$
$$W^V = \begin{bmatrix} W_1^V & W_2^V & \dots & W_h^V \end{bmatrix} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$$

### The 4-Step Tensor Reshape Pipeline:

```
[B, T, d_model] 
      │  (1) Single Big Linear MatMul: Q = X @ W^Q
      ▼
[B, T, d_model] = [B, T, h * d_k]
      │  (2) Reshape / View: split d_model into [h, d_k]
      ▼
[B, T, h, d_k]
      │  (3) Transpose / Permute: swap axes 1 and 2
      ▼
[B, h, T, d_k]
      │  (4) Batched MatMul across all (B * h) heads simultaneously!
      ▼
Scores = Q @ K^T: [B, h, T, T]
Attn_Weights = softmax(Scores / sqrt(d_k)): [B, h, T, T]
Context = Attn_Weights @ V: [B, h, T, d_k]
      │  (5) Transpose back: swap axes 1 and 2
      ▼
[B, T, h, d_k]
      │  (6) Flatten last two dimensions: [B, T, h * d_k] = [B, T, d_model]
      ▼
[B, T, d_model]
      │  (7) Output projection: Out @ W^O
      ▼
[B, T, d_model]
```

By transposing from $[B, T, h, d_k]$ to $[B, h, T, d_k]$, the batch dimension and head dimension combine into an effective batch size of $B \times h$. A single batched BLAS/cuBLAS matrix multiplication computes all heads for all sequences in parallel in one GPU clock cycle!

---

## 5. Architectural Takeaways

1. **Representation Capacity**: Different heads can simultaneously represent syntactic, semantic, and positional relationships without destructive interference.
2. **Computational Invariance**: By choosing $d_k = d_{\text{model}} / h$, multi-head attention has the exact same theoretical FLOP count as a single full-dimensional attention head.
3. **Subspace Mixing ($W^O$)**: The output projection $W^O$ provides the parameter capacity to translate distinct subspace observations back into a coherent token representation.
