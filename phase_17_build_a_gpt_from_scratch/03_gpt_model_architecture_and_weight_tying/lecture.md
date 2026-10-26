# Topic 17.3: GPT Model Architecture & Weight Tying

---

## 1. Executive Summary & The 12-Year-Old Analogy

The full Generative Pre-trained Transformer (GPT) architecture (Radford et al., 2018, 2019) unites the tokenization layer, embedding matrices, repeated causal Transformer blocks, and a final classification head into a singular, end-to-end differentiable autoregressive language model.

Given a sequence of token IDs $\mathbf{x} = (t_1, t_2, \dots, t_T)$, the model estimates the conditional next-token probability distribution:
$$P(t_{k+1} = v \mid t_1, \dots, t_k) = \text{softmax}(\mathbf{h}_k W_{\text{head}})_v$$

```
                         THE FULL GPT PIPELINE
                         
  Input Text: "AI is" ──> Tokenizer ──> Token IDs [65, 318]
                                                │
       ┌────────────────────────────────────────┴────────────────────────────────────────┐
       │                                                                                 │
       ▼                                                                                 ▼
Token Embeddings W_token [V, d_model]                              Position Embeddings W_pos [T_max, d_model]
       │                                                                                 │
       └────────────────────────────────────────(+)──────────────────────────────────────┘
                                                 │
                                           Hidden State x_0
                                                 │
                                                 ▼
                                     ┌───────────────────────┐
                                     │  Transformer Block 1  │
                                     ├───────────────────────┤
                                     │  Transformer Block 2  │
                                     ├───────────────────────┤
                                     │         ...           │
                                     ├───────────────────────┤
                                     │  Transformer Block N  │
                                     └───────────────────────┘
                                                 │
                                           Final LayerNorm
                                                 │
                                                 ▼
                                     Output LM Head (W_head)
                                   (Weight Tied: W_head = W_token^T)
                                                 │
                                                 ▼
                                    Logits over Vocab [B, T, V]
                                                 │
                                                 ▼
                                           Next-Token Probabilities
```

A cornerstone of modern parameter efficiency is **Weight Tying** (Press & Wolf, 2017): the input token embedding matrix $W_{\text{token}} \in \mathbb{R}^{V \times d_{\text{model}}}$ and the final projection head $W_{\text{head}} \in \mathbb{R}^{d_{\text{model}} \times V}$ are bound together as transposes:
$$W_{\text{head}} = W_{\text{token}}^T$$
In GPT-2 Small, this eliminates over **38.6 million redundant parameters** ($\approx 24\%$ of the total network) while compelling the model to align its input token semantics with its output probability predictions.

---

### The 12-Year-Old Analogy: The Master Printing Press with Reusable Type Blocks

Imagine a master printing press in 1450 making books:

1. **The Lead Letter Blocks (The Token & Position Embeddings):**
   - You have a giant wooden tray holding 50,257 lead character blocks ($W_{\text{token}}$).
   - You also have a shelf of 1,024 position stamps indicating which word slot on the page you are on ($W_{\text{pos}}$).
   - For every slot, you take the word stamp and position stamp, press them into soft wax, and create a 768-dimensional sculpture representing that specific word at that specific moment.

2. **The Factory of Editors (The Transformer Blocks):**
   - The wax sculptures roll down an assembly line of 12 distinct editing rooms.
   - In each room, an editor compares the current word to all preceding words, polishes the meaning, and passes it forward.

3. **The Master Stamp Recycling Trick (Weight Tying):**
   - At the final exit of the factory, we need to decide which of the 50,257 words to print next.
   - **The Wasteful Way:** Carve a brand new set of 50,257 giant copper stamps ($W_{\text{head}}$). That requires storing another 38 million pieces of heavy metal!
   - **The Master Printer's Secret (Weight Tying):** Re-use the exact same lead blocks from the entrance! If a block meant *"cat"* when entering the factory, touching the final sculpture with that same lead block tells you exactly how closely the final meaning matches *"cat"*. One tray of blocks does both jobs!

---

## 2. Mathematical Symbol Decoder

| Symbol | Mathematical Domain | Description & Meaning |
| :--- | :--- | :--- |
| $V$ | Integer ($\mathbb{N}$) | Vocabulary size (e.g., $50{,}257$ in GPT-2). |
| $T_{\max}$ | Integer ($\mathbb{N}$) | Maximum sequence context length (e.g., $1024$ in GPT-2). |
| $d_{\text{model}}$ | Integer ($\mathbb{N}$) | Hidden representation dimension ($768$ in GPT-2 Small). |
| $N$ | Integer ($\mathbb{N}$) | Number of stacked Transformer layers ($12$ in GPT-2 Small). |
| $W_{\text{token}}$ | Matrix $\mathbb{R}^{V \times d_{\text{model}}}$ | Learned token embedding lookup matrix (`wte`). |
| $W_{\text{pos}}$ | Matrix $\mathbb{R}^{T_{\max} \times d_{\text{model}}}$ | Learned positional embedding lookup matrix (`wpe`). |
| $W_{\text{head}}$ | Matrix $\mathbb{R}^{d_{\text{model}} \times V}$ | Final linear projection mapping hidden state to vocabulary logits. |
| $\mathbf{h}_t^{(L)}$ | Vector $\mathbb{R}^{d_{\text{model}}}$ | Final hidden state vector at sequence position $t$ after layer $L$. |
| $\mathbf{z}_t$ | Vector $\mathbb{R}^V$ | Unnormalized logit vector for next-token predictions at position $t$. |
| $\mathcal{L}_{\text{CE}}$ | Scalar ($\mathbb{R}$) | Cross-entropy loss averaged across all batch tokens. |

---

## 3. Detailed Component Architecture

### 3.1 Input Embedding Fusion
Given integer token indices $\mathbf{x} = [x_1, x_2, \dots, x_T]$ with $x_t \in \{0, \dots, V-1\}$:
1. **Token Embedding:**
   $$\mathbf{e}_t^{\text{token}} = W_{\text{token}}[x_t, :] \in \mathbb{R}^{d_{\text{model}}}$$
2. **Positional Embedding:** Unlike original Transformers which used fixed sinusoidal functions ($\sin(t / 10000^{2i/d})$), GPT uses **learned absolute positional embeddings**:
   $$\mathbf{e}_t^{\text{pos}} = W_{\text{pos}}[t, :] \in \mathbb{R}^{d_{\text{model}}}$$
3. **Element-wise Summation:**
   $$\mathbf{h}_t^{(0)} = \mathbf{e}_t^{\text{token}} + \mathbf{e}_t^{\text{pos}}$$

---

### 3.2 Transformer Block Stacking
The fused representation $\mathbf{H}^{(0)} \in \mathbb{R}^{B \times T \times d_{\text{model}}}$ passes sequentially through $N$ identical Pre-LN Transformer blocks:
$$\mathbf{H}^{(l)} = \text{TransformerBlock}_l(\mathbf{H}^{(l-1)}), \quad l = 1, 2, \dots, N$$

Each block preserves exact tensor dimensions $[B, T, d_{\text{model}}]$.

---

### 3.3 Final Normalization and Projection
After layer $N$, a final LayerNorm stabilizes the residual stream:
$$\mathbf{H}^{(\text{final})} = \text{LayerNorm}_f(\mathbf{H}^{(N)})$$

The logits vector $\mathbf{z}_t \in \mathbb{R}^V$ for each sequence position $t$ is computed via matrix multiplication:
$$\mathbf{z}_t = \mathbf{h}_t^{(\text{final})} W_{\text{head}}$$

---

## 4. The Mathematics of Weight Tying

In standard deep networks, input embedding and output classification are treated as independent parameters:
- Input embedding: $W_{\text{token}} \in \mathbb{R}^{V \times d_{\text{model}}}$ maps token index $v \to \mathbf{w}_v \in \mathbb{R}^{d_{\text{model}}}$.
- Output projection: $W_{\text{head}} \in \mathbb{R}^{d_{\text{model}} \times V}$ maps hidden vector $\mathbf{h} \to z_v = \mathbf{h} \cdot \mathbf{u}_v$.

**Weight Tying (Press & Wolf, 2017)** sets $\mathbf{u}_v \equiv \mathbf{w}_v$, which is equivalent to:
$$W_{\text{head}} = W_{\text{token}}^T$$

The logit for token $v$ is the inner product:
$$z_v = \mathbf{h}_t^{(\text{final})} \cdot W_{\text{token}}[v, :]^T$$

### Advantages of Weight Tying:
1. **Geometric Representation Alignment:** Forces input semantic vector space and output prediction vector space to share the same metric manifold. If two tokens are synonyms, their vectors are close both when being read and when being predicted.
2. **Huge Parameter Reduction:** Saves $V \times d_{\text{model}}$ parameters:
   - For GPT-2 Small ($V=50{,}257, d=768$): saves $38{,}597{,}376$ parameters ($\approx 38.6\text{M}$).
   - For LLaMA ($V=128{,}000, d=4096$): saves $524{,}288{,}000$ parameters ($\approx 524\text{M}$).
3. **Regularization:** Prevents the output matrix from overfitting on rare tokens.

---

## 5. Exact Parameter Counting: The Closed-Form Formula

Let us derive the exact closed-form parameter formula for any GPT architecture defined by:
- $V$: vocabulary size
- $T_{\max}$: maximum context length
- $L$: number of layers
- $d$: hidden dimension $d_{\text{model}}$
- Tied vs untied flag

### 1. Embeddings:
$$\text{Params}_{\text{wte}} = V \cdot d$$
$$\text{Params}_{\text{wpe}} = T_{\max} \cdot d$$

### 2. Single Transformer Block:
- **Attention:**
  - $W_q, W_k, W_v, W_o$: $4 \times (d \times d) = 4 d^2$
  - Biases: $4 \times d = 4d$
- **MLP:**
  - $W_1$: $d \times 4d = 4 d^2$, bias $\mathbf{b}_1 = 4d$
  - $W_2$: $4d \times d = 4 d^2$, bias $\mathbf{b}_2 = d$
  - Total MLP: $8 d^2 + 5d$
- **LayerNorms:**
  - 2 LayerNorms per block, each with $\boldsymbol{\gamma}$ and $\boldsymbol{\beta}$: $2 \times (2d) = 4d$
$$\text{Params}_{\text{block}} = (4d^2 + 4d) + (8d^2 + 5d) + 4d = 12 d^2 + 13 d$$

### 3. Final LayerNorm & LM Head:
$$\text{Params}_{\text{ln\_f}} = 2d$$
$$\text{Params}_{\text{lm\_head}} = \begin{cases} 0 & \text{if tied} \\ V \cdot d & \text{if untied} \end{cases}$$

### Total Parameters Formula:
$$\boxed{\text{Params}_{\text{Total}} = (V + T_{\max})d + L(12 d^2 + 13 d) + 2d + \mathbb{I}[\text{untied}] \cdot V d}$$

---

### The Official GPT-2 Family Parameters:

| Model | Layers ($L$) | $d_{\text{model}}$ | Heads ($H$) | Tied Params | Untied Params | Common Shorthand |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **GPT-2 Small** | 12 | 768 | 12 | **124,439,808** | 163,037,184 | **124M** |
| **GPT-2 Medium** | 24 | 1024 | 16 | **354,823,168** | 406,286,336 | **355M** |
| **GPT-2 Large** | 36 | 1280 | 20 | **774,030,080** | 838,359,040 | **774M** |
| **GPT-2 XL** | 48 | 1600 | 25 | **1,557,611,200** | 1,638,022,400 | **1.5B** |

---

## 6. Summary & Key Takeaways

1. **Clean Modular Hierarchy:** A GPT model is simply an embedding table, followed by $N$ identical Pre-LN Transformer blocks, a final LayerNorm, and a projection head.
2. **Weight Tying is Standard:** Reusing $W_{\text{token}}$ as $W_{\text{head}}^T$ saves millions of parameters and unifies token representation geometry.
3. **Quadratic Dependence on Depth & Width:** Block parameters scale as $12 \cdot L \cdot d^2$. Doubling the hidden dimension quadruples the block parameters!
