# Topic 19.4: PEFT Adapters, Prompt Tuning, and Prefix Tuning

---

## 1. Executive Summary & The 12-Year-Old Analogy

While LoRA is the most celebrated Parameter-Efficient Fine-Tuning (PEFT) technique today, parameter-efficient adaptation spans an entire family of algorithmic paradigms. Before LoRA's emergence in late 2021, the AI research community explored two primary alternatives:
1. **Bottleneck Adapters (Houlsby et al. 2019, He et al. 2021)**: Injecting compact, low-dimensional Multi-Layer Perceptrons (MLPs) sequentially or in parallel with existing Transformer layers.
2. **Soft Continuous Prompting (Prompt Tuning & Prefix Tuning)**: Prepending trainable virtual token vectors to input sequences or intermediate attention Key-Value states, steering pre-trained models without modifying a single weight matrix.

Understanding the structural trade-offs between **weight reparameterization (LoRA)**, **modular sub-layer insertion (Adapters)**, and **continuous activation steering (Prompt & Prefix Tuning)** is essential for modern AI systems architects designing multi-tenant serving engines, edge devices, and memory-constrained training pipelines.

```
                   THE SPECTRUM OF PEFT ARCHITECTURES
                   
    INPUT-LEVEL               KV-STATE ATTENTION              INTERNAL LAYERS
  (Prompt Tuning)              (Prefix Tuning)             (Adapters & LoRA)
  
  [ Virtual P ]                [ P_K ]   [ P_V ]              [ Base Layer W_0 ]
       │                          │         │                       │      │
       ▼                          ▼         ▼                       │      ▼
   [ Embeddings ] ───> ... ───> [ Multi-Head Attn ] ───> ...       │    [ Adapter / LoRA ]
   (Tokens 1..T)                (Attends over P + Seq)              ▼      │
                                                                   (+) ◄───┘
```

---

### The 12-Year-Old Analogy: Custom Tool Attachments for a Swiss Army Knife

Imagine you own a high-end Swiss Army knife with 20 stainless steel tools: blades, scissors, screwdrivers, and can openers. You are entering a competitive woodworking contest and need specialized chisels and whittling tips:

* **Full Fine-Tuning (Reforging the Knife):**
  You melt the entire Swiss Army knife down in a furnace and reforge it into a single giant woodworking chisel. It's great for whittling, but you can never use it as a Swiss Army knife again—the scissors, corkscrew, and blades are gone forever.

* **Prompt Tuning (The Instruction Post-It Note):**
  You take a tiny yellow Post-It note, write *"Act like a professional woodworking chisel"* on it, and tape it to the outside handle of the knife before opening a blade. It costs almost nothing, but if the wood is too dense (or the model is too small), the knife easily ignores the sticky note. Furthermore, the sticky note takes up pocket space (**Context Window penalty**).

* **Prefix Tuning (Internal Guide Rails on Every Tool):**
  You open up the knife and snap custom miniature mechanical guide rails directly onto the pivot joints of *every single folding blade inside the knife*. Every time a tool extends, the guide rail steers its angle with surgical precision.

* **Series Adapters & LoRA (Clip-On Magnetic Attachments):**
  You clip an ultra-compact magnetic precision guide onto the blade tip. When cutting, the guide clips on in a millisecond; when you're done, you unclip it and put it back in your pouch. The knife remains completely untouched and pristine!

---

## 2. Mathematical Symbol Decoder

| Symbol | Mathematical Domain | Description & Engineering Meaning |
| :--- | :--- | :--- |
| $L_p$ | Scalar $\in \mathbb{N}$ | Virtual prompt/prefix length (number of virtual tokens, e.g., $10 - 100$) |
| $P \in \mathbb{R}^{L_p \times d}$ | Continuous Embedding Tensor | Learnable virtual soft prompt tensor prepended to token embeddings |
| $P_K^{(l)}, P_V^{(l)}$ | $\mathbb{R}^{L_p \times d_k}$ | Learnable prefix Key and Value tensors injected at layer $l$ |
| $W_{\text{down}} \in \mathbb{R}^{d \times m}$ | Adapter Matrix | Bottleneck down-projection matrix ($m \ll d$, e.g., $m = 64$) |
| $W_{\text{up}} \in \mathbb{R}^{m \times d}$ | Adapter Matrix | Bottleneck up-projection matrix restoring hidden dimension $d$ |
| $m$ | Scalar $\in \mathbb{N}$ | Bottleneck dimension of an adapter sub-layer ($m \ll d$) |
| $f(\cdot)$ | Non-linear Function | Activation function inside adapter bottleneck (e.g., GeLU, ReLU) |
| $s$ | Scalar Multiplier | Scaling hyperparameter for adapter or LoRA residuals |

---

## 3. Taxonomy of Parameter-Efficient Adaptation

PEFT methodologies are grouped into three primary structural paradigms:

```
                                  PEFT TAXONOMY
                                        │
         ┌──────────────────────────────┼──────────────────────────────┐
         ▼                              ▼                              ▼
    Additive PEFT                Reparameterization             Selective PEFT
  (New parameters)                 (Factorization)            (Subset of weights)
   ├── Series Adapters (Houlsby)   ├── LoRA (Hu et al.)       ├── BitFit (Bias only)
   ├── Parallel Adapters (He)      ├── DoRA (Weight-Decomp)   └── Top-K Layer Freeze
   ├── Prompt Tuning (Lester)      └── AdaLoRA (Dynamic)
   └── Prefix Tuning (Li & Liang)
```

---

## 4. Series vs. Parallel Bottleneck Adapters

### 4.1 Houlsby Series Adapters (2019)
Neil Houlsby and colleagues at Google introduced the first modern adapter architecture for NLP. They inserted compact bottleneck MLP blocks sequentially into two positions within each Transformer layer:
1. Immediately following the Multi-Head Attention projection and layer norm.
2. Immediately following the Feed-Forward Network (FFN) block and layer norm.

$$\boxed{\text{Adapter}(h) = h + f\left(h W_{\text{down}} + b_d\right) W_{\text{up}} + b_u}$$

where:
- Input hidden state: $h \in \mathbb{R}^{B \times T \times d}$
- Down-projection: $W_{\text{down}} \in \mathbb{R}^{d \times m}$ projects into small bottleneck $m \ll d$ (e.g., $m = 64$ when $d = 4096$)
- Non-linearity: $f(\cdot) = \operatorname{GeLU}(\cdot)$
- Up-projection: $W_{\text{up}} \in \mathbb{R}^{m \times d}$ restores dimension back to $d$
- Residual connection: Preserves identity mapping if adapter weights are small.

#### The Sequential Bottleneck Problem:
Because series adapters are executed **in series** ($h \to \text{Attn} \to \text{Adapter}_1 \to \text{FFN} \to \text{Adapter}_2$), each adapter introduces a sequential GPU kernel launch barrier. In distributed inference engines, these sequential small GEMMs create latency bubbles, increasing inference latency by $15\% - 30\%$ despite the tiny parameter count.

### 4.2 Parallel Adapters (He et al. 2021)
To eliminate sequential GPU stalls, He et al. (2021) introduced **Parallel Adapters**:
$$\boxed{h_{\text{out}} = \text{BaseLayer}(x) + s \cdot \text{Adapter}(x)}$$

Both the frozen base layer (Attention or MLP) and the adapter receive the exact same input $x$ simultaneously. Modern GPU streams can execute both matrix multiplications concurrently, eliminating sequential dependency stalls.

---

## 5. Prompt Tuning (Lester et al. 2021)

Instead of altering internal layer architectures, **Prompt Tuning** freezes the entire model and prepends $L_p$ learnable continuous virtual token embeddings directly to the input sequence:

$$X_{\text{input}} = \left[ P_1, P_2, \dots, P_{L_p}; \; \mathbf{e}(w_1), \mathbf{e}(w_2), \dots, \mathbf{e}(w_T) \right] \in \mathbb{R}^{(L_p + T) \times d_{\text{model}}}$$

- Pre-trained token embeddings and all Transformer layers are **strictly frozen**.
- Only the $L_p \times d_{\text{model}}$ soft prompt matrix $P$ is updated via gradient descent.
- Total trainable parameters:
  $$P_{\text{PromptTuning}} = L_p \times d_{\text{model}}$$
  For $L_p = 20$ and $d_{\text{model}} = 4096$, trainable parameters $= 20 \times 4096 = \mathbf{81,920 \text{ parameters}}$ ($\mathbf{0.001\%}$ of an 8B model!).

### The "Power of Scale" Empirical Law
Brian Lester and the Google Brain team established a famous empirical scaling phenomenon:
- **Small Models ($<1\text{B}$ params):** Prompt tuning significantly underperforms full fine-tuning. Small models lack the representational capacity to be steered solely through soft input prefixes.
- **Large Models ($\ge 10\text{B}$ params):** The performance gap between Prompt Tuning and Full Fine-Tuning completely vanishes! Massive foundation models exhibit extraordinary prompt steering capability.

### Architectural Limitation:
Prompt tokens occupy space in the model's finite **context window length** ($T_{\max}$). If $L_p = 50$, every input sequence loses 50 tokens of available user context length.

---

## 6. Prefix Tuning (Li & Liang 2021)

While Prompt Tuning prepends virtual embeddings only at the input layer, its steering signal often attenuates as activations propagate through 32 or 80 deep Transformer layers.

Xiang Lisa Li and Percy Liang (Stanford) solved this with **Prefix Tuning**: injecting continuous learnable virtual Key and Value states **across every single attention layer**:

```
                       PREFIX TUNING KV-INJECTION
                       
                 Layer l Attention Block
                 
   Sequence Q_seq                  Prefix P_K       Sequence K_seq       Prefix P_V       Sequence V_seq
     [T x d_k]                    [L_p x d_k]        [T x d_k]          [L_p x d_k]        [T x d_k]
         │                             └──────────┬──────────┘               └──────────┬──────────┘
         │                                        ▼                                     ▼
         │                                   Concatenate                           Concatenate
         │                                   K_full = [P_K; K_seq]                 V_full = [P_V; V_seq]
         │                                   [(L_p + T) x d_k]                     [(L_p + T) x d_k]
         │                                        │                                     │
         └───────────────────────┬────────────────┘                                     │
                                 ▼                                                      │
                       Scaled Dot-Product MatMul                                        │
                       Q @ K_full^T / sqrt(d_k)                                         │
                       [T x (L_p + T)]                                                  │
                                 │                                                      │
                                 ▼                                                      │
                        Row-wise Softmax                                                │
                                 │                                                      │
                                 └───────────────────────┬──────────────────────────────┘
                                                         ▼
                                               Weighted Sum: Attn @ V_full
                                               Output Tensor [T x d_v]
```

### Mathematical Formulation:
For each attention layer $l \in \{1, \dots, L\}$:
$$K_{\text{full}}^{(l)} = \left[ P_K^{(l)}; \; K_{\text{seq}}^{(l)} \right] \in \mathbb{R}^{(L_p + T) \times d_k}$$
$$V_{\text{full}}^{(l)} = \left[ P_V^{(l)}; \; V_{\text{seq}}^{(l)} \right] \in \mathbb{R}^{(L_p + T) \times d_v}$$

The sequence query tokens $Q_{\text{seq}}^{(l)} \in \mathbb{R}^{T \times d_k}$ compute attention across all $L_p + T$ positions:
$$\text{Attention}(Q_{\text{seq}}, K_{\text{full}}, V_{\text{full}}) = \operatorname{softmax}\left( \frac{Q_{\text{seq}} (K_{\text{full}})^T}{\sqrt{d_k}} \right) V_{\text{full}}$$

### Re-parameterization Trick for Training Stability:
Directly optimizing $P_K, P_V$ via gradient descent is notoriously unstable and sensitive to learning rate initialization. Li & Liang stabilized training by reparameterizing prefixes via a small MLP:
$$P_K^{(l)}, P_V^{(l)} = \operatorname{MLP}_{\phi}\left( E_{\text{prefix}} \right)$$
After training converges, the MLP is discarded, and only the resulting static key-value tensors $P_K^{(l)}, P_V^{(l)}$ are saved for deployment.

---

## 7. Master Architectural Comparison Matrix

| Architectural Dimension | Full Fine-Tuning (FFT) | Series Adapters (Houlsby) | Parallel Adapters (He) | LoRA (Hu et al.) | Prefix Tuning (Li & Liang) | Prompt Tuning (Lester) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Trainable Params Ratio** | $100\%$ | $\approx 0.4\% - 1.0\%$ | $\approx 0.4\% - 1.0\%$ | $\approx 0.1\% - 0.6\%$ | $\approx 0.05\% - 0.1\%$ | $\approx 0.001\%$ |
| **Inference Latency Overhead** | **$0\text{ ms}$** (Baseline) | $+15\% - 30\%$ (Sequential GEMM) | $+5\% - 10\%$ (Parallel GEMM) | **$0\text{ ms}$** (Mergeable $W_0 + \Delta W$) | Minimal ($+2\% - 5\%$ KV cache) | **$0\text{ ms}$** (Prefixed tokens) |
| **Context Length Penalty** | $0$ tokens | $0$ tokens | $0$ tokens | $0$ tokens | $0$ tokens | $-L_p$ tokens (e.g. $-50$) |
| **Optimization Stability** | Medium (LR sensitive) | High | High | Very High ($\frac{\alpha}{r}$ scaling) | Medium (Needs MLP reparam) | Low for $<10\text{B}$; High for $>10\text{B}$ |
| **Multi-Tenant Serving** | Impossible on single GPU | Moderate (Kernel swaps) | Moderate | **Optimal** (S-LoRA / vLLM dynamic routing) | High (Prepend to KV cache) | High (Prepend to KV cache) |
| **Storage per Task** | Massive ($10 - 150\text{ GB}$) | Small ($30 - 100\text{ MB}$) | Small ($30 - 100\text{ MB}$) | Tiny ($10 - 50\text{ MB}$) | Tiny ($5 - 20\text{ MB}$) | Microscopic ($< 1\text{ MB}$) |

---

## 8. Summary & Takeaways

1. **Series Adapters** insert bottleneck MLPs sequentially into layers, but introduce inference latency overhead due to sequential kernel launch barriers.
2. **Parallel Adapters** resolve sequential stalls by executing concurrently with the base Transformer blocks.
3. **Prompt Tuning** prepends continuous soft tokens only at the input layer; it is extraordinarily lightweight ($0.001\%$) and achieves parity with FFT at scale ($\ge 10\text{B}$ parameters), but consumes context length budget.
4. **Prefix Tuning** injects learnable Key-Value prefixes across every attention layer, providing deep cross-layer steering without modifying base weights.
5. **LoRA** remains the dominant industry standard because its linear decomposition enables **exact mathematical weight merging ($W_{\text{merged}} = W_0 + \frac{\alpha}{r} B A$)**, delivering zero inference latency and zero context length penalty.
