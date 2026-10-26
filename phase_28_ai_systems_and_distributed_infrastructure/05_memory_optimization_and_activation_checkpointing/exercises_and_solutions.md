# Topic 05: Exercises & Solutions — Memory Optimization & Activation Checkpointing

## Exercise 1: Transformer Activation Memory Scaling
**Problem:**
Consider training one layer of LLaMA-70B ($D = 8{,}192$, attention heads $H = 64$, intermediate dimension $d_{\text{ffn}} = 28{,}672$) in FP16 ($2\text{ bytes/element}$) with micro-batch size $B = 2$.
The activation memory per token vector consists of:
- Attention $Q, K, V$ inputs: $2 \times B \times S \times D$
- Attention scores and weights: $2 \times B \times H \times S^2$
- Attention output projection: $2 \times B \times S \times D$
- MLP Gate & Up projections: $2 \times 2 \times B \times S \times d_{\text{ffn}}$
- MLP Down projection: $2 \times B \times S \times D$

1. Compute the activation memory for a single layer at context lengths:
   - $S = 2{,}048$ tokens
   - $S = 8{,}192$ tokens
   - $S = 32{,}768$ tokens
2. For an 80-layer model, calculate the total activation memory across all layers. Does it fit on an 80GB GPU at $S = 32{,}768$?
3. If activation checkpointing ($k = \sqrt{80} \approx 9$) is applied, calculate the new peak activation memory at $S = 32{,}768$.

<details>
<summary>Click to view Solution</summary>

### Solution:

#### 1. Activation Memory per Layer:
Let $c_{\text{linear}} = 2 \times B \times S \times (3D + 2 d_{\text{ffn}})$:
$$3D + 2 d_{\text{ffn}} = 3(8{,}192) + 2(28{,}672) = 24{,}576 + 57{,}344 = 81{,}920$$
For $B = 2$:
$$\text{Linear Activation Bytes} = 2 \times 2 \times S \times 81{,}920 = 327{,}680 \cdot S\text{ bytes}$$
Attention matrix bytes:
$$\text{Attn Matrix Bytes} = 2 \times B \times H \times S^2 = 2 \times 2 \times 64 \times S^2 = 256 \cdot S^2\text{ bytes}$$
Total bytes per layer:
$$A_{\text{layer}}(S) = 327{,}680 \cdot S + 256 \cdot S^2$$

- **At $S = 2{,}048$**:
  - Linear: $327{,}680 \times 2{,}048 \approx 671.09\text{ MB}$
  - Attn Matrix: $256 \times (2{,}048)^2 \approx 1{,}073.74\text{ MB} \approx 1.07\text{ GB}$
  - Total per layer $\approx 1.745\text{ GB}$.
- **At $S = 8{,}192$**:
  - Linear: $327{,}680 \times 8{,}192 \approx 2.68\text{ GB}$
  - Attn Matrix: $256 \times (8{,}192)^2 \approx 17.18\text{ GB}$
  - Total per layer $\approx 19.86\text{ GB}$.
- **At $S = 32{,}768$**:
  - Linear: $327{,}680 \times 32{,}768 \approx 10.74\text{ GB}$
  - Attn Matrix: $256 \times (32{,}768)^2 \approx 274.88\text{ GB}$
  - Total per layer $\approx \mathbf{285.62\text{ GB}}$!

#### 2. Total Activation Memory across 80 Layers:
- **At $S = 2{,}048$**: $80 \times 1.745\text{ GB} \approx \mathbf{139.6\text{ GB}}$ (exceeds single 80GB GPU).
- **At $S = 32{,}768$**: $80 \times 285.62\text{ GB} \approx \mathbf{22{,}849.6\text{ GB}} \approx \mathbf{22.85\text{ Terabytes}}$!
At $S = 32{,}768$, activation memory exceeds single-GPU capacity by nearly **$300\times$**!

#### 3. With Activation Checkpointing ($k = 9$):
Stored checkpoints: $\frac{80}{9} \approx 9$ checkpoints.
Transient recomputed segment: $9$ layers.
Peak memory at any moment:
$$M_{\text{peak}} \approx 2 \sqrt{L} \times A_{\text{layer}} \approx 18 \times A_{\text{linear\_layer}}$$
(Note: With FlashAttention, the quadratic $S^2$ term is never materialized in HBM, leaving only the linear term $10.74\text{ GB}$!).
With FlashAttention + Activation Checkpointing, peak activation memory at $S = 32{,}768$ is:
$$M_{\text{peak}} \approx 18 \times 0.67\text{ GB} \approx \mathbf{12.0\text{ GB}}$$
It fits easily inside the 80GB VRAM budget!
</details>

---

## Exercise 2: Hierarchical 2-Level Activation Checkpointing
**Problem:**
Standard activation checkpointing with 1 level saves checkpoints every $k$ layers, achieving $\mathcal{O}(\sqrt{L})$ peak memory.
Suppose you train an ultra-deep model with $L = 1{,}000$ layers.
Consider a **2-level hierarchical checkpointing** strategy:
- Level 1: Save a top-level checkpoint every $k_1$ layers.
- Level 2: Inside each top-level segment of length $k_1$, save sub-checkpoints every $k_2$ layers.

1. Write the objective function for peak activation memory $M(k_1, k_2)$ in terms of $L, k_1,$ and $k_2$.
2. Derive the optimal intervals $k_1^*$ and $k_2^*$ that minimize peak memory.
3. What is the asymptotic memory complexity in terms of $L$? For $L = 1{,}000$, compare peak memory between standard backprop, 1-level checkpointing, and 2-level checkpointing.

<details>
<summary>Click to view Solution</summary>

### Solution:

#### 1. Objective Function $M(k_1, k_2)$:
At peak execution:
1. Top-level checkpoints stored in memory: $\frac{L}{k_1}$
2. Sub-checkpoints stored inside the active Level 1 block: $\frac{k_1}{k_2}$
3. Scratch activations inside the active Level 2 block: $k_2$
$$M(k_1, k_2) = \frac{L}{k_1} + \frac{k_1}{k_2} + k_2$$

#### 2. Optimization:
Take partial derivatives with respect to $k_2$ and $k_1$:
$$\frac{\partial M}{\partial k_2} = -\frac{k_1}{k_2^2} + 1 = 0 \implies k_2^* = \sqrt{k_1}$$
Substitute $k_2 = \sqrt{k_1}$ into $M$:
$$M(k_1) = \frac{L}{k_1} + 2\sqrt{k_1}$$
Differentiate with respect to $k_1$:
$$\frac{\partial M}{\partial k_1} = -\frac{L}{k_1^2} + \frac{1}{\sqrt{k_1}} = 0 \implies \frac{1}{\sqrt{k_1}} = \frac{L}{k_1^2} \implies k_1^{3/2} = L \implies \mathbf{k_1^* = L^{2/3}}$$
Then:
$$\mathbf{k_2^* = \sqrt{k_1^*} = (L^{2/3})^{1/2} = L^{1/3}}$$

#### 3. Asymptotic Complexity & Numerical Evaluation ($L = 1{,}000$):
Minimum Peak Memory:
$$M^* = \frac{L}{L^{2/3}} + \frac{L^{2/3}}{L^{1/3}} + L^{1/3} = L^{1/3} + L^{1/3} + L^{1/3} = \mathbf{3 L^{1/3} \in \mathcal{O}(L^{1/3})}$$
For $L = 1{,}000$:
- **Standard Backpropagation**: $1{,}000\text{ units}$.
- **1-Level Checkpointing**: $2\sqrt{1{,}000} \approx 2 \times 31.62 \approx \mathbf{63.2\text{ units}}$ ($93.7\%$ savings).
- **2-Level Checkpointing**: $3 \times (1{,}000)^{1/3} = 3 \times 10 = \mathbf{30\text{ units}}$ ($97.0\%$ savings over standard).
</details>

---

## Exercise 3: FlashAttention HBM IO Reduction
**Problem:**
Calculate the exact HBM data traffic (reads and writes) for an attention head with sequence length $S = 16{,}384$ and head dimension $d = 128$ in FP16 ($2\text{ bytes/element}$).

1. In standard attention, compute the exact bytes transferred for:
   - Writing $\mathbf{S} = \mathbf{Q}\mathbf{K}^T$ to HBM.
   - Reading $\mathbf{S}$ and writing $\mathbf{P} = \text{Softmax}(\mathbf{S})$ to HBM.
   - Reading $\mathbf{P}$ and writing $\mathbf{O} = \mathbf{P}\mathbf{V}$ to HBM.
   - Total HBM bytes transferred.
2. In FlashAttention, compute the exact bytes transferred when reading $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ from HBM and writing output $\mathbf{O}$ to HBM.
3. Compute the memory IO reduction factor.

<details>
<summary>Click to view Solution</summary>

### Solution:

#### 1. Standard Attention HBM Traffic:
- Reading $\mathbf{Q}, \mathbf{K}$: $2 \times S \times d \times 2 = 2 \times 16{,}384 \times 128 \times 2 = 8{,}388{,}608\text{ B} = 8.39\text{ MB}$.
- Writing $\mathbf{S} \in \mathbb{R}^{S \times S}$: $(16{,}384)^2 \times 2\text{ B} = 268{,}435{,}456 \times 2 = 536{,}870{,}912\text{ B} = 536.87\text{ MB}$.
- Softmax (Read $\mathbf{S}$, Write $\mathbf{P}$): $2 \times 536.87\text{ MB} = 1{,}073.74\text{ MB}$.
- Compute $\mathbf{O}$ (Read $\mathbf{P}$, Read $\mathbf{V}$, Write $\mathbf{O}$):
  $536.87\text{ MB} + 4.19\text{ MB} + 4.19\text{ MB} = 545.25\text{ MB}$.
- **Total Standard HBM Traffic**:
  $$V_{\text{std}} = 8.39 + 536.87 + 1{,}073.74 + 545.25 = \mathbf{2{,}164.25\text{ MB}} \approx \mathbf{2.164\text{ GB per head}}$$

#### 2. FlashAttention HBM Traffic:
FlashAttention keeps the intermediate matrices inside on-chip SRAM:
- Read $\mathbf{Q}$: $S \times d \times 2 = 4.19\text{ MB}$.
- Read $\mathbf{K}$: $S \times d \times 2 = 4.19\text{ MB}$.
- Read $\mathbf{V}$: $S \times d \times 2 = 4.19\text{ MB}$.
- Write $\mathbf{O}$: $S \times d \times 2 = 4.19\text{ MB}$.
- **Total FlashAttention HBM Traffic**:
  $$V_{\text{flash}} = 4 \times 4.19\text{ MB} = \mathbf{16.78\text{ MB per head}}$$

#### 3. IO Reduction Factor:
$$\text{Reduction Factor} = \frac{2{,}164.25\text{ MB}}{16.78\text{ MB}} \approx \mathbf{129.0\times}$$
FlashAttention cuts off-chip memory traffic by **$129\times$**, completely freeing up the memory bus and preventing GPU stalls!
</details>
