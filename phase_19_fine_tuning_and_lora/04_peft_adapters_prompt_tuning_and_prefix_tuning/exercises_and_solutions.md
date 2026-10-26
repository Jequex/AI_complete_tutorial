# Topic 19.4: Exercises & Solutions — PEFT Adapters, Prompt Tuning, and Prefix Tuning

---

### Problem 1: Manual Prefix-Tuning Attention Allocation (Easy)

Consider a toy single-head attention layer with dimension $d_k = 2$.
A sequence contains a single query token $\mathbf{q}$ and a single sequence key $\mathbf{k}_{\text{seq}}$:
$$\mathbf{q} = \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix}, \quad \mathbf{k}_{\text{seq}} = \begin{bmatrix} 0.0 \\ 1.0 \end{bmatrix}$$

Under Prefix Tuning, a learnable virtual prefix key vector $\mathbf{p}_K \in \mathbb{R}^2$ is prepended:
$$\mathbf{p}_K = \begin{bmatrix} 1.0 \\ 1.0 \end{bmatrix}$$

The full key matrix is therefore:
$$K_{\text{full}} = \begin{bmatrix} \mathbf{p}_K^T \\ \mathbf{k}_{\text{seq}}^T \end{bmatrix} = \begin{bmatrix} 1.0 & 1.0 \\ 0.0 & 1.0 \end{bmatrix}$$

Using the scaled dot-product attention logit formula $z_j = \frac{\mathbf{q}^T \mathbf{k}_j}{\sqrt{d_k}}$:
1. Compute the raw dot-product scores for the prefix key ($\mathbf{q} \cdot \mathbf{p}_K$) and sequence key ($\mathbf{q} \cdot \mathbf{k}_{\text{seq}}$).
2. Compute the scaled attention logits $\mathbf{z} = [z_{\text{prefix}}, \; z_{\text{seq}}]$.
3. Compute the softmax attention weights $\mathbf{a} = [a_{\text{prefix}}, \; a_{\text{seq}}]$.
4. Given prefix value $\mathbf{p}_V = [5.0, 5.0]^T$ and sequence value $\mathbf{v}_{\text{seq}} = [1.0, 0.0]^T$, compute the contextualized output vector $\mathbf{h} = a_{\text{prefix}} \mathbf{p}_V + a_{\text{seq}} \mathbf{v}_{\text{seq}}$.

<details>
<summary>View Complete Solution</summary>

**Step 1: Compute Raw Dot Products**
- Prefix match:
  $$\mathbf{q} \cdot \mathbf{p}_K = (1.0)(1.0) + (0.0)(1.0) = 1.0 + 0.0 = 1.0$$
- Sequence match:
  $$\mathbf{q} \cdot \mathbf{k}_{\text{seq}} = (1.0)(0.0) + (0.0)(1.0) = 0.0 + 0.0 = 0.0$$

**Step 2: Scale by $\sqrt{d_k} = \sqrt{2} \approx 1.41421$**
$$z_{\text{prefix}} = \frac{1.0}{\sqrt{2}} \approx 0.70711$$
$$z_{\text{seq}} = \frac{0.0}{\sqrt{2}} = 0.00000$$

**Step 3: Compute Softmax Attention Distribution**
- Exponents:
  $$e^{z_{\text{prefix}}} = e^{0.70711} \approx 2.02811$$
  $$e^{z_{\text{seq}}} = e^{0.0} = 1.00000$$
- Denominator:
  $$\sum = 2.02811 + 1.00000 = 3.02811$$
- Attention Weights:
  $$a_{\text{prefix}} = \frac{2.02811}{3.02811} \approx \mathbf{0.6698} \quad (66.98\%)$$
  $$a_{\text{seq}} = \frac{1.00000}{3.02811} \approx \mathbf{0.3302} \quad (33.02\%)$$

**Step 4: Compute Contextualized Output $\mathbf{h}$**
$$\mathbf{h} = 0.6698 \begin{bmatrix} 5.0 \\ 5.0 \end{bmatrix} + 0.3302 \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix} = \begin{bmatrix} 3.3490 + 0.3302 \\ 3.3490 + 0.0000 \end{bmatrix} = \begin{bmatrix} \mathbf{3.6792} \\ \mathbf{3.3490} \end{bmatrix}$$

*(The learnable prefix captures over two-thirds of the attention mass, steering the output vector toward $[3.68, 3.35]$ without changing any sequence representations).*

</details>

---

### Problem 2: Parameter Budgeting Across Adaptation Paradigms (Medium)

Consider a compact 8-layer Transformer language model ($L = 8$) with:
- Hidden dimension $d = 512$
- Number of attention heads $h = 8$ ($d_k = 64$)
- MLP intermediate dimension $d_{\text{ffn}} = 2048$
- Vocabulary size $V = 16000$

We evaluate 4 PEFT configurations:
1. **Series Adapters (Houlsby):** 2 bottleneck MLPs per layer (after Attention and after FFN) with bottleneck dimension $m = 32$.
2. **LoRA:** Applied to $W_q, W_k, W_v, W_o$ with rank $r = 8$ and $\alpha = 16$.
3. **Prefix Tuning:** Prefix length $L_p = 10$ prepended to $K$ and $V$ at all 8 layers.
4. **Prompt Tuning:** Soft prompt length $L_p = 10$ prepended only to the input embedding layer.

**Tasks:**
1. Compute the total base model parameter count (embeddings + 8 layers of Attention & MLP).
2. Compute the exact number of trainable parameters for each of the 4 PEFT configurations.
3. Rank the methods from lowest to highest parameter efficiency.

<details>
<summary>View Complete Solution</summary>

**Step 1: Base Model Parameter Count**
- Embeddings: $V \times d = 16000 \times 512 = 8,192,000$ parameters.
- Single Layer:
  - Attention ($W_q, W_k, W_v, W_o$): $4 \times (512 \times 512) = 4 \times 262,144 = 1,048,576$
  - FFN ($W_1 \in \mathbb{R}^{2048 \times 512}, W_2 \in \mathbb{R}^{512 \times 2048}$): $2 \times (512 \times 2048) = 2,097,152$
  - Total per layer $= 1,048,576 + 2,097,152 = 3,145,728$ parameters.
- Across $L = 8$ layers: $8 \times 3,145,728 = 25,165,824$ parameters.
- Total Base Parameters:
  $$P_{\text{base}} = 8,192,000 + 25,165,824 = \mathbf{33,357,824 \text{ parameters}} \approx \mathbf{33.36\text{ M}}$$

---

**Step 2: Trainable Parameters per PEFT Method**

1. **Series Adapters (Houlsby, $m = 32$):**
   - Each adapter bottleneck MLP: $(d \times m) + m + (m \times d) + d = (512 \times 32) + 32 + (32 \times 512) + 512 = 16,384 + 32 + 16,384 + 512 = 33,312$ parameters.
   - 2 adapters per layer $\times 8$ layers $= 16 \times 33,312 = \mathbf{532,992 \text{ parameters}}$ ($1.60\%$ of base).

2. **LoRA ($r = 8$ on Attention):**
   - 4 attention matrices per layer: $4 \times [r(d + d)] = 4 \times [8 \times 1024] = 4 \times 8192 = 32,768$ parameters per layer.
   - Across 8 layers: $8 \times 32,768 = \mathbf{262,144 \text{ parameters}}$ ($0.79\%$ of base).

3. **Prefix Tuning ($L_p = 10$ on $K$ and $V$):**
   - In each layer: $P_K \in \mathbb{R}^{10 \times 512}$ and $P_V \in \mathbb{R}^{10 \times 512}$.
   - Parameters per layer: $2 \times (10 \times 512) = 10,240$ parameters.
   - Across 8 layers: $8 \times 10,240 = \mathbf{81,920 \text{ parameters}}$ ($0.25\%$ of base).

4. **Prompt Tuning ($L_p = 10$ on Input):**
   - Soft prompt tensor: $P \in \mathbb{R}^{10 \times 512}$.
   - Trainable parameters: $10 \times 512 = \mathbf{5,120 \text{ parameters}}$ ($\mathbf{0.015\%}$ of base).

---

**Step 3: Ranking by Parameter Count (Lowest to Highest)**
1. **Prompt Tuning:** $5,120$ params ($0.015\%$)
2. **Prefix Tuning:** $81,920$ params ($0.246\%$)
3. **LoRA:** $262,144$ params ($0.786\%$)
4. **Series Adapters:** $532,992$ params ($1.598\%$)

</details>

---

### Problem 3: Context Budget & FLOPs Penalty of Prompt Tuning vs. LoRA (Hard)

Consider an AI serving engine deployed to process user queries with context window $T = 2048$.
In self-attention, the computation of attention logits $Q K^T$ requires $2 \cdot T^2 \cdot d$ FLOPs per layer.

Compare two adaptation strategies:
- **Strategy A (LoRA):** Weights are merged into $W_{\text{merged}}$. Sequence length remains $T = 2048$.
- **Strategy B (Prompt Tuning):** $L_p = 64$ virtual prompt tokens are prepended to every sequence. Sequence length becomes $T' = T + L_p = 2048 + 64 = 2112$.

**Tasks:**
1. Calculate the percentage increase in self-attention FLOPs incurred by Strategy B compared to Strategy A.
2. If the user query has length $2000$ tokens, calculate the available response generation budget (in tokens) under Strategy A vs. Strategy B before hitting the maximum hardware context limit $T_{\max} = 2048$.

<details>
<summary>View Complete Solution</summary>

**Step 1: Attention FLOPs Increase**
The FLOPs required by the attention matrix multiply $Q K^T$ scale quadratically with sequence length:
$$\text{Ratio} = \frac{\text{FLOPs}(T')}{\text{FLOPs}(T)} = \frac{2 (T + L_p)^2 d}{2 T^2 d} = \left(\frac{T + L_p}{T}\right)^2$$

For $T = 2048$ and $L_p = 64$:
$$\frac{T + L_p}{T} = \frac{2112}{2048} \approx 1.03125$$
$$\text{Ratio} = (1.03125)^2 \approx 1.06348$$

Percentage increase in attention computation:
$$\Delta \text{FLOPs} = (1.06348 - 1) \times 100\% \approx \mathbf{+6.35\%}$$
*(Prompt tuning increases the quadratic attention computational burden across every single layer for every generation step).*

---

**Step 2: Generation Budget Reduction**
Let $T_{\max} = 2048$ and input query length $T_{\text{query}} = 2000$.

- **Strategy A (LoRA):**
  LoRA introduces zero virtual tokens into the input sequence.
  $$\text{Budget}_{\text{LoRA}} = T_{\max} - T_{\text{query}} = 2048 - 2000 = \mathbf{48 \text{ tokens}}$$

- **Strategy B (Prompt Tuning):**
  The model must prepend $L_p = 64$ virtual tokens before the query.
  Total input consumption $= 64 + 2000 = 2064$ tokens.
  Since $2064 > 2048$:
  $$\text{Budget}_{\text{PromptTuning}} = 2048 - 2064 = \mathbf{-16 \text{ tokens (CONTEXT OVERFLOW!)}}$$

The prompt-tuned model **cannot process this query at all**, throwing a context-length error. Even if the query were shortened to 1950 tokens, Prompt Tuning would leave only $2048 - (1950 + 64) = 34$ tokens of response room compared to $98$ tokens under LoRA.

**Key Insight:** While Prompt Tuning uses fewer parameters during training, it imposes a permanent runtime context length tax and quadratic FLOPs penalty on every single token generated during inference. LoRA avoids both penalties entirely.

</details>
