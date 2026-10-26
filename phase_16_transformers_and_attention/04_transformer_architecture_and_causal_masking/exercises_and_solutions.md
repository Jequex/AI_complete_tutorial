# Topic 16.4: Exercises & Solutions — Transformer Architecture & Causal Masking

---

### Problem 1: Manual Calculation of Causal Masked Attention (Easy)

Consider an attention logit matrix $S = Q K^T / \sqrt{d_k} \in \mathbb{R}^{3 \times 3}$ for a sequence of $T = 3$ tokens:
$$S = \begin{bmatrix} 2.0 & 4.0 & 1.0 \\ 3.0 & 1.0 & 5.0 \\ 0.0 & 2.0 & 2.0 \end{bmatrix}$$

1. Write out the explicit causal mask matrix $M \in \mathbb{R}^{3 \times 3}$.
2. Compute the masked logits matrix $Z = S + M$.
3. Compute the exact row-wise softmax attention weights $A \in \mathbb{R}^{3 \times 3}$ (round to 4 decimal places).
4. Verify that future token positions receive strictly zero attention weight.

<details>
<summary>View Complete Solution</summary>

**Step 1: The Causal Mask Matrix:**
$$M = \begin{bmatrix} 0.0 & -\infty & -\infty \\ 0.0 & 0.0 & -\infty \\ 0.0 & 0.0 & 0.0 \end{bmatrix}$$

**Step 2: Add Mask to Logits:**
$$Z = S + M = \begin{bmatrix} 2.0 & -\infty & -\infty \\ 3.0 & 1.0 & -\infty \\ 0.0 & 2.0 & 2.0 \end{bmatrix}$$

**Step 3: Row-Wise Softmax Normalization:**
- **Row 1 ($t = 1$):**
  - Exponents: $e^{2.0 - 2.0} = 1.0, \quad e^{-\infty} = 0.0, \quad e^{-\infty} = 0.0$
  - Sum $= 1.0$
  - $A_{11} = 1.0, \quad A_{12} = 0.0, \quad A_{13} = 0.0$

- **Row 2 ($t = 2$):**
  - Logits: $[3.0, 1.0, -\infty]$
  - Exponent shift $m = 3.0$:
    - $e^{3.0 - 3.0} = e^0 = 1.0$
    - $e^{1.0 - 3.0} = e^{-2.0} \approx 0.1353$
    - $e^{-\infty} = 0.0$
  - Sum $= 1.0 + 0.1353 = 1.1353$
  - $A_{21} = \frac{1.0}{1.1353} \approx 0.8808$
  - $A_{22} = \frac{0.1353}{1.1353} \approx 0.1192$
  - $A_{23} = 0.0$

- **Row 3 ($t = 3$):**
  - Logits: $[0.0, 2.0, 2.0]$
  - Exponent shift $m = 2.0$:
    - $e^{0.0 - 2.0} = e^{-2.0} \approx 0.1353$
    - $e^{2.0 - 2.0} = e^0 = 1.0$
    - $e^{2.0 - 2.0} = e^0 = 1.0$
  - Sum $= 0.1353 + 1.0 + 1.0 = 2.1353$
  - $A_{31} = \frac{0.1353}{2.1353} \approx 0.0634$
  - $A_{32} = \frac{1.0}{2.1353} \approx 0.4683$
  - $A_{33} = \frac{1.0}{2.1353} \approx 0.4683$

$$A = \begin{bmatrix} 1.0000 & 0.0000 & 0.0000 \\ 0.8808 & 0.1192 & 0.0000 \\ 0.0634 & 0.4683 & 0.4683 \end{bmatrix}$$

**Step 4: Verification:**
$A_{12} = 0.0, A_{13} = 0.0, A_{23} = 0.0$.
Every future token receives exact zero weight ($0.0$), and all row sums equal $1.0000$.

</details>

---

### Problem 2: Attention Memory Scaling & FlashAttention Motivation (Medium)

You are hosting an LLM with:
- Number of heads: $h = 32$
- Batch size: $B = 4$
- Precision: 16-bit floating point (FP16 or BF16, 2 bytes per element)

1. Write the formula for the total memory (in bytes) required to materialize the full attention weight matrix $A \in \mathbb{R}^{B \times h \times T \times T}$ for all layers in a single forward pass.
2. For $T = 2,048$, calculate the memory required for $A$ in a single layer.
3. For $T = 65,536$, calculate the memory required for $A$ across a 32-layer model.
4. Explain how **FlashAttention** eliminates this memory footprint.

<details>
<summary>View Complete Solution</summary>

**Step 1: Formula:**
$$\text{Memory} = B \times h \times T^2 \times 2 \text{ bytes}$$

**Step 2: Single Layer at $T = 2,048$:**
$$\text{Elements} = 4 \times 32 \times 2048^2 = 128 \times 4,194,304 = 536,870,912 \text{ elements}$$
$$\text{Bytes} = 536,870,912 \times 2 = 1,073,741,824 \text{ bytes} = 1.0 \text{ GiB}$$

A single layer requires $1.0\text{ GiB}$ of memory just to hold the attention matrix for batch size 4!

**Step 3: 32 Layers at $T = 65,536$:**
- Ratio of sequence length: $\frac{65536}{2048} = 32$.
- Because of $T^2$ scaling, memory per layer grows by $32^2 = 1024\times$:
  $$\text{Memory per layer} = 1.0 \text{ GiB} \times 1024 = 1,024 \text{ GiB} = 1 \text{ TiB}$$
- For 32 layers:
  $$\text{Total Memory} = 32 \times 1,024 \text{ GiB} = 32,768 \text{ GiB} \approx 32.8 \text{ Terabytes}!$$

It is physically impossible to store 32 TB of activation memory on even an 8-GPU H100 node ($8 \times 80\text{ GB} = 640\text{ GB}$).

**Step 4: FlashAttention Solution:**
FlashAttention computes scaled dot-product attention in blocks that fit entirely inside the high-speed **on-chip SRAM** (fast cache) of the GPU. It incrementally updates the softmax normalizer without ever writing the massive $T \times T$ matrix to High Bandwidth Memory (HBM). This reduces peak memory complexity from $\mathcal{O}(T^2)$ to $\mathcal{O}(T)$!

</details>

---

### Problem 3: Residual Stream Gradient Preservation: Pre-LN vs Post-LN (Hard)

Consider a deep network of $L$ sequential blocks. Let $x^{(l)}$ be the representation at layer $l$, and let $\mathcal{F}_l(\cdot)$ denote the attention/MLP sub-layer.

1. Write the recurrence equation for:
   - **Post-LN**: $x^{(l+1)} = \text{LN}(x^{(l)} + \mathcal{F}_l(x^{(l)}))$
   - **Pre-LN**: $x^{(l+1)} = x^{(l)} + \mathcal{F}_l(\text{LN}(x^{(l)}))$
2. Expand the Pre-LN recurrence from $l = 0$ to $l = L$.
3. Compute the Jacobian $\frac{\partial x^{(L)}}{\partial x^{(0)}}$ for Pre-LN.
4. Explain why Pre-LN guarantees that gradients cannot vanish as $L \to \infty$, while Post-LN causes gradient vanishing/explosion.

<details>
<summary>View Complete Solution</summary>

**Step 1: Recurrence Definitions:**
- Post-LN: $x^{(l+1)} = \text{LN}(x^{(l)} + \mathcal{F}_l(x^{(l)}))$
- Pre-LN: $x^{(l+1)} = x^{(l)} + \mathcal{F}_l(\text{LN}(x^{(l)}))$

**Step 2: Telescoping Sum of Pre-LN:**
Starting from $x^{(0)}$:
$$x^{(1)} = x^{(0)} + \mathcal{F}_0(\text{LN}(x^{(0)}))$$
$$x^{(2)} = x^{(1)} + \mathcal{F}_1(\text{LN}(x^{(1)})) = x^{(0)} + \mathcal{F}_0(\text{LN}(x^{(0)})) + \mathcal{F}_1(\text{LN}(x^{(1)}))$$
By induction, for any depth $L$:
$$x^{(L)} = x^{(0)} + \sum_{l=0}^{L-1} \mathcal{F}_l(\text{LN}(x^{(l)}))$$

**Step 3: Gradient Jacobian:**
Differentiating $x^{(L)}$ with respect to the input $x^{(0)}$:
$$\frac{\partial x^{(L)}}{\partial x^{(0)}} = \frac{\partial x^{(0)}}{\partial x^{(0)}} + \sum_{l=0}^{L-1} \frac{\partial \mathcal{F}_l(\text{LN}(x^{(l)}))}{\partial x^{(0)}} = I + \sum_{l=0}^{L-1} \frac{\partial \mathcal{F}_l}{\partial x^{(0)}}$$

By the multivariate chain rule, the loss gradient is:
$$\frac{\partial \mathcal{L}}{\partial x^{(0)}} = \frac{\partial \mathcal{L}}{\partial x^{(L)}} \frac{\partial x^{(L)}}{\partial x^{(0)}} = \frac{\partial \mathcal{L}}{\partial x^{(L)}} + \frac{\partial \mathcal{L}}{\partial x^{(L)}} \sum_{l=0}^{L-1} \frac{\partial \mathcal{F}_l}{\partial x^{(0)}}$$

**Step 4: Gradient Highway Comparison:**
- **In Pre-LN:** The term $\frac{\partial \mathcal{L}}{\partial x^{(L)}} \cdot I$ is preserved **verbatim**! Even if every single sub-layer $\mathcal{F}_l$ had zero gradient ($\frac{\partial \mathcal{F}_l}{\partial x} \approx 0$), the signal $\frac{\partial \mathcal{L}}{\partial x^{(L)}}$ travels unimpeded directly back to $x^{(0)}$. The residual stream acts as a non-decaying gradient superhighway!
- **In Post-LN:** Every step passes through $\text{LN}(\cdot)$. LayerNorm divides by standard deviation $\sigma_l \approx \sqrt{l}$. Across $L$ layers, gradients are multiplied by $\prod_{l=1}^L \frac{1}{\sigma_l}$, resulting in exponential decay and severe gradient vanishing near the bottom layers.

This explains why Pre-LN is the universal standard in all modern LLMs (GPT-3/4, LLaMA, PaLM, Mistral).

</details>
