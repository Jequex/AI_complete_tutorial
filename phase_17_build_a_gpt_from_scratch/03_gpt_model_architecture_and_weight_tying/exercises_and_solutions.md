# Topic 17.3: Exercises & Solutions — GPT Architecture & Weight Tying

---

### Problem 1: Manual Parameter Derivation for GPT-2 Small (Easy)
GPT-2 Small has the following architectural specifications:
- Vocabulary size $V = 50{,}257$
- Context block size $T_{\max} = 1024$
- Hidden dimension $d_{\text{model}} = 768$
- Number of layers $L = 12$
- Number of attention heads $H = 12$
- Weight tying enabled ($W_{\text{head}} = W_{\text{token}}^T$)

Using the closed-form component formulas:
1. Compute the parameters in the token embedding table `wte`.
2. Compute the parameters in the positional embedding table `wpe`.
3. Compute the parameters in all 12 Transformer blocks.
4. Compute the final LayerNorm parameters and output LM head parameters.
5. Sum them to find the exact total parameter count and compare it to the popular "124M" moniker.

<details>
<summary>View Complete Solution</summary>

**Step 1: Embeddings:**
- Token Embeddings:
  $$\text{wte} = V \times d_{\text{model}} = 50{,}257 \times 768 = 38{,}597{,}376$$
- Positional Embeddings:
  $$\text{wpe} = T_{\max} \times d_{\text{model}} = 1024 \times 768 = 786{,}432$$

**Step 2: Transformer Blocks:**
For one block with $d = 768$:
- Attention ($W_q, W_k, W_v, W_o$ plus biases):
  $$4 \times (768^2 + 768) = 4 \times (589{,}824 + 768) = 4 \times 590{,}592 = 2{,}362{,}368$$
- MLP ($W_1, \mathbf{b}_1, W_2, \mathbf{b}_2$):
  $$W_1: 768 \times 3072 + 3072 = 2{,}362{,}368$$
  $$W_2: 3072 \times 768 + 768 = 2{,}360{,}064$$
  $$\text{MLP} = 2{,}362{,}368 + 2{,}360{,}064 = 4{,}722{,}432$$
- LayerNorms (2 layers per block, $\boldsymbol{\gamma}$ and $\boldsymbol{\beta}$):
  $$2 \times (2 \times 768) = 3{,}072$$
- Total per block:
  $$\text{Params}_{\text{block}} = 2{,}362{,}368 + 4{,}722{,}432 + 3{,}072 = 7{,}087{,}872$$
- Total for 12 blocks:
  $$12 \times 7{,}087{,}872 = 85{,}054{,}464$$

**Step 3: Final LayerNorm & LM Head:**
- Final LayerNorm:
  $$\text{ln\_f} = 2 \times 768 = 1{,}536$$
- LM Head:
  $$\text{lm\_head} = 0 \quad (\text{Tied to } W_{\text{token}})$$

**Step 4: Total Parameters:**
$$\text{Total} = 38{,}597{,}376 + 786{,}432 + 85{,}054{,}464 + 1{,}536 + 0 = \mathbf{124{,}439{,}808}$$

This matches the exact figure of $124{,}439{,}808 \approx 124.44\text{ Million parameters}$ (popularly rounded to 124M).
</details>

---

### Problem 2: Impact of Untied Weights on VRAM Footprint (Medium)
Suppose an engineering team considers training a large language model with $V = 128{,}000$ (e.g. LLaMA 3 vocabulary) and hidden dimension $d_{\text{model}} = 4096$, using 16-bit floating point precision (FP16 / BF16, 2 bytes per parameter).
1. How many extra parameters does untying the output projection head add?
2. How many additional gigabytes (GB) of VRAM are required simply to store the model weights, Adam optimizer states (first moment $m$, second moment $v$, FP32 master weights = 16 bytes per parameter), and gradients (2 bytes per parameter) for this extra untied matrix?

<details>
<summary>View Complete Solution</summary>

**Step 1: Extra Parameters:**
$$\Delta P = V \times d_{\text{model}} = 128{,}000 \times 4096 = 524{,}288{,}000 \text{ parameters} \approx 524.29\text{ Million parameters}$$

**Step 2: Total Memory Footprint for Untied Matrix:**
In AdamW training with mixed precision:
- Model Weights (BF16): 2 bytes/param
- Gradients (BF16): 2 bytes/param
- Optimizer States:
  - FP32 Master Weights: 4 bytes/param
  - Momentum $m$ (FP32): 4 bytes/param
  - Variance $v$ (FP32): 4 bytes/param
- Total bytes per parameter during training:
  $$\text{Bytes/Param} = 2 + 2 + 4 + 4 + 4 = 16\text{ bytes}$$

Total additional VRAM consumed:
$$\text{Total Bytes} = 524{,}288{,}000 \times 16 = 8{,}388{,}608{,}000 \text{ bytes}$$
Converting to gigabytes (binary GiB vs metric GB):
$$\text{Metric GB} = \frac{8{,}388{,}608{,}000}{10^9} \approx 8.39\text{ GB}$$
$$\text{Binary GiB} = \frac{8{,}388{,}608{,}000}{1024^3} = 7.81\text{ GiB}$$

Untying the LM head requires an extra ~8 GB of high-bandwidth GPU VRAM just to store the output matrix and its Adam optimizer states, without contributing any deeper semantic reasoning capacity.
</details>

---

### Problem 3: Cross-Entropy Loss Numerical Stability Derivation (Hard)
Let $\mathbf{z} \in \mathbb{R}^V$ be the unnormalized logit vector output by the LM head at position $t$, and let $y \in \{0, \dots, V-1\}$ be the ground truth target token ID.
The cross-entropy loss is:
$$\mathcal{L} = -\log P(y) = -\log\left( \frac{\exp(z_y)}{\sum_{j=1}^V \exp(z_j)} \right)$$

1. Why does naively evaluating $\exp(z_j)$ cause catastrophic numerical overflow or underflow when $z_j$ is large (e.g. $z_j = 1000$)?
2. Derive the mathematically equivalent **Log-Sum-Exp (LSE)** trick using $c = \max_k z_k$.
3. Prove that the shifted exponent evaluation $\exp(z_j - c)$ is strictly bounded in $(0, 1]$, completely eliminating overflow.

<details>
<summary>View Complete Solution</summary>

**Step 1: Naive Evaluation Breakdown:**
Standard 32-bit floating point numbers (IEEE 754 float32) overflow to `+inf` when $x > 88.72$, because $\exp(88.72) \approx 3.4 \times 10^{38}$. If a single logit $z_j \ge 89$, $\exp(z_j) = \infty$, producing $\frac{\infty}{\infty} = \text{NaN}$.

**Step 2: Derivation of the Log-Sum-Exp Trick:**
Let $c = \max_{k=1}^V z_k$.
Factor $\exp(c)$ out of the summation:
$$\sum_{j=1}^V \exp(z_j) = \sum_{j=1}^V \exp(z_j - c + c) = \exp(c) \sum_{j=1}^V \exp(z_j - c)$$

Taking the logarithm:
$$\log\left(\sum_{j=1}^V \exp(z_j)\right) = \log\left(\exp(c) \sum_{j=1}^V \exp(z_j - c)\right) = c + \log\left(\sum_{j=1}^V \exp(z_j - c)\right)$$

Now substitute this into the loss formula:
$$\mathcal{L} = -\left( \log(\exp(z_y)) - \log\sum_{j=1}^V \exp(z_j) \right) = -z_y + c + \log\left(\sum_{j=1}^V \exp(z_j - c)\right)$$
Equivalently:
$$\mathcal{L} = -(z_y - c) + \log\left(\sum_{j=1}^V \exp(z_j - c)\right)$$

**Step 3: Bounding Proof:**
Since $c = \max_k z_k$, for every index $j \in \{1, \dots, V\}$:
$$z_j - c \le 0 \implies \exp(z_j - c) \in (0, 1]$$
At the maximum logit index $j^*$, $\exp(z_{j^*} - c) = \exp(0) = 1.0$.
Therefore, the maximum argument passed to the exponential function is exactly $0.0$, making overflow mathematically impossible!
</details>
