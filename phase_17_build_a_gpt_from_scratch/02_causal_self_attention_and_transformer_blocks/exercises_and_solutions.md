# Topic 17.2: Exercises & Solutions — Causal Self-Attention & Transformer Blocks

---

### Problem 1: Manual Causal Attention Softmax Calculation (Easy)
Consider a sequence of length $T = 3$ with query dimension $d_k = 4$. Suppose the raw unnormalized dot-product scores matrix $S = \frac{Q K^T}{\sqrt{d_k}}$ is:
$$S = \begin{bmatrix} 2.0 & 4.0 & 1.0 \\ 1.5 & 3.0 & 5.0 \\ 0.0 & 2.5 & 4.0 \end{bmatrix}$$

1. Write out the causal mask matrix $M \in \mathbb{R}^{3 \times 3}$.
2. Compute the masked logits matrix $\tilde{S} = S + M$.
3. Compute the final row-wise softmax attention weights matrix $A$.

<details>
<summary>View Complete Solution</summary>

**Step 1: Causal Mask Matrix $M$:**
$$M = \begin{bmatrix} 0 & -\infty & -\infty \\ 0 & 0 & -\infty \\ 0 & 0 & 0 \end{bmatrix}$$

**Step 2: Masked Logits $\tilde{S}$:**
$$\tilde{S} = \begin{bmatrix} 2.0 & -\infty & -\infty \\ 1.5 & 3.0 & -\infty \\ 0.0 & 2.5 & 4.0 \end{bmatrix}$$

**Step 3: Row-wise Softmax $A$:**
- **Row 1 ($i=1$):**
  $$\text{softmax}([2.0, -\infty, -\infty]) = [1.0, 0.0, 0.0]$$

- **Row 2 ($i=2$):**
  Logits: $[1.5, 3.0, -\infty]$.
  $\exp(1.5) \approx 4.4817$, $\exp(3.0) \approx 20.0855$, $\exp(-\infty) = 0$.
  Sum $= 4.4817 + 20.0855 = 24.5672$.
  - $A_{2,1} = 4.4817 / 24.5672 \approx 0.1824$
  - $A_{2,2} = 20.0855 / 24.5672 \approx 0.8176$
  - $A_{2,3} = 0.0$

- **Row 3 ($i=3$):**
  Logits: $[0.0, 2.5, 4.0]$.
  $\exp(0.0) = 1.0$, $\exp(2.5) \approx 12.1825$, $\exp(4.0) \approx 54.5982$.
  Sum $= 1.0 + 12.1825 + 54.5982 = 67.7807$.
  - $A_{3,1} = 1.0 / 67.7807 \approx 0.0148$
  - $A_{3,2} = 12.1825 / 67.7807 \approx 0.1797$
  - $A_{3,3} = 54.5982 / 67.7807 \approx 0.8055$

Final attention matrix:
$$A \approx \begin{bmatrix} 1.0000 & 0.0000 & 0.0000 \\ 0.1824 & 0.8176 & 0.0000 \\ 0.0148 & 0.1797 & 0.8055 \end{bmatrix}$$
Notice that $A$ is strictly lower-triangular, proving zero future leakage.
</details>

---

### Problem 2: Parameter Count & Memory Allocation of a Single Transformer Block (Medium)
Consider a standard Transformer block with residual dimension $d_{\text{model}} = 768$ and $H = 12$ heads (GPT-2 Small block):
- Multi-Head Attention has projection weights $W_q, W_k, W_v, W_o \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$ with corresponding bias vectors.
- MLP has $W_1 \in \mathbb{R}^{d_{\text{model}} \times 4 d_{\text{model}}}$ with bias $\mathbf{b}_1 \in \mathbb{R}^{4 d_{\text{model}}}$, and $W_2 \in \mathbb{R}^{4 d_{\text{model}} \times d_{\text{model}}}$ with bias $\mathbf{b}_2 \in \mathbb{R}^{d_{\text{model}}}$.
- Two LayerNorm layers ($\text{LN}_1, \text{LN}_2$), each with learnable gain $\boldsymbol{\gamma}$ and bias $\boldsymbol{\beta}$ in $\mathbb{R}^{d_{\text{model}}}$.

1. Calculate the exact parameter count of the Multi-Head Attention sub-layer.
2. Calculate the exact parameter count of the MLP sub-layer.
3. Calculate the parameter count of the two LayerNorm layers.
4. What is the total parameter count of one single GPT-2 Small Transformer block?

<details>
<summary>View Complete Solution</summary>

**Step 1: Attention Sub-Layer:**
- 4 linear projections ($Q, K, V, O$):
  $$\text{Weights} = 4 \times (d_{\text{model}} \times d_{\text{model}}) = 4 \times (768 \times 768) = 4 \times 589{,}824 = 2{,}359{,}296$$
  $$\text{Biases} = 4 \times d_{\text{model}} = 4 \times 768 = 3{,}072$$
  $$\text{Total Attention Params} = 2{,}359{,}296 + 3{,}072 = 2{,}362{,}368$$

**Step 2: MLP Sub-Layer ($d_{\text{ff}} = 4 \times 768 = 3072$):**
- Layer 1 ($768 \to 3072$):
  $$\text{Weights} = 768 \times 3072 = 2{,}359{,}296, \quad \text{Bias} = 3{,}072$$
- Layer 2 ($3072 \to 768$):
  $$\text{Weights} = 3072 \times 768 = 2{,}359{,}296, \quad \text{Bias} = 768$$
  $$\text{Total MLP Params} = (2{,}359{,}296 + 3072) + (2{,}359{,}296 + 768) = 4{,}722{,}432$$

Notice: The MLP contains exactly **twice** the parameters of the attention sub-layer!

**Step 3: LayerNorm Layers:**
- Each LN has $\boldsymbol{\gamma} \in \mathbb{R}^{768}$ and $\boldsymbol{\beta} \in \mathbb{R}^{768} \implies 2 \times 768 = 1{,}536$ params.
- Two LN layers: $2 \times 1{,}536 = 3{,}072$ params.

**Step 4: Total Parameters per Block:**
$$\text{Total Block Params} = 2{,}362{,}368 + 4{,}722{,}432 + 3{,}072 = 7{,}087{,}872 \approx 7.09\text{ Million parameters}$$
</details>

---

### Problem 3: Mathematical Proof of Gradient Preservation in Pre-LN vs Post-LN (Hard)
Let $x_l$ denote the activation along the residual stream at layer $l$, where $l \in \{1, \dots, L\}$.
1. Write the recurrence relation for Post-LN versus Pre-LN.
2. Expand the recurrence relation to express $x_L$ directly in terms of the input $x_0$ and intermediate sublayer outputs $F_i(x_i)$.
3. Use the chain rule $\frac{\partial \mathcal{L}}{\partial x_l}$ to prove why Pre-LN prevents gradient degradation across deep stacks.

<details>
<summary>View Complete Solution</summary>

**Step 1: Recurrence relations:**
- **Post-LN:**
  $$x_{l+1} = \text{LN}(x_l + F_l(x_l))$$
- **Pre-LN:**
  $$x_{l+1} = x_l + F_l(\text{LN}(x_l))$$

**Step 2: Unrolling the recurrence:**
For Pre-LN, expanding from layer 0 to layer $L$:
$$x_L = x_0 + \sum_{i=0}^{L-1} F_i(\text{LN}(x_i))$$
The output $x_L$ is a direct linear identity sum of the input $x_0$ plus all sublayer perturbations!

**Step 3: Gradient Flow:**
By applying the chain rule to the Pre-LN expression:
$$\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \frac{\partial x_L}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \left( \mathbf{I} + \frac{\partial}{\partial x_l} \sum_{i=l}^{L-1} F_i(\text{LN}(x_i)) \right)$$

Notice the identity matrix $\mathbf{I}$. Even if the sub-layer Jacobians $\frac{\partial F_i}{\partial x_l}$ approach zero or vanish, the loss gradient $\frac{\partial \mathcal{L}}{\partial x_L} \cdot \mathbf{I}$ flows backward directly to layer $l$ without attenuation. In Post-LN, every layer normalizes the sum, multiplying gradients by the Jacobian of LayerNorm repeatedly ($(\mathbf{J}_{\text{LN}})^L$), leading to severe gradient scaling issues in deep networks.
</details>
