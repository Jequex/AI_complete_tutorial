# Topic 16.1: Exercises & Solutions — Scaled Dot-Product Attention

---

### Problem 1: Manual Step-by-Step Scaled Dot-Product Attention (Easy)

Consider a toy attention layer with sequence length $T = 2$, key dimension $d_k = 2$, and value dimension $d_v = 2$.

The query matrix $Q$, key matrix $K$, and value matrix $V$ are:
$$Q = \begin{bmatrix} 2.0 & 0.0 \\ 0.0 & 2.0 \end{bmatrix}, \quad K = \begin{bmatrix} 2.0 & 0.0 \\ 1.0 & 1.0 \end{bmatrix}, \quad V = \begin{bmatrix} 1.0 & 3.0 \\ 4.0 & 2.0 \end{bmatrix}$$

Using the canonical formula:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

1. Calculate the unscaled affinity matrix $S = Q K^T$.
2. Divide by $\sqrt{d_k} = \sqrt{2} \approx 1.4142$ to obtain the scaled logits matrix $Z = S / \sqrt{2}$.
3. Compute the row-wise softmax attention weights $A \in \mathbb{R}^{2 \times 2}$ (round to 4 decimal places).
4. Compute the final contextualized output tensor $\text{Attention}(Q, K, V) \in \mathbb{R}^{2 \times 2}$.

<details>
<summary>View Complete Solution</summary>

**Step 1: Compute $S = Q K^T$:**
$$K^T = \begin{bmatrix} 2.0 & 1.0 \\ 0.0 & 1.0 \end{bmatrix}$$

$$S = \begin{bmatrix} 2.0 & 0.0 \\ 0.0 & 2.0 \end{bmatrix} \begin{bmatrix} 2.0 & 1.0 \\ 0.0 & 1.0 \end{bmatrix} = \begin{bmatrix} (2)(2) + (0)(0) & (2)(1) + (0)(1) \\ (0)(2) + (2)(0) & (0)(1) + (2)(1) \end{bmatrix} = \begin{bmatrix} 4.0 & 2.0 \\ 0.0 & 2.0 \end{bmatrix}$$

**Step 2: Scale by $\sqrt{d_k} = \sqrt{2} \approx 1.41421$:**
$$Z = \frac{S}{\sqrt{2}} = \begin{bmatrix} \frac{4.0}{1.41421} & \frac{2.0}{1.41421} \\ \frac{0.0}{1.41421} & \frac{2.0}{1.41421} \end{bmatrix} = \begin{bmatrix} 2.8284 & 1.4142 \\ 0.0 & 1.4142 \end{bmatrix}$$

**Step 3: Row-Wise Softmax Normalization:**
- **Row 1:** $z_1 = [2.8284, 1.4142]$
  - Exponent shift: $m = 2.8284$
  - Exponents: $e^{2.8284 - 2.8284} = e^0 = 1.0$; $e^{1.4142 - 2.8284} = e^{-1.4142} \approx 0.2431$
  - Denominator: $1.0 + 0.2431 = 1.2431$
  - $A_{1, 1} = \frac{1.0}{1.2431} \approx 0.8044$
  - $A_{1, 2} = \frac{0.2431}{1.2431} \approx 0.1956$
  - Check sum: $0.8044 + 0.1956 = 1.0000$ ✓

- **Row 2:** $z_2 = [0.0, 1.4142]$
  - Exponent shift: $m = 1.4142$
  - Exponents: $e^{0.0 - 1.4142} = e^{-1.4142} \approx 0.2431$; $e^{1.4142 - 1.4142} = e^0 = 1.0$
  - Denominator: $0.2431 + 1.0 = 1.2431$
  - $A_{2, 1} = \frac{0.2431}{1.2431} \approx 0.1956$
  - $A_{2, 2} = \frac{1.0}{1.2431} \approx 0.8044$
  - Check sum: $0.1956 + 0.8044 = 1.0000$ ✓

$$A = \begin{bmatrix} 0.8044 & 0.1956 \\ 0.1956 & 0.8044 \end{bmatrix}$$

**Step 4: Compute Output $= A V$:**
$$V = \begin{bmatrix} 1.0 & 3.0 \\ 4.0 & 2.0 \end{bmatrix}$$

$$\text{Row 1} = 0.8044 \cdot [1.0, 3.0] + 0.1956 \cdot [4.0, 2.0] = [0.8044 + 0.7824, 2.4132 + 0.3912] = [1.5868, 2.8044]$$

$$\text{Row 2} = 0.1956 \cdot [1.0, 3.0] + 0.8044 \cdot [4.0, 2.0] = [0.1956 + 3.2176, 0.5868 + 1.6088] = [3.4132, 2.1956]$$

$$\text{Attention}(Q, K, V) = \begin{bmatrix} 1.5868 & 2.8044 \\ 3.4132 & 2.1956 \end{bmatrix}$$

</details>

---

### Problem 2: Variance of Logit Differences & Scaling Justification (Medium)

In self-attention, the softmax probability depends strictly on the difference between logits:
$$A_{i1} - A_{i2} \propto z_{i1} - z_{i2}$$

Let $\mathbf{q} \in \mathbb{R}^{d_k}$ be a query vector and let $\mathbf{k}_1, \mathbf{k}_2 \in \mathbb{R}^{d_k}$ be two distinct key vectors. Assume all elements of $\mathbf{q}, \mathbf{k}_1, \mathbf{k}_2$ are drawn i.i.d. from $\mathcal{N}(0, 1)$ and are mutually independent.

Define the unscaled logits:
$$z_1 = \mathbf{q}^T \mathbf{k}_1 = \sum_{m=1}^{d_k} q_m k_{1, m}, \quad z_2 = \mathbf{q}^T \mathbf{k}_2 = \sum_{m=1}^{d_k} q_m k_{2, m}$$

1. Compute the expectation $\mathbb{E}[z_1 - z_2]$.
2. Compute the variance $\operatorname{Var}(z_1 - z_2)$ of the unscaled logit difference.
3. Compute the standard deviation $\sigma(z_1 - z_2)$ as a function of $d_k$.
4. Evaluate $\sigma(z_1 - z_2)$ for $d_k = 64$ and $d_k = 2048$. Explain what happens to the softmax distribution when two random logits differ by $3 \sigma$.

<details>
<summary>View Complete Solution</summary>

**Step 1: Expectation:**
$$\mathbb{E}[z_1 - z_2] = \mathbb{E}[z_1] - \mathbb{E}[z_2] = 0 - 0 = 0$$

**Step 2: Variance:**
Recall that $\operatorname{Var}(z_1 - z_2) = \operatorname{Var}(z_1) + \operatorname{Var}(z_2) - 2\operatorname{Cov}(z_1, z_2)$.

From Section 5.1 of the lecture:
$$\operatorname{Var}(z_1) = d_k, \quad \operatorname{Var}(z_2) = d_k$$

Now compute $\operatorname{Cov}(z_1, z_2)$:
$$\operatorname{Cov}(z_1, z_2) = \mathbb{E}[z_1 z_2] - \mathbb{E}[z_1]\mathbb{E}[z_2] = \mathbb{E}\left[\left(\sum_{m=1}^{d_k} q_m k_{1, m}\right)\left(\sum_{l=1}^{d_k} q_l k_{2, l}\right)\right] - 0$$
$$= \sum_{m=1}^{d_k} \sum_{l=1}^{d_k} \mathbb{E}[q_m q_l k_{1, m} k_{2, l}]$$

By independence of $\mathbf{k}_1$ and $\mathbf{k}_2$, $\mathbb{E}[k_{1, m} k_{2, l}] = \mathbb{E}[k_{1, m}]\mathbb{E}[k_{2, l}] = 0 \cdot 0 = 0$ for all pairs $(m, l)$.
Thus, $\operatorname{Cov}(z_1, z_2) = 0$.

Therefore:
$$\operatorname{Var}(z_1 - z_2) = d_k + d_k - 0 = 2 d_k$$

**Step 3: Standard deviation:**
$$\sigma(z_1 - z_2) = \sqrt{2 d_k}$$

**Step 4: Evaluation and Softmax Analysis:**
- For $d_k = 64$: $\sigma = \sqrt{128} \approx 11.31$
- For $d_k = 2048$: $\sigma = \sqrt{4096} = 64.00$

If two unscaled logits differ by just $3 \sigma$:
- For $d_k = 64$, $\Delta z \approx 3 \times 11.31 = 33.93$. The ratio of their softmax probabilities is $e^{33.93} \approx 5.4 \times 10^{14}$!
- For $d_k = 2048$, $\Delta z \approx 3 \times 64 = 192$. The ratio is $e^{192} \approx 2.4 \times 10^{83}$!

This causes total **winner-take-all collapse**: one token gets probability $1.0$, all others get $0.0$. The gradient $\frac{\partial S}{\partial z} = S_i(\delta_{ij} - S_j) \to 0$, freezing model parameters permanently.

</details>

---

### Problem 3: Softmax Jacobian Vanishing Gradient Proof (Hard)

Consider a softmax vector $\mathbf{S} = \text{softmax}(\mathbf{z}) \in \mathbb{R}^T$ defined by:
$$S_i = \frac{e^{z_i}}{\sum_{j=1}^T e^{z_j}}$$

1. Derive the Jacobian matrix elements $J_{ij} = \frac{\partial S_i}{\partial z_j}$ for both $i = j$ and $i \ne j$.
2. Suppose due to lack of variance scaling, one logit $z_1 = M$ while all other logits $z_2 = z_3 = \dots = z_T = 0$ with $M \gg 1$.
   - Express $S_1$ and $S_k$ ($k \ge 2$) in terms of $M$ and $T$.
   - Prove that as $M \to \infty$, the maximum entry in the Jacobian matrix satisfies $\|J\|_\infty \to 0$ with rate $\mathcal{O}(e^{-M})$.
3. Explain the downstream consequence on the weight update $\Delta W^Q = -\eta \frac{\partial \mathcal{L}}{\partial W^Q}$.

<details>
<summary>View Complete Solution</summary>

**Step 1: Jacobian derivation:**
Case 1: $i = j$. Using the quotient rule on $S_i = \frac{e^{z_i}}{\sum_k e^{z_k}}$:
$$\frac{\partial S_i}{\partial z_i} = \frac{e^{z_i} \left(\sum_k e^{z_k}\right) - e^{z_i} e^{z_i}}{\left(\sum_k e^{z_k}\right)^2} = \frac{e^{z_i}}{\sum_k e^{z_k}} - \left(\frac{e^{z_i}}{\sum_k e^{z_k}}\right)^2 = S_i - S_i^2 = S_i (1 - S_i)$$

Case 2: $i \ne j$. Here the numerator $e^{z_i}$ is constant with respect to $z_j$:
$$\frac{\partial S_i}{\partial z_j} = \frac{0 - e^{z_i} e^{z_j}}{\left(\sum_k e^{z_k}\right)^2} = - \left(\frac{e^{z_i}}{\sum_k e^{z_k}}\right)\left(\frac{e^{z_j}}{\sum_k e^{z_k}}\right) = -S_i S_j$$

Combining into Kronecker delta notation:
$$J_{ij} = \frac{\partial S_i}{\partial z_j} = S_i (\delta_{ij} - S_j)$$

**Step 2: Asymptotic analysis as $M \to \infty$:**
With $z_1 = M$ and $z_2 = \dots = z_T = 0$:
$$\sum_{j=1}^T e^{z_j} = e^M + (T - 1)e^0 = e^M + T - 1$$

$$S_1 = \frac{e^M}{e^M + T - 1} = \frac{1}{1 + (T - 1)e^{-M}} \approx 1 - (T - 1)e^{-M}$$
$$S_k = \frac{1}{e^M + T - 1} \approx e^{-M} \quad (\text{for } k \ge 2)$$

Now evaluate the Jacobian terms:
- For the diagonal element $J_{11}$:
  $$J_{11} = S_1(1 - S_1) \approx [1 - (T-1)e^{-M}][(T-1)e^{-M}] \approx (T-1)e^{-M} = \mathcal{O}(e^{-M})$$
- For other diagonal elements $J_{kk}$ ($k \ge 2$):
  $$J_{kk} = S_k(1 - S_k) \approx e^{-M}(1 - e^{-M}) \approx e^{-M} = \mathcal{O}(e^{-M})$$
- For off-diagonal elements $J_{1k}$ ($k \ge 2$):
  $$J_{1k} = -S_1 S_k \approx -(1)e^{-M} = -e^{-M} = \mathcal{O}(e^{-M})$$
- For off-diagonal elements $J_{kl}$ ($k, l \ge 2, k \ne l$):
  $$J_{kl} = -S_k S_l \approx -(e^{-M})(e^{-M}) = -e^{-2M} = \mathcal{O}(e^{-2M})$$

As $M \to \infty$, every single entry in the Jacobian decays exponentially to zero at rate $\mathcal{O}(e^{-M})$.

**Step 3: Downstream training consequence:**
By the multivariate chain rule:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{z}} = \frac{\partial \mathcal{L}}{\partial \mathbf{S}} \cdot J$$

Since $J \to \mathbf{0}$, $\frac{\partial \mathcal{L}}{\partial \mathbf{z}} \approx \mathbf{0}$.
Because $z_{ij} = \frac{1}{\sqrt{d_k}} \mathbf{q}_i^T \mathbf{k}_j$, the gradient with respect to $Q$ and $W^Q$:
$$\frac{\partial \mathcal{L}}{\partial W^Q} = \sum_{ij} \frac{\partial \mathcal{L}}{\partial z_{ij}} \frac{\partial z_{ij}}{\partial W^Q} \to 0$$
The gradient update $\Delta W^Q = -\eta \frac{\partial \mathcal{L}}{\partial W^Q} \to 0$. Training completely stalls on a flat plateau, proving that scaling by $\sqrt{d_k}$ is essential for gradient preservation.

</details>
