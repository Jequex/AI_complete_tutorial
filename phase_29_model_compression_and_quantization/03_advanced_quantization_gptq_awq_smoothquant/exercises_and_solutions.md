# Topic 03: Exercises & Solutions — Advanced Quantization: GPTQ, AWQ & SmoothQuant

---

### Exercise 1: Lagrange Multiplier Derivation for the Optimal Brain Surgeon (OBS) Update
**Problem:**
Let $\mathbf{H} \in \mathbb{R}^{K \times K}$ be a symmetric, positive-definite Hessian matrix.
We wish to quantize weight coordinate $q$ such that its displacement is constrained to $\mathbf{e}_q^\top \Delta \mathbf{w} = \delta_q \triangleq w_q - \hat{w}_q$, while minimizing the second-order loss perturbation:
$$\min_{\Delta \mathbf{w}} \frac{1}{2} \Delta \mathbf{w}^\top \mathbf{H} \Delta \mathbf{w} \quad \text{subject to} \quad \mathbf{e}_q^\top \Delta \mathbf{w} = \delta_q$$
1. Write down the Lagrangian function $\mathcal{L}(\Delta \mathbf{w}, \lambda)$ with scalar multiplier $\lambda$.
2. Derive the closed-form expression for $\Delta \mathbf{w}$ and scalar $\lambda$.
3. Compute the resulting minimal loss increase $\Delta E_q$.

<details>
<summary><strong>Solution</strong></summary>

1. **Lagrangian Formulation:**
   $$\mathcal{L}(\Delta \mathbf{w}, \lambda) = \frac{1}{2} \Delta \mathbf{w}^\top \mathbf{H} \Delta \mathbf{w} + \lambda (\mathbf{e}_q^\top \Delta \mathbf{w} - \delta_q)$$

2. **First-Order Optimality Conditions:**
   Setting the gradient with respect to $\Delta \mathbf{w}$ to zero:
   $$\nabla_{\Delta \mathbf{w}} \mathcal{L} = \mathbf{H} \Delta \mathbf{w} + \lambda \mathbf{e}_q = \mathbf{0}$$
   Since $\mathbf{H}$ is positive-definite, it is invertible:
   $$\Delta \mathbf{w} = -\lambda \mathbf{H}^{-1} \mathbf{e}_q$$
   Now substitute $\Delta \mathbf{w}$ into the equality constraint $\mathbf{e}_q^\top \Delta \mathbf{w} = \delta_q$:
   $$\mathbf{e}_q^\top \left( -\lambda \mathbf{H}^{-1} \mathbf{e}_q \right) = \delta_q$$
   $$(-\lambda) \cdot (\mathbf{e}_q^\top \mathbf{H}^{-1} \mathbf{e}_q) = \delta_q$$
   Notice that $\mathbf{e}_q^\top \mathbf{H}^{-1} \mathbf{e}_q = [\mathbf{H}^{-1}]_{qq}$, which is the $q$-th diagonal entry of the inverse Hessian.
   Therefore:
   $$\lambda = -\frac{\delta_q}{[\mathbf{H}^{-1}]_{qq}} = -\frac{w_q - \hat{w}_q}{[\mathbf{H}^{-1}]_{qq}}$$
   Substituting $\lambda$ back into $\Delta \mathbf{w}$:
   $$\boxed{\Delta \mathbf{w} = -\frac{w_q - \hat{w}_q}{[\mathbf{H}^{-1}]_{qq}} \mathbf{H}^{-1}_{:, q}}$$

3. **Minimal Loss Increase:**
   Substitute optimal $\Delta \mathbf{w}$ back into the quadratic objective:
   $$\Delta E_q = \frac{1}{2} \Delta \mathbf{w}^\top \mathbf{H} \Delta \mathbf{w} = \frac{1}{2} \left( -\lambda \mathbf{H}^{-1} \mathbf{e}_q \right)^\top \mathbf{H} \left( -\lambda \mathbf{H}^{-1} \mathbf{e}_q \right)$$
   $$= \frac{1}{2} \lambda^2 \mathbf{e}_q^\top \mathbf{H}^{-1} \mathbf{H} \mathbf{H}^{-1} \mathbf{e}_q = \frac{1}{2} \lambda^2 [\mathbf{H}^{-1}]_{qq}$$
   Substituting $\lambda^2 = \frac{\delta_q^2}{([\mathbf{H}^{-1}]_{qq})^2}$:
   $$\Delta E_q = \frac{1}{2} \frac{\delta_q^2}{[\mathbf{H}^{-1}]_{qq}} = \boxed{\frac{(w_q - \hat{w}_q)^2}{2 [\mathbf{H}^{-1}]_{qq}}}$$
</details>

---

### Exercise 2: Derivation of the Optimal SmoothQuant Scaling Factor
**Problem:**
Let $X_j$ denote the maximum absolute activation on channel $j$, and $W_j$ denote the maximum absolute weight on channel $j$.
To enable per-tensor or per-channel quantization of both $\hat{\mathbf{X}} = \mathbf{X} \mathbf{S}^{-1}$ and $\hat{\mathbf{W}} = \mathbf{S}\mathbf{W}$, we define the smoothed values:
$$\hat{X}_j = \frac{X_j}{s_j}, \quad \hat{W}_j = W_j \cdot s_j$$
We want to equalize the relative quantization difficulty of both matrices.
Show that setting $s_j = \frac{X_j^\alpha}{W_j^{1 - \alpha}}$ with $\alpha = 0.5$ makes the dynamic ranges of activations and weights identical for channel $j$.

<details>
<summary><strong>Solution</strong></summary>

Let us equate the smoothed maximums:
$$\hat{X}_j = \hat{W}_j \implies \frac{X_j}{s_j} = W_j \cdot s_j$$
Multiplying both sides by $s_j$:
$$s_j^2 = \frac{X_j}{W_j} \implies s_j = \sqrt{\frac{X_j}{W_j}} = \frac{X_j^{0.5}}{W_j^{0.5}}$$
Now compare this with the general SmoothQuant formula:
$$s_j = \frac{X_j^\alpha}{W_j^{1 - \alpha}}$$
Setting $\alpha = 0.5$ gives:
$$s_j = \frac{X_j^{0.5}}{W_j^{1 - 0.5}} = \frac{X_j^{0.5}}{W_j^{0.5}}$$
Under this choice:
$$\hat{X}_j = \frac{X_j}{X_j^{0.5} / W_j^{0.5}} = X_j^{0.5} W_j^{0.5} = \sqrt{X_j W_j}$$
$$\hat{W}_j = W_j \cdot \frac{X_j^{0.5}}{W_j^{0.5}} = W_j^{0.5} X_j^{0.5} = \sqrt{X_j W_j}$$
Thus, $\hat{X}_j = \hat{W}_j = \sqrt{X_j W_j}$. The dynamic ranges are mathematically identical, perfectly sharing the outlier dynamic range between both matrix operands!
</details>

---

### Exercise 3: Cholesky Factorization in GPTQ
**Problem:**
The naive Optimal Brain Quantizer updates the inverse Hessian $\mathbf{H}^{-1}$ by removing row and column $q$ at each step using a rank-1 update.
GPTQ instead performs Cholesky decomposition $\mathbf{H}^{-1} = \mathbf{L} \mathbf{L}^\top$ where $\mathbf{L}$ is lower-triangular.
1. Show how the $q$-th column of $\mathbf{H}^{-1}$ divided by $[\mathbf{H}^{-1}]_{qq}$ is directly related to the columns of $\mathbf{L}$.
2. Explain how this triangular property allows pre-computing all update vectors offline before quantizing a single weight.

<details>
<summary><strong>Solution</strong></summary>

1. Let $\mathbf{H}^{-1} = \mathbf{L} \mathbf{L}^\top$.
   The $q$-th diagonal entry of $\mathbf{H}^{-1}$ is:
   $$[\mathbf{H}^{-1}]_{qq} = \sum_{k=1}^q L_{qk}^2$$
   When eliminating weights sequentially from $q = 1$ to $K$, the effective inverse submatrix at step $q$ can be represented using the remaining columns of $\mathbf{L}$.
   Specifically, Frantar et al. (2022) prove that the sequential compensation vector for column $q$ is:
   $$\frac{\mathbf{H}^{-1}_{:, q}}{[\mathbf{H}^{-1}]_{qq}} = \frac{\mathbf{L}_{:, q}}{L_{qq}}$$
   
2. Because $\mathbf{L}$ is strictly lower-triangular, $L_{jq} = 0$ for all $j < q$.
   This guarantees that:
   - Quantizing weight $w_q$ only updates future weights $w_j$ for $j > q$.
   - It never perturbs already-quantized weights $w_k$ for $k < q$.
   - The entire Cholesky factor $\mathbf{L}$ depends only on the activation matrix $\mathbf{X}$ and can be computed **once** prior to quantization, enabling high-performance GPU block processing (128 columns at a time via `GEMM`).
</details>
