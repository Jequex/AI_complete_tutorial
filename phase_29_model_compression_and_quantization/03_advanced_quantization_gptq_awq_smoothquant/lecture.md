# Topic 03: Advanced Quantization: GPTQ, AWQ & SmoothQuant

## 1. Introduction & 12-Year-Old Intuition

Imagine you manage a synchronized acrobatic team of **100 gymnasts** balancing a massive pyramid.

### The Naive Approach (Round-To-Nearest / RTN)
A judge tells everyone: *"Everyone must immediately round their exact foot placement to the nearest painted grid line on the stage floor!"*
All 100 gymnasts simultaneously snap their feet to the nearest line. Because everyone made tiny individual rounding errors without coordinating with each other, the entire human pyramid wobbles and crashes down.

### The GPTQ Solution (Optimal Brain Surgeon)
Instead of everyone moving at once, the gymnasts move **one at a time**:
1. Acrobat #1 snaps their foot to the nearest grid line.
2. Acrobat #1 makes a tiny mistake of $+2$ inches.
3. Because the team knows the physics of the pyramid (the **Hessian matrix $\mathbf{H}$**), all 99 remaining acrobats instantly shift their weight by fractions of an inch to cancel out Acrobat #1's mistake!
4. Then Acrobat #2 snaps their foot, and the remaining 98 re-balance.
By the time the last acrobat snaps into place, the pyramid stands rock-solid!

### The SmoothQuant / AWQ Solution (The Backpack Rebalancing Trick)
Imagine two hikers on a mountain trail: **Hiker Activation** is carrying a ridiculously heavy **100-pound iron boulder** in their backpack (an activation outlier), while **Hiker Weight** is carrying an empty backpack weighing only 1 pound.
Hiker Activation is stumbling and about to collapse.
Instead of forcing Hiker Activation to squeeze their giant boulder into a tiny lunchbox, you take 50 pounds of iron out of their backpack and put it into Hiker Weight's backpack.
The total weight carried by the pair is **identical**, but now both hikers are carrying a reasonable 50-pound load that easily fits into standard packs!

---

## 2. Mathematical Symbol Decoder

| Symbol | Name | Formal Type | Intuition / Role |
| :--- | :--- | :--- | :--- |
| $\mathbf{X} \in \mathbb{R}^{M \times K}$ | Calibration Activations | Real matrix | Incoming activation features for a linear layer across $M$ tokens and $K$ channels. |
| $\mathbf{W} \in \mathbb{R}^{K \times N}$ | Weight Matrix | Real matrix | Layer weight parameters connecting $K$ inputs to $N$ output projections. |
| $\mathbf{H} = 2 \mathbf{X}^\top \mathbf{X}$ | Hessian Matrix | Symmetric PSD matrix $\in \mathbb{R}^{K \times K}$ | Curvature of the squared output error with respect to weight deviations $\Delta \mathbf{w}$. |
| $\mathbf{H}^{-1}$ | Inverse Hessian | $\mathbb{R}^{K \times K}$ | Sensitivity and cross-coupling inverse metric used to compensate quantization errors. |
| $[\mathbf{H}^{-1}]_{qq}$ | Diagonal Inverse Pivot | Positive scalar $\in \mathbb{R}^+$ | Quantization penalty cost for rounding weight coordinate $q$. |
| $\mathbf{S} = \text{diag}(s_1, \dots, s_K)$ | Smoothing / Scaling Matrix | Diagonal positive matrix | Channel-wise transformation migrating dynamic range between activations and weights. |
| $\alpha \in [0, 1]$ | Migration Exponent | Scalar hyperparameter | Tradeoff parameter allocating outlier burden between $\mathbf{X}$ and $\mathbf{W}$ (typically 0.5). |
| $\mathbf{s}_X \in \mathbb{R}^K$ | Activation Salience Vector | Vector: $\max_{m} |X_{m, j}|$ | Maximum per-channel magnitude identifying which weights must be protected. |

---

## 3. Second-Order Taylor Approximation & Optimal Brain Surgery

When compressing a trained model without access to full labeled retraining, we seek to minimize the change in model loss $\Delta \mathcal{L} = \mathcal{L}(\hat{\mathbf{w}}) - \mathcal{L}(\mathbf{w})$.

### 3.1 Taylor Series Expansion
Expanding the loss around the pre-trained continuous weights $\mathbf{w}$:
$$\mathcal{L}(\hat{\mathbf{w}}) = \mathcal{L}(\mathbf{w}) + \mathbf{g}^\top (\hat{\mathbf{w}} - \mathbf{w}) + \frac{1}{2} (\hat{\mathbf{w}} - \mathbf{w})^\top \mathbf{H} (\hat{\mathbf{w}} - \mathbf{w}) + \mathcal{O}(\|\hat{\mathbf{w}} - \mathbf{w}\|^3)$$
where:
- $\mathbf{g} = \nabla_{\mathbf{w}} \mathcal{L}$ is the loss gradient.
- $\mathbf{H} = \nabla_{\mathbf{w}}^2 \mathcal{L}$ is the Hessian matrix.

Because the pre-trained weights sit at a local minimum of the training loss, the gradient is approximately zero: $\mathbf{g} \approx \mathbf{0}$.
Neglecting third-order terms, the quantization penalty is governed entirely by the quadratic form:
$$\Delta \mathcal{L} \approx \frac{1}{2} \Delta \mathbf{w}^\top \mathbf{H} \Delta \mathbf{w}$$

### 3.2 The Layer-Wise Linear Regression Formulation
For a linear layer $\mathbf{Y} = \mathbf{X}\mathbf{W}$, minimizing the layer output perturbation $\|\mathbf{X}\mathbf{w} - \mathbf{X}\hat{\mathbf{w}}\|_2^2$ is an optimal proxy for preserving global task loss.
Let $\Delta \mathbf{w} = \mathbf{w} - \hat{\mathbf{w}}$. The squared error is:
$$\|\mathbf{X}\mathbf{w} - \mathbf{X}\hat{\mathbf{w}}\|_2^2 = (\mathbf{w} - \hat{\mathbf{w}})^\top (\mathbf{X}^\top \mathbf{X}) (\mathbf{w} - \hat{\mathbf{w}})$$
Thus, the empirical Hessian for layer $\mathbf{W}$ is:
$$\mathbf{H} = 2 \mathbf{X}^\top \mathbf{X}$$

---

## 4. GPTQ: Accurate Post-Training Quantization for Large Language Models

Originating from Optimal Brain Surgeon (OBS, Hassibi & Stork 1993) and Optimal Brain Quantizer (OBQ, Frantar et al. 2022), **GPTQ** (Frantar et al. 2022) scales second-order quantization to 175-billion parameter models in a matter of hours.

### 4.1 The Second-Order Weight Update Derivation
Suppose we quantize a single scalar weight $w_q$ to discrete grid value $\hat{w}_q$, introducing error $\delta_q = w_q - \hat{w}_q$.
We want to update all remaining unquantized weights $\mathbf{w}_F$ (where $F$ is the set of free weights) to minimize:
$$\min_{\Delta \mathbf{w}} \frac{1}{2} \Delta \mathbf{w}^\top \mathbf{H} \Delta \mathbf{w} \quad \text{subject to} \quad \mathbf{e}_q^\top \Delta \mathbf{w} = \delta_q$$

Using the method of Lagrange multipliers:
$$\mathcal{L}(\Delta \mathbf{w}, \lambda) = \frac{1}{2} \Delta \mathbf{w}^\top \mathbf{H} \Delta \mathbf{w} + \lambda (\mathbf{e}_q^\top \Delta \mathbf{w} - \delta_q)$$

Taking derivatives and setting to zero:
$$\nabla_{\Delta \mathbf{w}} \mathcal{L} = \mathbf{H} \Delta \mathbf{w} + \lambda \mathbf{e}_q = \mathbf{0} \implies \Delta \mathbf{w} = -\lambda \mathbf{H}^{-1} \mathbf{e}_q$$
Multiply both sides by $\mathbf{e}_q^\top$:
$$\mathbf{e}_q^\top \Delta \mathbf{w} = -\lambda \mathbf{e}_q^\top \mathbf{H}^{-1} \mathbf{e}_q = -\lambda [\mathbf{H}^{-1}]_{qq}$$
Since $\mathbf{e}_q^\top \Delta \mathbf{w} = \delta_q$:
$$\delta_q = -\lambda [\mathbf{H}^{-1}]_{qq} \implies \lambda = -\frac{\delta_q}{[\mathbf{H}^{-1}]_{qq}}$$

Substitute $\lambda$ back into the update expression:
$$\boxed{\Delta \mathbf{w} = -\frac{w_q - \hat{w}_q}{[\mathbf{H}^{-1}]_{qq}} \mathbf{H}^{-1}_{:, q}}$$

The corresponding increase in squared reconstruction error is:
$$\Delta E_q = \frac{(w_q - \hat{w}_q)^2}{2 [\mathbf{H}^{-1}]_{qq}}$$

### 4.2 GPTQ Algorithmic Pipeline
1. Compute $\mathbf{H} = 2 \mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I}$.
2. Compute the Cholesky decomposition of $\mathbf{H}^{-1} = \mathbf{L} \mathbf{L}^\top$.
3. **Block-Wise Lazy Updates**: Instead of updating all $K$ remaining weights after every single scalar quantization (which is memory-bandwidth bound), GPTQ processes weights in blocks (e.g. 128 columns) and updates the remainder using fast matrix-matrix multiplications (`GEMM`).

---

## 5. Activation-Aware Weight Quantization (AWQ)

Introduced by Song Han's lab (Lin et al. 2023), **AWQ** makes a profound discovery:
> **Weights in an LLM are NOT equally important.** Protecting only the top **1% most salient weights** preserves virtually all perplexity, even under 4-bit and 3-bit quantization!

```
Activation Channels:
[ Ch 0: 0.2 ] [ Ch 1: 0.1 ] [ Ch 2: 85.4 (Outlier!) ] [ Ch 3: 0.3 ]
                                     │
                                     ▼
Corresponding Weight Matrix Column:
                            [ W_{:, 2} is 100x MORE SALIENT! ]
```

### 5.1 Salience Metric
AWQ determines weight salience not by weight magnitude $|W|$, but by the **magnitude of the incoming activation channel**:
$$\mathbf{s}_X = \max_{\text{tokens}} |\mathbf{X}| \in \mathbb{R}^K$$
Channels with large activation spikes have high leverage on the final attention or MLP projections.

### 5.2 Mathematically Equivalent Scaling Transformation
Rather than storing the 1% salient weights in FP16 (which ruins regular integer tensor core kernels), AWQ applies a channel-wise per-channel transformation matrix $\mathbf{S} = \text{diag}(\mathbf{s})$:
$$\mathbf{Y} = \mathbf{X}\mathbf{W} = (\mathbf{X} \mathbf{S}^{-1}) (\mathbf{S}\mathbf{W})$$
By scaling up the salient weights $\mathbf{W}' = \mathbf{S}\mathbf{W}$ and scaling down activations $\mathbf{X}' = \mathbf{X}\mathbf{S}^{-1}$, the relative quantization error on the salient channels is compressed to near-zero.
The optimal scale $\mathbf{s}^*$ is found by grid-searching over:
$$\mathbf{s} = \mathbf{s}_X^\gamma$$
where $\gamma \in [0, 1]$ minimizes output reconstruction MSE.

---

## 6. SmoothQuant: Resolving the Activation Outlier Dilemma

In Large Language Models exceeding 6.7B parameters (e.g., OPT-13B, LLaMA-65B), **activation outliers** emerge systematically in specific feature dimensions (Dettmers et al. 2022). These outlier features can be $100\times$ larger than normal activations.

```
Activations X (Difficult: Outliers up to 100.0)    Weights W (Easy: Gaussian, max 0.5)
              │                                                 │
              └───────────────┬─────────────────────────────────┘
                              ▼
                     SmoothQuant Matrix S
                              ▼
Scaled Activations X * S^-1 (Max: 8.0)        Scaled Weights S * W (Max: 6.0)
     [Both easily fit into clean INT8 registers without clipping or precision loss!]
```

### 6.1 The Mathematical Migration
SmoothQuant (Xiao et al. 2023) observes that while activation quantization is hard due to outliers, weight quantization is easy because weights are uniformly distributed.
SmoothQuant introduces per-channel scaling matrix $\mathbf{S} = \text{diag}(s_1, s_2, \dots, s_K)$:
$$\mathbf{Y} = \mathbf{X}\mathbf{W} = (\mathbf{X} \mathbf{S}^{-1}) (\mathbf{S} \mathbf{W}) = \hat{\mathbf{X}} \hat{\mathbf{W}}$$

To balance the difficulty equally between activations and weights, the per-channel migration factor $s_j$ is defined as:
$$\boxed{s_j = \frac{\max(|X_j|)^\alpha}{\max(|W_j|)^{1 - \alpha}}}$$

### 6.2 The Migration Exponent $\alpha$
- When $\alpha = 1$: All activation outliers are migrated completely into weights.
- When $\alpha = 0$: Weights remain untouched (naive quantization).
- When $\alpha = 0.5$: The outlier dynamic range is split geometrically 50/50 between activations and weights, allowing both $\hat{\mathbf{X}}$ and $\hat{\mathbf{W}}$ to be cleanly quantized to INT8 (W8A8) with zero runtime latency overhead.

---

## 7. Comparative Architecture Matrix

| Feature | Naive RTN | GPTQ | AWQ | SmoothQuant |
| :--- | :--- | :--- | :--- | :--- |
| **Quantization Target** | Weights or Acts | Weights only (W4 / W3) | Weights only (W4 / W3) | Weights & Activations (W8A8) |
| **Hardware Core** | Any | Pure weight dequant (W4A16) | Pure weight dequant (W4A16) | Pure INT8 Tensor Core (W8A8) |
| **Core Principle** | Independent rounding | 2nd-order inverse Hessian compensation | Salient weight protection via scaling | Channel-wise outlier migration |
| **Calibration Cost** | None | Low (~128 tokens, ~1 hour) | Minimal (~32 tokens, ~10 mins) | Minimal (~512 tokens, ~5 mins) |
| **Speedup Source** | Memory bandwidth | Memory bandwidth | Memory bandwidth | Matrix compute + memory bandwidth |
