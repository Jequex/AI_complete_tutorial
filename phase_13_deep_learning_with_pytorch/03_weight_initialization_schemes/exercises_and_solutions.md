# Topic 13.3: Exercises & Solutions — Weight Initialization & Variance Dynamics

---

## Problem Set Overview

This exercise set explores the statistical mechanics of variance preservation in deep architectures, extending Kaiming initialization to generalized non-linearities (LeakyReLU) and analyzing residual connections.

---

### Problem 1: Generalized Kaiming Derivation for LeakyReLU

The LeakyReLU activation function allows a small non-zero slope $a \in [0, 1)$ for negative inputs:
$$\text{LeakyReLU}(z) = \begin{cases} z & \text{if } z > 0 \\ a z & \text{if } z \le 0 \end{cases}$$

Assume pre-activation $z \sim \mathcal{N}(0, \sigma_z^2)$ is symmetric around zero.
1. Compute the second moment $\mathbb{E}[\text{LeakyReLU}(z)^2]$ as a function of $\sigma_z^2$ and the slope parameter $a$.
2. For a linear layer $y = \sum_{i=1}^{n_{\text{in}}} w_i x_i$ followed by LeakyReLU, derive the required initialization variance $\sigma_w^2$ such that $\text{Var}(x_l) = \text{Var}(x_{l-1})$.
3. Verify that your formula reduces to standard Kaiming initialization when $a = 0$ (standard ReLU) and to linear Xavier when $a = 1$ (identity).

---

### Solution 1

1. **Second Moment Integration:**
   $$\mathbb{E}[\text{LeakyReLU}(z)^2] = \int_{-\infty}^0 (az)^2 p(z) \, dz + \int_0^\infty z^2 p(z) \, dz$$
   By symmetry of the Gaussian distribution $p(z) = p(-z)$, $\int_{-\infty}^0 z^2 p(z) dz = \int_0^\infty z^2 p(z) dz = \frac{1}{2} \sigma_z^2$.
   Therefore:
   $$\mathbb{E}[\text{LeakyReLU}(z)^2] = a^2 \left(\frac{1}{2} \sigma_z^2\right) + \left(\frac{1}{2} \sigma_z^2\right) = \frac{1 + a^2}{2} \sigma_z^2$$

2. **Derivation of Weight Variance $\sigma_w^2$:**
   From linear layer variance propagation:
   $$\sigma_z^2 = n_{\text{in}} \sigma_w^2 \text{Var}(x_{l-1})$$
   Passing through LeakyReLU:
   $$\text{Var}(x_l) = \frac{1 + a^2}{2} \sigma_z^2 = \frac{1 + a^2}{2} n_{\text{in}} \sigma_w^2 \text{Var}(x_{l-1})$$
   To maintain constant variance $\text{Var}(x_l) = \text{Var}(x_{l-1})$:
   $$\frac{1 + a^2}{2} n_{\text{in}} \sigma_w^2 = 1 \implies \sigma_w^2 = \frac{2}{(1 + a^2) n_{\text{in}}}$$

3. **Boundary Verification:**
   - For standard ReLU ($a = 0$):
     $$\sigma_w^2 = \frac{2}{(1 + 0^2) n_{\text{in}}} = \frac{2}{n_{\text{in}}} \quad (\text{Exact He/Kaiming Formula})$$
   - For identity / linear ($a = 1$):
     $$\sigma_w^2 = \frac{2}{(1 + 1^2) n_{\text{in}}} = \frac{1}{n_{\text{in}}} \quad (\text{Exact Glorot Forward Formula})$$

---

### Problem 2: Fan-In vs Fan-Out in Inverted Bottlenecks (MobileNetV2)

In an inverted bottleneck block with expansion factor $t=6$:
- Layer 1 (Expansion): $n_{\text{in}} = 32$, $n_{\text{out}} = 192$.
- Layer 2 (Projection): $n_{\text{in}} = 192$, $n_{\text{out}} = 32$.

1. Compute the Kaiming Normal standard deviation $\sigma_w$ under `mode='fan_in'` and `mode='fan_out'` for Layer 1.
2. In architectures with severe dimension mismatches ($n_{\text{in}} \ll n_{\text{out}}$ or $n_{\text{in}} \gg n_{\text{out}}$), why does choosing `mode='fan_in'` protect the forward pass while potentially destabilizing the backward pass?

---

### Solution 2

1. **Standard Deviation Computations (Layer 1):**
   - Under `mode='fan_in'`:
     $$\sigma_{\text{fan\_in}} = \sqrt{\frac{2}{n_{\text{in}}}} = \sqrt{\frac{2}{32}} = \sqrt{\frac{1}{16}} = 0.25$$
   - Under `mode='fan_out'`:
     $$\sigma_{\text{fan\_out}} = \sqrt{\frac{2}{n_{\text{out}}}} = \sqrt{\frac{2}{192}} = \sqrt{\frac{1}{96}} \approx 0.102$$

2. **Forward vs Backward Trade-off:**
   - `mode='fan_in'` scales weights inversely with input connections, guaranteeing that $\text{Var}(y) = \text{Var}(x)$ during the forward pass. However, when $n_{\text{out}} = 192 \gg n_{\text{in}} = 32$, the backward gradient variance $\text{Var}(\Delta x) = n_{\text{out}} \sigma_w^2 \text{Var}(\Delta y) = 192 \cdot (0.25)^2 \text{Var}(\Delta y) = 12 \cdot \text{Var}(\Delta y)$. The backward gradient expands by $12\times$ per layer!
   - In modern architectures, **Normalization layers (BatchNorm / LayerNorm)** absorb this discrepancy, making `mode='fan_in'` the universal safe default for forward stability.

---

### Problem 3: Variance Explosion in Residual Networks (ResNets)

Consider a deep ResNet where each residual block computes:
$$x_{l+1} = x_l + F(x_l, W_l)$$
Assume $x_l$ and $F(x_l, W_l)$ are uncorrelated, and $F$ is initialized such that $\text{Var}(F) = \text{Var}(x_l)$.
1. What is the variance of $x_L$ after $L=100$ residual blocks?
2. What initialization trick is used in production (e.g., Fixup or Zero-Init) to prevent this growth?

---

### Solution 3

1. **ResNet Variance Propagation:**
   $$\text{Var}(x_{l+1}) = \text{Var}(x_l) + \text{Var}(F) = \text{Var}(x_l) + \text{Var}(x_l) = 2 \cdot \text{Var}(x_l)$$
   After $L$ residual blocks:
   $$\text{Var}(x_L) = 2^L \text{Var}(x_0)$$
   For $L = 100$: $2^{100} \approx 1.26 \times 10^{30}$. The activations explode catastrophically!

2. **Production Fixes:**
   - **Zero-Init (Goyal et al., 2017):** Initialize the last normalization layer or final linear projection inside each residual branch to zero ($\gamma = 0$ or $W_{\text{final}} = 0$). This ensures $F(x) = 0$ at initialization, making the residual block an exact identity mapping ($x_{l+1} = x_l$) with $\text{Var}(x_L) = \text{Var}(x_0) = 1.0$.
   - **Residual Scaling (Transformers / GPT):** Multiply the residual branch by $\frac{1}{\sqrt{2L}}$ or $\frac{1}{\sqrt{2}}$.
