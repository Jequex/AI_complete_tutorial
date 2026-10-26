# Topic 13.4: Exercises & Solutions — Normalization Layers

---

## Problem Set Overview

This exercise set investigates the mathematical backpropagation mechanics of normalization layers, architectural placement (Pre-LN vs Post-LN), and hardware-level roofline performance.

---

### Problem 1: Analytical Derivation of BatchNorm Backward Pass

Let mini-batch input be $\mathbf{x} = [x_1, \dots, x_B]^T \in \mathbb{R}^B$ along a single feature dimension.
The forward operations are:
$$\mu = \frac{1}{B} \sum_{i=1}^B x_i$$
$$\sigma^2 = \frac{1}{B} \sum_{i=1}^B (x_i - \mu)^2$$
$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$
$$y_i = \gamma \hat{x}_i + \beta$$

Let the incoming upstream gradient be $\frac{\partial \mathcal{L}}{\partial y_i}$.
1. Derive $\frac{\partial \mathcal{L}}{\partial \gamma}$ and $\frac{\partial \mathcal{L}}{\partial \beta}$.
2. Derive the upstream gradient through normalized activation $\frac{\partial \mathcal{L}}{\partial \hat{x}_i}$.
3. Using the chain rule, prove that the total gradient w.r.t input $x_i$ is given by:
   $$\frac{\partial \mathcal{L}}{\partial x_i} = \frac{\gamma}{B \sqrt{\sigma^2 + \epsilon}} \left( B \frac{\partial \mathcal{L}}{\partial y_i} - \sum_{j=1}^B \frac{\partial \mathcal{L}}{\partial y_j} - \hat{x}_i \sum_{j=1}^B \frac{\partial \mathcal{L}}{\partial y_j} \hat{x}_j \right)$$

---

### Solution 1

1. **Gradients for Affine Parameters:**
   Since $y_i = \gamma \hat{x}_i + \beta$:
   $$\frac{\partial \mathcal{L}}{\partial \gamma} = \sum_{i=1}^B \frac{\partial \mathcal{L}}{\partial y_i} \hat{x}_i$$
   $$\frac{\partial \mathcal{L}}{\partial \beta} = \sum_{i=1}^B \frac{\partial \mathcal{L}}{\partial y_i}$$

2. **Gradient for $\hat{x}_i$:**
   $$\frac{\partial \mathcal{L}}{\partial \hat{x}_i} = \frac{\partial \mathcal{L}}{\partial y_i} \gamma$$

3. **Gradient for $x_i$ via Chain Rule:**
   Notice that $x_i$ affects $\mathcal{L}$ through three pathways:
   - Directly via $\hat{x}_i$ in the numerator: $\frac{\partial \hat{x}_i}{\partial x_i} = \frac{1}{\sqrt{\sigma^2 + \epsilon}}$.
   - Indirectly via the mean $\mu$: $\frac{\partial \mu}{\partial x_i} = \frac{1}{B}$.
   - Indirectly via the variance $\sigma^2$: $\frac{\partial \sigma^2}{\partial x_i} = \frac{2(x_i - \mu)}{B}$.

   First, compute $\frac{\partial \mathcal{L}}{\partial \sigma^2}$:
   $$\frac{\partial \hat{x}_j}{\partial \sigma^2} = (x_j - \mu) \left(-\frac{1}{2}\right) (\sigma^2 + \epsilon)^{-3/2} = -\frac{1}{2} \hat{x}_j \frac{1}{\sigma^2 + \epsilon}$$
   $$\frac{\partial \mathcal{L}}{\partial \sigma^2} = -\frac{1}{2 (\sigma^2 + \epsilon)} \sum_{j=1}^B \frac{\partial \mathcal{L}}{\partial \hat{x}_j} \hat{x}_j$$

   Next, compute $\frac{\partial \mathcal{L}}{\partial \mu}$:
   $$\frac{\partial \mathcal{L}}{\partial \mu} = \sum_{j=1}^B \frac{\partial \mathcal{L}}{\partial \hat{x}_j} \left(-\frac{1}{\sqrt{\sigma^2 + \epsilon}}\right) + \frac{\partial \mathcal{L}}{\partial \sigma^2} \frac{\partial \sigma^2}{\partial \mu}$$
   Since $\frac{\partial \sigma^2}{\partial \mu} = -\frac{2}{B} \sum_{j=1}^B (x_j - \mu) = 0$:
   $$\frac{\partial \mathcal{L}}{\partial \mu} = -\frac{1}{\sqrt{\sigma^2 + \epsilon}} \sum_{j=1}^B \frac{\partial \mathcal{L}}{\partial \hat{x}_j}$$

   Finally, assemble $\frac{\partial \mathcal{L}}{\partial x_i}$:
   $$\frac{\partial \mathcal{L}}{\partial x_i} = \frac{\partial \mathcal{L}}{\partial \hat{x}_i} \frac{\partial \hat{x}_i}{\partial x_i} + \frac{\partial \mathcal{L}}{\partial \mu} \frac{\partial \mu}{\partial x_i} + \frac{\partial \mathcal{L}}{\partial \sigma^2} \frac{\partial \sigma^2}{\partial x_i}$$
   Substitute:
   $$\frac{\partial \mathcal{L}}{\partial x_i} = \frac{1}{\sqrt{\sigma^2 + \epsilon}} \left( \frac{\partial \mathcal{L}}{\partial \hat{x}_i} - \frac{1}{B} \sum_{j=1}^B \frac{\partial \mathcal{L}}{\partial \hat{x}_j} - \frac{1}{B} \hat{x}_i \sum_{j=1}^B \frac{\partial \mathcal{L}}{\partial \hat{x}_j} \hat{x}_j \right)$$
   Factoring out $\frac{\gamma}{B \sqrt{\sigma^2 + \epsilon}}$ where $\frac{\partial \mathcal{L}}{\partial \hat{x}_i} = \gamma \frac{\partial \mathcal{L}}{\partial y_i}$:
   $$\frac{\partial \mathcal{L}}{\partial x_i} = \frac{\gamma}{B \sqrt{\sigma^2 + \epsilon}} \left( B \frac{\partial \mathcal{L}}{\partial y_i} - \sum_{j=1}^B \frac{\partial \mathcal{L}}{\partial y_j} - \hat{x}_i \sum_{j=1}^B \frac{\partial \mathcal{L}}{\partial y_j} \hat{x}_j \right)$$
   This completes the proof. Notice that $\sum_{i=1}^B \frac{\partial \mathcal{L}}{\partial x_i} \equiv 0$, guaranteeing zero-mean gradient updates!

---

### Problem 2: Pre-LN vs Post-LN in Deep Transformers

In early Transformers (Vaswani et al., 2017), LayerNorm was placed **after** the residual addition (**Post-LN**):
$$x_{l+1} = \text{LayerNorm}(x_l + \text{SubLayer}(x_l))$$

In modern Transformers (GPT-3, LLaMA), LayerNorm is placed **before** the sublayer (**Pre-LN**):
$$x_{l+1} = x_l + \text{SubLayer}(\text{LayerNorm}(x_l))$$

1. Analyze the gradient pathway from layer $L$ to layer $1$ for both architectures.
2. Why does Post-LN require a delicate learning rate warmup schedule, whereas Pre-LN trains stably out-of-the-box?

---

### Solution 2

1. **Gradient Pathway Analysis:**
   - **Post-LN:** The input $x_{l+1}$ is normalized at every block. Unrolling the recurrence, the gradient flowing back from layer $L$ to layer $l$ is multiplied by the Jacobian of the normalization operator at every step:
     $$\frac{\partial x_L}{\partial x_1} \propto \prod_{k=1}^L \mathbf{J}_{\text{LN}_k}$$
     As depth $L$ grows, this repeated product causes gradients in earlier layers to diminish exponentially (vanishing gradient near the input embeddings).
   - **Pre-LN:** The residual stream acts as an uninhibited gradient highway:
     $$x_L = x_0 + \sum_{l=0}^{L-1} \text{SubLayer}(\text{LN}(x_l))$$
     The derivative is:
     $$\frac{\partial x_L}{\partial x_0} = \mathbf{I} + \sum_{l=0}^{L-1} \frac{\partial \text{SubLayer}}{\partial x_0}$$
     The identity matrix $\mathbf{I}$ guarantees that gradients flow cleanly back to $x_0$ without decaying.

2. **Warmup Requirement:**
   In Post-LN, at initialization, the expected gradient norm in earlier layers is tiny compared to final layers. Without learning rate warmup (slowly ramping $\text{lr}$ from 0 to $\text{lr}_{\max}$), early layers fail to learn while final layers diverge. Pre-LN preserves identity gradient flow, eliminating the strict dependency on warmup.

---

### Problem 3: Memory Bandwidth Analysis of RMSNorm vs LayerNorm

Consider a Transformer hidden state $x \in \mathbb{R}^{B \times L \times D}$ where $B=8, L=2048, D=4096$ using 16-bit precision (FP16 / BF16 = 2 bytes per element).

1. How many total floating-point numbers are in this tensor, and what is its raw memory size?
2. Compare the number of memory reads/writes required for LayerNorm vs RMSNorm.
3. On an NVIDIA A100 GPU (memory bandwidth 2,039 GB/s), explain why reducing memory transactions in normalization layers directly translates to higher decoding tokens-per-second.

---

### Solution 3

1. **Tensor Size:**
   $$N = 8 \times 2048 \times 4096 = 67,108,864 \text{ elements}$$
   $$\text{Size} = 67,108,864 \times 2 \text{ bytes} = 134,217,728 \text{ bytes} = 128 \text{ MB}$$

2. **Memory Operations:**
   - **LayerNorm:**
     1. Read $X$ to compute $\mu$ (128 MB read).
     2. Read $X$ again to compute $\sigma^2 = \frac{1}{D}\sum (x - \mu)^2$ (128 MB read).
     3. Read $X$, $\gamma$, $\beta$ to write normalized $Y$ (128 MB read + 128 MB write).
     Total I/O $\approx 512$ MB.
   - **RMSNorm:**
     1. Read $X$ once to compute $\text{RMS} = \sqrt{\frac{1}{D}\sum x^2}$ (128 MB read).
     2. Read $X$ and $\gamma$ to write $Y$ (128 MB read + 128 MB write).
     Total I/O $\approx 384$ MB.
   RMSNorm reduces memory traffic by $25\%$ ($128$ MB saved per layer)!

3. **GPU Roofline Impact:**
   Normalization kernels have very low arithmetic intensity ($\sim 2-4$ FLOPs per byte loaded). The GPU compute cores (Tensor Cores) finish calculations in microseconds and spend the majority of time waiting for High Bandwidth Memory (HBM). By reducing data movement by 25%, RMSNorm directly eliminates memory bus bottlenecks, directly increasing tokens generated per second.
