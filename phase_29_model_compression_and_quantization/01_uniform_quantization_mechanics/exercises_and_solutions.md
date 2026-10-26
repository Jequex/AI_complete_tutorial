# Topic 01: Exercises & Solutions — Uniform Quantization Mechanics

---

### Exercise 1: Zero-Point Integer Constraining and Exact Zero Preservation
**Problem:**
In deep neural networks, exact zeros occur frequently due to zero-padding in convolutions, ReLU activations ($\max(0, x)$), and sparse embeddings.
1. Show mathematically that if $z$ were allowed to be an arbitrary real number $z \in \mathbb{R}$ rather than an integer $z \in \mathbb{Z}$, the quantized code $q$ for $x = 0$ would round to an integer, causing $x = 0$ to dequantize to a non-zero value $\hat{x} \neq 0$.
2. Derive the maximum dequantization error $|\hat{x} - 0|$ for real zero when $z$ is rounded to the nearest integer.

<details>
<summary><strong>Solution</strong></summary>

1. Let $x = 0.0$. Under the affine formula before integer rounding of the zero-point:
   $$z_{\text{real}} = -\frac{x_{\min}}{s} + q_{\min}$$
   If we evaluate quantization of $x = 0$:
   $$q(0) = \left\lfloor \frac{0}{s} + z_{\text{real}} \right\rceil = \lfloor z_{\text{real}} \rceil$$
   If $z_{\text{real}} \notin \mathbb{Z}$, then $\lfloor z_{\text{real}} \rceil \neq z_{\text{real}}$.
   When dequantized:
   $$\hat{x}(0) = s \left( q(0) - z_{\text{real}} \right) = s \left( \lfloor z_{\text{real}} \rceil - z_{\text{real}} \right)$$
   Because $\lfloor z_{\text{real}} \rceil - z_{\text{real}} \neq 0$, the reconstructed value $\hat{x}(0) \neq 0$. This destroys sparsity: every padded zero or ReLU-deactivated neuron in a tensor would become a non-zero floating-point number, causing severe numerical drift and breaking zero-skipping hardware optimizations.

2. By defining $z = \lfloor z_{\text{real}} \rceil \in \mathbb{Z}$ as part of the quantizer specification, the dequantization formula uses the exact integer $z$:
   $$\hat{x} = s(q - z)$$
   Now, evaluating at $x = 0$:
   $$q(0) = \left\lfloor \frac{0}{s} + z \right\rceil = \lfloor z \rceil = z$$
   Then dequantization yields:
   $$\hat{x}(0) = s(z - z) = 0.0 \quad \text{(Exact Zero)}$$
   The rounding of $z_{\text{real}}$ to $z$ shifts the effective dynamic range boundaries by at most $|z - z_{\text{real}}| \le 0.5$ LSBs, shifting the effective $x_{\min}$ by at most $\frac{s}{2}$, but guaranteeing that zero is represented with zero error.
</details>

---

### Exercise 2: Hardware Dot Product Cross-Terms in Asymmetric vs. Symmetric Quantization
**Problem:**
Consider two continuous vectors $\mathbf{x}, \mathbf{w} \in \mathbb{R}^K$. Let their quantized representations be $q_x^{(k)} \in [q_{\min}^x, q_{\max}^x]$ with scale $s_x$ and zero-point $z_x$, and $q_w^{(k)} \in [q_{\min}^w, q_{\max}^w]$ with scale $s_w$ and zero-point $z_w$.
1. Derive the algebraic expansion for the inner product $\mathbf{x}^\top \mathbf{w} \approx \hat{\mathbf{x}}^\top \hat{\mathbf{w}}$.
2. Identify which terms can be precomputed offline during compilation, and contrast the runtime computation of asymmetric quantization vs symmetric quantization ($z_x = 0, z_w = 0$).

<details>
<summary><strong>Solution</strong></summary>

1. Substituting the dequantized values $\hat{x}_k = s_x (q_x^{(k)} - z_x)$ and $\hat{w}_k = s_w (q_w^{(k)} - z_w)$:
   $$\hat{\mathbf{x}}^\top \hat{\mathbf{w}} = \sum_{k=1}^K s_x (q_x^{(k)} - z_x) \cdot s_w (q_w^{(k)} - z_w)$$
   Factoring out the scales $s_x s_w$:
   $$\hat{\mathbf{x}}^\top \hat{\mathbf{w}} = s_x s_w \sum_{k=1}^K \left( q_x^{(k)} q_w^{(k)} - z_w q_x^{(k)} - z_x q_w^{(k)} + z_x z_w \right)$$
   Expanding into four distinct summations:
   $$\hat{\mathbf{x}}^\top \hat{\mathbf{w}} = s_x s_w \left[ \underbrace{\sum_{k=1}^K q_x^{(k)} q_w^{(k)}}_{\text{Term 1: Integer MAC}} - \underbrace{z_w \sum_{k=1}^K q_x^{(k)}}_{\text{Term 2: Activation Reduction}} - \underbrace{z_x \sum_{k=1}^K q_w^{(k)}}_{\text{Term 3: Weight Reduction}} + \underbrace{K \cdot z_x z_w}_{\text{Term 4: Constant Offset}} \right]$$

2. **Offline Precomputations and Runtime Complexity**:
   - **Term 3 ($z_x \sum_{k=1}^K q_w^{(k)}$) and Term 4 ($K z_x z_w$)**: Weights $\mathbf{w}$ are static at inference time. Therefore, $\sum_{k=1}^K q_w^{(k)}$ is calculated once during model compilation/loading and stored as a constant bias vector!
   - **Term 2 ($z_w \sum_{k=1}^K q_x^{(k)}$)**: Depends on runtime activations $q_x$. If $z_w \neq 0$, the hardware must compute the row-sum of incoming activation tokens on the fly.
   - **Symmetric Weights ($z_w = 0$)**: In modern deep learning inference (e.g. TensorRT, ONNX Runtime, llama.cpp), weights are always symmetrically quantized ($z_w = 0$). Setting $z_w = 0$ immediately zeroes out Term 2 and Term 4!
   - **Fully Symmetric ($z_x = 0, z_w = 0$)**:
     $$\hat{\mathbf{x}}^\top \hat{\mathbf{w}} = s_x s_w \sum_{k=1}^K q_x^{(k)} q_w^{(k)}$$
     Only Term 1 remains: a pure, blistering fast integer multiply-accumulate (`dp4a` on NVIDIA GPUs or `sdot` on ARM NEON), scaled by a single floating-point scalar at the end.
</details>

---

### Exercise 3: Optimal Clipping Threshold for Gaussian Weights
**Problem:**
Let weight tensor elements be independent samples from standard normal distribution $W \sim \mathcal{N}(0, \sigma^2)$.
A symmetric $b$-bit quantizer has clipping boundary $[-c, c]$ with step size $\Delta = \frac{2c}{2^b - 1}$.
The total expected Mean Squared Error is the sum of rounding error and clipping error:
$$\text{MSE}(c) = \int_{-c}^c \frac{\Delta^2}{12} p(w) dw + 2 \int_c^\infty (w - c)^2 p(w) dw$$
Explain qualitatively and mathematically why setting $c = \max(|W|)$ is suboptimal for large tensors, and describe how percentile calibration reduces total MSE.

<details>
<summary><strong>Solution</strong></summary>

For a Gaussian tensor of size $N$ (e.g., $N = 10^7$ parameters in a transformer layer), extreme value theory shows that the expected maximum sample magnitude grows as:
$$\mathbb{E}[\max_{1 \le i \le N} |W_i|] \approx \sigma \sqrt{2 \ln(N)}$$
For $N = 10^7$, $\sqrt{2 \ln(10^7)} \approx 5.68 \sigma$.
1. If we set $c = \max(|W|) \approx 5.68 \sigma$ (MinMax calibration), the clipping error is zero. However, the step size becomes:
   $$\Delta = \frac{2 \times 5.68 \sigma}{2^b - 1}$$
   Because $\sigma_e^2 \propto \Delta^2 \propto c^2$, the rounding noise variance across all $99.9999\%$ of the weights in $[ -3\sigma, 3\sigma ]$ is inflated by a factor of $\left(\frac{5.68}{3}\right)^2 \approx 3.6\times$!
2. By clipping at $c \approx 3.0 \sigma$ to $3.5 \sigma$ (e.g. 99.9th percentile), only $0.1\%$ of the extreme tails get clamped to $c$. The penalty paid on those few outliers is vastly outweighed by shrinking $\Delta^2$ by $>60\%$ for the other $99.9\%$ of the parameters.
3. Therefore, the optimal clipping threshold $c^*$ that minimizes total $\text{MSE}(c)$ strictly satisfies $c^* < \max(|W|)$.
</details>
