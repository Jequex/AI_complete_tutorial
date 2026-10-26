# Topic 18.4: Exercises & Solutions — Modern LLM Architectures

---

## Exercise 1: KV-Cache Sizing & Inference Arithmetic Intensity Analysis

### Problem
An inference engine serves a 70-billion-parameter model ($L = 80$ layers, $d_{model} = 8192$, $H_Q = 64$ query heads, $d_{head} = 128$) on an NVIDIA H100 GPU ($80\text{ GB}$ HBM, $3.35\text{ TB/s}$ memory bandwidth, 1,000 TFLOPS BF16 compute).
Tokens are represented in 16-bit precision (2 bytes).
1. Formulate the KV-cache memory footprint for a context of $T = 32,768$ tokens for a single request under:
   - Multi-Head Attention (MHA: $H_{KV} = 64$)
   - Grouped-Query Attention (GQA-8: $H_{KV} = 8$)
2. During autoregressive decoding of 1 token with batch size $B = 1$:
   - The forward pass executes $\approx 2N = 140 \times 10^9$ FLOPs.
   - The GPU must read the entire model weights ($140\text{ GB}$) plus the KV-cache from HBM.
   Calculate the arithmetic intensity:
   $$\text{Arithmetic Intensity} = \frac{\text{Total FLOPs}}{\text{Total Bytes Transferred from HBM}}$$
   under both MHA and GQA-8.
3. What is the minimum batch size $B^*$ needed for the model to transition from memory-bandwidth bound to compute-bound on the H100?

---

### Solution

#### Step 1: KV-Cache Footprint per Single Request ($T = 32,768$)
$$\text{Formula: } \text{Bytes} = 2 \cdot L \cdot T \cdot H_{KV} \cdot d_{head} \cdot 2 \text{ bytes}$$

- **For Multi-Head Attention (MHA: $H_{KV} = 64$):**
  $$\text{Bytes}_{\text{MHA}} = 2 \cdot 80 \cdot 32,768 \cdot 64 \cdot 128 \cdot 2 = 85,899,345,920 \text{ bytes} \approx \mathbf{85.90 \text{ GB}}$$
  Notice: A single user's KV-cache at 32k context exceeds the entire 80 GB memory of an H100 GPU!

- **For Grouped-Query Attention (GQA-8: $H_{KV} = 8$):**
  $$\text{Bytes}_{\text{GQA}} = 2 \cdot 80 \cdot 32,768 \cdot 8 \cdot 128 \cdot 2 = 10,737,418,240 \text{ bytes} \approx \mathbf{10.74 \text{ GB}}$$
  GQA-8 fits cleanly in memory with $69.26\text{ GB}$ left over for weights and activations.

#### Step 2: Arithmetic Intensity at $B = 1$
At batch size $B = 1$:
$$\text{Compute} = 2 \times 70 \times 10^9 = 140 \times 10^9 \text{ FLOPs}$$
$$\text{Weight Read} = 70 \times 10^9 \times 2 = 140 \times 10^9 \text{ bytes (140 GB)}$$

- **Under MHA:**
  $$\text{Total Memory Read} = 140\text{ GB (weights)} + 85.90\text{ GB (KV)} = 225.90 \text{ GB}$$
  $$\text{Arithmetic Intensity}_{\text{MHA}} = \frac{140 \times 10^9 \text{ FLOPs}}{225.9 \times 10^9 \text{ bytes}} \approx \mathbf{0.62 \text{ FLOPs/byte}}$$

- **Under GQA-8:**
  $$\text{Total Memory Read} = 140\text{ GB (weights)} + 10.74\text{ GB (KV)} = 150.74 \text{ GB}$$
  $$\text{Arithmetic Intensity}_{\text{GQA}} = \frac{140 \times 10^9 \text{ FLOPs}}{150.74 \times 10^9 \text{ bytes}} \approx \mathbf{0.93 \text{ FLOPs/byte}}$$

#### Step 3: Hardware Ridge Point & Compute-Bound Transition
The hardware ridge point of the NVIDIA H100 is:
$$\text{Ridge Point} = \frac{\text{Peak Compute}}{\text{Peak Bandwidth}} = \frac{1,000 \times 10^{12} \text{ FLOPs/s}}{3.35 \times 10^{12} \text{ bytes/s}} \approx \mathbf{298.5 \text{ FLOPs/byte}}$$
Any operation with arithmetic intensity below $298.5$ is strictly memory-bandwidth bound.
With batch size $B$, weights ($140\text{ GB}$) are read once and amortized across $B$ queries:
$$\text{Intensity}(B) = \frac{B \cdot 140 \times 10^9 \text{ FLOPs}}{140 \times 10^9 + B \cdot \text{Cache}}$$
For $B \cdot \text{Cache} \gg 140$:
$$\text{Intensity}_{\max} = \frac{140 \times 10^9}{\text{Cache per query}} = \frac{140 \times 10^9}{10.74 \times 10^9} \approx 13.0 \text{ FLOPs/byte}$$
Even with large batches at 32k context, LLM generation remains memory-bound, proving that GQA's $8\times$ bandwidth reduction translates directly into an almost linear $4\times - 7\times$ speedup in real-world token generation throughput!

---

## Exercise 2: Mathematical Derivation of SwiGLU Backpropagation Gradient

### Problem
In a SwiGLU feedforward layer:
$$\text{SwiGLU}(x) = \left( (x W_g) \odot \text{SiLU}(x W_u) \right) W_d$$
Let $h_g = x W_g \in \mathbb{R}^{d_{ffn}}$ and $h_u = x W_u \in \mathbb{R}^{d_{ffn}}$.
The intermediate gated representation is:
$$z = h_g \odot \text{SiLU}(h_u) \in \mathbb{R}^{d_{ffn}}$$
and output is $y = z W_d \in \mathbb{R}^{d_{model}}$.
1. Recall the definition $\text{SiLU}(u) = u \cdot \sigma(u)$ where $\sigma(u) = \frac{1}{1 + e^{-u}}$. Derive the scalar derivative $\frac{d}{du}\text{SiLU}(u)$ in terms of $\sigma(u)$ and $u$.
2. Given upstream gradient $\delta_y = \frac{\partial \mathcal{L}}{\partial y} \in \mathbb{R}^{d_{model}}$, compute $\delta_z = \frac{\partial \mathcal{L}}{\partial z}$.
3. Using the multivariable chain rule, derive the analytical gradients with respect to the input $x$: $\frac{\partial \mathcal{L}}{\partial x}$.

---

### Solution

#### Step 1: Derivative of $\text{SiLU}(u)$
$$\text{SiLU}(u) = u \cdot \sigma(u)$$
Using the product rule:
$$\frac{d}{du}\text{SiLU}(u) = \sigma(u) + u \cdot \frac{d\sigma}{du}$$
Since $\frac{d\sigma}{du} = \sigma(u)(1 - \sigma(u))$:
$$\frac{d}{du}\text{SiLU}(u) = \sigma(u) + u \sigma(u)(1 - \sigma(u)) = \sigma(u) \left( 1 + u (1 - \sigma(u)) \right)$$

#### Step 2: Upstream Gradient to Gated Layer $\delta_z$
$$y = z W_d \implies \delta_z = \frac{\partial \mathcal{L}}{\partial z} = \delta_y W_d^T \in \mathbb{R}^{d_{ffn}}$$

#### Step 3: Backpropagation to Input $x$
The intermediate vector is the Hadamard product:
$$z = h_g \odot \text{SiLU}(h_u)$$
By element-wise chain rule:
$$\delta_{h_g} = \frac{\partial \mathcal{L}}{\partial h_g} = \delta_z \odot \text{SiLU}(h_u)$$
$$\delta_{h_u} = \frac{\partial \mathcal{L}}{\partial h_u} = \delta_z \odot h_g \odot \text{SiLU}'(h_u) = \delta_z \odot h_g \odot \left( \sigma(h_u) \left(1 + h_u \odot (1 - \sigma(h_u))\right) \right)$$

Now propagate back through the input projections:
$$h_g = x W_g \implies \left(\frac{\partial \mathcal{L}}{\partial x}\right)_g = \delta_{h_g} W_g^T$$
$$h_u = x W_u \implies \left(\frac{\partial \mathcal{L}}{\partial x}\right)_u = \delta_{h_u} W_u^T$$

Summing both gradient paths through $x$:
$$\frac{\partial \mathcal{L}}{\partial x} = \delta_{h_g} W_g^T + \delta_{h_u} W_u^T = \left( \delta_z \odot \text{SiLU}(x W_u) \right) W_g^T + \left( \delta_z \odot (x W_g) \odot \text{SiLU}'(x W_u) \right) W_u^T$$
Both pathways remain active and smoothly continuous, preventing vanishing gradients across deep Transformer networks.

---

## Exercise 3: Manual Step-by-Step Trace of FlashAttention Online Softmax

### Problem
Given the 4-element logit vector $x = [2.0, 4.0, 1.0, 5.0]$, trace the exact values produced by the FlashAttention online softmax algorithm when partitioned into two blocks of size 2:
- Block 1: $x^{(1)} = [2.0, 4.0]$
- Block 2: $x^{(2)} = [1.0, 5.0]$
1. Compute the exact statistics for Block 1 ($m_1, d_1$).
2. Process Block 2 and compute updated running statistics ($m_2, d_2$) using the online recurrence relations:
   $$m_2 = \max(m_1, \max(x^{(2)}))$$
   $$d_2 = d_1 e^{m_1 - m_2} + \sum_{j} e^{x_j^{(2)} - m_2}$$
3. Compute the final output probabilities $p_j = \frac{e^{x_j - m_2}}{d_2}$ and show that they match standard softmax.

---

### Solution

#### Step 1: Block 1 Statistics ($x^{(1)} = [2.0, 4.0]$)
$$m_1 = \max(2.0, 4.0) = 4.0$$
$$d_1 = e^{2.0 - 4.0} + e^{4.0 - 4.0} = e^{-2.0} + e^{0} \approx 0.135335 + 1.0 = 1.135335$$

#### Step 2: Block 2 Update ($x^{(2)} = [1.0, 5.0]$)
Block 2 local maximum:
$$\tilde{m}_2 = \max(1.0, 5.0) = 5.0$$
Global running maximum:
$$m_2 = \max(m_1, \tilde{m}_2) = \max(4.0, 5.0) = 5.0$$

Rescale Block 1 denominator and add Block 2 exponentials:
$$\text{Rescaling Factor for Block 1} = e^{m_1 - m_2} = e^{4.0 - 5.0} = e^{-1.0} \approx 0.367879$$
$$d_1 \cdot e^{m_1 - m_2} = 1.135335 \times 0.367879 \approx 0.417666$$

Block 2 exponentials shifted by global $m_2 = 5.0$:
$$e^{1.0 - 5.0} + e^{5.0 - 5.0} = e^{-4.0} + e^{0} \approx 0.018316 + 1.0 = 1.018316$$

New global denominator $d_2$:
$$d_2 = 0.417666 + 1.018316 = \mathbf{1.435982}$$

#### Step 3: Evaluate Final Probabilities
With global $m_2 = 5.0$ and $d_2 = 1.435982$:
- $p_0 = \frac{e^{2.0 - 5.0}}{d_2} = \frac{e^{-3.0}}{1.435982} = \frac{0.049787}{1.435982} \approx \mathbf{0.03467}$
- $p_1 = \frac{e^{4.0 - 5.0}}{d_2} = \frac{e^{-1.0}}{1.435982} = \frac{0.367879}{1.435982} \approx \mathbf{0.25619}$
- $p_2 = \frac{e^{1.0 - 5.0}}{d_2} = \frac{e^{-4.0}}{1.435982} = \frac{0.018316}{1.435982} \approx \mathbf{0.01275}$
- $p_3 = \frac{e^{5.0 - 5.0}}{d_2} = \frac{e^{0}}{1.435982} = \frac{1.0}{1.435982} \approx \mathbf{0.69639}$

Sum check:
$$0.03467 + 0.25619 + 0.01275 + 0.69639 = 1.00000 \quad \checkmark$$

Comparison with standard monolithic softmax:
$$m = 5.0, \quad \sum_{j=0}^3 e^{x_j - 5.0} = e^{-3} + e^{-1} + e^{-4} + e^0 = 0.049787 + 0.367879 + 0.018316 + 1.0 = 1.435982$$
$$p_{\text{standard}} = [0.03467, 0.25619, 0.01275, 0.69639]$$
The results match identically to all decimal places without ever storing or materializing the full vector at once.
