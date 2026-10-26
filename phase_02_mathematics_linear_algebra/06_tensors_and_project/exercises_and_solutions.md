# Topic 2.6: Exercises & Solutions — Tensors, Stride Memory & Capstone PCA Engine

Welcome to the problem set for **Topic 2.6: Tensors and Capstone Linear Algebra Engine**. These exercises bridge theoretical multi-dimensional linear algebra with the practical architecture of modern neural networks.

---

## Problem Set Overview

| Problem | Title | Concept Focus | Difficulty |
| :--- | :--- | :--- | :--- |
| **1** | **Tensor Shape Tracking in Conv2D & Attention** | Forward-pass tensor dimension calculus | Medium |
| **2** | **Memory Footprint & Physical Stride Offsets** | GPU memory sizing (FP32/FP16/INT8) & row-major indexing | Medium |
| **3** | **Broadcasting Rules & Step-by-Step PCA by Hand** | Aligning dimensions & end-to-end PCA math from scratch | Challenging |

---

## Problem 1: Tensor Shape Tracking in Conv2D & Transformer Multi-Head Attention

### The Context
When deep neural networks train on GPUs, every layer is a linear transformation changing the shape of multi-dimensional tensors. Tracking shapes accurately is the primary skill needed to design and debug architectures without runtime shape-mismatch crashes.

### Part A: Convolutional Feature Map
A computer vision model receives a batch of RGB images with shape:
$$X \in \mathbb{R}^{B \times C_{\text{in}} \times H_{\text{in}} \times W_{\text{in}}}$$
Where:
- Batch size $B = 8$
- Input channels $C_{\text{in}} = 3$ (Red, Green, Blue)
- Height $H_{\text{in}} = 64$
- Width $W_{\text{in}} = 64$

The input is passed through a 2D convolution layer with:
- Number of output filters / channels: $C_{\text{out}} = 16$
- Kernel spatial size: $K = 5 \times 5$
- Stride: $S = 2$
- Padding: $P = 2$

1. Write the general mathematical formula for the output spatial dimensions $H_{\text{out}}$ and $W_{\text{out}}$.
2. Calculate the exact resulting shape of the output tensor $Y$.
3. What is the total number of learnable weight parameters in this convolutional layer (including 1 bias per output channel)?

### Part B: Transformer Multi-Head Attention
A Transformer language model processes a batch of text tokens.
- Batch size: $B = 4$
- Sequence length: $T = 512$ tokens
- Embedding hidden dimension: $d_{\text{model}} = 768$
- Number of attention heads: $h = 12$
- Head dimension: $d_k = d_{\text{model}} / h = 64$

The input sequence tensor is $X \in \mathbb{R}^{4 \times 512 \times 768}$.
1. $X$ is multiplied by the Query weight matrix $W_Q \in \mathbb{R}^{768 \times 768}$. What is the shape of $Q$?
2. $Q$ is reshaped and transposed to split the hidden dimension across heads. The standard multi-head shape is $(B, h, T, d_k)$. Write the exact dimensions.
3. In scaled dot-product attention, the engine computes $A = Q K^T$, where $K$ has shape $(B, h, T, d_k)$ and $K^T$ transposes the last two dimensions. What is the exact shape of the attention score matrix $A$?

---

### Solution 1

#### Part A: Convolutional Spatial Dimensions
1. **Formula:**
   For an input dimension $D_{\text{in}}$, kernel size $K$, padding $P$, and stride $S$:
   $$D_{\text{out}} = \left\lfloor \frac{D_{\text{in}} + 2P - K}{S} \right\rfloor + 1$$

2. **Calculation:**
   $$H_{\text{out}} = \left\lfloor \frac{64 + 2(2) - 5}{2} \right\rfloor + 1 = \left\lfloor \frac{64 + 4 - 5}{2} \right\rfloor + 1 = \left\lfloor \frac{63}{2} \right\rfloor + 1 = 31 + 1 = 32$$
   $$W_{\text{out}} = 32$$
   The output tensor shape is:
   $$Y \in \mathbb{R}^{8 \times 16 \times 32 \times 32}$$

3. **Learnable Parameters:**
   Each of the $C_{\text{out}} = 16$ filters connects to all $C_{\text{in}} = 3$ channels across a $5 \times 5$ window:
   $$\text{Weights} = C_{\text{out}} \times C_{\text{in}} \times K \times K = 16 \times 3 \times 5 \times 5 = 16 \times 75 = 1{,}200$$
   $$\text{Biases} = C_{\text{out}} = 16$$
   $$\text{Total Parameters} = 1{,}200 + 16 = 1{,}216$$

#### Part B: Transformer Attention Shapes
1. **Query Tensor $Q$:**
   $$\text{Shape}(Q) = (B, T, d_{\text{model}}) = (4, 512, 768)$$
2. **Reshaped Multi-Head Query:**
   $$\text{Shape}(Q_{\text{heads}}) = (B, h, T, d_k) = (4, 12, 512, 64)$$
3. **Attention Score Matrix $A = Q K^T$:**
   $K$ has shape $(4, 12, 512, 64)$. Transposing its last two dimensions yields $K^T \in \mathbb{R}^{4 \times 12 \times 64 \times 512}$.
   Multiplying batch-by-batch and head-by-head:
   $$(512 \times 64) \times (64 \times 512) \longrightarrow (512 \times 512)$$
   $$\text{Shape}(A) = (4, 12, 512, 512)$$
   *Insight:* Each head computes a $512 \times 512$ matrix where entry $(i, j)$ represents how strongly token $i$ attends to token $j$!

---

## Problem 2: Memory Footprint & Physical Stride Offsets

### The Context
Tensors appear multi-dimensional in Python or PyTorch, but physical computer hardware (RAM and VRAM) is strictly a one-dimensional tape of memory addresses. Understanding strides and precision sizing prevents Out-Of-Memory (OOM) errors.

### Part A: Precision Memory Sizing
A Large Language Model (LLM) contains $7 \times 10^9$ (7 Billion) parameters.
1. Compute the raw parameter memory footprint in Gigabytes ($1 \text{ GB} = 10^9 \text{ bytes}$ or $1 \text{ GiB} = 2^{30} \text{ bytes}$) under:
   - FP32 (Full precision, 32-bit float = 4 bytes per parameter)
   - FP16 / BF16 (Half precision, 16-bit float = 2 bytes per parameter)
   - INT8 (Quantized 8-bit integer = 1 byte per parameter)
   - INT4 (Quantized 4-bit integer = 0.5 bytes per parameter)
2. If an NVIDIA RTX 4090 GPU has $24 \text{ GB}$ of VRAM, which precision formats can fit the model weights with sufficient space remaining for KV cache activations?

### Part B: Physical 1D Memory Stride Offsets
Consider a 3D tensor $T$ with shape $(D_0, D_1, D_2) = (3, 4, 5)$ stored in standard row-major (C-contiguous) format.
1. What is the total number of elements in $T$?
2. Compute the stride tuple $(S_0, S_1, S_2)$ in units of elements.
3. Write the indexing equation mapping coordinate $(i, j, k)$ to the 1D flat memory index $idx$.
4. Calculate the exact flat index for element $(2, 1, 4)$.
5. What coordinate corresponds to flat index $idx = 37$?

---

### Solution 2

#### Part A: LLM Memory Footprint
1. **Calculations for 7 Billion Parameters ($N = 7 \times 10^9$):**
   - **FP32:** $7 \times 10^9 \times 4 \text{ bytes} = 28 \times 10^9 \text{ bytes} = 28.0 \text{ GB} \approx 26.08 \text{ GiB}$
   - **FP16 / BF16:** $7 \times 10^9 \times 2 \text{ bytes} = 14 \times 10^9 \text{ bytes} = 14.0 \text{ GB} \approx 13.04 \text{ GiB}$
   - **INT8:** $7 \times 10^9 \times 1 \text{ byte} = 7 \times 10^9 \text{ bytes} = 7.0 \text{ GB} \approx 6.52 \text{ GiB}$
   - **INT4:** $7 \times 10^9 \times 0.5 \text{ bytes} = 3.5 \times 10^9 \text{ bytes} = 3.5 \text{ GB} \approx 3.26 \text{ GiB}$

2. **GPU Sizing:**
   On a $24 \text{ GB}$ RTX 4090:
   - **FP32 (28 GB):** Cannot fit at all. OOM before running any tokens.
   - **FP16 (14 GB):** Fits the weights ($14 \text{ GB}$), leaving $10 \text{ GB}$ for context tokens, activations, and KV cache.
   - **INT8 (7 GB) & INT4 (3.5 GB):** Fit comfortably with massive headroom for huge batch sizes and long context windows (32k+ tokens).

#### Part B: Row-Major Stride Offsets
1. **Total Elements:**
   $$N = 3 \times 4 \times 5 = 60 \text{ elements}$$

2. **Stride Tuple:**
   In row-major order:
   - $S_2 = 1$
   - $S_1 = D_2 = 5$
   - $S_0 = D_1 \times D_2 = 4 \times 5 = 20$
   $$\text{Strides} = (20, 5, 1)$$

3. **Indexing Formula:**
   $$idx(i, j, k) = i \cdot S_0 + j \cdot S_1 + k \cdot S_2 = 20i + 5j + k$$

4. **Flat Index for $(2, 1, 4)$:**
   $$idx(2, 1, 4) = 2(20) + 1(5) + 4(1) = 40 + 5 + 4 = 49$$

5. **Inverse Mapping of Flat Index $idx = 37$:**
   - $i = \lfloor 37 / 20 \rfloor = 1$ (Remainder: $37 - 20 = 17$)
   - $j = \lfloor 17 / 5 \rfloor = 3$ (Remainder: $17 - 15 = 2$)
   - $k = 2 / 1 = 2$
   $$\text{Coordinate} = (1, 3, 2)$$
   *Check:* $1(20) + 3(5) + 2(1) = 20 + 15 + 2 = 37$. Exact match!

---

## Problem 3: Broadcasting Rules & Step-by-Step PCA by Hand

### The Context
Principal Component Analysis (PCA) finds the orthogonal axes of maximum variance in high-dimensional data by computing the eigenvectors of the data covariance matrix.

### Part A: NumPy/PyTorch Broadcasting Check
For each pair of tensor shapes, determine if they are broadcast-compatible according to the standard NumPy/PyTorch rules. If compatible, state the resulting shape. If not, explain why.
1. Tensor $A$: `(8, 1, 64)`, Tensor $B$: `(8, 32, 64)`
2. Tensor $A$: `(5, 3)`, Tensor $B$: `(3,)`
3. Tensor $A$: `(4, 3, 2)`, Tensor $B$: `(4, 2)`

### Part B: End-to-End PCA by Hand
Consider a tiny dataset of 3 points in 2D space:
$$x_1 = \begin{bmatrix} 2 \\ 4 \end{bmatrix}, \quad x_2 = \begin{bmatrix} 4 \\ 6 \end{bmatrix}, \quad x_3 = \begin{bmatrix} 6 \\ 8 \end{bmatrix}$$

1. **Calculate the Centroid (Mean Vector $\mu$):**
   Compute $\mu = \frac{1}{N} \sum_{i=1}^3 x_i$.
2. **Mean-Center the Data:**
   Subtract $\mu$ from each point to form the zero-centered matrix $X_c$ of shape $(3 \times 2)$.
3. **Compute the Sample Covariance Matrix $\Sigma$:**
   $$\Sigma = \frac{1}{N - 1} X_c^T X_c$$
4. **Find the Eigenvalues and Eigenvectors of $\Sigma$:**
   Determine the principal eigenvector $v_1$ (unit length) corresponding to the largest eigenvalue $\lambda_1$.
5. **Project Data onto 1D Principal Component:**
   Compute the 1D projected coordinate $z_i = (x_i - \mu) \cdot v_1$ for each of the three points.
6. **Variance Explained:**
   What percentage of the total variance is captured by this single 1D axis?

---

### Solution 3

#### Part A: Broadcasting Evaluations
1. **`(8, 1, 64)` and `(8, 32, 64)`:**
   - Align right to left:
     - Dim 2: $64 == 64$ (Compatible)
     - Dim 1: $1$ vs $32$ ($1$ stretches to $32$, Compatible)
     - Dim 0: $8 == 8$ (Compatible)
   - **Result: Compatible $\to$ Shape: `(8, 32, 64)`**

2. **`(5, 3)` and `(3,)`:**
   - Prepend leading $1$ to $B$: `(1, 3)`
   - Align right to left:
     - Dim 1: $3 == 3$
     - Dim 0: $5$ vs $1$ ($1$ stretches to $5$)
   - **Result: Compatible $\to$ Shape: `(5, 3)`**

3. **`(4, 3, 2)` and `(4, 2)`:**
   - Prepend leading $1$ to $B$: `(1, 4, 2)`
   - Align right to left:
     - Dim 2: $2 == 2$
     - Dim 1: $3$ vs $4$ (Neither is 1, and $3 \neq 4$ $\to$ **Incompatible!**)
   - **Result: Shape mismatch error!**

---

#### Part B: End-to-End PCA Step-by-Step

##### 1. Centroid (Mean Vector $\mu$)
$$\mu_x = \frac{2 + 4 + 6}{3} = \frac{12}{3} = 4$$
$$\mu_y = \frac{4 + 6 + 8}{3} = \frac{18}{3} = 6$$
$$\mu = \begin{bmatrix} 4 \\ 6 \end{bmatrix}$$

##### 2. Mean-Centered Data Matrix $X_c$
Subtract $\begin{bmatrix} 4 \\ 6 \end{bmatrix}$ from each point:
$$x_{c, 1} = \begin{bmatrix} 2 - 4 \\ 4 - 6 \end{bmatrix} = \begin{bmatrix} -2 \\ -2 \end{bmatrix}$$
$$x_{c, 2} = \begin{bmatrix} 4 - 4 \\ 6 - 6 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$
$$x_{c, 3} = \begin{bmatrix} 6 - 4 \\ 8 - 6 \end{bmatrix} = \begin{bmatrix} 2 \\ 2 \end{bmatrix}$$

$$X_c = \begin{bmatrix} -2 & -2 \\ 0 & 0 \\ 2 & 2 \end{bmatrix}$$

##### 3. Covariance Matrix $\Sigma$
With sample size $N = 3$, degrees of freedom $N - 1 = 2$:
$$X_c^T X_c = \begin{bmatrix} -2 & 0 & 2 \\ -2 & 0 & 2 \end{bmatrix} \begin{bmatrix} -2 & -2 \\ 0 & 0 \\ 2 & 2 \end{bmatrix} = \begin{bmatrix} (-2)^2 + 0^2 + 2^2 & (-2)(-2) + 0 + 2(2) \\ (-2)(-2) + 0 + 2(2) & (-2)^2 + 0^2 + 2^2 \end{bmatrix} = \begin{bmatrix} 8 & 8 \\ 8 & 8 \end{bmatrix}$$

$$\Sigma = \frac{1}{2} \begin{bmatrix} 8 & 8 \\ 8 & 8 \end{bmatrix} = \begin{bmatrix} 4 & 4 \\ 4 & 4 \end{bmatrix}$$

##### 4. Eigenvalues and Eigenvectors of $\Sigma$
Solve $\det(\Sigma - \lambda I) = 0$:
$$\det \begin{bmatrix} 4 - \lambda & 4 \\ 4 & 4 - \lambda \end{bmatrix} = (4 - \lambda)^2 - 16 = \lambda^2 - 8\lambda + 16 - 16 = \lambda(\lambda - 8) = 0$$
$$\lambda_1 = 8, \quad \lambda_2 = 0$$

Find eigenvector for $\lambda_1 = 8$:
$$\begin{bmatrix} 4 - 8 & 4 \\ 4 & 4 - 8 \end{bmatrix} \begin{bmatrix} v_x \\ v_y \end{bmatrix} = \begin{bmatrix} -4 & 4 \\ 4 & -4 \end{bmatrix} \begin{bmatrix} v_x \\ v_y \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$
$$-4 v_x + 4 v_y = 0 \implies v_x = v_y$$

Normalizing to unit length $\|v_1\| = 1$:
$$v_1 = \begin{bmatrix} \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} \end{bmatrix} \approx \begin{bmatrix} 0.7071 \\ 0.7071 \end{bmatrix}$$

##### 5. 1D Projected Coordinates
For each point $x_{c, i}$, $z_i = x_{c, i}^T v_1$:
$$z_1 = \begin{bmatrix} -2 & -2 \end{bmatrix} \begin{bmatrix} \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} \end{bmatrix} = -\frac{4}{\sqrt{2}} = -2\sqrt{2} \approx -2.8284$$
$$z_2 = \begin{bmatrix} 0 & 0 \end{bmatrix} \begin{bmatrix} \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} \end{bmatrix} = 0$$
$$z_3 = \begin{bmatrix} 2 & 2 \end{bmatrix} \begin{bmatrix} \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} \end{bmatrix} = \frac{4}{\sqrt{2}} = 2\sqrt{2} \approx +2.8284$$

##### 6. Variance Explained
$$\text{Total Variance} = \lambda_1 + \lambda_2 = 8 + 0 = 8$$
$$\text{Variance Explained by PC1} = \frac{\lambda_1}{\lambda_1 + \lambda_2} = \frac{8}{8} = 100\%$$

*Insight:* Because the 3 original points $(2,4), (4,6), (6,8)$ lie perfectly along the line $y = x + 2$, the second eigenvalue is exactly $0$. A single 1D principal component preserves $100\%$ of the information without any reconstruction error whatsoever!

