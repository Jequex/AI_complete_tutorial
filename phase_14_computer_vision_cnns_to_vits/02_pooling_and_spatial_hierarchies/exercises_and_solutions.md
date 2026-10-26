# Topic 14.2: Exercises & Solutions — Pooling & Spatial Hierarchies

---

### Problem 1: Manual Forward and Backward Pass of 2D Max Pooling (Easy)
Consider a single-channel $4 \times 4$ input matrix $X$:
$$X = \begin{bmatrix}
3.0 & 1.0 & 2.0 & 4.0 \\
5.0 & 2.0 & 6.0 & 1.0 \\
7.0 & 4.0 & 1.0 & 3.0 \\
2.0 & 8.0 & 5.0 & 0.0
\end{bmatrix}$$

We apply a MaxPooling2D operation with window size $K = 2$ and non-overlapping stride $S = 2$.
1. Compute the output matrix $Y \in \mathbb{R}^{2 \times 2}$.
2. Construct the binary argmax routing mask $M \in \{0, 1\}^{4 \times 4}$.
3. Suppose the upstream loss gradient arriving at this layer is:
   $$\frac{\partial \mathcal{L}}{\partial Y} = \begin{bmatrix} 1.5 & -2.0 \\ 3.0 & 0.5 \end{bmatrix}$$
   Compute the exact numerical downstream gradient $\frac{\partial \mathcal{L}}{\partial X} \in \mathbb{R}^{4 \times 4}$.

<details>
<summary>View Complete Solution</summary>

**Part 1: Forward Pass ($K=2, S=2$):**
- Top-left quadrant (rows 0..1, cols 0..1):
  $$\max(3.0, 1.0, 5.0, 2.0) = 5.0 \quad \text{at index } (1, 0)$$
- Top-right quadrant (rows 0..1, cols 2..3):
  $$\max(2.0, 4.0, 6.0, 1.0) = 6.0 \quad \text{at index } (1, 2)$$
- Bottom-left quadrant (rows 2..3, cols 0..1):
  $$\max(7.0, 4.0, 2.0, 8.0) = 8.0 \quad \text{at index } (3, 1)$$
- Bottom-right quadrant (rows 2..3, cols 2..3):
  $$\max(1.0, 3.0, 5.0, 0.0) = 5.0 \quad \text{at index } (3, 2)$$

$$Y = \begin{bmatrix} 5.0 & 6.0 \\ 8.0 & 5.0 \end{bmatrix}$$

**Part 2: Argmax Routing Mask $M$:**
Mark $1$ at the winning positions and $0$ elsewhere:
$$M = \begin{bmatrix}
0 & 0 & 0 & 0 \\
1 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 \\
0 & 1 & 1 & 0
\end{bmatrix}$$

**Part 3: Backward Gradient Routing:**
Each upstream gradient $\frac{\partial \mathcal{L}}{\partial Y_{i, j}}$ routes exclusively to its cached argmax location:
- Coordinate $(1, 0)$ receives $\frac{\partial \mathcal{L}}{\partial Y_{0, 0}} = 1.5$
- Coordinate $(1, 2)$ receives $\frac{\partial \mathcal{L}}{\partial Y_{0, 1}} = -2.0$
- Coordinate $(3, 1)$ receives $\frac{\partial \mathcal{L}}{\partial Y_{1, 0}} = 3.0$
- Coordinate $(3, 2)$ receives $\frac{\partial \mathcal{L}}{\partial Y_{1, 1}} = 0.5$
All non-argmax coordinates receive $0$.

$$\frac{\partial \mathcal{L}}{\partial X} = \begin{bmatrix}
0.0 & 0.0 & 0.0 & 0.0 \\
1.5 & 0.0 & -2.0 & 0.0 \\
0.0 & 0.0 & 0.0 & 0.0 \\
0.0 & 3.0 & 0.5 & 0.0
\end{bmatrix}$$
</details>

---

### Problem 2: Overlapping Pooling Windows & Gradient Accumulation (Medium)
In AlexNet (Krizhevsky et al., 2012), the authors observed that **overlapping pooling** reduced top-1 error by $0.4\%$ and alleviated overfitting. Consider an overlapping $3 \times 3$ pooling window with stride $S = 2$ applied to a $5 \times 5$ matrix:
$$H_{\text{out}} = \frac{5 - 3}{2} + 1 = 2, \quad W_{\text{out}} = \frac{5 - 3}{2} + 1 = 2$$

Consider an input matrix where a prominent feature activation at position $(2, 2)$ is the maximum value for **multiple** overlapping windows:
1. Explain mathematically why a single input pixel $(m, n)$ can receive gradients from multiple output units when $S < K$.
2. State the exact summation formula for the input gradient under overlapping pooling.
3. If an input pixel $(2, 2)$ is the argmax for all 4 output windows in a $2 \times 2$ output grid, and the upstream gradients are:
   $$\frac{\partial \mathcal{L}}{\partial Y} = \begin{bmatrix} 2.0 & 1.5 \\ -1.0 & 3.5 \end{bmatrix}$$
   Compute the total accumulated gradient $\frac{\partial \mathcal{L}}{\partial X_{2, 2}}$.

<details>
<summary>View Complete Solution</summary>

**Part 1: Mathematical Origin of Multi-Window Overlap:**
When stride $S < K$, the spacing between adjacent window origins is smaller than the window width. Therefore, adjacent windows overlap:
$$\Omega_{i, j} \cap \Omega_{i, j+1} \neq \emptyset$$
A pixel $(m, n)$ located in the intersection belongs to the receptive field of more than one output neuron: $(m, n) \in \Omega_{i, j}$ and $(m, n) \in \Omega_{i, j+1}$. By the multivariate chain rule, its total gradient is the sum of partial derivatives across all paths.

**Part 2: Summation Formula:**
$$\frac{\partial \mathcal{L}}{\partial X_{m, n}} = \sum_{i} \sum_{j} \frac{\partial \mathcal{L}}{\partial Y_{i, j}} \cdot \mathbf{1}[(m, n) = \operatorname{argmax}_{(r, c) \in \Omega_{i, j}} X_{r, c}]$$

**Part 3: Gradient Accumulation Calculation:**
Because pixel $(2, 2)$ is the argmax for all 4 windows:
$$\frac{\partial \mathcal{L}}{\partial X_{2, 2}} = \frac{\partial \mathcal{L}}{\partial Y_{0, 0}} + \frac{\partial \mathcal{L}}{\partial Y_{0, 1}} + \frac{\partial \mathcal{L}}{\partial Y_{1, 0}} + \frac{\partial \mathcal{L}}{\partial Y_{1, 1}}$$
$$\frac{\partial \mathcal{L}}{\partial X_{2, 2}} = 2.0 + 1.5 + (-1.0) + 3.5 = 6.0$$
The gradient accumulates linearly to $6.0$.
</details>

---

### Problem 3: VGG vs. ResNet Classifier Head: Flattening vs. Global Average Pooling (Hard)
Compare two architectural approaches for connecting a convolutional backbone to a 1,000-class classification head:
- **Approach A (VGG-style Flattening + Fully Connected Layers):**
  - Final conv feature map volume: $[C=512, H=7, W=7]$
  - Flattened to dimension $d = 512 \times 7 \times 7 = 25,088$
  - Dense Layer 1: $25,088 \to 4096$ with bias
  - Dense Layer 2: $4096 \to 4096$ with bias
  - Classifier Head: $4096 \to 1000$ with bias
- **Approach B (ResNet-style Global Average Pooling):**
  - Final conv feature map volume: $[C=512, H=7, W=7]$
  - Global Average Pooling (GAP) collapses $[512, 7, 7] \to 512$
  - Classifier Head: $512 \to 1000$ with bias

1. Compute the total number of learnable parameters in the classification head for Approach A and Approach B.
2. Calculate the percentage parameter reduction achieved by Approach B over Approach A.
3. Compute the memory (in Megabytes, assuming 32-bit float = 4 bytes) required just to store the weights of the classification head in both approaches.
4. Explain why Approach A is heavily prone to spatial overfitting and requires heavy dropout ($p=0.5$), while Approach B is naturally regularized.

<details>
<summary>View Complete Solution</summary>

**Part 1: Parameter Calculations:**
- **Approach A (Flatten + Dense Layers):**
  - FC1: $(25,088 \times 4096) + 4096 = 102,760,448 + 4096 = 102,764,544$
  - FC2: $(4096 \times 4096) + 4096 = 16,777,216 + 4096 = 16,781,312$
  - Head: $(4096 \times 1000) + 1000 = 4,096,000 + 1000 = 4,097,000$
  - Total Approach A = $102,764,544 + 16,781,312 + 4,097,000 = \mathbf{123,642,856 \text{ parameters}}$

- **Approach B (GAP + Linear Head):**
  - GAP: $0$ parameters
  - Head: $(512 \times 1000) + 1000 = 512,000 + 1000 = \mathbf{513,000 \text{ parameters}}$

**Part 2: Percentage Parameter Reduction:**
$$\text{Reduction} = \frac{123,642,856 - 513,000}{123,642,856} \times 100\% = \frac{123,129,856}{123,642,856} \times 100\% \approx \mathbf{99.585\%}$$
Approach B eliminates $99.59\%$ of the classification head parameters!

**Part 3: Memory Footprint (FP32, 4 bytes/param):**
- Approach A: $123,642,856 \times 4 \text{ bytes} = 494,571,424 \text{ bytes} \approx \mathbf{471.7 \text{ MB}}$
- Approach B: $513,000 \times 4 \text{ bytes} = 2,052,000 \text{ bytes} \approx \mathbf{1.96 \text{ MB}}$

**Part 4: Overfitting and Regularization Analysis:**
- In Approach A, every single spatial coordinate $(r, c)$ in each channel connects via independent weights to each hidden neuron. This means the model can memorize exact pixel positions (e.g., "if feature 32 appears specifically at coordinate $(1, 4)$, predict dog"). This memorization breaks translation invariance and causes severe overfitting, requiring $50\%$ dropout to train.
- In Approach B, Global Average Pooling sums and averages the activations across the entire spatial grid, completely stripping spatial coordinate information. The classifier is forced to base its predictions on the *overall presence and semantic intensity* of features anywhere in the image, acting as a built-in structural regularizer.
</details>
