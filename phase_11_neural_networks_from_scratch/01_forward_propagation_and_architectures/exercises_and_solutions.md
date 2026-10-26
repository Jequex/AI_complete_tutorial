# Topic 11.1: Exercises & Solutions — Forward Propagation & Architectures

---

### Problem 1: Manual Forward Pass for a Single Layer (Easy)
Consider a batch of two input vectors $X \in \mathbb{R}^{2 \times 3}$:
$$X = \begin{bmatrix} 2.0 & -1.0 & 0.5 \\ 0.0 & 3.0 & -2.0 \end{bmatrix}$$

This batch is fed into a linear layer with 2 output neurons ($d_{\text{out}} = 2$). The weight matrix $W \in \mathbb{R}^{2 \times 3}$ and bias vector $\mathbf{b} \in \mathbb{R}^2$ are:
$$W = \begin{bmatrix} 1.0 & 0.5 & -2.0 \\ -0.5 & 1.0 & 1.0 \end{bmatrix}, \quad \mathbf{b} = \begin{bmatrix} 0.5 \\ -1.0 \end{bmatrix}$$

Using the batch forward formula $Z = X W^T + \mathbf{1} \mathbf{b}^T$:
1. State the shape of $W^T$, $X W^T$, and $Z$.
2. Compute the exact numerical values of the matrix $Z \in \mathbb{R}^{2 \times 2}$.

<details>
<summary>View Complete Solution</summary>

**Step 1: Tensor shapes:**
- $X$ has shape $[2, 3]$.
- $W$ has shape $[2, 3]$, so $W^T$ has shape $[3, 2]$.
- $X W^T$ has shape $[2, 3] \times [3, 2] = [2, 2]$.
- Adding bias $\mathbf{b}^T \in \mathbb{R}^{1 \times 2}$ broadcast across 2 rows results in $Z$ of shape $[2, 2]$.

**Step 2: Linear product calculation $X W^T$:**
$$W^T = \begin{bmatrix} 1.0 & -0.5 \\ 0.5 & 1.0 \\ -2.0 & 1.0 \end{bmatrix}$$

For Sample 1 ($\mathbf{x}_1 = [2.0, -1.0, 0.5]$):
- $(X W^T)_{1, 1} = 2.0(1.0) + (-1.0)(0.5) + 0.5(-2.0) = 2.0 - 0.5 - 1.0 = 0.5$
- $(X W^T)_{1, 2} = 2.0(-0.5) + (-1.0)(1.0) + 0.5(1.0) = -1.0 - 1.0 + 0.5 = -1.5$

For Sample 2 ($\mathbf{x}_2 = [0.0, 3.0, -2.0]$):
- $(X W^T)_{2, 1} = 0.0(1.0) + 3.0(0.5) + (-2.0)(-2.0) = 0.0 + 1.5 + 4.0 = 5.5$
- $(X W^T)_{2, 2} = 0.0(-0.5) + 3.0(1.0) + (-2.0)(1.0) = 0.0 + 3.0 - 2.0 = 1.0$

$$X W^T = \begin{bmatrix} 0.5 & -1.5 \\ 5.5 & 1.0 \end{bmatrix}$$

**Step 3: Add broadcast bias:**
$$\mathbf{b} = [0.5, -1.0]$$

$$Z_{1, 1} = 0.5 + 0.5 = 1.0$$
$$Z_{1, 2} = -1.5 + (-1.0) = -2.5$$
$$Z_{2, 1} = 5.5 + 0.5 = 6.0$$
$$Z_{2, 2} = 1.0 + (-1.0) = 0.0$$

Final pre-activation matrix:
$$Z = \begin{bmatrix} 1.0 & -2.5 \\ 6.0 & 0.0 \end{bmatrix}$$
</details>

---

### Problem 2: Deep Network Parameter Counting & Memory Bookkeeping (Medium)
You are designing a deep feedforward network for audio spectrogram classification:
- Input feature dimension: $d_{\text{in}} = 128$
- Layer 1: 256 neurons
- Layer 2: 128 neurons
- Layer 3: 64 neurons
- Layer 4 (Output): 10 classes
- Mini-batch size: $B = 32$

1. Compute the total number of trainable parameters (weights + biases) in the network.
2. For each layer $l \in \{1, 2, 3, 4\}$, state the exact shape of:
   - Weight matrix $W^{[l]}$
   - Bias vector $\mathbf{b}^{[l]}$
   - Batch pre-activation matrix $Z^{[l]}$
3. How many floating point numbers (assuming IEEE 754 32-bit floats, 4 bytes each) must be stored in memory to cache the intermediate activations $\{A^{[0]}, A^{[1]}, A^{[2]}, A^{[3]}, A^{[4]}\}$ for a single batch forward pass?

<details>
<summary>View Complete Solution</summary>

**Step 1: Trainable parameter count:**
- Layer 1: $W^{[1]} \in \mathbb{R}^{256 \times 128}$ has $256 \times 128 = 32,768$ weights. $\mathbf{b}^{[1]}$ has $256$ biases. Subtotal = $33,024$.
- Layer 2: $W^{[2]} \in \mathbb{R}^{128 \times 256}$ has $128 \times 256 = 32,768$ weights. $\mathbf{b}^{[2]}$ has $128$ biases. Subtotal = $32,896$.
- Layer 3: $W^{[3]} \in \mathbb{R}^{64 \times 128}$ has $64 \times 128 = 8,192$ weights. $\mathbf{b}^{[3]}$ has $64$ biases. Subtotal = $8,256$.
- Layer 4: $W^{[4]} \in \mathbb{R}^{10 \times 64}$ has $10 \times 64 = 640$ weights. $\mathbf{b}^{[4]}$ has $10$ biases. Subtotal = $650$.

$$\text{Total Parameters} = 33,024 + 32,896 + 8,256 + 650 = 74,826$$

**Step 2: Tensor shape table:**
- Layer 1: $W^{[1]}: [256, 128]$, $\mathbf{b}^{[1]}: [256]$, $Z^{[1]}: [32, 256]$
- Layer 2: $W^{[2]}: [128, 256]$, $\mathbf{b}^{[2]}: [128]$, $Z^{[2]}: [32, 128]$
- Layer 3: $W^{[3]}: [64, 128]$, $\mathbf{b}^{[3]}: [64]$, $Z^{[3]}: [32, 64]$
- Layer 4: $W^{[4]}: [10, 64]$, $\mathbf{b}^{[4]}: [10]$, $Z^{[4]}: [32, 10]$

**Step 3: Activation memory cache:**
- $A^{[0]} \equiv X$: $32 \times 128 = 4,096$ values
- $A^{[1]}$: $32 \times 256 = 8,192$ values
- $A^{[2]}$: $32 \times 128 = 4,096$ values
- $A^{[3]}$: $32 \times 64 = 2,048$ values
- $A^{[4]}$: $32 \times 10 = 320$ values

$$\text{Total activation floats} = 4,096 + 8,192 + 4,096 + 2,048 + 320 = 18,752 \text{ floats}$$
$$\text{Memory in Bytes} = 18,752 \times 4 \text{ bytes} = 75,008 \text{ bytes} \approx 73.25 \text{ KB}$$
</details>

---

### Problem 3: Expressivity Analysis — Depth vs Width (Hard)
Let $f: [0, 1] \to [0, 1]$ be the "triangle sawtooth wave" function defined by:
$$g(x) = \begin{cases} 2x & \text{if } 0 \le x \le 0.5 \\ 2 - 2x & \text{if } 0.5 < x \le 1 \end{cases}$$

Notice that $g(x)$ has 2 linear pieces (1 peak).
1. Prove that composing $g$ with itself $k$ times, $f_k(x) = (g \circ g \circ \dots \circ g)(x)$ ($k$ compositions), produces a continuous piecewise-linear function with $2^k$ linear segments and $2^{k-1}$ peaks.
2. How many layers and total neurons does a deep ReLU network require to compute $f_k(x)$ exactly?
3. How many neurons would a shallow 1-hidden-layer ReLU network require to represent $f_k(x)$?

<details>
<summary>View Complete Solution</summary>

**Step 1: Composition of sawtooth pieces:**
Notice that $g(x) = 2x - 4\max(0, x - 0.5)$. It maps $[0, 0.5]$ linearly to $[0, 1]$ and $[0.5, 1]$ linearly back down to $[0, 1]$.
When we compose $g(g(x))$, on each of the 2 monotonic halves of the input interval $[0, 1]$, $g(x)$ sweeps across the full range $[0, 1]$. Therefore, $g$ applies its two-piece folding to each half, generating $2 \times 2 = 4 = 2^2$ linear segments.
By mathematical induction, each additional composition doubles the number of monotonic linear slopes. Hence, $k$ compositions yield $2^k$ linear segments and $2^{k-1}$ alternating peaks.

**Step 2: Deep network construction:**
Each single sawtooth function $g(x) = 2x - 4\text{ReLU}(x - 0.5)$ can be computed by a 2-neuron ReLU layer.
To compute $f_k(x)$, we chain $k$ such 2-neuron layers sequentially:
$$\text{Number of layers} = k, \quad \text{Total neurons} = 2k$$
The total number of parameters is $\mathcal{O}(k)$, growing **linearly** with $k$.

**Step 3: Shallow network requirements:**
A shallow 1-hidden-layer network with ReLU activations is a linear combination of hinges:
$$F(x) = c_0 + \sum_{i=1}^M c_i \max(0, w_i x + b_i)$$
Each ReLU neuron can introduce at most **one** change of slope (breakpoint).
To form $2^k$ distinct linear pieces, a function must have at least $2^k - 1$ interior breakpoints!
Since each hidden neuron contributes at most one breakpoint, the shallow network must possess at least:
$$M \ge 2^k - 1 \text{ neurons}$$
This demonstrates Telgarsky's celebrated theorem (2016): a deep network with $\mathcal{O}(k)$ neurons can compute a function that requires $\mathcal{O}(2^k)$ neurons in a shallow network—an **exponential advantage for depth**!
</details>
