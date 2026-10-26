# Topic 14.1: Exercises & Solutions — Convolutions & Feature Maps

---

### Problem 1: Manual 2D Convolution Calculation with Padding & Stride (Easy)
You are given a $4 \times 4$ single-channel input matrix $X$ and a $3 \times 3$ kernel $K$ with bias $b = -2.0$:

$$X = \begin{bmatrix}
1 & 2 & 0 & 3 \\
0 & 1 & 1 & 2 \\
2 & 0 & 3 & 1 \\
1 & 2 & 0 & 1
\end{bmatrix}, \quad
K = \begin{bmatrix}
1 & 0 & -1 \\
1 & 0 & -1 \\
1 & 0 & -1
\end{bmatrix}, \quad
b = -2.0$$

1. If you apply zero-padding $P = 1$ and stride $S = 2$, compute the spatial shape of the output feature map $Y$.
2. Compute the exact numerical values of the output matrix $Y$.

<details>
<summary>View Complete Solution</summary>

**Step 1: Compute output dimensions:**
Using the spatial arithmetic formula:
$$H_{\text{out}} = \left\lfloor \frac{H - K_h + 2P}{S} \right\rfloor + 1 = \left\lfloor \frac{4 - 3 + 2(1)}{2} \right\rfloor + 1 = \left\lfloor \frac{3}{2} \right\rfloor + 1 = 1 + 1 = 2$$
$$W_{\text{out}} = \left\lfloor \frac{4 - 3 + 2(1)}{2} \right\rfloor + 1 = 2$$
The output matrix $Y$ has shape $2 \times 2$.

**Step 2: Construct zero-padded matrix $X_{\text{pad}}$ ($6 \times 6$):**
$$X_{\text{pad}} = \begin{bmatrix}
0 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 2 & 0 & 3 & 0 \\
0 & 0 & 1 & 1 & 2 & 0 \\
0 & 2 & 0 & 3 & 1 & 0 \\
0 & 1 & 2 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 0
\end{bmatrix}$$

**Step 3: Evaluate kernel windows at stride $S = 2$:**

- **Top-left element $Y_{0, 0}$ (Window rows 0..2, cols 0..2):**
  Window:
  $$\begin{bmatrix} 0 & 0 & 0 \\ 0 & 1 & 2 \\ 0 & 0 & 1 \end{bmatrix}$$
  Element-wise multiplication with $K$:
  $$0(1) + 0(0) + 0(-1) + 0(1) + 1(0) + 2(-1) + 0(1) + 0(0) + 1(-1) = 0 - 2 - 1 = -3$$
  Add bias $b = -2.0$:
  $$Y_{0, 0} = -3 + (-2) = -5.0$$

- **Top-right element $Y_{0, 1}$ (Window rows 0..2, cols 2..4):**
  Window:
  $$\begin{bmatrix} 0 & 0 & 0 \\ 0 & 3 & 0 \\ 1 & 2 & 0 \end{bmatrix}$$
  Element-wise multiplication with $K$:
  $$0(1) + 0(0) + 0(-1) + 0(1) + 3(0) + 0(-1) + 1(1) + 2(0) + 0(-1) = 1$$
  Add bias $b = -2.0$:
  $$Y_{0, 1} = 1 + (-2) = -1.0$$

- **Bottom-left element $Y_{1, 0}$ (Window rows 2..4, cols 0..2):**
  Window:
  $$\begin{bmatrix} 0 & 0 & 1 \\ 0 & 2 & 0 \\ 0 & 1 & 2 \end{bmatrix}$$
  Element-wise multiplication with $K$:
  $$0(1) + 0(0) + 1(-1) + 0(1) + 2(0) + 0(-1) + 0(1) + 1(0) + 2(-1) = -1 - 2 = -3$$
  Add bias $b = -2.0$:
  $$Y_{1, 0} = -3 + (-2) = -5.0$$

- **Bottom-right element $Y_{1, 1}$ (Window rows 2..4, cols 2..4):**
  Window:
  $$\begin{bmatrix} 1 & 1 & 2 \\ 3 & 1 & 0 \\ 0 & 1 & 0 \end{bmatrix}$$
  Element-wise multiplication with $K$:
  $$1(1) + 1(0) + 2(-1) + 3(1) + 1(0) + 0(-1) + 0(1) + 1(0) + 0(-1) = (1 - 2) + 3 = 2$$
  Add bias $b = -2.0$:
  $$Y_{1, 1} = 2 + (-2) = 0.0$$

**Final Output Matrix:**
$$Y = \begin{bmatrix} -5.0 & -1.0 \\ -5.0 & 0.0 \end{bmatrix}$$
</details>

---

### Problem 2: Multi-Channel Parameter & FLOPs Analysis (Medium)
An autonomous vehicle camera feed of resolution $H = 256, W = 512$ with $C_{\text{in}} = 3$ color channels is passed into a convolutional layer configured as follows:
- Number of filters ($C_{\text{out}}$): 64
- Kernel spatial size ($K_h \times K_w$): $5 \times 5$
- Padding ($P$): 2 ("same" padding)
- Stride ($S$): 2

1. Compute the spatial dimensions of the output feature map ($H_{\text{out}}, W_{\text{out}}$).
2. Calculate the total number of learnable parameters (weights + biases).
3. Calculate the total number of Multiply-Accumulate operations (MACs) and approximate FLOPs required for a single image forward pass.

<details>
<summary>View Complete Solution</summary>

**Part 1: Output Spatial Dimensions:**
$$H_{\text{out}} = \left\lfloor \frac{256 - 5 + 2(2)}{2} \right\rfloor + 1 = \left\lfloor \frac{255}{2} \right\rfloor + 1 = 127 + 1 = 128$$
$$W_{\text{out}} = \left\lfloor \frac{512 - 5 + 2(2)}{2} \right\rfloor + 1 = \left\lfloor \frac{511}{2} \right\rfloor + 1 = 255 + 1 = 256$$
Output feature map volume: $[64, 128, 256]$.

**Part 2: Parameter Count:**
Each of the 64 filters spans all 3 input channels with a $5 \times 5$ spatial footprint, plus 1 bias scalar per output channel:
$$\text{Weights} = C_{\text{out}} \times C_{\text{in}} \times K_h \times K_w = 64 \times 3 \times 5 \times 5 = 64 \times 75 = 4,800$$
$$\text{Biases} = C_{\text{out}} = 64$$
$$\text{Total Parameters} = 4,800 + 64 = 4,864$$

**Part 3: MACs and FLOPs:**
At each of the $H_{\text{out}} \times W_{\text{out}}$ spatial locations, each of the $C_{\text{out}}$ filters performs $C_{\text{in}} \times K_h \times K_w$ MAC operations:
$$\text{MACs} = H_{\text{out}} \times W_{\text{out}} \times C_{\text{out}} \times (C_{\text{in}} \cdot K_h \cdot K_w)$$
$$\text{MACs} = 128 \times 256 \times 64 \times 75 = 32,768 \times 4,800 = 157,286,400 \text{ MACs} \approx 157.3 \text{ MMACs}$$

Since $1 \text{ MAC} = 1 \text{ multiplication} + 1 \text{ addition} = 2 \text{ FLOPs}$:
$$\text{FLOPs} \approx 2 \times 157,286,400 = 314,572,800 \text{ FLOPs} \approx 314.6 \text{ MFLOPs}$$
</details>

---

### Problem 3: Receptive Field Propagation & VGG-Style Filter Stacking Proof (Hard)
In 2014, the Visual Geometry Group (VGG) at Oxford proved that stacking multiple small $3 \times 3$ convolutional layers is mathematically and computationally superior to using larger $5 \times 5$ or $7 \times 7$ kernels.

1. Using the recursive formula $RF_l = RF_{l-1} + (k_l - 1) \cdot J_{l-1}$, prove that stacking **two** successive $3 \times 3$ conv layers (with stride 1) achieves the exact same receptive field as a single $5 \times 5$ conv layer.
2. Prove that stacking **three** successive $3 \times 3$ conv layers achieves the exact same receptive field as a single $7 \times 7$ conv layer.
3. Assuming all intermediate layers have $C$ channels (i.e. $C_{\text{in}} = C_{\text{out}} = C$):
   - Compute the parameter ratio of two stacked $3 \times 3$ layers vs. one $5 \times 5$ layer.
   - Compute the parameter ratio of three stacked $3 \times 3$ layers vs. one $7 \times 7$ layer.
4. What qualitative advantage beyond parameter efficiency does stacking multiple $3 \times 3$ layers provide?

<details>
<summary>View Complete Solution</summary>

**Part 1: Receptive field of two stacked $3 \times 3$ layers:**
Initial condition: $RF_0 = 1$, $J_0 = 1$.
- Layer 1 ($k_1 = 3, s_1 = 1$):
  $$RF_1 = RF_0 + (k_1 - 1) \cdot J_0 = 1 + (3 - 1) \cdot 1 = 3$$
  $$J_1 = J_0 \cdot s_1 = 1 \cdot 1 = 1$$
- Layer 2 ($k_2 = 3, s_2 = 1$):
  $$RF_2 = RF_1 + (k_2 - 1) \cdot J_1 = 3 + (3 - 1) \cdot 1 = 5$$
A single $5 \times 5$ conv layer has $RF = 1 + (5 - 1) \cdot 1 = 5$.
Thus, two stacked $3 \times 3$ layers match the receptive field of one $5 \times 5$ layer identically ($5 \times 5$).

**Part 2: Receptive field of three stacked $3 \times 3$ layers:**
- Layer 3 ($k_3 = 3, s_3 = 1$):
  $$RF_3 = RF_2 + (k_3 - 1) \cdot J_2 = 5 + (3 - 1) \cdot 1 = 7$$
A single $7 \times 7$ conv layer has $RF = 1 + (7 - 1) \cdot 1 = 7$.
Thus, three stacked $3 \times 3$ layers match the receptive field of one $7 \times 7$ layer identically ($7 \times 7$).

**Part 3: Parameter Comparisons (excluding bias):**
- **Two $3 \times 3$ vs. One $5 \times 5$:**
  - Two $3 \times 3$ layers: $2 \times (C \times C \times 3 \times 3) = 18 C^2$ parameters.
  - One $5 \times 5$ layer: $1 \times (C \times C \times 5 \times 5) = 25 C^2$ parameters.
  - Ratio: $\frac{18 C^2}{25 C^2} = \frac{18}{25} = 0.72 \implies \mathbf{28\% \text{ reduction in parameters}}$.

- **Three $3 \times 3$ vs. One $7 \times 7$:**
  - Three $3 \times 3$ layers: $3 \times (C \times C \times 3 \times 3) = 27 C^2$ parameters.
  - One $7 \times 7$ layer: $1 \times (C \times C \times 7 \times 7) = 49 C^2$ parameters.
  - Ratio: $\frac{27 C^2}{49 C^2} \approx 0.551 \implies \mathbf{44.9\% \text{ reduction in parameters}}$.

**Part 4: Qualitative Architectural Advantage:**
Stacking multiple smaller layers incorporates multiple non-linear activation functions (e.g. ReLU) instead of just one.
- One $7 \times 7$ conv has only 1 activation function.
- Three stacked $3 \times 3$ convs have 3 consecutive non-linear activation functions ($\text{Conv} \to \text{ReLU} \to \text{Conv} \to \text{ReLU} \to \text{Conv} \to \text{ReLU}$).
This substantially increases the representational capacity (expressivity) of the network, enabling it to learn much more discriminative, non-linear feature representations.
</details>
