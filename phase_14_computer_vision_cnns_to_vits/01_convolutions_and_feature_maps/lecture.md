# Topic 14.1: Convolutions & Feature Maps

---

## 1. Executive Summary & The 12-Year-Old Analogy

In fully connected feedforward networks (MLPs), every input neuron connects to every output neuron. While this architecture is theoretically capable of approximating any continuous function, it fails catastrophically when applied to high-resolution visual data. A modest $1024 \times 1024$ color image possesses over 3 million scalar values; connecting it to a hidden layer of 1024 units would require over 3 billion parameters, instantly resulting in computational gridlock and destructive overfitting.

More fundamentally, MLPs treat images as flat, 1D vectors, ignoring the essential spatial geometry of the physical world:
1. **Local Connectivity (Locality):** Nearby pixels correlate strongly to form edges, curves, and textures. A pixel at $(0, 0)$ rarely interacts directly with a pixel at $(1023, 1023)$ without an intermediate hierarchy.
2. **Translation Equivariance:** A dog in the top-left corner of an image has the same visual characteristics as a dog in the bottom-right corner. The feature detector that recognizes fur or ears should not need to be re-learned at every coordinate of the pixel grid.

The **Convolutional Neural Network (CNN)** solves these challenges through **parameter sharing** and **sparse local interactions**. Instead of full matrix multiplication across all pixels, CNNs slide compact mathematical templates—called **kernels** or **filters**—across the image, producing structured 2D activations known as **feature maps**.

```
                   THE 2D CONVOLUTION (CROSS-CORRELATION) OPERATION
                   
     Input Feature Map (I)                Kernel (K)              Output Feature Map (S)
    ┌─────┬─────┬─────┬─────┐                                     ┌─────┬─────┐
    │  1  │  2  │  0  │  0  │             ┌─────┬─────┐           │  8  │  7  │
    ├─────┼─────┼─────┼─────┤             │  1  │  2  │           ├─────┼─────┤
    │  0  │  3  │  1  │  0  │      *      ├─────┼─────┤     =     │ 10  │  6  │
    ├─────┼─────┼─────┼─────┤             │  0  │  1  │           └─────┴─────┘
    │  1  │  2  │  1  │  0  │             └─────┴─────┘
    ├─────┼─────┼─────┼─────┤
    │  0  │  0  │  1  │  1  │
    └─────┴─────┴─────┴─────┘
            [4 x 4]                           [2 x 2]                 [2 x 2] (Stride 2)
```

---

### The 12-Year-Old Analogy: The Sliding Cookie Cutter & Edge Highlighter

Imagine you are looking at a giant coloring book page with a secret hidden drawing:

1. **The Stencil / Cookie Cutter (The Kernel):**
   - You hold a small magnifying glass stencil that is shaped specifically to look for sharp vertical lines (bright on the left, dark on the right).
   
2. **Sliding Across the Page (The Convolutional Sweep):**
   - You start at the top-left corner of the page. You look through the stencil at a small $3 \times 3$ patch of pixels.
   - If the patch matches your stencil perfectly, your stencil "glows bright green" (a large positive number!).
   - If the patch is flat or opposite, your stencil stays dark (zero or negative).

3. **Writing on the New Tracing Paper (The Feature Map):**
   - On a fresh sheet of paper, you mark down a single dot showing how brightly your stencil glowed at that exact location.
   - Then you take a step to the right (the **stride**) and check the next patch.
   - Once you reach the edge, you move down and scan the next row.

4. **Multi-Channel Coloring (Looking at Red, Green, and Blue):**
   - If your picture is in full color, your stencil has three layers: one looking at red light, one at green, and one at blue. It checks all three at the same time and combines them into one score.

5. **The Magic Hierarchy:**
   - Layer 1 finds simple edges and corners.
   - Layer 2 looks at the edges and spots shapes like circles and triangles.
   - Layer 3 looks at shapes and identifies eyes, wheels, and cat ears!

---

## 2. Mathematical Symbol Decoder

| Symbol | Type / Domain | Mathematical Meaning |
| :--- | :--- | :--- |
| $I \in \mathbb{R}^{H \times W}$ | 2D Matrix | Input image or feature map of height $H$ and width $W$ |
| $K \in \mathbb{R}^{K_h \times K_w}$ | 2D Matrix | Spatial filter / kernel of height $K_h$ and width $K_w$ |
| $X \in \mathbb{R}^{C_{\text{in}} \times H \times W}$ | 3D Tensor | Multi-channel input tensor with $C_{\text{in}}$ channels |
| $W \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times K_h \times K_w}$ | 4D Tensor | Convolutional weight tensor containing $C_{\text{out}}$ filters |
| $\mathbf{b} \in \mathbb{R}^{C_{\text{out}}}$ | 1D Vector | Per-filter learnable additive bias vector |
| $Y \in \mathbb{R}^{C_{\text{out}} \times H_{\text{out}} \times W_{\text{out}}}$ | 3D Tensor | Output feature map volume |
| $P \in \mathbb{Z}_{\ge 0}$ | Integer | Zero-padding width applied symmetrically to spatial borders |
| $S \in \mathbb{Z}_{\ge 1}$ | Integer | Stride: step size in pixels between successive window placements |
| $RF_l \in \mathbb{Z}_{\ge 1}$ | Integer | Spatial diameter of the Receptive Field at network layer $l$ |
| $J_l \in \mathbb{Z}_{\ge 1}$ | Integer | Cumulative stride (jump) at layer $l$: $J_l = \prod_{i=1}^l S_i$ |
| $\text{MACs}$ | Integer | Multiply-Accumulate operations required for one forward pass |
| $\text{FLOPs}$ | Integer | Floating-point operations ($\approx 2 \times \text{MACs}$) |

---

## 3. Mathematical Foundations: Convolution vs. Cross-Correlation

### 3.1 The Continuous and Discrete Mathematical Convolution
In pure mathematics and signal processing, the continuous convolution of two functions $f$ and $g$ is defined as:
$$(f * g)(t) = \int_{-\infty}^{\infty} f(\tau) g(t - \tau) d\tau$$

In 2D discrete space, for an image $I$ and kernel $K$, the mathematical convolution is defined with a double summation where the kernel indices are **flipped** (reflected across both axes):
$$(I * K)(i, j) = \sum_{m=-\infty}^{\infty} \sum_{n=-\infty}^{\infty} I(i - m, j - n) K(m, n)$$

### 3.2 Cross-Correlation in Deep Learning
In deep learning frameworks (PyTorch, TensorFlow, JAX), the operation implemented under the name `conv2d` is actually **cross-correlation** (denoted $\star$):
$$(I \star K)(i, j) = \sum_{m=0}^{K_h - 1} \sum_{n=0}^{K_w - 1} I(i + m, j + n) K(m, n)$$

**Why omit the kernel flip?**
1. **Symmetry in Learning:** The weights in $K$ are not fixed predetermined coefficients; they are initialized randomly and learned via gradient descent ($\frac{\partial \mathcal{L}}{\partial K}$). A flipped learned kernel is mathematically equivalent to an un-flipped learned kernel.
2. **Computational Efficiency:** Eliminating the spatial index reflection saves memory copies and simplifies tensor pointer arithmetic in CUDA GEMM kernels.

---

## 4. Spatial Arithmetic: Padding, Stride, and Output Dimensions

When a kernel of size $K_h \times K_w$ slides over an input of size $H \times W$ with stride $S$ and zero-padding $P$, how large is the resulting output feature map?

### 4.1 The Spatial Output Dimension Formula
$$H_{\text{out}} = \left\lfloor \frac{H - K_h + 2P}{S} \right\rfloor + 1$$
$$W_{\text{out}} = \left\lfloor \frac{W - K_w + 2P}{S} \right\rfloor + 1$$

Where:
- $H, W$: Input spatial dimensions.
- $K_h, K_w$: Kernel height and width.
- $P$: Zero-padding added to each side (top/bottom or left/right).
- $S$: Stride (subsampling factor).
- $\lfloor \cdot \rfloor$: Floor division function.

```
       ZERO PADDING: PRESERVING BOUNDARY INFORMATION (P=1)
       
              0     0     0     0     0     0
                 ┌─────┬─────┬─────┬─────┐
              0  │ x₀₀ │ x₀₁ │ x₀₂ │ x₀₃ │  0
                 ├─────┼─────┼─────┼─────┤
              0  │ x₁₀ │ x₁₁ │ x₁₂ │ x₁₃ │  0
                 ├─────┼─────┼─────┼─────┤
              0  │ x₂₀ │ x₂₁ │ x₂₂ │ x₂₃ │  0
                 ├─────┼─────┼─────┼─────┤
              0  │ x₃₀ │ x₃₁ │ x₃₂ │ x₃₃ │  0
                 └─────┴─────┴─────┴─────┘
              0     0     0     0     0     0
```

### 4.2 Standard Padding Modes
1. **Valid Padding ($P = 0$):**
   - No padding is added. The kernel never ventures beyond original pixel boundaries.
   - Spatial dimension shrinks: $O = \lfloor \frac{W - K}{S} \rfloor + 1 < W$.
   - Boundary pixels are visited far fewer times than central pixels, causing border information loss.

2. **Same Padding ($O = W$ when $S = 1$):**
   - The spatial dimensions of the output match the input exactly.
   - For an odd kernel size $K$, setting:
     $$P = \frac{K - 1}{2}$$
     yields:
     $$O = \frac{W - K + 2\left(\frac{K-1}{2}\right)}{1} + 1 = W - K + K - 1 + 1 = W$$
   - Example: A $3 \times 3$ kernel requires $P = 1$; a $5 \times 5$ kernel requires $P = 2$; a $7 \times 7$ kernel requires $P = 3$.

---

## 5. Multi-Channel Convolutions & Tensor Volumes

Real-world images are not 2D matrices; they are multi-channel tensors (e.g., RGB images with $C_{\text{in}} = 3$). Subsequent hidden layers can contain dozens or hundreds of channels ($C_{\text{in}} = 64, 128, 512$).

### 5.1 Mechanics of Multi-Channel Filtering
- A single convolutional filter is a **3D volume** of shape $[C_{\text{in}}, K_h, K_w]$.
- To produce an output volume with $C_{\text{out}}$ channels, we stack $C_{\text{out}}$ distinct 3D filters, forming a **4D parameter tensor**:
  $$\mathcal{W} \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times K_h \times K_w}$$
- For output channel $k \in \{0, \dots, C_{\text{out}} - 1\}$ at coordinate $(i, j)$:
  $$Y_{k, i, j} = b_k + \sum_{c=0}^{C_{\text{in}}-1} \sum_{m=0}^{K_h-1} \sum_{n=0}^{K_w-1} X_{c, \, i\cdot S + m, \, j\cdot S + n} \cdot \mathcal{W}_{k, c, m, n}$$

Notice that the sum over all input channels $c \in \{0, \dots, C_{\text{in}}-1\}$ **collapses the channel dimension**, converting $C_{\text{in}}$ 2D responses into a single scalar activation, to which a single scalar bias $b_k$ is added.

```
                  MULTI-CHANNEL CONVOLUTION VOLUME TRANSFORMATION
                  
       Input Tensor X               Weight Tensor W                Output Tensor Y
      [C_in x H x W]         [C_out x C_in x K_h x K_w]       [C_out x H_out x W_out]
       
         ┌──────┐                     ┌──────┐                      ┌──────┐
        ╱ C_in ╱│                    ╱ C_in ╱│                     ╱C_out ╱│
       ┌──────┐ │     *             ┌──────┐ │      =             ┌──────┐ │
       │      │ │      K_out        │      │ │                    │      │ │
       │  H   │ │     filters       │ K_h  │ │                    │H_out │ │
       │      │╱                    │      │╱                     │      │╱
       └──────┘                     └──────┘                      └──────┘
          W                           K_w                           W_out
```

---

## 6. Computational Complexity: FLOPs and Parameter Counts

Understanding CNN computational budgets is vital for edge deployment (mobile, autonomous vehicles, embedded vision).

### 6.1 Parameter Count
Every filter in output channel $k$ contains $C_{\text{in}} \times K_h \times K_w$ weights plus 1 bias scalar:
$$\text{Parameters} = C_{\text{out}} \times (C_{\text{in}} \cdot K_h \cdot K_w + 1)$$

*Crucial Insight:* The number of parameters is **independent of the spatial size of the input image ($H, W$)**! This is the fundamental property of **parameter sharing**.

### 6.2 Multiply-Accumulate Operations (MACs) and FLOPs
For each of the $H_{\text{out}} \times W_{\text{out}}$ spatial locations, across all $C_{\text{out}}$ filters:
- We compute $C_{\text{in}} \times K_h \times K_w$ multiplications and additions.
- Therefore:
  $$\text{MACs} = H_{\text{out}} \cdot W_{\text{out}} \cdot C_{\text{out}} \cdot (C_{\text{in}} \cdot K_h \cdot K_w)$$
- Because one MAC equals 1 multiply + 1 addition (2 floating-point operations):
  $$\text{FLOPs} \approx 2 \times \text{MACs} = 2 \cdot H_{\text{out}} \cdot W_{\text{out}} \cdot C_{\text{out}} \cdot C_{\text{in}} \cdot K_h \cdot K_w$$

---

## 7. Receptive Field (RF) Propagation

The **Receptive Field (RF)** of a unit in layer $l$ is the spatial region in the original input image that contributes to that unit's activation.

```
                        RECEPTIVE FIELD (RF) EXPANSION
                        
   Input Image (L0)       Layer 1 (k=3, s=1)        Layer 2 (k=3, s=1)
   ┌─┬─┬─┬─┬─┬─┬─┐          ┌─┬─┬─┬─┬─┐               ┌─┬─┬─┐
   │ │ │ │ │ │ │ │          │ │ │ │ │ │               │ │■│ │  <-- Output neuron
   ├─┼─┼─┼─┼─┼─┼─┤          ├─┼─┼─┼─┼─┤               └─┴─┴─┘
   │ │█│█│█│█│█│ │          │ │█│█│█│ │                       Receptive Field = 5x5
   ├─┼─┼─┼─┼─┼─┼─┤          ├─┼─┼─┼─┼─┤
   │ │█│█│█│█│█│ │          │ │█│█│█│ │
   ├─┼─┼─┼─┼─┼─┼─┤          ├─┼─┼─┼─┼─┤
   │ │█│█│█│█│█│ │          │ │█│█│█│ │
   ├─┼─┼─┼─┼─┼─┼─┤          └─┴─┴─┴─┴─┘
   │ │█│█│█│█│█│ │
   ├─┼─┼─┼─┼─┼─┼─┤
   │ │█│█│█│█│█│ │
   └─┴─┴─┴─┴─┴─┴─┘
```

### 7.1 The Recursive Receptive Field Formula
Let:
- $k_l$: Kernel size at layer $l$.
- $s_l$: Stride at layer $l$.
- $J_l$: Cumulative stride (jump) at layer $l$, where $J_0 = 1$ and $J_l = J_{l-1} \cdot s_l$.

The receptive field $RF_l$ propagates recursively from input layer $0$ ($RF_0 = 1$):
$$RF_l = RF_{l-1} + (k_l - 1) \cdot J_{l-1}$$

### 7.2 Worked Example
Consider a 3-layer network:
- Layer 1: $k_1 = 3, s_1 = 1 \implies RF_1 = 1 + (3 - 1) \cdot 1 = 3$, $J_1 = 1$.
- Layer 2: $k_2 = 3, s_2 = 2 \implies RF_2 = 3 + (3 - 1) \cdot 1 = 5$, $J_2 = 1 \cdot 2 = 2$.
- Layer 3: $k_3 = 3, s_3 = 1 \implies RF_3 = 5 + (3 - 1) \cdot 2 = 9$, $J_3 = 2 \cdot 1 = 2$.

Even though each individual filter only spans $3 \times 3$ locally, a neuron in Layer 3 has an effective window of $9 \times 9$ over the input image!

### 7.3 Effective vs. Theoretical Receptive Field
While the **theoretical receptive field** grows linearly with depth, the **effective receptive field (ERF)** follows a 2D Gaussian distribution centered at the unit's receptive center. Pixels near the center contribute through many more computational paths than pixels near the boundary, giving center pixels exponentially higher gradient influence.

---

## 8. Classic Edge Detection Kernels: The Sobel Operator

Before deep networks learned filters automatically, computer vision engineers designed hand-crafted kernels based on discrete approximations of the image gradient $\nabla I = \left[ \frac{\partial I}{\partial x}, \frac{\partial I}{\partial y} \right]^T$.

### 8.1 The Sobel Kernels
- **Horizontal Gradient ($G_x$, detects vertical edges):**
  $$G_x = \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix}$$
- **Vertical Gradient ($G_y$, detects horizontal edges):**
  $$G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ +1 & +2 & +1 \end{bmatrix}$$

Notice that $G_x$ combines a difference operator in the horizontal direction with a smoothing Gaussian blur $[1, 2, 1]^T$ in the vertical direction to suppress high-frequency sensor noise.

The total gradient magnitude is:
$$G = \sqrt{G_x^2 + G_y^2}$$
And edge orientation is given by:
$$\theta = \text{atan2}(G_y, G_x)$$

---

## 9. Architectural Takeaways

1. **Parameter Efficiency:** Parameter sharing allows CNNs to scale to megapixel images without blowing up weight counts.
2. **Translation Equivariance:** Shifting the input results in an equivalent shift in the feature map: $f(T_x(I)) = T_x(f(I))$.
3. **Hierarchy of Abstraction:** Stacking small local filters naturally yields larger receptive fields, transforming raw pixels into edges, edges into motifs, and motifs into semantic objects.
