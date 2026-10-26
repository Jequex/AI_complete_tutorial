# Topic 12.4: Matrix/Tensor Backpropagation & Gradient Checking

---

## 1. Executive Summary & The 12-Year-Old Analogy

While scalar autograd (Topic 12.3) illuminates the graph theory of differentiation, real-world deep learning cannot operate on scalar loops. Modern hardware (NVIDIA GPUs, Google TPUs) is architecturally designed around dense parallel matrix multiplication units (Tensor Cores / Systolic Arrays).

To train modern neural networks, we must vectorize backpropagation: propagating **entire matrices and tensors of adjoints** in a single vectorized BLAS/GEMM call.

Once we derive analytical tensor gradients, how do we know our matrix transpose derivations and tensor index manipulations are 100% bug-free? We use **Gradient Checking (GradCheck)**: rigorously verifying analytical gradients against high-precision numerical finite differences.

```
┌────────────────────────────────────────────────────────────────────────┐
│                     THE MATRIX AFFINE LAYER: Y = XW + b                │
│                                                                        │
│   FORWARD PASS:                                                        │
│   X (B x Din)  ──►  [ * W (Din x Dout) ]  ──►  [ + b (1 x Dout) ]  ──► Y
│                                                                        │
│   BACKWARD PASS:                                                       │
│   dL/dX (B x Din)  ◄──  [ dL/dY · W^T ]                                │
│   dL/dW (Din x Dout) ◄── [ X^T · dL/dY ]                               │
│   dL/db (1 x Dout)  ◄──  [ sum_batch(dL/dY) ]                          │
└────────────────────────────────────────────────────────────────────────┘
```

---

### The 12-Year-Old Analogy: The Laser Beam & The Wooden Ruler Double-Check

Imagine you are an engineer building an ultra-precise rocket component:
1. **The Laser Beam (Analytical Tensor Backprop):**
   You have a sophisticated, lightning-fast laser measurement system. In a microsecond, it scans the entire metal structure and outputs the exact mechanical stress at every coordinate. It is incredibly fast, but if you coded a plus sign instead of a minus sign in the laser software, the whole measurement could be quietly inverted!
2. **The Wooden Ruler (Finite Difference GradCheck):**
   To verify the laser, you grab a simple, heavy wooden ruler. You physically push on the metal rod by a millimeter ($\epsilon = 0.001$), measure the deflection with your ruler, push it the other way ($-\epsilon$), and calculate the ratio $\frac{\Delta y}{2\epsilon}$.
   - It takes 500 times longer to measure with the ruler.
   - You would never use the wooden ruler to guide the rocket in real-time.
   - But if the ruler's physical measurement and the laser's software output agree to within one-millionth of an inch, **you know with 100% certainty that your math is correct!**

In deep learning:
- **Analytical Backpropagation** is the laser: blazing fast, runs on GPUs during training.
- **GradCheck** is the wooden ruler: used only during unit testing to prove the analytical formulas have zero bugs!

---

## 2. Mathematical Symbol Decoder

| Symbol | Tensor Shape | Description & Operational Role |
| :--- | :--- | :--- |
| $X$ | $(B, D_{\text{in}})$ | Input batch tensor ($B$ samples, $D_{\text{in}}$ features) |
| $W$ | $(D_{\text{in}}, D_{\text{out}})$ | Weight parameter matrix |
| $b$ | $(1, D_{\text{out}})$ | Bias row vector (broadcast across batch dimension) |
| $Y = XW + b$ | $(B, D_{\text{out}})$ | Forward linear activation tensor |
| $\frac{\partial L}{\partial Y}$ | $(B, D_{\text{out}})$ | Incoming adjoint tensor from subsequent layers |
| $\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} W^T$ | $(B, D_{\text{in}})$ | Upstream gradient passed to preceding layer |
| $\frac{\partial L}{\partial W} = X^T \frac{\partial L}{\partial Y}$ | $(D_{\text{in}}, D_{\text{out}})$ | Weight gradient accumulator |
| $\frac{\partial L}{\partial b} = \sum_{i=1}^B \left(\frac{\partial L}{\partial Y}\right)_{i, :}$ | $(1, D_{\text{out}})$ | Bias gradient (reduction sum across batch) |
| $g_{\text{num}}$ | Scalar / Tensor | Two-sided numerical finite difference approximation |
| $\text{RelErr}$ | Scalar $\in [0, 1]$ | Relative error metric between analytical and numerical gradients |

---

## 3. Matrix Calculus: First-Principles Derivation of Linear Layer Backprop

Consider the affine transformation:
$$Y = X W + b$$
where $X \in \mathbb{R}^{B \times D_{\text{in}}}$, $W \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}$, and $b \in \mathbb{R}^{1 \times D_{\text{out}}}$.

Let $L$ be a scalar loss function. Suppose we are given the upstream gradient tensor:
$$\bar{Y} \equiv \frac{\partial L}{\partial Y} \in \mathbb{R}^{B \times D_{\text{out}}}, \quad \text{where } \bar{Y}_{ij} = \frac{\partial L}{\partial Y_{ij}}$$

### 3.1 Derivation of $\frac{\partial L}{\partial W}$ (Weight Gradients)
In index notation, each entry of $Y$ is:
$$Y_{ij} = \sum_{k=1}^{D_{\text{in}}} X_{ik} W_{kj} + b_j$$

By the multivariable chain rule, the sensitivity of $L$ to parameter $W_{kj}$ is the sum over all elements of $Y$ that depend on $W_{kj}$:
$$\frac{\partial L}{\partial W_{kj}} = \sum_{i=1}^B \sum_{m=1}^{D_{\text{out}}} \frac{\partial L}{\partial Y_{im}} \frac{\partial Y_{im}}{\partial W_{kj}}$$

Notice that $\frac{\partial Y_{im}}{\partial W_{kj}} \ne 0$ if and only if $m = j$:
$$\frac{\partial Y_{ij}}{\partial W_{kj}} = X_{ik}$$

Substituting this in:
$$\frac{\partial L}{\partial W_{kj}} = \sum_{i=1}^B \frac{\partial L}{\partial Y_{ij}} X_{ik} = \sum_{i=1}^B X_{ik} \bar{Y}_{ij} = \sum_{i=1}^B (X^T)_{ki} \bar{Y}_{ij}$$

Recognizing the definition of matrix multiplication:
$$\boxed{\frac{\partial L}{\partial W} = X^T \frac{\partial L}{\partial Y} = X^T \bar{Y}}$$

**Dimensional Verification Check:**
- $X^T$ has shape $(D_{\text{in}} \times B)$
- $\bar{Y}$ has shape $(B \times D_{\text{out}})$
- Matrix product $(D_{\text{in}} \times B) \times (B \times D_{\text{out}}) = (D_{\text{in}} \times D_{\text{out}})$
- Matches the shape of $W$ perfectly! $\checkmark$

---

### 3.2 Derivation of $\frac{\partial L}{\partial X}$ (Input / Activation Gradients)
Similarly, to propagate the gradient backward to earlier layers, we compute $\frac{\partial L}{\partial X_{ik}}$:
$$\frac{\partial L}{\partial X_{ik}} = \sum_{m=1}^B \sum_{j=1}^{D_{\text{out}}} \frac{\partial L}{\partial Y_{mj}} \frac{\partial Y_{mj}}{\partial X_{ik}}$$

Since $X_{ik}$ only affects row $i$ of $Y$:
$$\frac{\partial Y_{ij}}{\partial X_{ik}} = W_{kj}$$

Substituting:
$$\frac{\partial L}{\partial X_{ik}} = \sum_{j=1}^{D_{\text{out}}} \bar{Y}_{ij} W_{kj} = \sum_{j=1}^{D_{\text{out}}} \bar{Y}_{ij} (W^T)_{jk}$$

In matrix notation:
$$\boxed{\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} W^T = \bar{Y} W^T}$$

**Dimensional Verification Check:**
- $\bar{Y}$ has shape $(B \times D_{\text{out}})$
- $W^T$ has shape $(D_{\text{out}} \times D_{\text{in}})$
- Matrix product $(B \times D_{\text{out}}) \times (D_{\text{out}} \times D_{\text{in}}) = (B \times D_{\text{in}})$
- Matches the shape of $X$ perfectly! $\checkmark$

---

### 3.3 Derivation of $\frac{\partial L}{\partial b}$ (Bias Gradients)
Because the bias vector $b \in \mathbb{R}^{1 \times D_{\text{out}}}$ is broadcast across all $B$ batch samples:
$$Y_{ij} = (XW)_{ij} + b_j \implies \frac{\partial Y_{ij}}{\partial b_j} = 1.0 \quad \text{for all } i \in \{1, \dots, B\}$$

Summing over the batch:
$$\frac{\partial L}{\partial b_j} = \sum_{i=1}^B \frac{\partial L}{\partial Y_{ij}} \frac{\partial Y_{ij}}{\partial b_j} = \sum_{i=1}^B \bar{Y}_{ij}$$

In matrix notation:
$$\boxed{\frac{\partial L}{\partial b} = \mathbf{1}_{1 \times B} \cdot \bar{Y} = \sum_{i=1}^B \bar{Y}_{i, :}}$$

The bias gradient is simply the column-wise sum of the output adjoints across all batch elements!

---

## 4. The "Dimension Bookkeeping" Heuristic

When working with matrix backprop, you do not need to expand full indices every time. You can use the **Dimension Matching Rule**:

1. You know $\frac{\partial L}{\partial W}$ must have the same shape as $W$: $(D_{\text{in}}, D_{\text{out}})$.
2. You have $X$ of shape $(B, D_{\text{in}})$ and $\bar{Y}$ of shape $(B, D_{\text{out}})$.
3. The only matrix multiplication between $X$ and $\bar{Y}$ that cancels the batch dimension $B$ and yields shape $(D_{\text{in}}, D_{\text{out}})$ is:
$$X^T \cdot \bar{Y} \quad [(D_{\text{in}} \times B) \times (B \times D_{\text{out}}) = (D_{\text{in}} \times D_{\text{out}})]$$
4. Similarly, $\frac{\partial L}{\partial X}$ must have shape $(B, D_{\text{in}})$. The only product between $\bar{Y}$ and $W$ that yields $(B, D_{\text{in}})$ is:
$$\bar{Y} \cdot W^T \quad [(B \times D_{\text{out}}) \times (D_{\text{out}} \times D_{\text{in}}) = (B \times D_{\text{in}})]$$

---

## 5. Gradient Checking (GradCheck)

Gradient checking compares your analytical gradient tensor against a numerical finite difference approximation.

### 5.1 Why Two-Sided Central Differences?
- **One-Sided Difference:**
  $$g_{\text{one-sided}} = \frac{f(\theta + \epsilon) - f(\theta)}{\epsilon}$$
  By Taylor expansion:
  $$f(\theta + \epsilon) = f(\theta) + \epsilon f'(\theta) + \frac{1}{2} \epsilon^2 f''(\theta) + \mathcal{O}(\epsilon^3)$$
  $$\implies \frac{f(\theta + \epsilon) - f(\theta)}{\epsilon} = f'(\theta) + \underbrace{\frac{1}{2}\epsilon f''(\theta)}_{\mathcal{O}(\epsilon) \text{ Error!}}$$
  The truncation error is first-order $\mathcal{O}(\epsilon)$. For $\epsilon = 10^{-7}$, error is $\sim 10^{-7}$.

- **Two-Sided Central Difference:**
  $$g_{\text{central}} = \frac{f(\theta + \epsilon) - f(\theta - \epsilon)}{2\epsilon}$$
  Subtracting Taylor expansions:
  $$f(\theta + \epsilon) - f(\theta - \epsilon) = 2\epsilon f'(\theta) + 2 \frac{\epsilon^3}{3!} f'''(\theta) + \dots$$
  Dividing by $2\epsilon$:
  $$g_{\text{central}} = f'(\theta) + \underbrace{\mathcal{O}(\epsilon^2)}_{\text{Second-Order Error!}}$$
  The second-order error term **vanishes completely**! For $\epsilon = 10^{-7}$, the truncation error is $\mathcal{O}(\epsilon^2) \sim 10^{-14}$ (down near machine precision).

```
                      Slope of secant line (Two-sided difference)
                                    /
                                   /  * f(theta + eps)
                                  /  /
                                 /  /
        f(theta - eps) *        /  /
                         \     /  /
                          \   /  /
                           \ /  /
                            *  f(theta)
```

---

### 5.2 The Relative Error Metric
Never compare gradients using absolute differences $|g_{\text{ana}} - g_{\text{num}}|$, because a difference of $0.01$ is catastrophic if $g = 0.001$, but negligible if $g = 1,000,000$.

We use the scale-invariant **Frobenius Relative Error**:
$$\boxed{\text{RelErr} = \frac{\|g_{\text{analytical}} - g_{\text{numerical}}\|_F}{\|g_{\text{analytical}}\|_F + \|g_{\text{numerical}}\|_F + \epsilon_{\text{denom}}}}$$

### 5.3 GradCheck Interpretation Guide

| Relative Error Band | Diagnosis & Action Required |
| :--- | :--- |
| $\text{RelErr} < 10^{-7}$ | **Flawless Match.** Derivation and code are provably correct. |
| $10^{-7} \le \text{RelErr} < 10^{-5}$ | **Acceptable.** Minor floating point rounding or near non-smooth kink. |
| $10^{-4} \le \text{RelErr} < 10^{-2}$ | **Warning / Defective.** High probability of incorrect constant or sign. |
| $\text{RelErr} \ge 10^{-1}$ | **Catastrophic Failure.** Fundamental bug in derivation, transposition, or indexing. |

---

## 6. Summary: Best Practices for Writing Custom Layers

1. **Always double-check inner dimensions:** Tensor contractions must match shapes before and after transposition.
2. **Batch reduction:** Any parameter broadcasted across batch samples ($b$) must reduce via summation in backward.
3. **Always run GradCheck in float64:** Floating point 32 has only 24 bits of mantissa ($\approx 7$ decimal digits), which contaminates finite difference subtraction with cancellation noise. Double precision (`float64`, Python `float`) provides 53 mantissa bits ($\approx 15$ digits), enabling pristine $\epsilon = 10^{-7}$ verification.
