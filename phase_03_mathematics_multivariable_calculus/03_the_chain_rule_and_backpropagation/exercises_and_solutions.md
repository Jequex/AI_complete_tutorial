# Topic 3.3: Exercises & Solutions — The Chain Rule & Backpropagation

Welcome to the problem set for **Topic 3.3: The Chain Rule, Computational Graphs & Backpropagation**.

Follow along with pen and paper. Solve each problem before expanding the collapsible solution panels.

---

## 📝 Problem Set Overview

| Problem | Focus | Difficulty | Practical AI Connection |
| :---: | :--- | :---: | :--- |
| **1** | **Step-by-Step Manual Backpropagation on a Computational Graph** | Medium | Tracing PyTorch's internal tape engine |
| **2** | **Multi-Path Gradient Accumulation Calculus** | Medium | Residual connections ($x + f(x)$) in ResNets and Transformers |
| **3** | **Vectorized Matrix Backpropagation Derivation ($Y = XW + b$)** | Challenging | The exact linear layer backward pass inside torch.nn.Linear |

---

## Problem 1: Step-by-Step Manual Backpropagation on a Computational Graph

### 🧒 The Context: Maya's Multiply-Add Calculator
Maya has built a simple arithmetic toy module that computes:
$$f(x, y, z) = (x + y) \cdot z$$
For the specific inputs:
$$x = 2.0, \quad y = 3.0, \quad z = 4.0$$

Define the intermediate addition node:
$$q = x + y$$
$$f = q \cdot z$$

### Tasks:
1. Run the **Forward Pass**: Compute the exact values of $q$ and $f$.
2. Compute the four **Local Derivatives**: $\frac{\partial f}{\partial q}$, $\frac{\partial f}{\partial z}$, $\frac{\partial q}{\partial x}$, and $\frac{\partial q}{\partial y}$.
3. Run the **Backward Pass**: Compute the adjoint gradients $\bar{f}$, $\bar{z}$, $\bar{q}$, $\bar{x}$, and $\bar{y}$.
4. Sensitivity Verification: If Maya nudges $x$ by $\Delta x = 0.01$, by how much does the output $f$ change?

---

<details>
<summary>Click to view solution for Problem 1</summary>

### Detailed Step-by-Step Solution:

#### Step 1: Forward Pass
- $q = x + y = 2.0 + 3.0 = \mathbf{5.0}$
- $f = q \cdot z = 5.0 \cdot 4.0 = \mathbf{20.0}$

#### Step 2: Compute Local Derivatives at Current Activation Values
- $\frac{\partial f}{\partial q} = z = 4.0$
- $\frac{\partial f}{\partial z} = q = 5.0$
- $\frac{\partial q}{\partial x} = 1.0$
- $\frac{\partial q}{\partial y} = 1.0$

#### Step 3: Backward Pass (Adjoint Chain Rule)
1. Initialize seed gradient at output:
   $$\bar{f} = \frac{\partial f}{\partial f} = \mathbf{1.0}$$
2. Backpropagate through the multiplication node ($f = q \cdot z$):
   $$\bar{z} = \bar{f} \cdot \frac{\partial f}{\partial z} = 1.0 \times 5.0 = \mathbf{5.0}$$
   $$\bar{q} = \bar{f} \cdot \frac{\partial f}{\partial q} = 1.0 \times 4.0 = \mathbf{4.0}$$
3. Backpropagate through the addition node ($q = x + y$):
   $$\bar{x} = \bar{q} \cdot \frac{\partial q}{\partial x} = 4.0 \times 1.0 = \mathbf{4.0}$$
   $$\bar{y} = \bar{q} \cdot \frac{\partial q}{\partial y} = 4.0 \times 1.0 = \mathbf{4.0}$$

#### Step 4: Sensitivity Check
Since $\bar{x} = \frac{\partial f}{\partial x} = 4.0$:
$$\Delta f \approx \bar{x} \cdot \Delta x = 4.0 \times 0.01 = \mathbf{+0.04}$$
Testing directly: $f(2.01, 3, 4) = (5.01) \cdot 4 = 20.04$. The change is exactly $+0.04$!
</details>

---

## Problem 2: Multi-Path Gradient Accumulation Calculus

### 🧒 The Context: Timmy's Split-Stream Water Flow
In Transformer attention layers and residual networks ($x + \text{Attention}(x)$), a variable splits into multiple pathways and recombines later.
Consider the function:
$$f(x, y) = x^2 y + x y^2$$

### Tasks:
1. Differentiate directly with respect to $x$ using standard single-variable calculus rules.
2. Break the function into intermediate branch nodes:
   - $u = x^2 y$
   - $v = x y^2$
   - $f = u + v$
3. Apply the Multivariable Chain Rule:
   $$\frac{\partial f}{\partial x} = \frac{\partial f}{\partial u}\frac{\partial u}{\partial x} + \frac{\partial f}{\partial v}\frac{\partial v}{\partial x}$$
   and show that it yields the identical formula.
4. Evaluate numerically at $(x = 2, y = 3)$.

---

<details>
<summary>Click to view solution for Problem 2</summary>

### Detailed Step-by-Step Solution:

#### Step 1: Direct Differentiation
$$\frac{\partial f}{\partial x} = \frac{\partial}{\partial x}(x^2 y) + \frac{\partial}{\partial x}(x y^2) = \mathbf{2xy + y^2}$$

#### Step 2: Intermediate Branch Derivatives
- Path 1 ($u = x^2 y$):
  $$\frac{\partial u}{\partial x} = 2xy, \quad \frac{\partial f}{\partial u} = 1.0$$
- Path 2 ($v = x y^2$):
  $$\frac{\partial v}{\partial x} = y^2, \quad \frac{\partial f}{\partial v} = 1.0$$

#### Step 3: Multivariable Chain Rule Verification
$$\frac{\partial f}{\partial x} = \underbrace{\frac{\partial f}{\partial u} \frac{\partial u}{\partial x}}_{\text{Flow through Path 1}} + \underbrace{\frac{\partial f}{\partial v} \frac{\partial v}{\partial x}}_{\text{Flow through Path 2}} = (1.0)(2xy) + (1.0)(y^2) = \mathbf{2xy + y^2} \quad \blacksquare$$

#### Step 4: Numerical Evaluation at $(x = 2, y = 3)$
$$\frac{\partial f}{\partial x}(2, 3) = 2(2)(3) + 3^2 = 12 + 9 = \mathbf{21.0}$$
Both paths contribute to the total sensitivity: Path 1 contributes $12.0$, and Path 2 contributes $9.0$!
</details>

---

## Problem 3: Vectorized Matrix Backpropagation Derivation ($Y = XW + b$)

### 🧒 The Context: Maya's High-Speed Neural Layer
In deep learning libraries, layers process batches of data simultaneously using matrix operations:
$$Y = X W + b$$
Where:
- $X \in \mathbb{R}^{B \times D_{\text{in}}}$ ($B$ samples, $D_{\text{in}}$ features)
- $W \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}$ (Weight matrix)
- $b \in \mathbb{R}^{D_{\text{out}}}$ (Bias vector)
- $Y \in \mathbb{R}^{B \times D_{\text{out}}}$ (Layer activations)

Assume the downstream gradient $\frac{\partial \mathcal{L}}{\partial Y} \in \mathbb{R}^{B \times D_{\text{out}}}$ is already known.

### Tasks:
1. Write index formula for entry $Y_{i, j}$.
2. Derive the weight gradient matrix formula:
   $$\frac{\partial \mathcal{L}}{\partial W} = X^T \left( \frac{\partial \mathcal{L}}{\partial Y} \right)$$
3. Derive the input gradient matrix formula:
   $$\frac{\partial \mathcal{L}}{\partial X} = \left( \frac{\partial \mathcal{L}}{\partial Y} \right) W^T$$
4. Derive the bias gradient formula $\frac{\partial \mathcal{L}}{\partial b}$.

---

<details>
<summary>Click to view solution for Problem 3</summary>

### Detailed Step-by-Step Solution:

#### Step 1: Index Formulation
$$Y_{i, j} = \sum_{k=1}^{D_{\text{in}}} X_{i, k} W_{k, j} + b_j$$

#### Step 2: Weight Gradient Derivation
By the multivariable chain rule, entry $(k, j)$ of $\frac{\partial \mathcal{L}}{\partial W}$ is:
$$\frac{\partial \mathcal{L}}{\partial W_{k, j}} = \sum_{i=1}^B \frac{\partial \mathcal{L}}{\partial Y_{i, j}} \frac{\partial Y_{i, j}}{\partial W_{k, j}}$$
From Step 1, $\frac{\partial Y_{i, j}}{\partial W_{k, j}} = X_{i, k}$. Therefore:
$$\frac{\partial \mathcal{L}}{\partial W_{k, j}} = \sum_{i=1}^B X_{i, k} \left(\frac{\partial \mathcal{L}}{\partial Y}\right)_{i, j} = \sum_{i=1}^B (X^T)_{k, i} \left(\frac{\partial \mathcal{L}}{\partial Y}\right)_{i, j}$$
Writing in full matrix form:
$$\mathbf{\frac{\partial \mathcal{L}}{\partial W} = X^T \left( \frac{\partial \mathcal{L}}{\partial Y} \right)} \quad \blacksquare$$

**Dimension Check:**
- $X^T$ has dimension $(D_{\text{in}} \times B)$.
- $\frac{\partial \mathcal{L}}{\partial Y}$ has dimension $(B \times D_{\text{out}})$.
- Product dimension: $(D_{\text{in}} \times D_{\text{out}})$, matching $W$ exactly!

#### Step 3: Input Gradient Derivation
$$\frac{\partial \mathcal{L}}{\partial X_{i, k}} = \sum_{j=1}^{D_{\text{out}}} \frac{\partial \mathcal{L}}{\partial Y_{i, j}} \frac{\partial Y_{i, j}}{\partial X_{i, k}} = \sum_{j=1}^{D_{\text{out}}} \left(\frac{\partial \mathcal{L}}{\partial Y}\right)_{i, j} W_{k, j} = \sum_{j=1}^{D_{\text{out}}} \left(\frac{\partial \mathcal{L}}{\partial Y}\right)_{i, j} (W^T)_{j, k}$$
In matrix form:
$$\mathbf{\frac{\partial \mathcal{L}}{\partial X} = \left( \frac{\partial \mathcal{L}}{\partial Y} \right) W^T} \quad \blacksquare$$

**Dimension Check:**
- $\frac{\partial \mathcal{L}}{\partial Y}$ has dimension $(B \times D_{\text{out}})$.
- $W^T$ has dimension $(D_{\text{out}} \times D_{\text{in}})$.
- Product dimension: $(B \times D_{\text{in}})$, matching $X$ exactly!

#### Step 4: Bias Gradient Derivation
Since each bias $b_j$ is added to every sample $i$ in the batch:
$$\frac{\partial \mathcal{L}}{\partial b_j} = \sum_{i=1}^B \frac{\partial \mathcal{L}}{\partial Y_{i, j}} \cdot 1$$
In PyTorch vectorized code:
$$\mathbf{\frac{\partial \mathcal{L}}{\partial b} = \sum_{i=1}^B \left( \frac{\partial \mathcal{L}}{\partial Y} \right)_{i, :}} \quad (\text{Sum across batch dimension } 0)$$
</details>
