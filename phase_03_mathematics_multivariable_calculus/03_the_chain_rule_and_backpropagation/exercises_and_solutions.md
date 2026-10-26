# Topic 3.3: Exercises & Solutions — The Chain Rule & Backpropagation

---

## Problem Set Overview

| Problem | Focus | Difficulty |
| :--- | :--- | :--- |
| **1** | **Step-by-Step Manual Backpropagation on a Computational Graph** | Medium |
| **2** | **Multi-Path Gradient Accumulation Calculus** | Medium |
| **3** | **Vectorized Matrix Backpropagation Derivation ($Y = XW + b$)** | Challenging |

---

## Problem 1: Step-by-Step Manual Backpropagation on a Computational Graph

### The Context
Consider the computational graph evaluating:
$$f(x, y, z) = (x + y) \cdot z$$
Given input values:
$$x = 2.0, \quad y = 3.0, \quad z = 4.0$$

Define intermediate node:
$$q = x + y$$
$$f = q \cdot z$$

### Task
1. Execute the **Forward Pass**: Compute the numerical value of $q$ and $f$.
2. Compute the **Local Derivatives**: $\frac{\partial f}{\partial q}$, $\frac{\partial f}{\partial z}$, $\frac{\partial q}{\partial x}$, and $\frac{\partial q}{\partial y}$.
3. Execute the **Backward Pass**: Compute the adjoints $\bar{f}$, $\bar{z}$, $\bar{q}$, $\bar{x}$, and $\bar{y}$.
4. State the interpretation of $\bar{x}$: If $x$ increases by $0.01$, by how much does $f$ change?

---

### Solution 1

#### 1. Forward Pass
- $q = x + y = 2.0 + 3.0 = \mathbf{5.0}$
- $f = q \cdot z = 5.0 \cdot 4.0 = \mathbf{20.0}$

#### 2. Local Derivatives
- $\frac{\partial f}{\partial q} = z = 4.0$
- $\frac{\partial f}{\partial z} = q = 5.0$
- $\frac{\partial q}{\partial x} = 1.0$
- $\frac{\partial q}{\partial y} = 1.0$

#### 3. Backward Pass (Adjoint Accumulation)
1. Initialize at output:
   $$\bar{f} = \frac{\partial f}{\partial f} = \mathbf{1.0}$$
2. Backprop through multiplication node $f = q \cdot z$:
   $$\bar{z} = \bar{f} \cdot \frac{\partial f}{\partial z} = 1.0 \times 5.0 = \mathbf{5.0}$$
   $$\bar{q} = \bar{f} \cdot \frac{\partial f}{\partial q} = 1.0 \times 4.0 = \mathbf{4.0}$$
3. Backprop through addition node $q = x + y$:
   $$\bar{x} = \bar{q} \cdot \frac{\partial q}{\partial x} = 4.0 \times 1.0 = \mathbf{4.0}$$
   $$\bar{y} = \bar{q} \cdot \frac{\partial q}{\partial y} = 4.0 \times 1.0 = \mathbf{4.0}$$

#### 4. Sensitivity Interpretation
Since $\bar{x} = \frac{\partial f}{\partial x} = 4.0$:
$$\Delta f \approx \bar{x} \cdot \Delta x = 4.0 \times 0.01 = \mathbf{+0.04}$$

---

## Problem 2: Multi-Path Gradient Accumulation Calculus

### The Context
In Transformer attention layers and residual connections ($x + f(x)$), gradients flow through multiple parallel paths.

Consider the function:
$$f(x, y) = x^2 y + x y^2$$

### Task
1. Differentiate directly with respect to $x$ using standard calculus rules.
2. Break the function into intermediate nodes:
   - $u = x^2 y$
   - $v = x y^2$
   - $f = u + v$
3. Apply the multivariable chain rule $\frac{\partial f}{\partial x} = \frac{\partial f}{\partial u}\frac{\partial u}{\partial x} + \frac{\partial f}{\partial v}\frac{\partial v}{\partial x}$ and prove it produces the exact same expression.
4. Evaluate at $(x = 2, y = 3)$.

---

### Solution 2

#### 1. Direct Differentiation
$$\frac{\partial f}{\partial x} = \frac{\partial}{\partial x}(x^2 y) + \frac{\partial}{\partial x}(x y^2) = \mathbf{2xy + y^2}$$

#### 2. Intermediate Nodes
- $u = x^2 y \implies \frac{\partial u}{\partial x} = 2xy$
- $v = x y^2 \implies \frac{\partial v}{\partial x} = y^2$
- $f = u + v \implies \frac{\partial f}{\partial u} = 1, \quad \frac{\partial f}{\partial v} = 1$

#### 3. Multivariable Chain Rule
$$\frac{\partial f}{\partial x} = \frac{\partial f}{\partial u} \frac{\partial u}{\partial x} + \frac{\partial f}{\partial v} \frac{\partial v}{\partial x} = (1)(2xy) + (1)(y^2) = \mathbf{2xy + y^2} \quad \blacksquare$$

#### 4. Numerical Evaluation at $(x = 2, y = 3)$
$$\frac{\partial f}{\partial x}(2, 3) = 2(2)(3) + 3^2 = 12 + 9 = \mathbf{21.0}$$

---

## Problem 3: Vectorized Matrix Backpropagation Derivation ($Y = XW + b$)

### The Context
In PyTorch, neural layers do not operate on scalars; they operate on 2D matrices representing batches:
$$Y = X W + b$$
Where:
- $X \in \mathbb{R}^{B \times D_{\text{in}}}$ (Input batch of size $B$)
- $W \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}$ (Weight matrix)
- $b \in \mathbb{R}^{D_{\text{out}}}$ (Bias vector)
- $Y \in \mathbb{R}^{B \times D_{\text{out}}}$ (Output activations)

Let $\mathcal{L}$ be the scalar loss, and assume the upstream gradient $\frac{\partial \mathcal{L}}{\partial Y} \in \mathbb{R}^{B \times D_{\text{out}}}$ has already been backpropagated from downstream layers.

### Task
1. Using index notation, write entry $Y_{i, j}$ in terms of $X$, $W$, and $b$.
2. Derive the weight gradient matrix $\frac{\partial \mathcal{L}}{\partial W} \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}$. Show that:
   $$\frac{\partial \mathcal{L}}{\partial W} = X^T \left( \frac{\partial \mathcal{L}}{\partial Y} \right)$$
3. Derive the input gradient matrix $\frac{\partial \mathcal{L}}{\partial X} \in \mathbb{R}^{B \times D_{\text{in}}}$. Show that:
   $$\frac{\partial \mathcal{L}}{\partial X} = \left( \frac{\partial \mathcal{L}}{\partial Y} \right) W^T$$
4. Derive the bias gradient $\frac{\partial \mathcal{L}}{\partial b} \in \mathbb{R}^{D_{\text{out}}}$.

---

### Solution 3

#### 1. Index Notation
$$Y_{i, j} = \sum_{k=1}^{D_{\text{in}}} X_{i, k} W_{k, j} + b_j$$

#### 2. Derivation of $\frac{\partial \mathcal{L}}{\partial W_{k, j}}$
By the chain rule:
$$\frac{\partial \mathcal{L}}{\partial W_{k, j}} = \sum_{i=1}^B \frac{\partial \mathcal{L}}{\partial Y_{i, j}} \frac{\partial Y_{i, j}}{\partial W_{k, j}}$$
From the formula for $Y_{i, j}$, $\frac{\partial Y_{i, j}}{\partial W_{k, j}} = X_{i, k}$. Therefore:
$$\frac{\partial \mathcal{L}}{\partial W_{k, j}} = \sum_{i=1}^B X_{i, k} \left( \frac{\partial \mathcal{L}}{\partial Y} \right)_{i, j} = \sum_{i=1}^B (X^T)_{k, i} \left( \frac{\partial \mathcal{L}}{\partial Y} \right)_{i, j}$$
In matrix form:
$$\mathbf{\frac{\partial \mathcal{L}}{\partial W} = X^T \left( \frac{\partial \mathcal{L}}{\partial Y} \right)} \quad \blacksquare$$
*(Dimension check: $(D_{\text{in}} \times B) \times (B \times D_{\text{out}}) \longrightarrow (D_{\text{in}} \times D_{\text{out}})$. Perfect!)*

#### 3. Derivation of $\frac{\partial \mathcal{L}}{\partial X_{i, k}}$
$$\frac{\partial \mathcal{L}}{\partial X_{i, k}} = \sum_{j=1}^{D_{\text{out}}} \frac{\partial \mathcal{L}}{\partial Y_{i, j}} \frac{\partial Y_{i, j}}{\partial X_{i, k}} = \sum_{j=1}^{D_{\text{out}}} \left( \frac{\partial \mathcal{L}}{\partial Y} \right)_{i, j} W_{k, j} = \sum_{j=1}^{D_{\text{out}}} \left( \frac{\partial \mathcal{L}}{\partial Y} \right)_{i, j} (W^T)_{j, k}$$
In matrix form:
$$\mathbf{\frac{\partial \mathcal{L}}{\partial X} = \left( \frac{\partial \mathcal{L}}{\partial Y} \right) W^T} \quad \blacksquare$$
*(Dimension check: $(B \times D_{\text{out}}) \times (D_{\text{out}} \times D_{\text{in}}) \longrightarrow (B \times D_{\text{in}})$. Perfect!)*

#### 4. Derivation of Bias Gradient $\frac{\partial \mathcal{L}}{\partial b_j}$
Since bias $b_j$ is added to every sample $i \in \{1, \dots, B\}$:
$$\frac{\partial \mathcal{L}}{\partial b_j} = \sum_{i=1}^B \frac{\partial \mathcal{L}}{\partial Y_{i, j}} \frac{\partial Y_{i, j}}{\partial b_j} = \sum_{i=1}^B \left( \frac{\partial \mathcal{L}}{\partial Y} \right)_{i, j} \cdot 1$$
In vectorized code:
$$\mathbf{\frac{\partial \mathcal{L}}{\partial b} = \sum_{i=1}^B \left( \frac{\partial \mathcal{L}}{\partial Y} \right)_{i, :}}$$
*Result: The bias gradient is simply the sum of incoming gradients across the batch dimension!*
