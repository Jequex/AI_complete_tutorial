# Topic 3.4: Jacobians & Vector Calculus in Deep Learning

---

## 1. Executive Summary & Intuitive Foundations

### What is a Vector-Valued Function?
In single-variable calculus, we dealt with functions that take one number and spit out one number:
$$f: \mathbb{R} \to \mathbb{R}$$

In Topic 3.2, we met multivariable scalar functions (like loss functions) that take a whole vector of weights and spit out a single scalar score:
$$f: \mathbb{R}^n \to \mathbb{R}$$

Now we reach the ultimate generalization used in neural network layers: **vector-valued functions of multiple variables**. These functions take a vector of inputs and output an entirely new vector of outputs:
$$\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$$

$$\mathbf{f}(\mathbf{x}) = \begin{bmatrix} f_1(x_1, x_2, \dots, x_n) \\ f_2(x_1, x_2, \dots, x_n) \\ \vdots \\ f_m(x_1, x_2, \dots, x_n) \end{bmatrix}$$

Every hidden layer in a neural network, every normalization layer (LayerNorm, BatchNorm), and the final classification layer (**Softmax**) is a vector-valued function!

---

### The 12-Year-Old Analogy: The Music Producer's Sound Mixing Console
Imagine you are at an audio mixing console with **$n$ sliders** (the inputs $\mathbf{x}$, like bass, treble, echo, volume) and **$m$ output speakers** (the outputs $\mathbf{f}(\mathbf{x})$, like front-left, front-right, subwoofer).

If you nudge **slider 3** slightly upward:
- Speaker 1 gets a little louder.
- Speaker 2 stays the same.
- Subwoofer 3 gets much bassier.

How do you record how every single slider affects every single speaker? You need a **grid (matrix) of sensitivity numbers**!
- Row $i$ corresponds to **Speaker $i$**.
- Column $j$ corresponds to **Slider $j$**.
- Entry $(i, j)$ tells you: *"If I nudge Slider $j$ by 1 millimeter, how many decibels does Speaker $i$ change?"*

That grid is the **Jacobian Matrix**!

---

## 2. Mathematical Definition of the Jacobian Matrix

Let $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$ be a differentiable vector-valued function. The **Jacobian matrix** $J \in \mathbb{R}^{m \times n}$ (often denoted $J_{\mathbf{f}}(\mathbf{x})$ or $\frac{\partial \mathbf{f}}{\partial \mathbf{x}}$) collects all $m \times n$ first-order partial derivatives:

$$J = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \begin{bmatrix} 
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}$$

### Structural Rules of the Jacobian:
1. **Number of Rows = Dimension of Output ($m$)**: Each row is the transpose of the gradient of one output component:
   $$\text{Row } i = \nabla f_i(\mathbf{x})^T = \left[ \frac{\partial f_i}{\partial x_1}, \dots, \frac{\partial f_i}{\partial x_n} \right]$$
2. **Number of Columns = Dimension of Input ($n$)**: Each column represents the rate of change of the entire output vector with respect to a single input variable $x_j$:
   $$\text{Col } j = \frac{\partial \mathbf{f}}{\partial x_j} = \begin{bmatrix} \frac{\partial f_1}{\partial x_j} \\ \vdots \\ \frac{\partial f_m}{\partial x_j} \end{bmatrix}$$
3. **Element-wise Formula**:
   $$J_{ij} = \frac{\partial f_i}{\partial x_j}$$

---

## 3. Mathematical Decoder Table

| Symbol | Mathematical Name | Plain-English Meaning | Dimension |
| :--- | :--- | :--- | :--- |
| $\mathbf{x}$ | Input vector | Coordinates entering the transformation | $n \times 1$ |
| $\mathbf{f}(\mathbf{x})$ | Output vector | Transformed coordinates | $m \times 1$ |
| $J \text{ or } \frac{\partial \mathbf{f}}{\partial \mathbf{x}}$ | Jacobian Matrix | Complete matrix of all first-order partial derivatives | $m \times n$ |
| $J_{ij}$ | Entry $(i, j)$ of $J$ | Rate of change of $i$-th output with respect to $j$-th input ($\frac{\partial f_i}{\partial x_j}$) | Scalar |
| $\delta_{ij}$ | Kronecker Delta | Equals $1$ if $i = j$, and $0$ if $i \neq j$ | Scalar |
| $\mathbf{z}$ | Logits vector | Raw, unnormalized class scores entering Softmax | $K \times 1$ |
| $\mathbf{p}$ | Probability vector | Softmax output vector ($\sum p_i = 1$) | $K \times 1$ |
| $\det(J)$ | Jacobian Determinant | Local volume expansion/compression factor of space | Scalar |
| $\mathbf{v}^T J$ | Vector-Jacobian Product (VJP) | Reverse-mode autodiff primitive (adjoint vector backpropagation) | $1 \times n$ |

---

## 4. First-Order Linear Approximation with the Jacobian

In multivariable calculus, just as the tangent line approximates a 1D curve and the tangent plane approximates a 2D surface, the **Jacobian matrix provides the best local linear transformation** for any vector-valued function:

$$\mathbf{f}(\mathbf{x} + \Delta\mathbf{x}) \approx \mathbf{f}(\mathbf{x}) + J \Delta\mathbf{x}$$

Where:
- $\mathbf{f}(\mathbf{x}) \in \mathbb{R}^m$ is the base output.
- $\Delta\mathbf{x} \in \mathbb{R}^n$ is a tiny perturbation input vector.
- $J \Delta\mathbf{x} \in \mathbb{R}^m$ is standard matrix-vector multiplication producing the resulting change $\Delta\mathbf{f} \in \mathbb{R}^m$.

---

## 5. The Grand Derivation: The Softmax Jacobian

In classification models (LLMs predicting the next token, vision models classifying images), the final layer is the **Softmax function**. It converts a vector of raw unbounded real numbers (logits $\mathbf{z} \in \mathbb{R}^K$) into a probability distribution $\mathbf{p} \in \mathbb{R}^K$:

$$p_i = \operatorname{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{k=1}^K e^{z_k}}$$

Notice that every single output probability $p_i$ depends on **all** input logits $z_1, z_2, \dots, z_K$ through the denominator!
Therefore, its derivative is a $K \times K$ Jacobian matrix $J \in \mathbb{R}^{K \times K}$ where:
$$J_{ij} = \frac{\partial p_i}{\partial z_j}$$

Let us derive this step-by-step using the quotient rule:
$$\left( \frac{u}{v} \right)' = \frac{u' v - u v'}{v^2}$$

Let:
- Numerator $u = e^{z_i}$
- Denominator $v = \sum_{k=1}^K e^{z_k}$
- $p_i = \frac{u}{v}$

Now we must analyze two distinct cases:
1. **Case 1: Diagonal entries ($i = j$)** — We differentiate $p_i$ with respect to its own logit $z_i$.
2. **Case 2: Off-diagonal entries ($i \neq j$)** — We differentiate $p_i$ with respect to a competing logit $z_j$.

---

### Step 1: Differentiating the Denominator $v$
Notice that for any $j \in \{1, \dots, K\}$:
$$\frac{\partial v}{\partial z_j} = \frac{\partial}{\partial z_j} \left( \sum_{k=1}^K e^{z_k} \right) = \frac{\partial}{\partial z_j} (e^{z_1} + \dots + e^{z_j} + \dots + e^{z_K})$$
Since only the $j$-th term in the sum contains $z_j$, all other terms have derivative zero:
$$\frac{\partial v}{\partial z_j} = e^{z_j}$$

---

### Step 2: Deriving Case 1 ($i = j$)
When $j = i$:
- $u = e^{z_i} \implies \frac{\partial u}{\partial z_i} = e^{z_i}$
- $v = \sum_k e^{z_k} \implies \frac{\partial v}{\partial z_i} = e^{z_i}$

Applying the quotient rule:
$$\frac{\partial p_i}{\partial z_i} = \frac{\left( \frac{\partial u}{\partial z_i} \right) v - u \left( \frac{\partial v}{\partial z_i} \right)}{v^2}$$

$$\frac{\partial p_i}{\partial z_i} = \frac{e^{z_i} \left( \sum_k e^{z_k} \right) - e^{z_i} e^{z_i}}{\left( \sum_k e^{z_k} \right)^2}$$

Split the fraction into two terms:
$$\frac{\partial p_i}{\partial z_i} = \frac{e^{z_i} \left( \sum_k e^{z_k} \right)}{\left( \sum_k e^{z_k} \right)^2} - \frac{e^{z_i} e^{z_i}}{\left( \sum_k e^{z_k} \right)^2}$$

Simplify each term:
$$\frac{\partial p_i}{\partial z_i} = \frac{e^{z_i}}{\sum_k e^{z_k}} - \left( \frac{e^{z_i}}{\sum_k e^{z_k}} \right) \left( \frac{e^{z_i}}{\sum_k e^{z_k}} \right)$$

Recognizing that $p_i = \frac{e^{z_i}}{\sum_k e^{z_k}}$:
$$\mathbf{\frac{\partial p_i}{\partial z_i} = p_i - p_i^2 = p_i(1 - p_i)}$$

*Interpretation:* If logit $z_i$ increases, its own probability $p_i$ increases (since $0 < p_i < 1$, $p_i(1 - p_i) > 0$).

---

### Step 3: Deriving Case 2 ($i \neq j$)
When $j \neq i$:
- $u = e^{z_i}$. Does $u$ contain $z_j$? **No!** Therefore: $\frac{\partial u}{\partial z_j} = 0$.
- $v = \sum_k e^{z_k} \implies \frac{\partial v}{\partial z_j} = e^{z_j}$.

Applying the quotient rule:
$$\frac{\partial p_i}{\partial z_j} = \frac{\left( \frac{\partial u}{\partial z_j} \right) v - u \left( \frac{\partial v}{\partial z_j} \right)}{v^2}$$

$$\frac{\partial p_i}{\partial z_j} = \frac{0 \cdot v - e^{z_i} e^{z_j}}{v^2} = -\frac{e^{z_i} e^{z_j}}{\left( \sum_k e^{z_k} \right)^2}$$

Factor into the product of probabilities:
$$\frac{\partial p_i}{\partial z_j} = -\left( \frac{e^{z_i}}{\sum_k e^{z_k}} \right) \left( \frac{e^{z_j}}{\sum_k e^{z_k}} \right) = \mathbf{-p_i p_j}$$

*Interpretation:* If a competing logit $z_j$ increases, the probability $p_i$ of category $i$ **decreases** (negative derivative!).

---

### Step 4: Unifying with the Kronecker Delta
Using the Kronecker delta $\delta_{ij}$:
$$\delta_{ij} = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases}$$

We can write the complete Softmax Jacobian in a single master equation:
$$\mathbf{J_{ij} = \frac{\partial p_i}{\partial z_j} = p_i (\delta_{ij} - p_j)}$$

In matrix notation:
$$\mathbf{J_{\text{softmax}} = \operatorname{diag}(\mathbf{p}) - \mathbf{p} \mathbf{p}^T}$$

Where $\operatorname{diag}(\mathbf{p})$ is a diagonal matrix containing the probabilities, and $\mathbf{p}\mathbf{p}^T$ is the outer product of $\mathbf{p}$ with itself!

---

## 6. The Beautiful Gradient Collapse: Softmax + Cross-Entropy

In neural networks, Softmax is almost always paired with the **Categorical Cross-Entropy Loss**:
$$\mathcal{L} = -\sum_{k=1}^K y_k \ln(p_k)$$
where $\mathbf{y}$ is the one-hot target vector ($y_c = 1$ for the correct class $c$, and $y_k = 0$ for all $k \neq c$).

What is the gradient of the loss with respect to the input logits $\mathbf{z}$?
By the multivariable chain rule:
$$\frac{\partial \mathcal{L}}{\partial z_j} = \sum_{i=1}^K \frac{\partial \mathcal{L}}{\partial p_i} \frac{\partial p_i}{\partial z_j}$$

1. First, compute $\frac{\partial \mathcal{L}}{\partial p_i}$:
   $$\frac{\partial \mathcal{L}}{\partial p_i} = \frac{\partial}{\partial p_i} \left( -\sum_{k=1}^K y_k \ln(p_k) \right) = -\frac{y_i}{p_i}$$

2. Substitute this and the Softmax Jacobian $J_{ij} = p_i(\delta_{ij} - p_j)$ into the chain rule:
   $$\frac{\partial \mathcal{L}}{\partial z_j} = \sum_{i=1}^K \left( -\frac{y_i}{p_i} \right) \left[ p_i (\delta_{ij} - p_j) \right]$$

3. Notice how $p_i$ in the numerator and denominator **cancel out completely**:
   $$\frac{\partial \mathcal{L}}{\partial z_j} = -\sum_{i=1}^K y_i (\delta_{ij} - p_j) = -\sum_{i=1}^K (y_i \delta_{ij} - y_i p_j)$$

4. Split the sum:
   $$\frac{\partial \mathcal{L}}{\partial z_j} = -\sum_{i=1}^K y_i \delta_{ij} + p_j \sum_{i=1}^K y_i$$
   - Since $\delta_{ij} = 0$ unless $i = j$, $\sum_{i=1}^K y_i \delta_{ij} = y_j$.
   - Since $\mathbf{y}$ is a one-hot distribution, $\sum_{i=1}^K y_i = 1$.

5. The equation collapses into pure elegance:
   $$\frac{\partial \mathcal{L}}{\partial z_j} = -y_j + p_j (1) = \mathbf{p_j - y_j}$$

In vector notation:
$$\mathbf{\nabla_{\mathbf{z}} \mathcal{L} = \mathbf{p} - \mathbf{y}}$$

> [!IMPORTANT]
> **The Miracle of Softmax + Cross-Entropy:**
> Even though both the Softmax Jacobian and the Cross-Entropy derivative look complicated with exponentials, fractions, and outer products, when you chain them together, all non-linear complexities vanish. The gradient is simply:
> $$\text{Error Vector} = \text{Predicted Probabilities} - \text{True Labels}$$
> This is why AI engineers compute this combined gradient directly rather than materializing the full $K \times K$ Jacobian matrix!

---

## 7. Vector-Jacobian Products (VJP) and Autodiff Efficiency

Why do PyTorch and JAX almost never compute or store the explicit Jacobian matrix $J$?
Because in deep learning, an intermediate layer might have $100{,}000$ dimensions (e.g. an LLM vocabulary size).
A full Jacobian matrix for a vocabulary of size $V = 100{,}000$ would be:
$$100{,}000 \times 100{,}000 = 10^{10} \text{ floats} \approx \mathbf{40 \text{ Gigabytes of VRAM!}}$$

Just to store the derivatives for a single token!

Instead, reverse-mode automatic differentiation only ever computes the **Vector-Jacobian Product (VJP)**:
$$\mathbf{v}^T J$$
where $\mathbf{v} = \frac{\partial \mathcal{L}}{\partial \mathbf{f}} \in \mathbb{R}^{1 \times m}$ is the incoming upstream gradient row vector!

In the Softmax case, instead of forming the $K \times K$ matrix $J = \operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^T$:
$$\mathbf{v}^T J = \mathbf{v}^T \left( \operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^T \right) = (\mathbf{v} \odot \mathbf{p}) - (\mathbf{v} \cdot \mathbf{p}) \mathbf{p}$$
This requires only **$O(K)$ time and $O(K)$ memory**, bypassing the $O(K^2)$ matrix completely!

---

## 8. Summary of Essential Takeaways

1. **The Jacobian $J$** is the matrix of all first-order partial derivatives of a vector-valued function $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$, with shape $m \times n$.
2. **First-order local linearity:** $\mathbf{f}(\mathbf{x} + \Delta\mathbf{x}) \approx \mathbf{f}(\mathbf{x}) + J \Delta\mathbf{x}$.
3. **Softmax Jacobian:** $J_{ij} = p_i(\delta_{ij} - p_j)$, or in matrix terms $J = \operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^T$.
4. **Softmax + Cross-Entropy:** Collapses cleanly to $\nabla_{\mathbf{z}} \mathcal{L} = \mathbf{p} - \mathbf{y}$.
5. **Modern Autodiff:** Computes Vector-Jacobian Products (VJPs) directly in $O(n)$ space without ever materializing giant $O(n \times m)$ Jacobian matrices.
