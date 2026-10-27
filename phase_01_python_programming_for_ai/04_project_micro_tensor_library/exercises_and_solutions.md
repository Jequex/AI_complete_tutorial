# Topic 1.4: Exercises & Solutions — Capstone Micro-Tensor Library

Welcome to the problem set for **Topic 1.4: Capstone Project — Building a Micro-Tensor Library**!

Test your understanding of stride mechanics, tensor broadcasting, and neural network forward passes. Work through each problem carefully before opening the collapsible solution.

---

## 📝 Problem Set

### Problem 1: Stride Calculus & Contiguous Memory Views

#### The Context
In deep learning frameworks, calling `.view()` or `.reshape()` avoids copying memory if the tensor is C-contiguous. If the tensor has been transposed (`.T`), its strides are non-contiguous, requiring `.contiguous()` before reshaping.

#### Tasks
1. State the general recurrence relation for row-major C-contiguous strides $(S_0, S_1, \dots, S_{k-1})$ given dimensions $(D_0, D_1, \dots, D_{k-1})$.
2. Calculate the exact strides for a 3D tensor of shape $(B=2, T=3, D=4)$.
3. If matrix $A$ of shape $(2, 3)$ has strides $(3, 1)$ and is transposed to $A^T$ of shape $(3, 2)$ with strides $(1, 3)$, prove why $A^T$ is not C-contiguous.

<details>
<summary>Click to view solution for Problem 1</summary>

### Solution 1: Stride Calculus & Contiguous Memory Views

#### Part 1: Stride Recurrence Relation
For a $k$-dimensional tensor with shape $(D_0, D_1, \dots, D_{k-1})$:
$$S_{k-1} = 1$$
$$S_i = S_{i+1} \times D_{i+1} \quad \text{for } i = k-2, k-3, \dots, 0$$
$$\underbrace{S_i}_{\text{Stride for dimension } i} = \prod_{j=i+1}^{k-1} D_j$$

#### Part 2: Calculation for Shape `(2, 3, 4)`
- $S_2 = 1$
- $S_1 = S_2 \times D_2 = 1 \times 4 = \mathbf{4}$
- $S_0 = S_1 \times D_1 = 4 \times 3 = \mathbf{12}$
$$\text{Strides} = (12, 4, 1)$$

#### Part 3: Contiguity Proof of $A^T$
- A 2D matrix of shape $(3, 2)$ is C-contiguous if and only if its elements along rows are adjacent in memory:
  $$S_1^{\text{contiguous}} = 1, \qquad S_0^{\text{contiguous}} = D_1 = 2 \implies (2, 1)$$
- But $A^T$ has strides $(1, 3)$: moving right by 1 column steps by 3 elements in memory! Because adjacent column entries are separated by 3 units in RAM, $A^T$ is non-contiguous.
</details>

---

### Problem 2: Broadcasting Mechanics for Bias Addition $(M, N) + (N,)$

In deep learning, neural network layers add a 1D bias vector $\mathbf{b} \in \mathbb{R}^N$ to a 2D batch output $Z \in \mathbb{R}^{M \times N}$:
$$Y_{i, j} = Z_{i, j} + b_j$$

1. If $Z = \begin{bmatrix} 1.0 & 2.0 \\ 3.0 & 4.0 \end{bmatrix}$ and $\mathbf{b} = \begin{bmatrix} 10.0 & 20.0 \end{bmatrix}$, compute $Y = Z + \mathbf{b}$ step-by-step.
2. How does stride manipulation allow broadcasting $\mathbf{b}$ across $M$ rows with **zero memory copies**?

<details>
<summary>Click to view solution for Problem 2</summary>

### Solution 2: Broadcasting Mechanics

#### Step 1: Step-by-Step Calculation
Add $b_j$ across each row $i$:
- Row 0: $[1.0 + 10.0, \quad 2.0 + 20.0] = [\mathbf{11.0}, \mathbf{22.0}]$
- Row 1: $[3.0 + 10.0, \quad 4.0 + 20.0] = [\mathbf{13.0}, \mathbf{24.0}]$
$$Y = \begin{bmatrix} 11.0 & 22.0 \\ 13.0 & 24.0 \end{bmatrix}$$

#### Step 2: Zero-Copy Stride Broadcasting
To broadcast $\mathbf{b}$ from shape $(N,)$ to $(M, N)$ without duplicating memory:
- Set Shape to: $(M, N)$
- Set Strides to: $(0, 1)$!
- Notice that stride $S_0 = 0$: when row index $r$ increases, the flat memory pointer advances by $r \times 0 = 0$! Every row re-reads the exact same $N$ numbers in memory!
</details>

---

### Problem 3: End-to-End Hand Calculation of a 2-Layer Neural Network

Consider a 2-layer MLP:
$$z_1 = x @ W_1 + b_1, \qquad a_1 = \text{ReLU}(z_1), \qquad \hat{y} = a_1 @ W_2 + b_2$$
Given:
$$x = \begin{bmatrix} 2.0 & -1.0 \end{bmatrix}, \qquad W_1 = \begin{bmatrix} 1.0 & 2.0 \\ -3.0 & 0.5 \end{bmatrix}, \qquad b_1 = \begin{bmatrix} 0.5 & -1.0 \end{bmatrix}$$
$$W_2 = \begin{bmatrix} 2.0 \\ -1.0 \end{bmatrix}, \qquad b_2 = \begin{bmatrix} 1.5 \end{bmatrix}$$

1. Compute $z_1 = x @ W_1 + b_1$.
2. Compute $a_1 = \text{ReLU}(z_1)$.
3. Compute final prediction $\hat{y}$.

<details>
<summary>Click to view solution for Problem 3</summary>

### Solution 3: End-to-End Neural Network Hand Calculation

#### Step 1: Layer 1 Linear Transformation
- Dot product with Col 0 of $W_1$: $(2.0 \times 1.0) + (-1.0 \times -3.0) = 2.0 + 3.0 = 5.0$.
  Add bias $b_{1, 0}$: $5.0 + 0.5 = \mathbf{5.5}$.
- Dot product with Col 1 of $W_1$: $(2.0 \times 2.0) + (-1.0 \times 0.5) = 4.0 - 0.5 = 3.5$.
  Add bias $b_{1, 1}$: $3.5 + (-1.0) = \mathbf{2.5}$.
$$z_1 = \begin{bmatrix} 5.5 & 2.5 \end{bmatrix}$$

#### Step 2: ReLU Non-linearity
$$a_1 = \text{ReLU}(z_1) = [\max(0, 5.5), \max(0, 2.5)] = \begin{bmatrix} 5.5 & 2.5 \end{bmatrix}$$

#### Step 3: Layer 2 Output
- Matrix product $a_1 @ W_2$:
  $$(5.5 \times 2.0) + (2.5 \times -1.0) = 11.0 - 2.5 = 8.5$$
- Add output bias $b_2$:
  $$\hat{y} = 8.5 + 1.5 = \mathbf{10.0}$$
</details>
