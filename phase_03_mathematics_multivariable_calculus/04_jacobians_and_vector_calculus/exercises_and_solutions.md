# Topic 3.4: Exercises & Solutions — Jacobians & Vector Calculus

---

## Problem Set Overview

| Problem | Focus | Difficulty |
| :--- | :--- | :--- |
| **1** | **Softmax Jacobian Calculation & The Zero Column-Sum Invariant** | Medium |
| **2** | **Softmax + Cross-Entropy Step-by-Step Gradient Collapse** | Medium |
| **3** | **2D Coordinate Transformation Jacobian & Determinant in Generative Flows** | Challenging |

---

## Problem 1: Softmax Jacobian Calculation & The Zero Column-Sum Invariant

### The Context
Suppose a 3-class classification model outputs predicted probabilities:
$$\mathbf{p} = \begin{bmatrix} p_1 \\ p_2 \\ p_3 \end{bmatrix} = \begin{bmatrix} 0.50 \\ 0.30 \\ 0.20 \end{bmatrix}$$

Recall the Softmax Jacobian formula:
$$J_{ij} = \frac{\partial p_i}{\partial z_j} = p_i (\delta_{ij} - p_j) = \begin{cases} p_i(1 - p_i) & \text{if } i = j \\ -p_i p_j & \text{if } i \neq j \end{cases}$$

### Task
1. Calculate all 9 numerical entries of the $3 \times 3$ Jacobian matrix $J$.
2. Compute the sum of the entries in **each column** of $J$:
   $$\sum_{i=1}^3 J_{ij} \quad \text{for } j \in \{1, 2, 3\}$$
3. Provide the deep mathematical reason why every column of the Softmax Jacobian must sum to exactly **zero**. *(Hint: What is the constant sum $\sum_{i=1}^K p_i$?)*

---

### Solution 1

#### 1. Calculating Entries of the $3 \times 3$ Jacobian
**Diagonal Elements ($i = j$):**
- $J_{11} = p_1(1 - p_1) = 0.50 \times (1 - 0.50) = 0.50 \times 0.50 = \mathbf{0.25}$
- $J_{22} = p_2(1 - p_2) = 0.30 \times (1 - 0.30) = 0.30 \times 0.70 = \mathbf{0.21}$
- $J_{33} = p_3(1 - p_3) = 0.20 \times (1 - 0.20) = 0.20 \times 0.80 = \mathbf{0.16}$

**Off-Diagonal Elements ($i \neq j$):**
- $J_{12} = -p_1 p_2 = -(0.50)(0.30) = \mathbf{-0.15}$
- $J_{13} = -p_1 p_3 = -(0.50)(0.20) = \mathbf{-0.10}$
- $J_{21} = -p_2 p_1 = -(0.30)(0.50) = \mathbf{-0.15}$
- $J_{23} = -p_2 p_3 = -(0.30)(0.20) = \mathbf{-0.06}$
- $J_{31} = -p_3 p_1 = -(0.20)(0.50) = \mathbf{-0.10}$
- $J_{32} = -p_3 p_2 = -(0.20)(0.30) = \mathbf{-0.06}$

Writing the full Jacobian matrix:
$$J = \begin{bmatrix}
+0.25 & -0.15 & -0.10 \\
-0.15 & +0.21 & -0.06 \\
-0.10 & -0.06 & +0.16
\end{bmatrix}$$

#### 2. Summing Columns
- **Column 1:** $+0.25 + (-0.15) + (-0.10) = \mathbf{0.00}$
- **Column 2:** $-0.15 + 0.21 + (-0.06) = \mathbf{0.00}$
- **Column 3:** $-0.10 + (-0.06) + 0.16 = \mathbf{0.00}$

#### 3. Mathematical Reason
By definition of a probability distribution, the sum of all probabilities is identically 1 for all possible input logits $\mathbf{z}$:
$$\sum_{i=1}^K p_i(\mathbf{z}) \equiv 1.0$$
Differentiating both sides with respect to any input logit $z_j$:
$$\frac{\partial}{\partial z_j} \left( \sum_{i=1}^K p_i \right) = \frac{\partial}{\partial z_j}(1.0)$$
By linearity of differentiation:
$$\sum_{i=1}^K \frac{\partial p_i}{\partial z_j} = 0 \implies \mathbf{\sum_{i=1}^K J_{ij} = 0} \quad \blacksquare$$
*Intuition:* If increasing logit $z_j$ causes some probabilities to increase, other probabilities must decrease by the exact same total amount so the sum stays locked at 1.0!

---

## Problem 2: Softmax + Cross-Entropy Step-by-Step Gradient Collapse

### The Context
A vision model receives an image of a cat. The true target class is **Cat** (class index $k=2$, 1-indexed):
$$\mathbf{y} = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}$$
The model outputs logits $\mathbf{z} = [z_1, z_2, z_3]^T$ resulting in Softmax probabilities:
$$\mathbf{p} = \begin{bmatrix} 0.10 \\ 0.70 \\ 0.20 \end{bmatrix}$$
The categorical cross-entropy loss is:
$$\mathcal{L} = -\sum_{k=1}^3 y_k \ln(p_k) = -\ln(p_2)$$

### Task
1. Compute the loss value $\mathcal{L}$ for this sample (using natural log, $\ln(0.70) \approx -0.3567$).
2. Compute the upstream gradient vector $\frac{\partial \mathcal{L}}{\partial \mathbf{p}}$.
3. Using the full chain rule with the Softmax Jacobian $J$:
   $$\frac{\partial \mathcal{L}}{\partial z_j} = \sum_{i=1}^3 \frac{\partial \mathcal{L}}{\partial p_i} J_{ij}$$
   Calculate $\frac{\partial \mathcal{L}}{\partial z_1}$, $\frac{\partial \mathcal{L}}{\partial z_2}$, and $\frac{\partial \mathcal{L}}{\partial z_3}$.
4. Verify that the result matches the short-cut collapsed formula:
   $$\nabla_{\mathbf{z}} \mathcal{L} = \mathbf{p} - \mathbf{y}$$

---

### Solution 2

#### 1. Loss Value
$$\mathcal{L} = -\ln(0.70) \approx \mathbf{0.3567}$$

#### 2. Upstream Gradient Vector $\frac{\partial \mathcal{L}}{\partial \mathbf{p}}$
For $k = 1$: $\frac{\partial \mathcal{L}}{\partial p_1} = -\frac{y_1}{p_1} = -\frac{0}{0.10} = \mathbf{0.0}$
For $k = 2$: $\frac{\partial \mathcal{L}}{\partial p_2} = -\frac{y_2}{p_2} = -\frac{1}{0.70} \approx \mathbf{-1.42857}$
For $k = 3$: $\frac{\partial \mathcal{L}}{\partial p_3} = -\frac{y_3}{p_3} = -\frac{0}{0.20} = \mathbf{0.0}$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{p}} = \begin{bmatrix} 0 \\ -1.42857 \\ 0 \end{bmatrix}$$

#### 3. Chain Rule with Softmax Jacobian
Because $\frac{\partial \mathcal{L}}{\partial p_1} = 0$ and $\frac{\partial \mathcal{L}}{\partial p_3} = 0$, only the second term $i=2$ survives:
$$\frac{\partial \mathcal{L}}{\partial z_j} = \frac{\partial \mathcal{L}}{\partial p_2} J_{2j} = \left(-\frac{1}{p_2}\right) J_{2j}$$

- For $j = 1$:
  $$\frac{\partial \mathcal{L}}{\partial z_1} = \left(-\frac{1}{p_2}\right) (-p_2 p_1) = +p_1 = \mathbf{+0.10}$$
- For $j = 2$:
  $$\frac{\partial \mathcal{L}}{\partial z_2} = \left(-\frac{1}{p_2}\right) \left[ p_2(1 - p_2) \right] = -(1 - p_2) = p_2 - 1 = 0.70 - 1 = \mathbf{-0.30}$$
- For $j = 3$:
  $$\frac{\partial \mathcal{L}}{\partial z_3} = \left(-\frac{1}{p_2}\right) (-p_2 p_3) = +p_3 = \mathbf{+0.20}$$

#### 4. Verification with Collapsed Formula
$$\nabla_{\mathbf{z}} \mathcal{L} = \mathbf{p} - \mathbf{y} = \begin{bmatrix} 0.10 \\ 0.70 \\ 0.20 \end{bmatrix} - \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix} = \begin{bmatrix} +0.10 \\ -0.30 \\ +0.20 \end{bmatrix}$$
The results are identical!
- The true class $z_2$ has a **negative gradient** ($-0.30$), meaning increasing $z_2$ decreases the loss.
- The wrong classes $z_1, z_3$ have **positive gradients** ($+0.10, +0.20$), meaning increasing them increases the loss.

---

## Problem 3: 2D Coordinate Transformation Jacobian & Determinant in Generative Flows

### The Context
In generative modeling (specifically **Normalizing Flows** such as RealNVP, Glow), deep networks learn invertible transformations $\mathbf{x} = \mathbf{g}(\mathbf{z})$.
To evaluate exact likelihoods, the probability density formula requires the absolute value of the **Jacobian determinant**:
$$p_X(\mathbf{x}) = p_Z(\mathbf{z}) \left| \det\left( \frac{\partial \mathbf{g}}{\partial \mathbf{z}} \right) \right|^{-1}$$

Consider the 2D transformation from polar coordinates $(r, \theta)$ to Cartesian coordinates $(x, y)$:
$$\mathbf{g}(r, \theta) = \begin{bmatrix} x(r, \theta) \\ y(r, \theta) \end{bmatrix} = \begin{bmatrix} r \cos \theta \\ r \sin \theta \end{bmatrix}$$

### Task
1. Compute the $2 \times 2$ Jacobian matrix $J = \frac{\partial(x, y)}{\partial(r, \theta)}$.
2. Compute the determinant $\det(J)$.
3. Simplify the expression using the trigonometric identity $\cos^2\theta + \sin^2\theta = 1$.
4. Explain what $\det(J) = r$ means geometrically when transforming an infinitesimal area element $dA = dr \, d\theta$.

---

### Solution 3

#### 1. Calculating the $2 \times 2$ Jacobian Matrix
$$J = \begin{bmatrix} 
\frac{\partial x}{\partial r} & \frac{\partial x}{\partial \theta} \\
\frac{\partial y}{\partial r} & \frac{\partial y}{\partial \theta} 
\end{bmatrix}$$

- $\frac{\partial x}{\partial r} = \frac{\partial}{\partial r}(r \cos\theta) = \cos\theta$
- $\frac{\partial x}{\partial \theta} = \frac{\partial}{\partial \theta}(r \cos\theta) = -r \sin\theta$
- $\frac{\partial y}{\partial r} = \frac{\partial}{\partial r}(r \sin\theta) = \sin\theta$
- $\frac{\partial y}{\partial \theta} = \frac{\partial}{\partial \theta}(r \sin\theta) = r \cos\theta$

Thus:
$$J = \begin{bmatrix} 
\cos\theta & -r \sin\theta \\ 
\sin\theta & r \cos\theta 
\end{bmatrix}$$

#### 2. Computing the Determinant
For a $2 \times 2$ matrix $\begin{bmatrix} a & b \\ c & d \end{bmatrix}$, $\det = ad - bc$:
$$\det(J) = (\cos\theta)(r \cos\theta) - (-r \sin\theta)(\sin\theta)$$
$$\det(J) = r \cos^2\theta + r \sin^2\theta$$

#### 3. Trigonometric Simplification
$$\det(J) = r (\cos^2\theta + \sin^2\theta) = r (1) = \mathbf{r} \quad \blacksquare$$

#### 4. Geometric Interpretation in Generative AI
- The Jacobian determinant measures how much local volume (or area in 2D) is stretched or compressed by the mapping.
- An infinitesimal patch in polar space $dr \, d\theta$ maps to an area $dx \, dy = |\det(J)| \, dr \, d\theta = \mathbf{r \, dr \, d\theta}$.
- As $r$ grows further from the origin, a small angular change $d\theta$ sweeps out a proportionally larger physical arc length ($r \, d\theta$).
- In Normalizing Flows, when a neural network stretches latent space ($|\det(J)| > 1$), probability density $p_X(x)$ dilutes; when it compresses space ($|\det(J)| < 1$), probability density concentrates!
