# Topic 3.4: Exercises & Solutions — Jacobians & Vector Calculus

Welcome to the problem set for **Topic 3.4: Jacobians & Vector Calculus in Deep Learning**.

Solve each problem on paper before expanding the collapsible solution panels below.

---

## 📝 Problem Set Overview

| Problem | Focus | Difficulty | Practical AI Connection |
| :---: | :--- | :---: | :--- |
| **1** | **Softmax Jacobian Calculation & The Zero Column-Sum Invariant** | Medium | Verifying probability distribution conservation in multi-class heads |
| **2** | **Softmax + Cross-Entropy Step-by-Step Gradient Collapse** | Medium | The primary loss backward pass used in GPT and vision models |
| **3** | **2D Coordinate Transformation Jacobian & Determinant in Generative Flows** | Challenging | Probability density tracking in Normalizing Flows & Flow Matching |

---

## Problem 1: Softmax Jacobian Calculation & The Zero Column-Sum Invariant

### 🧒 The Context: Maya's 3-Class Animal Classifier
Maya builds a neural network that predicts whether an image is a Dog, Cat, or Bunny.
The model outputs predicted probabilities:
$$\mathbf{p} = \begin{bmatrix} p_1 \\ p_2 \\ p_3 \end{bmatrix} = \begin{bmatrix} 0.50 \\ 0.30 \\ 0.20 \end{bmatrix}$$

Recall the Softmax Jacobian formula:
$$J_{ij} = \frac{\partial p_i}{\partial z_j} = p_i (\delta_{ij} - p_j) = \begin{cases} p_i(1 - p_i) & \text{if } i = j \\ -p_i p_j & \text{if } i \neq j \end{cases}$$

### Tasks:
1. Calculate all 9 numerical entries of the $3 \times 3$ Jacobian matrix $J$.
2. Compute the sum of the entries in **each column** of $J$: $\sum_{i=1}^3 J_{ij}$ for $j \in \{1, 2, 3\}$.
3. Explain the deep mathematical reason why every column of the Softmax Jacobian must sum to exactly **zero**.

---

<details>
<summary>Click to view solution for Problem 1</summary>

### Detailed Step-by-Step Solution:

#### Step 1: Calculate Diagonal Entries ($i = j$)
- $J_{11} = p_1(1 - p_1) = 0.50 \times (1 - 0.50) = 0.50 \times 0.50 = \mathbf{0.25}$
- $J_{22} = p_2(1 - p_2) = 0.30 \times (1 - 0.30) = 0.30 \times 0.70 = \mathbf{0.21}$
- $J_{33} = p_3(1 - p_3) = 0.20 \times (1 - 0.20) = 0.20 \times 0.80 = \mathbf{0.16}$

#### Step 2: Calculate Off-Diagonal Entries ($i \neq j$)
Notice symmetry: $J_{ij} = -p_i p_j = J_{ji}$.
- $J_{12} = J_{21} = -(0.50)(0.30) = \mathbf{-0.15}$
- $J_{13} = J_{31} = -(0.50)(0.20) = \mathbf{-0.10}$
- $J_{23} = J_{32} = -(0.30)(0.20) = \mathbf{-0.06}$

Writing the full $3 \times 3$ Jacobian matrix:
$$J = \begin{bmatrix}
+0.25 & -0.15 & -0.10 \\
-0.15 & +0.21 & -0.06 \\
-0.10 & -0.06 & +0.16
\end{bmatrix}$$

#### Step 3: Sum the Columns
- **Column 1:** $+0.25 + (-0.15) + (-0.10) = \mathbf{0.00}$
- **Column 2:** $-0.15 + 0.21 + (-0.06) = \mathbf{0.00}$
- **Column 3:** $-0.10 + (-0.06) + 0.16 = \mathbf{0.00}$

#### Step 4: The Deep Mathematical Reason
By definition, Softmax probabilities must sum to $1.0$ at all times:
$$\sum_{i=1}^K p_i = 1$$
Differentiating both sides with respect to any input logit $z_j$:
$$\frac{\partial}{\partial z_j} \left( \sum_{i=1}^K p_i \right) = \frac{\partial}{\partial z_j}(1) = 0$$
Swapping the sum and derivative:
$$\sum_{i=1}^K \frac{\partial p_i}{\partial z_j} = \sum_{i=1}^K J_{ij} = \mathbf{0}$$

**🧒 Intuition:** Probability is a conserved pie. If Maya increases logit $z_1$, the probability of Dog expands ($+0.25$). To keep total probability at $100\%$, Cat and Bunny *must* shrink by the exact same total amount ($-0.15 - 0.10 = -0.25$)!
</details>

---

## Problem 2: Softmax + Cross-Entropy Step-by-Step Gradient Collapse

### 🧒 The Context: Maya's Cat Prediction Error
Continuing Maya's 3-class animal classifier where:
$$\mathbf{p} = \begin{bmatrix} 0.50 \\ 0.30 \\ 0.20 \end{bmatrix} \quad (\text{Dog: } 0.50, \text{ Cat: } 0.30, \text{ Bunny: } 0.20)$$

Suppose the true label is a **Cat** (class 2):
$$\mathbf{y} = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}$$

Cross-Entropy loss is:
$$\mathcal{L} = -\ln(p_2) = -\ln(0.30) \approx 1.204$$

### Tasks:
1. Compute the upstream gradient vector $\mathbf{v} = \frac{\partial \mathcal{L}}{\partial \mathbf{p}} \in \mathbb{R}^{1 \times 3}$.
2. Using the explicit Jacobian matrix $J$ from Problem 1, compute the Vector-Jacobian Product $\nabla_{\mathbf{z}} \mathcal{L} = \mathbf{v} J$.
3. Verify that the result matches the collapsed formula $\nabla_{\mathbf{z}} \mathcal{L} = \mathbf{p} - \mathbf{y}$ directly.

---

<details>
<summary>Click to view solution for Problem 2</summary>

### Detailed Step-by-Step Solution:

#### Step 1: Upstream Gradient Vector $\mathbf{v} = \frac{\partial \mathcal{L}}{\partial \mathbf{p}}$
Since $\mathcal{L} = -\sum_{k=1}^3 y_k \ln(p_k) = -\ln(p_2)$:
- $\frac{\partial \mathcal{L}}{\partial p_1} = 0$
- $\frac{\partial \mathcal{L}}{\partial p_2} = -\frac{1}{p_2} = -\frac{1}{0.30} \approx -3.3333$
- $\frac{\partial \mathcal{L}}{\partial p_3} = 0$
$$\mathbf{v} = \begin{bmatrix} 0 & -\frac{1}{0.30} & 0 \end{bmatrix}$$

#### Step 2: Compute $\mathbf{v} J$ Using Matrix Multiplication
$$\mathbf{v} J = \begin{bmatrix} 0 & -\frac{1}{0.30} & 0 \end{bmatrix} \begin{bmatrix}
+0.25 & -0.15 & -0.10 \\
-0.15 & +0.21 & -0.06 \\
-0.10 & -0.06 & +0.16
\end{bmatrix}$$

Evaluating entry by entry:
1. Component 1 ($z_1$):
   $$\left(-\frac{1}{0.30}\right) \times (-0.15) = +\frac{0.15}{0.30} = \mathbf{+0.50}$$
2. Component 2 ($z_2$):
   $$\left(-\frac{1}{0.30}\right) \times (+0.21) = -\frac{0.21}{0.30} = \mathbf{-0.70}$$
3. Component 3 ($z_3$):
   $$\left(-\frac{1}{0.30}\right) \times (-0.06) = +\frac{0.06}{0.30} = \mathbf{+0.20}$$

$$\nabla_{\mathbf{z}} \mathcal{L} = \begin{bmatrix} +0.50 \\ -0.70 \\ +0.20 \end{bmatrix}$$

#### Step 3: Check Collapsed Formula $\mathbf{p} - \mathbf{y}$
$$\mathbf{p} - \mathbf{y} = \begin{bmatrix} 0.50 \\ 0.30 \\ 0.20 \end{bmatrix} - \begin{bmatrix} 0.00 \\ 1.00 \\ 0.00 \end{bmatrix} = \begin{bmatrix} +0.50 \\ -0.70 \\ +0.20 \end{bmatrix} \quad \text{✅ Exact Match!}$$

**🧒 Intuition:**
- The gradient for Dog ($z_1$) is $+0.50$, so descent steps $-\eta(+0.50)$ will decrease Dog logit!
- The gradient for Cat ($z_2$) is $-0.70$, so descent steps $-\eta(-0.70)$ will increase Cat logit!
- The gradient for Bunny ($z_3$) is $+0.20$, so descent steps will decrease Bunny logit!
The model automatically pushes down incorrect classes and lifts up the true class!
</details>

---

## Problem 3: 2D Coordinate Transformation Jacobian & Determinant in Generative Flows

### 🧒 The Context: Maya's Stretched Rubber Sheet (Normalizing Flows)
In generative models (Normalizing Flows), an AI transforms simple Gaussian noise into complex images using invertible mappings $\mathbf{y} = \mathbf{f}(\mathbf{x})$.
Consider the 2D non-linear transformation:
$$y_1 = x_1^2 + x_2$$
$$y_2 = 3x_1 + 2x_2$$

### Tasks:
1. Compute the $2 \times 2$ Jacobian matrix $J(\mathbf{x}) = \frac{\partial \mathbf{y}}{\partial \mathbf{x}}$.
2. Compute the Jacobian determinant $\det(J(\mathbf{x}))$.
3. Evaluate $J$ and $\det(J)$ at $\mathbf{x} = [2, 1]^T$.
4. What is the physical meaning of $\det(J) = 5$? If a tiny patch of input noise has area $0.02$, what is the area of the transformed patch?

---

<details>
<summary>Click to view solution for Problem 3</summary>

### Detailed Step-by-Step Solution:

#### Step 1: Compute the $2 \times 2$ Jacobian Matrix
$$J = \begin{bmatrix}
\frac{\partial y_1}{\partial x_1} & \frac{\partial y_1}{\partial x_2} \\
\frac{\partial y_2}{\partial x_1} & \frac{\partial y_2}{\partial x_2}
\end{bmatrix}$$

- $\frac{\partial y_1}{\partial x_1} = 2x_1$
- $\frac{\partial y_1}{\partial x_2} = 1$
- $\frac{\partial y_2}{\partial x_1} = 3$
- $\frac{\partial y_2}{\partial x_2} = 2$

$$J(\mathbf{x}) = \begin{bmatrix} 2x_1 & 1 \\ 3 & 2 \end{bmatrix}$$

#### Step 2: Compute the Determinant
For a $2 \times 2$ matrix $\begin{bmatrix} a & b \\ c & d \end{bmatrix}$, $\det = ad - bc$:
$$\det(J) = (2x_1)(2) - (1)(3) = \mathbf{4x_1 - 3}$$

#### Step 3: Numerical Evaluation at $\mathbf{x} = [2, 1]^T$
$$J(2, 1) = \begin{bmatrix} 2(2) & 1 \\ 3 & 2 \end{bmatrix} = \begin{bmatrix} 4 & 1 \\ 3 & 2 \end{bmatrix}$$
$$\det(J(2, 1)) = (4)(2) - (1)(3) = 8 - 3 = \mathbf{5}$$

#### Step 4: Physical Meaning of the Determinant
The Jacobian determinant measures the **local area/volume expansion factor** of the transformation:
$$\text{Area}_{\text{new}} \approx |\det(J)| \times \text{Area}_{\text{old}}$$
$$\text{Area}_{\text{new}} = 5 \times 0.02 = \mathbf{0.10}$$

**AI Connection:** In Normalizing Flows, when space is stretched by factor $5$, probability density drops by factor $5$ ($p_Y(\mathbf{y}) = \frac{p_X(\mathbf{x})}{|\det(J)|}$) so total probability remains conserved at $1.0$!
</details>
