# Exercises & Detailed Solutions: Topic 2.1

Welcome to the problem set for **Topic 2.1: Scalars, Vectors, and Matrices**.

Work through each problem on paper before checking the step-by-step arithmetic below.

---

## 📝 Problem Set

### Problem 1: Norms and Unit Vector Normalization
Given the 3-dimensional vector:
$$\mathbf{v} = \begin{bmatrix} 3 \\ -4 \\ 12 \end{bmatrix}$$

- **Part A:** Compute the $L_1$ Manhattan norm $\|\mathbf{v}\|_1$.
- **Part B:** Compute the $L_2$ Euclidean norm $\|\mathbf{v}\|_2$.
- **Part C:** Compute the $L_\infty$ Max norm $\|\mathbf{v}\|_\infty$.
- **Part D:** Create a unit vector $\hat{\mathbf{v}}$ by normalizing $\mathbf{v}$ with its $L_2$ norm:
  $$\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2}$$
  Verify that the $L_2$ norm of $\hat{\mathbf{v}}$ is exactly $1.0$.

---

### Problem 2: Dot Products, Angles, and Orthogonality
Consider two vectors in $\mathbb{R}^3$:
$$\mathbf{a} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}, \qquad \mathbf{b} = \begin{bmatrix} 4 \\ -2 \\ 0 \end{bmatrix}$$

- **Part A:** Compute the algebraic dot product $\mathbf{a} \cdot \mathbf{b} = \mathbf{a}^T \mathbf{b}$.
- **Part B:** Based on the dot product, what is the angle $\theta$ between $\mathbf{a}$ and $\mathbf{b}$? Are they orthogonal?
- **Part C:** Compute the cosine similarity between $\mathbf{a}$ and a third vector $\mathbf{c} = [2, 4, 6]^T$. What does your answer reveal?

---

### Problem 3: Dimension Compatibility and Matrix Multiplication
Let:
$$A = \begin{bmatrix} 1 & 0 & 2 \\ -1 & 3 & 1 \end{bmatrix} \in \mathbb{R}^{2 \times 3}, \qquad B = \begin{bmatrix} 3 & 1 \\ 2 & -1 \\ 0 & 4 \end{bmatrix} \in \mathbb{R}^{3 \times 2}$$

- **Part A:** Can we compute $AB$? What is the dimension of the resulting product matrix?
- **Part B:** Can we compute $BA$? What is the dimension of the resulting product matrix?
- **Part C:** Compute the product matrix $C = AB$ showing every intermediate multiplication and addition.

---

# 💡 Complete Solutions & Walkthroughs

---

### Solution 1: Norms and Unit Vector Normalization

#### Part A: $L_1$ Manhattan Norm
$$\|\mathbf{v}\|_1 = \sum_{i=1}^3 |v_i| = |3| + |-4| + |12| = 3 + 4 + 12 = \mathbf{19}$$

#### Part B: $L_2$ Euclidean Norm
$$\|\mathbf{v}\|_2 = \sqrt{3^2 + (-4)^2 + 12^2} = \sqrt{9 + 16 + 144} = \sqrt{169} = \mathbf{13}$$

#### Part C: $L_\infty$ Max Norm
$$\|\mathbf{v}\|_\infty = \max(|3|, |-4|, |12|) = \mathbf{12}$$

#### Part D: Unit Vector Normalization
Divide each component by $\|\mathbf{v}\|_2 = 13$:
$$\hat{\mathbf{v}} = \frac{1}{13} \begin{bmatrix} 3 \\ -4 \\ 12 \end{bmatrix} = \begin{bmatrix} \frac{3}{13} \\ -\frac{4}{13} \\ \frac{12}{13} \end{bmatrix}$$

**Verification:**
$$\|\hat{\mathbf{v}}\|_2 = \sqrt{\left(\frac{3}{13}\right)^2 + \left(-\frac{4}{13}\right)^2 + \left(\frac{12}{13}\right)^2} = \sqrt{\frac{9 + 16 + 144}{169}} = \sqrt{\frac{169}{169}} = \sqrt{1} = \mathbf{1.0} \quad \blacksquare$$

---

### Solution 2: Dot Products, Angles, and Orthogonality

#### Part A: Algebraic Dot Product
$$\mathbf{a} \cdot \mathbf{b} = (1 \times 4) + (2 \times -2) + (3 \times 0) = 4 - 4 + 0 = \mathbf{0}$$

#### Part B: Angle and Orthogonality
Since $\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \|\mathbf{b}\| \cos\theta = 0$:
$$\cos\theta = 0 \implies \theta = 90^\circ = \frac{\pi}{2} \text{ radians}$$
Yes! Because their dot product is zero, vectors $\mathbf{a}$ and $\mathbf{b}$ are **perpendicular (orthogonal)** to each other in 3D space.

#### Part C: Cosine Similarity with $\mathbf{c}$
Notice that $\mathbf{c} = [2, 4, 6]^T = 2 \mathbf{a}$.
1. Compute $\mathbf{a} \cdot \mathbf{c}$:
   $$\mathbf{a} \cdot \mathbf{c} = (1 \times 2) + (2 \times 4) + (3 \times 6) = 2 + 8 + 18 = 28$$
2. Compute norms:
   $$\|\mathbf{a}\| = \sqrt{1^2 + 2^2 + 3^2} = \sqrt{1 + 4 + 9} = \sqrt{14}$$
   $$\|\mathbf{c}\| = \sqrt{2^2 + 4^2 + 6^2} = \sqrt{4 + 16 + 36} = \sqrt{56} = 2\sqrt{14}$$
3. Compute cosine similarity:
   $$\cos\theta = \frac{28}{\sqrt{14} \times 2\sqrt{14}} = \frac{28}{2 \times 14} = \frac{28}{28} = \mathbf{1.0}$$

**Insight:** Because $\mathbf{c}$ is just a scaled copy of $\mathbf{a}$, the angle between them is $0^\circ$ ($\cos\theta = 1.0$). Cosine similarity detects that both vectors point in the identical semantic direction, ignoring length!

---

### Solution 3: Dimension Compatibility & Matrix Multiplication

#### Part A: Dimension of $AB$
- $A \in \mathbb{R}^{2 \times 3}$ and $B \in \mathbb{R}^{3 \times 2}$.
- Inner dimensions match ($3 = 3$).
- Outer dimensions dictate the result: $C = AB \in \mathbb{R}^{\mathbf{2 \times 2}}$.

#### Part B: Dimension of $BA$
- $B \in \mathbb{R}^{3 \times 2}$ and $A \in \mathbb{R}^{2 \times 3}$.
- Inner dimensions match ($2 = 2$).
- Resulting product is in $\mathbb{R}^{\mathbf{3 \times 3}}$.
*(Notice that matrix multiplication is non-commutative: $AB \neq BA$!)*

#### Part C: Calculating $C = AB$
$$C = \begin{bmatrix} c_{11} & c_{12} \\ c_{21} & c_{22} \end{bmatrix}$$

1. **Entry $c_{11}$ (Row 1 of $A \cdot$ Column 1 of $B$):**
   $$c_{11} = (1 \times 3) + (0 \times 2) + (2 \times 0) = 3 + 0 + 0 = \mathbf{3}$$

2. **Entry $c_{12}$ (Row 1 of $A \cdot$ Column 2 of $B$):**
   $$c_{12} = (1 \times 1) + (0 \times -1) + (2 \times 4) = 1 + 0 + 8 = \mathbf{9}$$

3. **Entry $c_{21}$ (Row 2 of $A \cdot$ Column 1 of $B$):**
   $$c_{21} = (-1 \times 3) + (3 \times 2) + (1 \times 0) = -3 + 6 + 0 = \mathbf{3}$$

4. **Entry $c_{22}$ (Row 2 of $A \cdot$ Column 2 of $B$):**
   $$c_{22} = (-1 \times 1) + (3 \times -1) + (1 \times 4) = -1 - 3 + 4 = \mathbf{0}$$

Result:
$$C = \begin{bmatrix} 3 & 9 \\ 3 & 0 \end{bmatrix}$$

