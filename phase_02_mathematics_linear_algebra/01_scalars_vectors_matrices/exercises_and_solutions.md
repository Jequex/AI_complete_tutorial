# Topic 2.1: Exercises & Solutions — Scalars, Vectors, and Matrices

Welcome to the problem set for **Topic 2.1: Scalars, Vectors, and Matrices**!

Work through each problem with pen and paper before revealing the step-by-step arithmetic solution.

---

## 📝 Problem Set

### Problem 1: Norms and Unit Vector Normalization

Given the 3-dimensional vector:
$$\mathbf{v} = \begin{bmatrix} 3 \\ -4 \\ 12 \end{bmatrix}$$

1. Compute the $L_1$ Manhattan norm $\|\mathbf{v}\|_1$.
2. Compute the $L_2$ Euclidean norm $\|\mathbf{v}\|_2$.
3. Compute the $L_\infty$ Max norm $\|\mathbf{v}\|_\infty$.
4. Create a unit vector $\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2}$ and verify that $\|\hat{\mathbf{v}}\|_2 = 1.0$.

<details>
<summary>Click to view solution for Problem 1</summary>

### Solution 1: Norms and Unit Vector Normalization

#### Part 1: $L_1$ Manhattan Norm
$$\|\mathbf{v}\|_1 = \sum_{i=1}^3 |v_i| = |3| + |-4| + |12| = 3 + 4 + 12 = \mathbf{19}$$

#### Part 2: $L_2$ Euclidean Norm
$$\|\mathbf{v}\|_2 = \sqrt{3^2 + (-4)^2 + 12^2} = \sqrt{9 + 16 + 144} = \sqrt{169} = \mathbf{13}$$

#### Part 3: $L_\infty$ Max Norm
$$\|\mathbf{v}\|_\infty = \max(|3|, |-4|, |12|) = \mathbf{12}$$

#### Part 4: Unit Vector Normalization
$$\hat{\mathbf{v}} = \frac{\mathbf{v}}{13} = \begin{bmatrix} 3/13 \\ -4/13 \\ 12/13 \end{bmatrix}$$
Verify Euclidean norm:
$$\|\hat{\mathbf{v}}\|_2 = \sqrt{\left(\frac{3}{13}\right)^2 + \left(\frac{-4}{13}\right)^2 + \left(\frac{12}{13}\right)^2} = \sqrt{\frac{9 + 16 + 144}{169}} = \sqrt{\frac{169}{169}} = \mathbf{1.0}$$
</details>

---

### Problem 2: Dot Products, Angles, and Orthogonality

Consider two vectors in $\mathbb{R}^3$:
$$\mathbf{a} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}, \qquad \mathbf{b} = \begin{bmatrix} 4 \\ -2 \\ 0 \end{bmatrix}$$

1. Compute the algebraic dot product $\mathbf{a} \cdot \mathbf{b} = \mathbf{a}^T \mathbf{b}$.
2. Based on your result, what is the geometric angle $\theta$ between $\mathbf{a}$ and $\mathbf{b}$? Are they orthogonal?
3. Compute the cosine similarity between $\mathbf{a}$ and $\mathbf{c} = \begin{bmatrix} 2 \\ 4 \\ 6 \end{bmatrix}$. What does this reveal?

<details>
<summary>Click to view solution for Problem 2</summary>

### Solution 2: Dot Products, Angles, and Orthogonality

#### Part 1: Dot Product
$$\mathbf{a} \cdot \mathbf{b} = (1 \times 4) + (2 \times -2) + (3 \times 0) = 4 - 4 + 0 = \mathbf{0}$$

#### Part 2: Angle & Orthogonality
Since $\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \|\mathbf{b}\| \cos\theta = 0$ (and neither vector has zero length):
$$\cos\theta = 0 \implies \theta = \mathbf{90^\circ} \quad (\text{or } \pi/2\text{ radians})$$
**Yes, the vectors are exactly orthogonal (perpendicular)!**

#### Part 3: Cosine Similarity with $\mathbf{c} = [2, 4, 6]^T$
Notice that $\mathbf{c} = 2 \mathbf{a}$.
- $\mathbf{a} \cdot \mathbf{c} = (1 \times 2) + (2 \times 4) + (3 \times 6) = 2 + 8 + 18 = 28$
- $\|\mathbf{a}\| = \sqrt{1^2 + 2^2 + 3^2} = \sqrt{14}$
- $\|\mathbf{c}\| = \sqrt{2^2 + 4^2 + 6^2} = \sqrt{4 + 16 + 36} = \sqrt{56} = 2\sqrt{14}$
$$\cos\theta = \frac{28}{\sqrt{14} \times 2\sqrt{14}} = \frac{28}{2 \times 14} = \frac{28}{28} = \mathbf{1.0}$$
*Insight:* The cosine similarity is exactly $1.0$ because $\mathbf{c}$ points in the identical direction as $\mathbf{a}$ ($\theta = 0^\circ$). Scaling an embedding vector does not change its semantic meaning!
</details>

---

### Problem 3: Dimension Compatibility and Matrix Multiplication

Let:
$$A = \begin{bmatrix} 1 & 0 & 2 \\ -1 & 3 & 1 \end{bmatrix} \in \mathbb{R}^{2 \times 3}, \qquad B = \begin{bmatrix} 3 & 1 \\ 2 & -1 \\ 0 & 4 \end{bmatrix} \in \mathbb{R}^{3 \times 2}$$

1. Can we compute $AB$? State the dimensions of $AB$.
2. Can we compute $BA$? State the dimensions of $BA$.
3. Compute the full product matrix $C = AB$ step-by-step.

<details>
<summary>Click to view solution for Problem 3</summary>

### Solution 3: Dimension Compatibility and Matrix Multiplication

#### Part 1: Product $AB$
- Dimensions: $(2 \times 3) \times (3 \times 2)$.
- Inner dimensions match ($3 = 3$).
- The product $AB$ is valid and has dimension **$2 \times 2$**.

#### Part 2: Product $BA$
- Dimensions: $(3 \times 2) \times (2 \times 3)$.
- Inner dimensions match ($2 = 2$).
- The product $BA$ is valid and has dimension **$3 \times 3$**.
- *Takeaway:* Matrix multiplication is non-commutative ($AB \neq BA$).

#### Part 3: Computing $C = AB$
$$C = \begin{bmatrix} c_{11} & c_{12} \\ c_{21} & c_{22} \end{bmatrix}$$
1. $c_{11} = (1 \times 3) + (0 \times 2) + (2 \times 0) = 3 + 0 + 0 = \mathbf{3}$
2. $c_{12} = (1 \times 1) + (0 \times -1) + (2 \times 4) = 1 + 0 + 8 = \mathbf{9}$
3. $c_{21} = (-1 \times 3) + (3 \times 2) + (1 \times 0) = -3 + 6 + 0 = \mathbf{3}$
4. $c_{22} = (-1 \times 1) + (3 \times -1) + (1 \times 4) = -1 - 3 + 4 = \mathbf{0}$

$$C = AB = \begin{bmatrix} 3 & 9 \\ 3 & 0 \end{bmatrix}$$
</details>
