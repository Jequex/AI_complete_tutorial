# Exercises & Detailed Solutions: Topic 2.2

Welcome to the problem set for **Topic 2.2: Linear Transformations & Determinants**.

---

## 📝 Problem Set

### Problem 1: Matrix Synthesis from Basis Vector Landings
A secret linear transformation $T: \mathbb{R}^2 \to \mathbb{R}^2$ acts on the 2D Cartesian plane:
- It sends the horizontal basis arrow $\hat{i} = [1, 0]^T$ to landing coordinates:
  $$T(\hat{i}) = \begin{bmatrix} 2 \\ 5 \end{bmatrix}$$
- It sends the vertical basis arrow $\hat{j} = [0, 1]^T$ to landing coordinates:
  $$T(\hat{j}) = \begin{bmatrix} -3 \\ 1 \end{bmatrix}$$

- **Part A:** Write down the $2 \times 2$ transformation matrix $A$ representing $T$.
- **Part B:** Predict the landing position of the point $\mathbf{v} = [4, -2]^T$ under this transformation.
- **Part C:** Calculate the determinant $\det(A)$. By what factor does this transformation expand or shrink areas?

---

### Problem 2: Composition of Rotations
The 2D rotation matrix for counterclockwise rotation by $\theta$ is:
$$R_\theta = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

- **Part A:** Write the matrix $R_{90^\circ}$ for a $90^\circ$ ($\frac{\pi}{2}$ radian) counterclockwise rotation.
- **Part B:** Compute the matrix product $R_{90^\circ} \times R_{90^\circ}$ (rotating by $90^\circ$ twice in a row).
- **Part C:** Compare your result to $R_{180^\circ}$. Does matrix multiplication correctly model sequential geometric transformations?

---

### Problem 3: The Singular Inversion Paradox ($\det = 0$)
Consider the matrix:
$$M = \begin{bmatrix} 2 & 6 \\ 1 & 3 \end{bmatrix}$$

- **Part A:** Compute $\det(M) = ad - bc$.
- **Part B:** Where do $\hat{i}$ and $\hat{j}$ land? Describe what happens to the entire 2D coordinate plane geometrically.
- **Part C:** Explain why it is impossible to create an inverse matrix $M^{-1}$ that reverses this transformation.

---

# 💡 Complete Solutions & Walkthroughs

---

### Solution 1: Matrix Synthesis from Basis Vector Landings

#### Part A: Writing Matrix $A$
The columns of a transformation matrix are literally the landing coordinates of the standard basis vectors:
$$A = \begin{bmatrix} T(\hat{i}) & T(\hat{j}) \end{bmatrix} = \begin{bmatrix} 2 & -3 \\ 5 & 1 \end{bmatrix}$$

#### Part B: Transforming $\mathbf{v} = [4, -2]^T$
Multiply $A$ by $\mathbf{v}$:
$$T(\mathbf{v}) = A \mathbf{v} = \begin{bmatrix} 2 & -3 \\ 5 & 1 \end{bmatrix} \begin{bmatrix} 4 \\ -2 \end{bmatrix}$$
$$\text{Row 1: } (2 \times 4) + (-3 \times -2) = 8 + 6 = \mathbf{14}$$
$$\text{Row 2: } (5 \times 4) + (1 \times -2) = 20 - 2 = \mathbf{18}$$
$$T(\mathbf{v}) = \begin{bmatrix} 14 \\ 18 \end{bmatrix}$$

#### Part C: Determinant Calculation
$$\det(A) = ad - bc = (2 \times 1) - (-3 \times 5) = 2 - (-15) = 2 + 15 = \mathbf{17}$$
Any shape drawn on the original grid has its area **multiplied by $17$** under this transformation!

---

### Solution 2: Composition of Rotations

#### Part A: Rotation by $90^\circ$
Since $\cos(90^\circ) = 0$ and $\sin(90^\circ) = 1$:
$$R_{90^\circ} = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$$

#### Part B: Matrix Product $R_{90^\circ} \times R_{90^\circ}$
$$R_{90^\circ} R_{90^\circ} = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix} \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$$
- Entry $(1, 1): (0 \times 0) + (-1 \times 1) = \mathbf{-1}$
- Entry $(1, 2): (0 \times -1) + (-1 \times 0) = \mathbf{0}$
- Entry $(2, 1): (1 \times 0) + (0 \times 1) = \mathbf{0}$
- Entry $(2, 2): (1 \times -1) + (0 \times 0) = \mathbf{-1}$

$$R_{90^\circ} R_{90^\circ} = \begin{bmatrix} -1 & 0 \\ 0 & -1 \end{bmatrix} = -I$$

#### Part C: Comparison with $R_{180^\circ}$
For a $180^\circ$ rotation: $\cos(180^\circ) = -1$ and $\sin(180^\circ) = 0$:
$$R_{180^\circ} = \begin{bmatrix} -1 & 0 \\ 0 & -1 \end{bmatrix}$$
The results are identical! Matrix multiplication $AB$ exactly corresponds to **function composition** $T_A(T_B(\mathbf{x}))$—applying transformation $B$ first, then transformation $A$.

---

### Solution 3: The Singular Inversion Paradox

#### Part A: Determinant
$$\det(M) = (2 \times 3) - (6 \times 1) = 6 - 6 = \mathbf{0}$$

#### Part B: Landing Sites and Geometry
- $\hat{i}$ lands at $[2, 1]^T$.
- $\hat{j}$ lands at $[6, 3]^T = 3 \begin{bmatrix} 2 \\ 1 \end{bmatrix}$.
Notice that both basis vectors land on the **exact same straight line**: $y = \frac{1}{2} x$!
The entire infinite 2D plane has been squashed flat into a single 1D diagonal line.

#### Part C: Why No Inverse $M^{-1}$ Can Exist
To undo a transformation, the inverse function must answer: *"Which original 2D point produced output point $\mathbf{w}$?"*
Because infinitely many 2D points were squashed onto the exact same output on the line, you cannot run time backwards. A single point cannot be uniquely "unsquashed" into an entire plane. Hence, whenever $\det(A) = 0$, the transformation is irreversible, and **$A^{-1}$ does not exist**.

