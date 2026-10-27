# Topic 2.2: Exercises & Solutions — Linear Transformations & Determinants

Welcome to the problem set for **Topic 2.2: Linear Transformations & Determinants**!

Work through each problem carefully before clicking to reveal the solution.

---

## 📝 Problem Set

### Problem 1: Matrix Synthesis from Basis Vector Landings

A secret linear transformation $T: \mathbb{R}^2 \to \mathbb{R}^2$ acts on the 2D plane:
- It lands $\hat{i} = [1, 0]^T$ at $T(\hat{i}) = \begin{bmatrix} 2 \\ 5 \end{bmatrix}$.
- It lands $\hat{j} = [0, 1]^T$ at $T(\hat{j}) = \begin{bmatrix} -3 \\ 1 \end{bmatrix}$.

1. Write down the $2 \times 2$ matrix $A$ representing transformation $T$.
2. Compute the transformed position of vector $\mathbf{v} = [4, -2]^T$.
3. Compute the determinant $\det(A)$ and state by what factor areas expand under $T$.

<details>
<summary>Click to view solution for Problem 1</summary>

### Solution 1: Matrix Synthesis from Basis Vector Landings

#### Part 1: Writing Matrix $A$
The columns of matrix $A$ are literally the landing sites of the basis vectors:
$$A = \begin{bmatrix} T(\hat{i}) & T(\hat{j}) \end{bmatrix} = \begin{bmatrix} 2 & -3 \\ 5 & 1 \end{bmatrix}$$

#### Part 2: Transforming $\mathbf{v} = [4, -2]^T$
$$A \mathbf{v} = \begin{bmatrix} 2 & -3 \\ 5 & 1 \end{bmatrix} \begin{bmatrix} 4 \\ -2 \end{bmatrix} = \begin{bmatrix} (2)(4) + (-3)(-2) \\ (5)(4) + (1)(-2) \end{bmatrix} = \begin{bmatrix} 8 + 6 \\ 20 - 2 \end{bmatrix} = \begin{bmatrix} 14 \\ 18 \end{bmatrix}$$

#### Part 3: Determinant & Area Factor
$$\det(A) = ad - bc = (2 \times 1) - (-3 \times 5) = 2 - (-15) = \mathbf{17}$$
*Interpretation:* Every 2D shape expands in area by a factor of exactly **$17$**!
</details>

---

### Problem 2: Composition of Rotations ($R_{90^\circ} \times R_{90^\circ}$)

The 2D rotation matrix is given by:
$$R_\theta = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

1. Write down $R_{90^\circ}$ ($\theta = \pi/2$ rad).
2. Compute the matrix product $R_{90^\circ} \times R_{90^\circ}$.
3. Verify that the result matches $R_{180^\circ}$.

<details>
<summary>Click to view solution for Problem 2</summary>

### Solution 2: Composition of Rotations

#### Part 1: Matrix $R_{90^\circ}$
Since $\cos(90^\circ) = 0$ and $\sin(90^\circ) = 1$:
$$R_{90^\circ} = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$$

#### Part 2: Matrix Multiplication
$$R_{90^\circ} \times R_{90^\circ} = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix} \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$$
- Row 1, Col 1: $(0)(0) + (-1)(1) = \mathbf{-1}$
- Row 1, Col 2: $(0)(-1) + (-1)(0) = \mathbf{0}$
- Row 2, Col 1: $(1)(0) + (0)(1) = \mathbf{0}$
- Row 2, Col 2: $(1)(-1) + (0)(0) = \mathbf{-1}$
$$R_{90^\circ}^2 = \begin{bmatrix} -1 & 0 \\ 0 & -1 \end{bmatrix}$$

#### Part 3: Verification with $R_{180^\circ}$
Since $\cos(180^\circ) = -1$ and $\sin(180^\circ) = 0$:
$$R_{180^\circ} = \begin{bmatrix} -1 & 0 \\ 0 & -1 \end{bmatrix}$$
The matrix product exactly equals $R_{180^\circ}$! Matrix multiplication perfectly models composite physical transformations!
</details>

---

### Problem 3: The Singular Inversion Paradox ($\det(M) = 0$)

Consider the matrix:
$$M = \begin{bmatrix} 2 & 6 \\ 1 & 3 \end{bmatrix}$$

1. Compute $\det(M) = ad - bc$.
2. Explain geometrically what happens to the entire 2D Cartesian plane under $M$.
3. Why is it mathematically impossible to invert $M$ ($M^{-1}$ does not exist)?

<details>
<summary>Click to view solution for Problem 3</summary>

### Solution 3: The Singular Inversion Paradox

#### Part 1: Determinant
$$\det(M) = (2 \times 3) - (6 \times 1) = 6 - 6 = \mathbf{0}$$

#### Part 2: Geometric Collapse
- $\hat{i}$ lands at $[2, 1]^T$.
- $\hat{j}$ lands at $[6, 3]^T = 3 [2, 1]^T$.
Both basis vectors land on the **exact same 1D line** ($y = \frac{1}{2}x$). The entire 2D plane is squashed flat into a single 1D line. The area of the transformed space is $0$!

#### Part 3: Non-Invertibility
To invert $M$, an inverse matrix $M^{-1}$ would need to take a single 1D line and reconstruct the original 2D plane. But infinitely many points in the original plane collapsed onto the exact same point on the line (e.g., both $[3, -1]^T$ and $[0, 0]^T$ map to $[0, 0]^T$). You cannot uniquely determine which original point produced an output. Information has been irreversibly destroyed!
</details>
