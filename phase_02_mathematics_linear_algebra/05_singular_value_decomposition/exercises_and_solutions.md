# Topic 2.5: Exercises & Solutions — Singular Value Decomposition (SVD)

Welcome to the problem set for **Topic 2.5: Singular Value Decomposition (SVD)**!

Work through each problem carefully before clicking to reveal the step-by-step arithmetic and proof breakdown.

---

## 📝 Problem Set

### Problem 1: Step-by-Step SVD of a Rank-1 Matrix

Consider the matrix:
$$A = \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}$$

1. Compute the symmetric matrix $A^T A$.
2. Find the eigenvalues of $A^T A$ and deduce singular values $\sigma_1, \sigma_2$. Why is $\sigma_2 = 0$?
3. Find the right singular vectors $\mathbf{v}_1, \mathbf{v}_2$.
4. Compute the first left singular vector $\mathbf{u}_1 = \frac{1}{\sigma_1} A \mathbf{v}_1$.
5. Verify that $A = \sigma_1 \mathbf{u}_1 \mathbf{v}_1^T$.

<details>
<summary>Click to view solution for Problem 1</summary>

### Solution 1: Step-by-Step SVD of a Rank-1 Matrix

#### Part 1: Computing $A^T A$
$$A^T A = \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix} \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix} = \begin{bmatrix} 1+4 & 2+8 \\ 2+8 & 4+16 \end{bmatrix} = \begin{bmatrix} 5 & 10 \\ 10 & 20 \end{bmatrix}$$

#### Part 2: Eigenvalues & Singular Values
$$\det(A^T A - \lambda I) = (5 - \lambda)(20 - \lambda) - 100 = \lambda^2 - 25\lambda = 0$$
$$\lambda_1 = 25, \qquad \lambda_2 = 0$$
$$\sigma_1 = \sqrt{25} = \mathbf{5.0}, \qquad \sigma_2 = \sqrt{0} = \mathbf{0.0}$$
*Insight:* $\sigma_2 = 0$ because Row 2 is $2 \times$ Row 1; the matrix has rank 1 and collapses 2D space to a 1D line.

#### Part 3: Right Singular Vectors $V$
For $\lambda_1 = 25$:
$$(A^T A - 25I)\mathbf{v} = \begin{bmatrix} -20 & 10 \\ 10 & -5 \end{bmatrix}\mathbf{v} = \mathbf{0} \implies -2v_1 + v_2 = 0 \implies \mathbf{v}_1 = \frac{1}{\sqrt{5}} \begin{bmatrix} 1 \\ 2 \end{bmatrix}$$
For $\lambda_2 = 0$:
$$v_1 + 2v_2 = 0 \implies \mathbf{v}_2 = \frac{1}{\sqrt{5}} \begin{bmatrix} -2 \\ 1 \end{bmatrix}$$

#### Part 4: Left Singular Vector $\mathbf{u}_1$
$$\mathbf{u}_1 = \frac{1}{\sigma_1} A \mathbf{v}_1 = \frac{1}{5} \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix} \left(\frac{1}{\sqrt{5}} \begin{bmatrix} 1 \\ 2 \end{bmatrix}\right) = \frac{1}{5\sqrt{5}} \begin{bmatrix} 1 + 4 \\ 2 + 8 \end{bmatrix} = \frac{1}{5\sqrt{5}} \begin{bmatrix} 5 \\ 10 \end{bmatrix} = \frac{1}{\sqrt{5}} \begin{bmatrix} 1 \\ 2 \end{bmatrix}$$

#### Part 5: Outer Product Verification
$$\sigma_1 \mathbf{u}_1 \mathbf{v}_1^T = 5 \left(\frac{1}{\sqrt{5}} \begin{bmatrix} 1 \\ 2 \end{bmatrix}\right) \left(\frac{1}{\sqrt{5}} \begin{bmatrix} 1 & 2 \end{bmatrix}\right) = 5 \left(\frac{1}{5}\right) \begin{bmatrix} 1 \\ 2 \end{bmatrix} \begin{bmatrix} 1 & 2 \end{bmatrix} = \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix} = A \quad \text{✅ (Matches!)}$$
</details>

---

### Problem 2: Eckart-Young Low-Rank Approximation Error

A data matrix $X \in \mathbb{R}^{100 \times 100}$ has singular values:
$$\sigma_1 = 10.0, \quad \sigma_2 = 6.0, \quad \sigma_3 = 2.0, \quad \sigma_4 = 1.0, \quad \sigma_5 = \dots = \sigma_{100} = 0.0$$

1. What is the exact rank of matrix $X$?
2. Compute the Frobenius norm $\|X\|_F = \sqrt{\sum \sigma_i^2}$.
3. By the Eckart-Young Theorem, compute the exact reconstruction error $\|X - X_2\|_F$ when using a rank-2 approximation $X_2$.
4. What percentage of total energy ($\frac{\sigma_1^2 + \sigma_2^2}{\|X\|_F^2}$) is retained by $X_2$?

<details>
<summary>Click to view solution for Problem 2</summary>

### Solution 2: Eckart-Young Low-Rank Approximation Error

#### Part 1: Exact Rank
There are exactly 4 non-zero singular values, so $\text{Rank}(X) = \mathbf{4}$.

#### Part 2: Frobenius Norm $\|X\|_F$
$$\|X\|_F^2 = 10^2 + 6^2 + 2^2 + 1^2 = 100 + 36 + 4 + 1 = 141$$
$$\|X\|_F = \sqrt{141} \approx \mathbf{11.874}$$

#### Part 3: Reconstruction Error $\|X - X_2\|_F$
By the Eckart-Young-Mirsky Theorem:
$$\|X - X_2\|_F = \sqrt{\sum_{j=3}^4 \sigma_j^2} = \sqrt{2^2 + 1^2} = \sqrt{4 + 1} = \sqrt{5} \approx \mathbf{2.236}$$

#### Part 4: Retained Energy Percentage
$$\text{Energy Retained} = \frac{\sigma_1^2 + \sigma_2^2}{\|X\|_F^2} \times 100\% = \frac{100 + 36}{141} \times 100\% = \frac{136}{141} \times 100\% \approx \mathbf{96.45\%}$$
*Conclusion:* Keeping just 2 singular values captures over **$96.4\%$** of the entire dataset's variance and information!
</details>

---

### Problem 3: The Moore-Penrose Pseudoinverse ($A^+$)

Consider rectangular matrix:
$$A = \begin{bmatrix} 4 & 0 \\ 0 & 2 \\ 0 & 0 \end{bmatrix} \in \mathbb{R}^{3 \times 2}$$

1. Identify the singular values $\sigma_1, \sigma_2$ by inspection.
2. Construct $\Sigma^+$ and compute the pseudoinverse $A^+$.
3. Verify that $A A^+ A = A$.

<details>
<summary>Click to view solution for Problem 3</summary>

### Solution 3: Moore-Penrose Pseudoinverse

#### Part 1: Singular Values by Inspection
Because $A$ has orthogonal columns with norms $4$ and $2$:
$$\sigma_1 = \mathbf{4.0}, \qquad \sigma_2 = \mathbf{2.0}$$

#### Part 2: Pseudoinverse $A^+$
For a rectangular diagonal matrix with entries $\sigma_1, \sigma_2$:
$$\Sigma = \begin{bmatrix} 4 & 0 \\ 0 & 2 \\ 0 & 0 \end{bmatrix} \implies \Sigma^+ = \begin{bmatrix} 1/4 & 0 & 0 \\ 0 & 1/2 & 0 \end{bmatrix} = \begin{bmatrix} 0.25 & 0 & 0 \\ 0 & 0.50 & 0 \end{bmatrix}$$
Here, $U = I_3$ and $V = I_2$, so:
$$A^+ = V \Sigma^+ U^T = \begin{bmatrix} 0.25 & 0 & 0 \\ 0 & 0.50 & 0 \end{bmatrix}$$

#### Part 3: Verification $A A^+ A = A$
$$A^+ A = \begin{bmatrix} 0.25 & 0 & 0 \\ 0 & 0.50 & 0 \end{bmatrix} \begin{bmatrix} 4 & 0 \\ 0 & 2 \\ 0 & 0 \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = I_2$$
$$A (A^+ A) = A I_2 = A \quad \text{✅ (Matches!)}$$
</details>
