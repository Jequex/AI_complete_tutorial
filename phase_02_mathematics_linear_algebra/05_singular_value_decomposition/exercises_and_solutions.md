# Exercises & Detailed Solutions: Topic 2.5

Welcome to the problem set for **Topic 2.5: Singular Value Decomposition (SVD)**.

---

## 📝 Problem Set

### Problem 1: Step-by-Step SVD of a Rank-1 Matrix
Consider the $2 \times 2$ matrix:
$$A = \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}$$

- **Part A:** Compute the symmetric matrix $A^T A$.
- **Part B:** Find the eigenvalues of $A^T A$ and deduce the singular values $\sigma_1, \sigma_2$ of $A$. Why is $\sigma_2 = 0$?
- **Part C:** Find the right singular vectors $\mathbf{v}_1, \mathbf{v}_2$ (columns of $V$).
- **Part D:** Compute the first left singular vector $\mathbf{u}_1 = \frac{1}{\sigma_1} A \mathbf{v}_1$.
- **Part E:** Verify that $A = \sigma_1 \mathbf{u}_1 \mathbf{v}_1^T$.

---

### Problem 2: The Eckart-Young Low-Rank Approximation Error
A data matrix $X \in \mathbb{R}^{100 \times 100}$ has singular values:
$$\sigma_1 = 10.0, \quad \sigma_2 = 6.0, \quad \sigma_3 = 2.0, \quad \sigma_4 = 1.0, \quad \sigma_5 = \dots = \sigma_{100} = 0.0$$

- **Part A:** What is the exact rank of matrix $X$?
- **Part B:** The Frobenius norm of a matrix is $\|X\|_F = \sqrt{\sum_{i=1}^r \sigma_i^2}$. Compute $\|X\|_F$.
- **Part C:** By the Eckart-Young-Mirsky Theorem, what is the exact reconstruction error $\|X - X_2\|_F$ when approximating $X$ with a rank-2 matrix $X_2$?
- **Part D:** What percentage of total energy (defined as $\frac{\sum_{i=1}^k \sigma_i^2}{\sum_{i=1}^r \sigma_i^2}$) is retained by $X_2$?

---

### Problem 3: The Moore-Penrose Pseudoinverse ($A^+$)
Consider the rectangular matrix:
$$A = \begin{bmatrix} 4 & 0 \\ 0 & 2 \\ 0 & 0 \end{bmatrix} \in \mathbb{R}^{3 \times 2}$$

- **Part A:** Identify the singular values $\sigma_1, \sigma_2$ by inspection.
- **Part B:** Construct $\Sigma \in \mathbb{R}^{3 \times 2}$ and its pseudoinverse $\Sigma^+ \in \mathbb{R}^{2 \times 3}$.
- **Part C:** Compute the pseudoinverse $A^+$.
- **Part D:** Verify that $A A^+ A = A$.

---

# 💡 Complete Solutions & Walkthroughs

---

### Solution 1: Step-by-Step SVD of a Rank-1 Matrix

#### Part A: Computing $A^T A$
$$A^T A = \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix} \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix} = \begin{bmatrix} 1(1) + 2(2) & 1(2) + 2(4) \\ 2(1) + 4(2) & 2(2) + 4(4) \end{bmatrix} = \begin{bmatrix} 5 & 10 \\ 10 & 20 \end{bmatrix}$$

---

#### Part B: Eigenvalues and Singular Values
Find eigenvalues of $A^T A$:
$$\det \begin{bmatrix} 5 - \lambda & 10 \\ 10 & 20 - \lambda \end{bmatrix} = (5 - \lambda)(20 - \lambda) - 100 = 0$$
$$\lambda^2 - 25\lambda + 100 - 100 = 0 \implies \lambda(\lambda - 25) = 0$$
$$\lambda_1 = 25, \qquad \lambda_2 = 0$$

Singular values are the square roots of the eigenvalues:
$$\sigma_1 = \sqrt{\lambda_1} = \sqrt{25} = \mathbf{5}$$
$$\sigma_2 = \sqrt{\lambda_2} = \sqrt{0} = \mathbf{0}$$
**Insight:** $\sigma_2 = 0$ because Row 2 of $A$ is a scalar multiple of Row 1 ($[2, 4] = 2[1, 2]$). The matrix has rank 1, meaning it squashes space down to a 1D line!

---

#### Part C: Right Singular Vectors ($V$)
For $\lambda_1 = 25$:
$$(A^T A - 25I)\mathbf{v}_1 = \begin{bmatrix} -20 & 10 \\ 10 & -5 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$
$$-20 v_1 + 10 v_2 = 0 \implies v_2 = 2 v_1$$
A unit eigenvector is:
$$\mathbf{v}_1 = \frac{1}{\sqrt{1^2 + 2^2}} \begin{bmatrix} 1 \\ 2 \end{bmatrix} = \begin{bmatrix} \frac{1}{\sqrt{5}} \\ \frac{2}{\sqrt{5}} \end{bmatrix}$$

For $\lambda_2 = 0$: Any unit vector perpendicular to $\mathbf{v}_1$:
$$\mathbf{v}_2 = \begin{bmatrix} -\frac{2}{\sqrt{5}} \\ \frac{1}{\sqrt{5}} \end{bmatrix}$$

---

#### Part D: Left Singular Vector ($\mathbf{u}_1$)
$$\mathbf{u}_1 = \frac{1}{\sigma_1} A \mathbf{v}_1 = \frac{1}{5} \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix} \begin{bmatrix} \frac{1}{\sqrt{5}} \\ \frac{2}{\sqrt{5}} \end{bmatrix} = \frac{1}{5\sqrt{5}} \begin{bmatrix} 1(1) + 2(2) \\ 2(1) + 4(2) \end{bmatrix} = \frac{1}{5\sqrt{5}} \begin{bmatrix} 5 \\ 10 \end{bmatrix} = \begin{bmatrix} \frac{1}{\sqrt{5}} \\ \frac{2}{\sqrt{5}} \end{bmatrix}$$

---

#### Part E: Verifying $A = \sigma_1 \mathbf{u}_1 \mathbf{v}_1^T$
$$\sigma_1 \mathbf{u}_1 \mathbf{v}_1^T = 5 \begin{bmatrix} \frac{1}{\sqrt{5}} \\ \frac{2}{\sqrt{5}} \end{bmatrix} \begin{bmatrix} \frac{1}{\sqrt{5}} & \frac{2}{\sqrt{5}} \end{bmatrix} = 5 \cdot \frac{1}{5} \begin{bmatrix} 1(1) & 1(2) \\ 2(1) & 2(2) \end{bmatrix} = \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix} = A \quad \blacksquare$$

---

### Solution 2: Eckart-Young Low-Rank Approximation Error

#### Part A: Rank of $X$
The rank is the number of strictly positive singular values. Since only 4 singular values are non-zero:
$$\text{Rank}(X) = \mathbf{4}$$

#### Part B: Total Frobenius Norm
$$\|X\|_F = \sqrt{\sum_{i=1}^4 \sigma_i^2} = \sqrt{10^2 + 6^2 + 2^2 + 1^2} = \sqrt{100 + 36 + 4 + 1} = \sqrt{141} \approx \mathbf{11.874}$$

#### Part C: Eckart-Young Reconstruction Error
The error of the optimal rank-2 approximation $X_2$ is the root sum of squares of the discarded singular values ($\sigma_3, \sigma_4$):
$$\|X - X_2\|_F = \sqrt{\sigma_3^2 + \sigma_4^2} = \sqrt{2^2 + 1^2} = \sqrt{4 + 1} = \sqrt{5} \approx \mathbf{2.236}$$

#### Part D: Energy Retention
$$\text{Energy Retention} = \frac{\sigma_1^2 + \sigma_2^2}{\sum_{i=1}^4 \sigma_i^2} = \frac{100 + 36}{141} = \frac{136}{141} \approx \mathbf{96.45\%}$$
By keeping only a rank-2 matrix ($2\%$ of the original data), we retain over **$96.4\%$ of the total information**!

---

### Solution 3: The Moore-Penrose Pseudoinverse

#### Part A & B: Singular Values and Matrices
Since $A$ is already diagonalized on its non-zero entries:
$$\sigma_1 = 4, \qquad \sigma_2 = 2$$
$$\Sigma = \begin{bmatrix} 4 & 0 \\ 0 & 2 \\ 0 & 0 \end{bmatrix} \in \mathbb{R}^{3 \times 2}$$

To form $\Sigma^+$, invert every non-zero singular value and transpose from $3 \times 2$ to $2 \times 3$:
$$\Sigma^+ = \begin{bmatrix} 1/4 & 0 & 0 \\ 0 & 1/2 & 0 \end{bmatrix} \in \mathbb{R}^{2 \times 3}$$

#### Part C: Computing $A^+$
Here $U = I_{3 \times 3}$ and $V = I_{2 \times 2}$. Therefore:
$$A^+ = V \Sigma^+ U^T = I_{2 \times 2} \Sigma^+ I_{3 \times 3} = \mathbf{\begin{bmatrix} 1/4 & 0 & 0 \\ 0 & 1/2 & 0 \end{bmatrix}}$$

#### Part D: Verification of $A A^+ A = A$
1. Compute $A A^+$:
   $$A A^+ = \begin{bmatrix} 4 & 0 \\ 0 & 2 \\ 0 & 0 \end{bmatrix} \begin{bmatrix} 1/4 & 0 & 0 \\ 0 & 1/2 & 0 \end{bmatrix} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{bmatrix}$$
2. Multiply by $A$:
   $$(A A^+) A = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{bmatrix} \begin{bmatrix} 4 & 0 \\ 0 & 2 \\ 0 & 0 \end{bmatrix} = \begin{bmatrix} 4 & 0 \\ 0 & 2 \\ 0 & 0 \end{bmatrix} = A \quad \blacksquare$$

