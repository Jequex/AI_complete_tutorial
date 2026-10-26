# Topic 9.2: Exercises & Solutions — Dimensionality Reduction (PCA & SVD)

---

### Problem 1: 2D Hand Calculation of PCA (Easy)
Consider three centered 2D data points:
$$X_c = \begin{pmatrix} -2 & -1 \\ 0 & 0 \\ 2 & 1 \end{pmatrix}$$

1. Compute the sample covariance matrix $C = \frac{1}{N-1} X_c^T X_c$ (with $N = 3$).
2. Compute the eigenvalues $\lambda_1, \lambda_2$ of $C$.
3. Compute the normalized first principal component $w_1$.
4. What percentage of the total variance is explained by the first principal component?

<details>
<summary>View Complete Solution</summary>

**Step 1: Compute sample covariance matrix:**
$$X_c^T X_c = \begin{pmatrix} -2 & 0 & 2 \\ -1 & 0 & 1 \end{pmatrix} \begin{pmatrix} -2 & -1 \\ 0 & 0 \\ 2 & 1 \end{pmatrix} = \begin{pmatrix} (-2)^2 + 0^2 + 2^2 & (-2)(-1) + 0 + (2)(1) \\ (-1)(-2) + 0 + (1)(2) & (-1)^2 + 0^2 + 1^2 \end{pmatrix}$$
$$X_c^T X_c = \begin{pmatrix} 8 & 4 \\ 4 & 2 \end{pmatrix}$$

Dividing by $N-1 = 3 - 1 = 2$:
$$C = \begin{pmatrix} 4 & 2 \\ 2 & 1 \end{pmatrix}$$

**Step 2: Compute eigenvalues:**
Solve the characteristic equation $\det(C - \lambda I) = 0$:
$$\det \begin{pmatrix} 4 - \lambda & 2 \\ 2 & 1 - \lambda \end{pmatrix} = (4 - \lambda)(1 - \lambda) - 4 = \lambda^2 - 5\lambda + 4 - 4 = \lambda(\lambda - 5) = 0$$

The eigenvalues are:
$$\lambda_1 = 5, \quad \lambda_2 = 0$$

**Step 3: Compute first eigenvector $w_1$:**
Solve $(C - 5I) w_1 = 0$:
$$\begin{pmatrix} -1 & 2 \\ 2 & -4 \end{pmatrix} \begin{pmatrix} w_{11} \\ w_{12} \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix} \implies -w_{11} + 2 w_{12} = 0 \implies w_{11} = 2 w_{12}$$

Choosing $w_{12} = 1 \implies w = \begin{pmatrix} 2 \\ 1 \end{pmatrix}$.
Normalizing to unit length:
$$\|w\|_2 = \sqrt{2^2 + 1^2} = \sqrt{5}$$
$$w_1 = \begin{pmatrix} \frac{2}{\sqrt{5}} \\ \frac{1}{\sqrt{5}} \end{pmatrix} \approx \begin{pmatrix} 0.8944 \\ 0.4472 \end{pmatrix}$$

**Step 4: Explained variance ratio:**
$$\text{Total Variance} = \text{Tr}(C) = 4 + 1 = 5$$
$$\text{EVR}_1 = \frac{\lambda_1}{\lambda_1 + \lambda_2} = \frac{5}{5 + 0} = 1.0 \quad (100\%)$$
The 3 data points lie perfectly on the line $y = 0.5x$, so a single 1D component captures 100% of all information!
</details>

---

### Problem 2: Lagrange Multiplier Proof of Orthogonality (Medium)
When deriving the second principal component $w_2$, we maximize variance $w_2^T C w_2$ subject to:
1. Unit norm: $w_2^T w_2 = 1$
2. Orthogonality to $w_1$: $w_1^T w_2 = 0$

Formulate the Lagrangian and prove that the Lagrange multiplier associated with the orthogonality constraint must equal zero.

<details>
<summary>View Complete Solution</summary>

**Form the Lagrangian:**
Let $\lambda_2$ be the multiplier for $\|w_2\|^2 = 1$ and $\phi$ be the multiplier for $w_1^T w_2 = 0$:
$$\mathcal{L}(w_2, \lambda_2, \phi) = w_2^T C w_2 - \lambda_2 (w_2^T w_2 - 1) - \phi (w_1^T w_2)$$

**Compute the gradient with respect to $w_2$ and set to zero:**
$$\nabla_{w_2} \mathcal{L} = 2 C w_2 - 2 \lambda_2 w_2 - \phi w_1 = 0$$

**Multiply from the left by $w_1^T$:**
$$2 w_1^T C w_2 - 2 \lambda_2 (w_1^T w_2) - \phi (w_1^T w_1) = 0$$

Use the known properties:
- $w_1^T w_1 = 1$ (unit length of $w_1$)
- $w_1^T w_2 = 0$ (orthogonality constraint)
- $C$ is symmetric, so $w_1^T C w_2 = (C w_1)^T w_2$.
- Since $w_1$ is an eigenvector of $C$ with eigenvalue $\lambda_1$, $C w_1 = \lambda_1 w_1$.
- Therefore, $w_1^T C w_2 = (\lambda_1 w_1)^T w_2 = \lambda_1 (w_1^T w_2) = \lambda_1(0) = 0$.

Substitute these into the equation:
$$2(0) - 2 \lambda_2(0) - \phi(1) = 0 \implies -\phi = 0 \implies \phi = 0$$

**Conclusion:**
Since $\phi = 0$, the gradient equation reduces to:
$$2 C w_2 - 2 \lambda_2 w_2 = 0 \implies C w_2 = \lambda_2 w_2$$
Thus, $w_2$ is simply the second eigenvector of $C$, and orthogonality to $w_1$ is inherently satisfied without needing a non-zero constraint penalty! $\blacksquare$
</details>

---

### Problem 3: SVD Singular Values vs Covariance Eigenvalues (Medium)
Suppose an centered data matrix $X_c \in \mathbb{R}^{101 \times 4}$ has singular values:
$$\sigma = (20.0, 10.0, 4.0, 1.0)$$

1. Compute the eigenvalues $\lambda_1, \lambda_2, \lambda_3, \lambda_4$ of the sample covariance matrix $C = \frac{1}{N-1} X_c^T X_c$ (where $N = 101$).
2. Compute the Explained Variance Ratio (EVR) for each of the 4 components.
3. How many components must be retained to explain at least 95% of the total variance?

<details>
<summary>View Complete Solution</summary>

**Step 1: Compute eigenvalues:**
Using $\lambda_j = \frac{\sigma_j^2}{N-1}$ with $N - 1 = 100$:
- $\lambda_1 = \frac{20.0^2}{100} = \frac{400}{100} = 4.0$
- $\lambda_2 = \frac{10.0^2}{100} = \frac{100}{100} = 1.0$
- $\lambda_3 = \frac{4.0^2}{100} = \frac{16}{100} = 0.16$
- $\lambda_4 = \frac{1.0^2}{100} = \frac{1}{100} = 0.01$

Total variance:
$$\text{Total} = \sum_{j=1}^4 \lambda_j = 4.0 + 1.0 + 0.16 + 0.01 = 5.17$$

Notice this is identically $\frac{\sum \sigma_j^2}{100} = \frac{400 + 100 + 16 + 1}{100} = \frac{517}{100} = 5.17$.

**Step 2: Compute Explained Variance Ratios:**
- $\text{EVR}_1 = \frac{4.0}{5.17} \approx 0.7737$ (77.37%)
- $\text{EVR}_2 = \frac{1.0}{5.17} \approx 0.1934$ (19.34%)
- $\text{EVR}_3 = \frac{0.16}{5.17} \approx 0.0309$ (3.09%)
- $\text{EVR}_4 = \frac{0.01}{5.17} \approx 0.0019$ (0.19%)

**Step 3: Cumulative variance:**
- 1 component: $77.37\% < 95\%$
- 2 components: $77.37\% + 19.34\% = 96.71\% \ge 95\%$

Therefore, keeping **2 components** is sufficient to capture 96.71% (> 95%) of the variance.
</details>

---

### Problem 4: Frobenius Norm Reconstruction Error (Hard)
State the Eckart-Young-Mirsky Theorem. Given centered matrix $X_c$ with singular values $\sigma_1 = 10, \sigma_2 = 6, \sigma_3 = 2$, compute the exact Frobenius norm reconstruction error $\|X_c - \hat{X}_2\|_F$ when truncating to rank 2.

<details>
<summary>View Complete Solution</summary>

**Eckart-Young-Mirsky Theorem:**
Let $X \in \mathbb{R}^{N \times d}$ have singular value decomposition $X = \sum_{j=1}^r \sigma_j u_j v_j^T$. For any rank $k \le r$, the truncated SVD matrix $\hat{X}_k = \sum_{j=1}^k \sigma_j u_j v_j^T$ satisfies:
$$\min_{\text{rank}(A) \le k} \|X - A\|_F = \|X - \hat{X}_k\|_F = \sqrt{\sum_{j=k+1}^r \sigma_j^2}$$

**Calculation for $k = 2$:**
The omitted singular values are $\sigma_3 = 2$:
$$\|X_c - \hat{X}_2\|_F^2 = \sigma_3^2 = 2^2 = 4$$
$$\|X_c - \hat{X}_2\|_F = \sqrt{4} = 2.0$$

The squared reconstruction error is exactly 4.0, representing the variance along the discarded third dimension.
</details>
