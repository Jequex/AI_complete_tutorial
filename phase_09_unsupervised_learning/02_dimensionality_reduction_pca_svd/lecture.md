# Topic 9.2: Dimensionality Reduction — PCA, SVD, & Subspace Learning

---

## 1. Executive Summary & The 12-Year-Old Analogy

Modern machine learning datasets frequently contain hundreds or thousands of features: gene expression microarrays have 20,000 dimensions, high-resolution images have millions of pixels, and word embeddings span hundreds of latent dimensions. Yet, data points in these high-dimensional spaces almost never fill the space uniformly; they live on low-dimensional sub-manifolds and linear subspaces.

**Principal Component Analysis (PCA)** and **Singular Value Decomposition (SVD)** are the foundational mathematical techniques for linear dimensionality reduction: discovering the optimal orthogonal directions of maximum variance and projecting hyper-dimensional data into compressed representations while minimizing information loss.

```
                  PCA PROJECTION: FINDING THE AXIS OF MAXIMUM VARIANCE
       y ^
         │                  * (High variance direction: PC 1)
         │               * *   /
         │             * * *  /
         │           * * *   / 
         │         * * *    /
         │       * *       / 
         │      *         /  <-- Second axis (PC 2, orthogonal, minimal variance)
         └────────────────────────────────────────> x
```

---

### The 12-Year-Old Analogy: Casting 3D Shadow Puppets onto a Flat Wall

Imagine you have a detailed, 3D plastic toy airplane suspended in mid-air inside a dark room. You want to capture the airplane's shape as accurately as possible on a flat, 2D sheet of paper using a single flashlight beam (projecting a shadow).

- **The Bad Angle:** If you hold the flashlight directly in front of the airplane's nose, the shadow on the wall is just a tiny, boring circle. You lost the wings! You lost the fuselage! The shadow has almost zero variance.
- **The Optimal Angle (Principal Component 1):** You slowly walk around the airplane until you find the exact flashlight angle where the shadow spreads out as wide and long as possible—revealing the entire wingspan and the long fuselage. This 2D shadow captures 95% of everything you need to know about the airplane's 3D shape.
- **The Reconstruction:** If someone only sees your 2D shadow drawing, they can reconstruct the original 3D airplane with incredible accuracy. The only thing they missed is the tiny thickness of the plastic wing, which had almost no variance anyway!

---

## 2. Mathematical Symbol Decoder

| Symbol | Mathematical Domain | Description & Meaning |
| :--- | :--- | :--- |
| $X \in \mathbb{R}^{N \times d}$ | Matrix | Dataset of $N$ observations with $d$ features |
| $\bar{x} \in \mathbb{R}^d$ | Vector | Empirical feature mean vector: $\bar{x} = \frac{1}{N} \sum_{i=1}^N x_i$ |
| $X_c = X - \mathbf{1} \bar{x}^T$ | Matrix | Mean-centered data matrix ($\sum_{i=1}^N (X_c)_{ij} = 0$) |
| $C \in \mathbb{R}^{d \times d}$ | Symmetric positive semi-definite | Sample covariance matrix: $C = \frac{1}{N-1} X_c^T X_c$ |
| $w \in \mathbb{R}^d$ | Unit vector ($\|w\|_2 = 1$) | Projection direction (principal component axis / loading vector) |
| $\lambda_k \in \mathbb{R}_{\ge 0}$ | Scalar | $k$-th eigenvalue of $C$, representing variance along $w_k$ |
| $W_k \in \mathbb{R}^{d \times k}$ | Orthogonal columns | Matrix of top $k$ principal eigenvectors: $W_k = [w_1, w_2, \dots, w_k]$ |
| $Z = X_c W_k \in \mathbb{R}^{N \times k}$ | Matrix | Low-dimensional principal component scores / projected coordinates |
| $\hat{X} = Z W_k^T + \mathbf{1} \bar{x}^T$ | Matrix | Reconstructed data matrix in original $d$-dimensional space |
| $U \Sigma V^T$ | Factorization | Singular Value Decomposition (SVD) of centered matrix $X_c$ |
| $\sigma_j \ge 0$ | Singular values | Diagonal entries of $\Sigma$, where $\sigma_j = \sqrt{(N-1) \lambda_j}$ |

---

## 3. PCA from First Principles: The Maximum Variance Derivation

Let $X_c \in \mathbb{R}^{N \times d}$ be the mean-centered data matrix. We wish to find a unit projection vector $w_1 \in \mathbb{R}^d$ ($\|w_1\|_2 = 1$) such that the variance of the projected scalar coordinates $z_i = x_{c,i}^T w_1$ is maximized.

### 3.1 Variance of Projected Data
The sample mean of the projected coordinates is zero:
$$\bar{z} = \frac{1}{N} \sum_{i=1}^N x_{c,i}^T w_1 = \left(\frac{1}{N} \sum_{i=1}^N x_{c,i}^T\right) w_1 = 0^T w_1 = 0$$

The empirical sample variance along direction $w_1$ is:
$$\text{Var}(z) = \frac{1}{N-1} \sum_{i=1}^N (z_i - \bar{z})^2 = \frac{1}{N-1} \sum_{i=1}^N (w_1^T x_{c,i})(x_{c,i}^T w_1) = w_1^T \left( \frac{1}{N-1} X_c^T X_c \right) w_1 = w_1^T C w_1$$

where $C \equiv \frac{1}{N-1} X_c^T X_c$ is the $d \times d$ sample covariance matrix.

### 3.2 Rayleigh Quotient Maximization via Lagrange Multipliers

We formulate the constrained optimization problem:
$$\max_{w_1 \in \mathbb{R}^d} w_1^T C w_1 \quad \text{subject to} \quad w_1^T w_1 = 1$$

Form the Lagrangian with multiplier $\lambda_1$:
$$\mathcal{L}(w_1, \lambda_1) = w_1^T C w_1 - \lambda_1 (w_1^T w_1 - 1)$$

Taking the gradient with respect to $w_1$:
$$\nabla_{w_1} \mathcal{L} = 2 C w_1 - 2 \lambda_1 w_1 = 0$$

Dividing by 2 gives the fundamental **eigenvalue equation**:
$$C w_1 = \lambda_1 w_1$$

Multiplying both sides on the left by $w_1^T$:
$$w_1^T C w_1 = \lambda_1 w_1^T w_1 = \lambda_1 (1) = \lambda_1$$

> **Key Insight:** The variance of the data along direction $w_1$ is precisely equal to the eigenvalue $\lambda_1$! Therefore, to maximize variance, $w_1$ must be the eigenvector corresponding to the **largest eigenvalue** $\lambda_{\max}(C)$.

---

### 3.3 Subsequent Principal Components & Orthogonality

To find the second principal component $w_2$, we maximize variance subject to two constraints: unit length $\|w_2\|_2 = 1$ and orthogonality to the first component $w_1^T w_2 = 0$:

$$\max_{w_2} w_2^T C w_2 \quad \text{s.t.} \quad w_2^T w_2 = 1, \quad w_1^T w_2 = 0$$

The Lagrangian is:
$$\mathcal{L}(w_2, \lambda_2, \phi) = w_2^T C w_2 - \lambda_2 (w_2^T w_2 - 1) - \phi (w_1^T w_2)$$

Differentiating with respect to $w_2$:
$$2 C w_2 - 2 \lambda_2 w_2 - \phi w_1 = 0$$

Pre-multiplying by $w_1^T$:
$$2 w_1^T C w_2 - 2 \lambda_2 (w_1^T w_2) - \phi (w_1^T w_1) = 0$$

Since $C$ is symmetric, $w_1^T C w_2 = (C w_1)^T w_2 = (\lambda_1 w_1)^T w_2 = \lambda_1 (w_1^T w_2) = 0$. Since $w_1^T w_1 = 1$, we find:
$$\phi = 0$$

The equation simplifies back to:
$$C w_2 = \lambda_2 w_2$$

Thus, $w_2$ is simply the eigenvector corresponding to the **second largest eigenvalue** $\lambda_2 \le \lambda_1$, and by the Spectral Theorem for symmetric real matrices, $w_1 \perp w_2$ automatically!

---

## 4. Singular Value Decomposition (SVD) & Mathematical Equivalence

While computing the covariance matrix $C = \frac{1}{N-1} X_c^T X_c$ works well conceptually, real-world numerical packages (like SciPy, NumPy, and PyTorch) almost always compute PCA using **Singular Value Decomposition (SVD)** on $X_c$ directly.

### 4.1 The SVD Factorization
Any real matrix $X_c \in \mathbb{R}^{N \times d}$ can be decomposed as:
$$X_c = U \Sigma V^T$$
where:
- $U \in \mathbb{R}^{N \times N}$ is an orthogonal matrix of left singular vectors ($U^T U = I_N$).
- $\Sigma \in \mathbb{R}^{N \times d}$ is a rectangular diagonal matrix containing singular values $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r \ge 0$.
- $V \in \mathbb{R}^{d \times d}$ is an orthogonal matrix of right singular vectors ($V^T V = I_d$).

### 4.2 Exact Equivalence Proof: SVD vs Covariance Eigenvectors
Let us compute the matrix product $X_c^T X_c$:
$$X_c^T X_c = (U \Sigma V^T)^T (U \Sigma V^T) = V \Sigma^T (U^T U) \Sigma V^T$$

Since $U$ is orthogonal, $U^T U = I$:
$$X_c^T X_c = V (\Sigma^T \Sigma) V^T$$

Notice that $\Sigma^T \Sigma = \text{diag}(\sigma_1^2, \sigma_2^2, \dots, \sigma_d^2)$.
Dividing by $N-1$:
$$C = \frac{1}{N-1} X_c^T X_c = V \left( \frac{\Sigma^2}{N-1} \right) V^T$$

Post-multiplying by $V$ (since $V^T V = I$):
$$C V = V \left( \frac{\Sigma^2}{N-1} \right) \implies C v_j = \left( \frac{\sigma_j^2}{N-1} \right) v_j$$

> **The Mathematical Equivalence:**
> 1. The **right singular vectors** $V = [v_1, \dots, v_d]$ of the centered matrix $X_c$ are **identical** to the principal component loading vectors $W = [w_1, \dots, w_d]$.
> 2. The eigenvalues of the covariance matrix are directly related to the singular values:
>    $$\lambda_j = \frac{\sigma_j^2}{N-1}$$
> 3. The low-dimensional projection coordinates are:
>    $$Z = X_c V = (U \Sigma V^T) V = U \Sigma$$

### 4.3 Why SVD is Numerically Superior to $X^T X$
Computing $X_c^T X_c$ explicitly squares the condition number of the matrix:
$$\kappa(X_c^T X_c) = \kappa(X_c)^2$$
If $X_c$ has condition number $10^8$ (common in ill-conditioned datasets), $X_c^T X_c$ has condition number $10^{16}$, causing standard 64-bit floating point arithmetic to completely lose precision. SVD computes $V$ and $\Sigma$ directly on $X_c$ without ever forming $X_c^T X_c$.

---

## 5. Reconstruction Error & The Eckart-Young-Mirsky Theorem

When we project from $d$ dimensions down to $k < d$ principal components, how much information is lost?

### 5.1 The Low-Rank Reconstruction
The projected scores are $Z = X_c W_k \in \mathbb{R}^{N \times k}$. The reconstructed data points $\hat{X}_c \in \mathbb{R}^{N \times d}$ are:
$$\hat{X}_c = Z W_k^T = X_c W_k W_k^T$$

The total reconstruction error (squared Frobenius norm) across the dataset is:
$$\|X_c - \hat{X}_c\|_F^2 = \sum_{i=1}^N \|x_{c,i} - \hat{x}_{c,i}\|_2^2 = (N-1) \sum_{j=k+1}^d \lambda_j = \sum_{j=k+1}^d \sigma_j^2$$

> **Eckart-Young-Mirsky Theorem (1936):** The rank-$k$ truncated SVD approximation $\hat{X}_k = \sum_{j=1}^k \sigma_j u_j v_j^T$ is the **optimal** rank-$k$ matrix approximation in both the spectral norm and Frobenius norm:
> $$\min_{\text{rank}(A) \le k} \|X_c - A\|_F = \|X_c - \hat{X}_k\|_F = \sqrt{\sum_{j=k+1}^d \sigma_j^2}$$

### 5.2 Explained Variance Ratio
To choose the number of components $k$, we evaluate the **Explained Variance Ratio (EVR)**:
$$\text{EVR}_j = \frac{\lambda_j}{\sum_{m=1}^d \lambda_m} = \frac{\sigma_j^2}{\sum_{m=1}^d \sigma_m^2}$$

A common heuristic is to choose $k$ such that the cumulative explained variance reaches a threshold (e.g. 95% or 99%):
$$\sum_{j=1}^k \text{EVR}_j \ge 0.95$$

---

## 6. Summary & Key Takeaways

1. **Maximum Variance = Minimum Reconstruction Error:** Maximizing the variance of projected points is mathematically identical to minimizing the mean squared reconstruction error between original points and projections.
2. **Eigenvectors of Covariance:** The principal axes $w_k$ are the orthogonal eigenvectors of the sample covariance matrix $C = \frac{1}{N-1} X_c^T X_c$, and the eigenvalues $\lambda_k$ represent the variance along each axis.
3. **SVD Equivalence:** The right singular vectors $V$ of the centered data $X_c = U \Sigma V^T$ match the principal components, and $\lambda_k = \frac{\sigma_k^2}{N-1}$.
4. **Limitation:** PCA is strictly a **linear** dimensionality reduction method. If data lies on a curved, non-linear manifold (such as a Swiss Roll or hypersphere surface), PCA fails to capture the intrinsic geometry.

In **Topic 9.3**, we will explore non-linear manifold learning algorithms: **t-SNE** and **UMAP**.
