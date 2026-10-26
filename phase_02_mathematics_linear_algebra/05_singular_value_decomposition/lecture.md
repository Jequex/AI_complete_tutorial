# Topic 2.5: Singular Value Decomposition (SVD)

If Eigendecomposition is a superpower for square matrices, **Singular Value Decomposition (SVD)** is the king of all matrix factorizations. 

Why? Because **every matrix in the universe—tall, wide, square, invertible, or non-invertible—has an SVD**:

$$\boxed{A = U \Sigma V^T}$$

SVD is the mathematical engine powering image compression, recommendation algorithms (like the Netflix Prize), Principal Component Analysis (PCA), pseudoinverses, and modern LLM weight pruning.

---

## 1. 🧒 The 12-Year-Old Analogy: The 3-Step Space Dance

Imagine you have a ball of dough on a table. Any matrix transformation $A$ can be decomposed into **three simple dance steps**:

```text
    STEP 1: ROTATE               STEP 2: STRETCH                 STEP 3: ROTATE
       ( Vᵀ )                       ( Σ )                           ( U )
       
   Spin the dough             Pull & squash into               Spin the ellipse
   without changing           an oval / ellipse by             into its final
   its spherical shape        factors σ₁, σ₂, ...              orientation
```

```text
       Unit Circle              Rotated Circle           Stretched Ellipse            Final Output
            y                         y                          y                         y
            ▲                         ▲                          ▲                         ▲
         ┌──●──┐                   ┌──●──┐                    ┌───●───┐                 /───●───\
         │     │  ───( Vᵀ )───►    │     │   ───( Σ )───►     │       │  ───( U )───►  /         \
         └──●──┘                   └──●──┘                    └───●───┘                \───●───/
            └─────► x                 └─────► x                  └────────► x              └─────► x
```

No matter how complicated a matrix looks, **it is always just: Rotate $\to$ Stretch $\to$ Rotate!**

---

## 2. Deconstructing the Formula: $A = U \Sigma V^T$

Let $A \in \mathbb{R}^{m \times n}$ be any matrix with $m$ rows and $n$ columns. Its SVD is:

$$A = U \Sigma V^T$$

### 1. $U \in \mathbb{R}^{m \times m}$ (Left Singular Vectors)
* An **orthogonal matrix**: $U^T U = I$.
* The columns $\mathbf{u}_1, \dots, \mathbf{u}_m$ are called **Left Singular Vectors**.
* They represent the axes of the stretched ellipse in the output space.
* They are the eigenvectors of $A A^T$!

### 2. $\Sigma \in \mathbb{R}^{m \times n}$ (Singular Values)
* A rectangular diagonal matrix:
  $$\Sigma = \begin{bmatrix} \sigma_1 & 0 & 0 \\ 0 & \sigma_2 & 0 \\ 0 & 0 & \ddots \end{bmatrix}$$
* The numbers $\sigma_1, \sigma_2, \dots$ along the diagonal are called **Singular Values**.
* **Crucial Rule:** They are always real, non-negative, and sorted in descending order:
  $$\sigma_1 \ge \sigma_2 \ge \sigma_3 \ge \dots \ge 0$$
* $\sigma_i$ measures the stretch factor along the $i$-th principal axis.

### 3. $V \in \mathbb{R}^{n \times n}$ (Right Singular Vectors)
* An **orthogonal matrix**: $V^T V = I$.
* The columns $\mathbf{v}_1, \dots, \mathbf{v}_n$ are called **Right Singular Vectors**.
* They are the eigenvectors of $A^T A$!
* $V^T$ represents the first rotation.

---

### 📋 SVD Symbol Decoder Table

| Symbol | Matrix Shape | Mathematical Property | Geometric Meaning |
| :---: | :---: | :--- | :--- |
| $A$ | $m \times n$ | Any real matrix | The overall transformation |
| $U$ | $m \times m$ | Orthogonal ($U^T U = I$) | Output rotation matrix |
| $\Sigma$ | $m \times n$ | Rectangular diagonal | Stretch amounts ($\sigma_1 \ge \sigma_2 \ge \dots \ge 0$) |
| $V^T$ | $n \times n$ | Orthogonal ($V^T V = I$) | Input rotation matrix |
| $\sigma_i$ | Scalar $\ge 0$ | $\sqrt{\lambda_i(A^T A)}$ | Length of the $i$-th principal axis of the ellipse |

---

## 3. How to Compute SVD: The Link to Eigenvalues

Notice what happens when we multiply $A$ by its transpose:

$$A^T A = (U \Sigma V^T)^T (U \Sigma V^T) = V \Sigma^T \underbrace{U^T U}_{I} \Sigma V^T = V (\Sigma^T \Sigma) V^T$$

Look at that formula: $A^T A = V (\Sigma^T \Sigma) V^T$.
This is the exact **Eigendecomposition of $A^T A$**!

1. **To find $V$ and $\sigma_i$:**
   - Compute symmetric matrix $A^T A$.
   - Find its eigenvalues $\lambda_i$ and eigenvectors $\mathbf{v}_i$.
   - The singular values are the square roots: $\sigma_i = \sqrt{\lambda_i}$.
   - The eigenvectors $\mathbf{v}_i$ form the columns of $V$.

2. **To find $U$:**
   - For each non-zero singular value $\sigma_i > 0$:
     $$\mathbf{u}_i = \frac{1}{\sigma_i} A \mathbf{v}_i$$

---

## 4. The Sum of Outer Products View

There is a second, brilliant way to write SVD:

$$\boxed{A = \sum_{i=1}^r \sigma_i \mathbf{u}_i \mathbf{v}_i^T = \sigma_1 \mathbf{u}_1 \mathbf{v}_1^T + \sigma_2 \mathbf{u}_2 \mathbf{v}_2^T + \dots + \sigma_r \mathbf{u}_r \mathbf{v}_r^T}$$

where $r = \text{rank}(A)$.

Notice what each term $\mathbf{u}_i \mathbf{v}_i^T$ is:
- $\mathbf{u}_i \in \mathbb{R}^{m \times 1}$ and $\mathbf{v}_i^T \in \mathbb{R}^{1 \times n}$.
- Their product $\mathbf{u}_i \mathbf{v}_i^T$ is an **$m \times n$ matrix of rank 1**!
- SVD decomposes **any big matrix into a weighted sum of simple rank-1 layers**, where the weights are the singular values $\sigma_i$!

---

## 5. Truncated SVD & The Eckart-Young Theorem (Data Compression)

Because the singular values are sorted ($\sigma_1 \ge \sigma_2 \ge \dots$), the first few terms contain **almost all the energy and information** of the matrix!

If we keep only the top $k$ terms:

$$A_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^T \approx A$$

### 🔬 The Eckart-Young-Mirsky Theorem
> **Theorem:** For any matrix $A$, the truncated SVD matrix $A_k$ is the **mathematically optimal rank-$k$ approximation** that minimizes the reconstruction error in Frobenius norm:
> $$\min_{\text{rank}(B) \le k} \|A - B\|_F = \|A - A_k\|_F = \sqrt{\sum_{j=k+1}^r \sigma_j^2}$$

### 🧒 Real AI Application: Image & Weight Compression
Suppose you have a high-resolution image of size $1000 \times 1000$ pixels ($1,000,000$ numbers).
- If you compute SVD and keep only the top $k = 20$ singular values:
  - $U_k$: $1000 \times 20 = 20,000$ numbers
  - $\Sigma_k$: $20$ numbers
  - $V_k^T$: $20 \times 1000 = 20,000$ numbers
  - **Total stored:** $20,000 + 20 + 20,000 = \mathbf{40,020} \text{ numbers}$.
- That is a **$96\%$ storage savings**, yet the reconstructed image $A_{20}$ retains almost all visible features!

---

## 6. The Moore-Penrose Pseudoinverse: $A^+$

What if a matrix $A$ is not square, or has $\det(A) = 0$? How do we "invert" it to solve $A\mathbf{x} = \mathbf{b}$?

Using SVD, we construct the **Pseudoinverse** $A^+$:

$$\boxed{A^+ = V \Sigma^+ U^T}$$

where $\Sigma^+$ is formed by inverting every non-zero singular value: $\sigma_i \to \frac{1}{\sigma_i}$, and taking the transpose.

The solution:
$$\hat{\mathbf{x}} = A^+ \mathbf{b}$$
is the famous **minimum-norm least-squares solution** used in linear regression!

---

## 📌 Summary Checklist for Topic 2.5

- [x] **SVD ($A = U \Sigma V^T$)** exists for **every matrix**, regardless of shape.
- [x] Geometric meaning: **Rotate ($V^T$) $\to$ Stretch ($\Sigma$) $\to$ Rotate ($U$)**.
- [x] Singular values $\sigma_i = \sqrt{\lambda_i(A^T A)}$ are non-negative and sorted in descending order ($\sigma_1 \ge \sigma_2 \ge \dots \ge 0$).
- [x] **Sum of Rank-1 Matrices:** $A = \sum_{i=1}^r \sigma_i \mathbf{u}_i \mathbf{v}_i^T$.
- [x] **Eckart-Young Theorem:** Truncating SVD to rank $k$ provides the mathematically optimal low-rank compression.
- [x] **Pseudoinverse ($A^+ = V \Sigma^+ U^T$)** solves linear systems even when $A$ is singular or rectangular.

---

Next up: **Run `experiment.py` in this folder** to compress an ASCII matrix image from scratch using SVD!

