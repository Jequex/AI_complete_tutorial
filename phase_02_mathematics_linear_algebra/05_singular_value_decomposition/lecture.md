# Topic 2.5: Singular Value Decomposition (SVD)

If Eigendecomposition is a superpower for square matrices, **Singular Value Decomposition (SVD)** is the undisputed king of all matrix factorizations. 

Why? Because **every matrix in the universe—tall, wide, square, invertible, or non-invertible—has an SVD**:

$$\boxed{A = U \Sigma V^T}$$

SVD is the mathematical engine powering image compression, recommendation systems (like the Netflix Prize), Principal Component Analysis (PCA), pseudoinverses, and modern LLM weight pruning.

---

## 1. 🧒 The 12-Year-Old Analogy: The 3-Step Space Dance

Imagine you have a ball of dough sitting on a kitchen table. Any matrix transformation $A$—no matter how messy it looks—can be broken down into **three simple dance steps**:

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

Let $A \in \mathbb{R}^{m \times n}$ be any real matrix with $m$ rows and $n$ columns:

$$A = U \Sigma V^T$$

Let us break down each term:

$$\underbrace{A}_{\substack{\text{Original Matrix} \\ (m \times n)}} = \underbrace{U}_{\substack{\text{Left Singular Vectors} \\ \text{Output Rotation Matrix } (m \times m)}} \times \underbrace{\Sigma}_{\substack{\text{Singular Values} \\ \text{Stretch Diagonal Matrix } (m \times n)}} \times \underbrace{V^T}_{\substack{\text{Right Singular Vectors} \\ \text{Input Rotation Matrix } (n \times n)}}$$

### 1. $U \in \mathbb{R}^{m \times m}$ (Left Singular Vectors)
* An **orthogonal matrix**: $U^T U = I$.
* The columns $\mathbf{u}_1, \dots, \mathbf{u}_m$ are called **Left Singular Vectors**.
* They represent the principal axes of the stretched ellipse in the output space.
* They are the eigenvectors of $A A^T$!

### 2. $\Sigma \in \mathbb{R}^{m \times n}$ (Singular Values)
* A rectangular diagonal matrix holding non-negative stretch factors:
  $$\Sigma = \begin{bmatrix} \sigma_1 & 0 & 0 \\ 0 & \sigma_2 & 0 \\ 0 & 0 & \ddots \end{bmatrix}$$
* **Crucial Rule:** The singular values are always real, non-negative, and sorted in descending order:
  $$\sigma_1 \ge \sigma_2 \ge \sigma_3 \ge \dots \ge 0$$
* $\sigma_i$ measures the stretch factor along the $i$-th principal axis.

### 3. $V \in \mathbb{R}^{n \times n}$ (Right Singular Vectors)
* An **orthogonal matrix**: $V^T V = I$.
* The columns $\mathbf{v}_1, \dots, \mathbf{v}_n$ are called **Right Singular Vectors**.
* They are the eigenvectors of $A^T A$!

---

## 3. Concrete Numerical Walkthrough by Hand

Let us compute the SVD of the symmetric matrix:
$$A = \begin{bmatrix} 3 & 2 \\ 2 & 3 \end{bmatrix}$$

### Step 1: Compute $A^T A$
$$A^T A = \begin{bmatrix} 3 & 2 \\ 2 & 3 \end{bmatrix} \begin{bmatrix} 3 & 2 \\ 2 & 3 \end{bmatrix} = \begin{bmatrix} (9+4) & (6+6) \\ (6+6) & (4+9) \end{bmatrix} = \begin{bmatrix} 13 & 12 \\ 12 & 13 \end{bmatrix}$$

### Step 2: Find Eigenvalues of $A^T A$
$$\det(A^T A - \lambda I) = (13 - \lambda)^2 - 12^2 = 0$$
$$(13 - \lambda)^2 - 144 = 0 \implies 13 - \lambda = \pm 12$$
$$\lambda_1 = 13 + 12 = \mathbf{25}, \qquad \lambda_2 = 13 - 12 = \mathbf{1}$$

### Step 3: Compute Singular Values $\sigma_i$
The singular values are the square roots of the eigenvalues of $A^T A$:
$$\sigma_1 = \sqrt{\lambda_1} = \sqrt{25} = \mathbf{5.0}$$
$$\sigma_2 = \sqrt{\lambda_2} = \sqrt{1} = \mathbf{1.0}$$
$$\Sigma = \begin{bmatrix} 5.0 & 0 \\ 0 & 1.0 \end{bmatrix}$$

### Step 4: Find Right Singular Vectors $V$
- For $\lambda_1 = 25$: $(A^T A - 25I)\mathbf{v} = \begin{bmatrix} -12 & 12 \\ 12 & -12 \end{bmatrix}\mathbf{v} = \mathbf{0} \implies \mathbf{v}_1 = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 1 \end{bmatrix}$
- For $\lambda_2 = 1$: $(A^T A - 1I)\mathbf{v} = \begin{bmatrix} 12 & 12 \\ 12 & 12 \end{bmatrix}\mathbf{v} = \mathbf{0} \implies \mathbf{v}_2 = \frac{1}{\sqrt{2}} \begin{bmatrix} -1 \\ 1 \end{bmatrix}$
$$V = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & -1 \\ 1 & 1 \end{bmatrix}$$

### Step 5: Find Left Singular Vectors $U$
Using $\mathbf{u}_i = \frac{1}{\sigma_i} A \mathbf{v}_i$:
$$\mathbf{u}_1 = \frac{1}{5} \begin{bmatrix} 3 & 2 \\ 2 & 3 \end{bmatrix} \left(\frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 1 \end{bmatrix}\right) = \frac{1}{5\sqrt{2}} \begin{bmatrix} 5 \\ 5 \end{bmatrix} = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 1 \end{bmatrix}$$
$$\mathbf{u}_2 = \frac{1}{1} \begin{bmatrix} 3 & 2 \\ 2 & 3 \end{bmatrix} \left(\frac{1}{\sqrt{2}} \begin{bmatrix} -1 \\ 1 \end{bmatrix}\right) = \frac{1}{\sqrt{2}} \begin{bmatrix} -1 \\ 1 \end{bmatrix}$$
$$U = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & -1 \\ 1 & 1 \end{bmatrix}$$

We have derived the complete SVD: $A = U \Sigma V^T$!

---

## 4. The Sum of Outer Products View (The Cellophane Sheets)

### 🧒 The 12-Year-Old Analogy: Transparent Cellophane Layers
Imagine you want to paint a masterpiece portrait:
- Instead of painting all details on one canvas, you stack sheets of transparent plastic (cellophane).
- **Sheet 1 ($\sigma_1 \mathbf{u}_1 \mathbf{v}_1^T$):** The broad strokes (the main outline of the face). It carries $\sigma_1 = 5.0$ worth of information!
- **Sheet 2 ($\sigma_2 \mathbf{u}_2 \mathbf{v}_2^T$):** The fine brushstrokes (eyelashes, wrinkles). It carries only $\sigma_2 = 1.0$ worth of information.

$$A = \sum_{i=1}^r \sigma_i \mathbf{u}_i \mathbf{v}_i^T = \sigma_1 \mathbf{u}_1 \mathbf{v}_1^T + \sigma_2 \mathbf{u}_2 \mathbf{v}_2^T + \dots + \sigma_r \mathbf{u}_r \mathbf{v}_r^T$$

$$\underbrace{A}_{\text{Full Matrix}} = \sum_{i=1}^r \underbrace{\sigma_i}_{\text{Importance weight}} \times \underbrace{\left( \mathbf{u}_i \mathbf{v}_i^T \right)}_{\substack{\text{Simple rank-1 matrix} \\ (m \times n)}}$$

**🧒 What the math is actually doing:**  
SVD proves that every complex matrix is just a sum of simple rank-1 layers weighted by how much information ($\sigma_i$) they contribute!

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

where $\Sigma^+$ is formed by inverting every non-zero singular value ($\sigma_i \to \frac{1}{\sigma_i}$) and taking the transpose.

The solution:
$$\hat{\mathbf{x}} = A^+ \mathbf{b}$$
is the famous **minimum-norm least-squares solution** used in linear regression!

---

## 7. 📋 Mathematical & Technical Symbol Decoder Table

| Symbol | Pronunciation | Plain English Meaning | Concrete Real-World Analogy / Flashcard Example |
| :---: | :---: | :--- | :--- |
| $A = U \Sigma V^T$ | "A equals U Sigma V transpose" | The Singular Value Decomposition of any matrix $A$ | Rotate $\to$ Stretch $\to$ Rotate space dance |
| $U$ | "U matrix (left singular vectors)" | Orthogonal matrix containing output rotation axes | The final orientation spin of the stretched ellipse |
| $\Sigma$ | "capital Sigma" | Rectangular diagonal matrix containing singular values | The stretch multipliers along each principal axis |
| $V^T$ | "V transpose (right singular vectors)"| Orthogonal matrix containing input rotation axes | The initial spin of the dough ball before stretching |
| $\sigma_i$ | "sigma-i" | The $i$-th singular value ($\sigma_i = \sqrt{\lambda_i(A^T A)}$) | The brightness/thickness of the $i$-th cellophane sheet |
| $A_k$ | "A sub k (rank-k truncated SVD)" | The optimal rank-$k$ approximation of matrix $A$ | Compressing an image by keeping only the top 20 color layers |
| $A^+$ | "A-plus / A-dagger / pseudoinverse"| Moore-Penrose generalized inverse | An undo button that works even on rectangular or collapsed matrices |
| $\|A\|_F$ | "Frobenius norm of A" | The square root of the sum of squared matrix entries | The total geometric energy or pixel weight of a matrix |

---

## 8. 📌 Summary & Mathematical Checklist

- [x] **SVD ($A = U \Sigma V^T$)** exists for **every matrix in the universe**, regardless of shape.
- [x] Geometric meaning: **Rotate ($V^T$) $\to$ Stretch ($\Sigma$) $\to$ Rotate ($U$)**.
- [x] Singular values $\sigma_i = \sqrt{\lambda_i(A^T A)}$ are non-negative and sorted in descending order ($\sigma_1 \ge \sigma_2 \ge \dots \ge 0$).
- [x] **Sum of Rank-1 Layers:** $A = \sum_{i=1}^r \sigma_i \mathbf{u}_i \mathbf{v}_i^T$.
- [x] **Eckart-Young Theorem:** Truncated SVD provides the provably optimal rank-$k$ approximation.
- [x] **Pseudoinverse ($A^+ = V \Sigma^+ U^T$)** solves linear regression and inverse problems when $A$ is singular.

---

Next up: **Run `python3 experiment.py`** in this folder to compress an ASCII matrix image from scratch using SVD!
