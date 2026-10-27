# Topic 01: Spectral Theory & SVD Geometry

## 1. Introduction & 12-Year-Old Intuition

Imagine you have a perfectly round, flexible rubber ball in your hands. This ball represents the unit sphere in geometry—every point on its skin is exactly 1 unit of distance away from the center.

Now, imagine taking this rubber ball and stretching it, twisting it, and placing it onto a table. What shape does it become? 
It turns into a smooth, stretched **ellipse** (or an oval in 2D, and an **ellipsoid** in higher dimensions).

No matter how complicated, lopsided, or strange the transformation is, three fundamental geometric facts always remain true:
1. **The Straight Pins (Singular Vectors)**: Inside the original ball, there was an orderly set of perpendicular needles pointing straight out from the center (like the $x$, $y$, and $z$ axes).
2. **The Stretch Factors (Singular Values)**: When the ball was stretched into an egg, those needles got stretched or squished by specific numbers: the longest axis was stretched by $\sigma_1$, the next by $\sigma_2$, and the shortest by $\sigma_3$.
3. **The Final Alignment (Output Directions)**: The stretched needles ended up pointing in new perpendicular directions in the room.

This is the entire magic of the **Singular Value Decomposition (SVD)**:
$$\mathbf{A} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^\top$$

Any linear transformation $\mathbf{A}$—no matter how giant, rectangular, or complex—can be broken down into three simple, intuitive actions:
1. **Rotate** the input space ($\mathbf{V}^\top$).
2. **Stretch or squash** along the coordinate axes by factors $\sigma_i$ ($\boldsymbol{\Sigma}$).
3. **Rotate** into the output space ($\mathbf{U}$).

In modern deep learning and AI research, high-dimensional matrices represent everything from weight matrices in Large Language Models (LLMs) to covariance matrices in diffusion models and representations in mechanistic interpretability. Understanding the geometry of SVD allows researchers to compress 70-billion parameter models using Low-Rank Adaptation (LoRA), diagnose vanishing and exploding gradients via condition numbers, and uncover low-dimensional manifolds in neural latent spaces.

---

## 2. Mathematical Symbol Decoder

| Symbol | Mathematical Entity | Dimensionality / Domain | Operational & Research Role |
| :--- | :--- | :--- | :--- |
| $\mathbf{A}$ | Arbitrary Real Matrix | $\mathbb{R}^{m \times n}$ | Linear operator mapping input vectors $\mathbf{x} \in \mathbb{R}^n$ to $\mathbf{y} \in \mathbb{R}^m$. |
| $\mathbf{S}$ | Symmetric Real Matrix | $\mathbb{R}^{n \times n}, \mathbf{S} = \mathbf{S}^\top$ | Self-adjoint operator (e.g., Hessian $\nabla^2 f$, covariance $\mathbf{X}^\top \mathbf{X}$, Fisher matrix). |
| $\lambda_i$ | Eigenvalue | $\mathbb{R}$ | Scaling factor along invariant direction $\mathbf{q}_i$: $\mathbf{S}\mathbf{q}_i = \lambda_i \mathbf{q}_i$. |
| $\mathbf{q}_i$ | Eigenvector | $\mathbb{R}^n, \|\mathbf{q}_i\|_2 = 1$ | Invariant geometric direction whose spatial orientation is preserved under $\mathbf{S}$. |
| $\mathbf{Q}$ | Orthogonal Matrix | $\mathbb{R}^{n \times n}, \mathbf{Q}^\top \mathbf{Q} = \mathbf{I}$ | Rigid rotation / reflection matrix preserving Euclidean lengths and angles. |
| $\boldsymbol{\Lambda}$ | Diagonal Eigenvalue Matrix | $\mathbb{R}^{n \times n}$ | $\text{diag}(\lambda_1, \dots, \lambda_n)$, ordered $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_n$. |
| $\sigma_i$ | Singular Value | $\mathbb{R}_{\ge 0}$ | Non-negative square root of eigenvalue of $\mathbf{A}^\top \mathbf{A}$: $\sigma_i = \sqrt{\lambda_i(\mathbf{A}^\top \mathbf{A})}$. |
| $\mathbf{u}_i$ | Left Singular Vector | $\mathbb{R}^m, \|\mathbf{u}_i\|_2 = 1$ | $i$-th column of $\mathbf{U}$; principal axis direction of the output hyperellipsoid. |
| $\mathbf{v}_i$ | Right Singular Vector | $\mathbb{R}^n, \|\mathbf{v}_i\|_2 = 1$ | $i$-th column of $\mathbf{V}$; orthonormal domain vector mapped to $\sigma_i \mathbf{u}_i$. |
| $\boldsymbol{\Sigma}$ | Diagonal Singular Matrix | $\mathbb{R}^{m \times n}$ | Rectangular diagonal matrix containing singular values $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$. |
| $\mathbf{A}_k$ | Truncated Rank-$k$ Matrix | $\mathbb{R}^{m \times n}$ | Best rank-$k$ approximation: $\sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^\top$. |
| $\|\mathbf{A}\|_F$ | Frobenius Norm | $\mathbb{R}_{\ge 0}$ | Elementwise Euclidean norm: $\sqrt{\sum_{i,j} A_{ij}^2} = \sqrt{\sum_i \sigma_i^2}$. |
| $\|\mathbf{A}\|_2$ | Spectral (Operator) Norm | $\mathbb{R}_{\ge 0}$ | Maximum magnification factor: $\sup_{\mathbf{x} \neq \mathbf{0}} \frac{\|\mathbf{A}\mathbf{x}\|_2}{\|\mathbf{x}\|_2} = \sigma_1$. |
| $\kappa(\mathbf{A})$ | Condition Number | $[1, \infty]$ | Ratio of maximum to minimum singular value: $\sigma_{\max} / \sigma_{\min}$. |

---

## 3. The Spectral Theorem for Real Symmetric Matrices

### 3.1 Theorem Statement
Let $\mathbf{S} \in \mathbb{R}^{n \times n}$ be a symmetric matrix ($\mathbf{S}^\top = \mathbf{S}$). Then:
1. All eigenvalues $\lambda_1, \dots, \lambda_n$ of $\mathbf{S}$ are strictly real: $\lambda_i \in \mathbb{R}$.
2. Eigenvectors corresponding to distinct eigenvalues are mutually orthogonal.
3. There exists an orthonormal basis of $\mathbb{R}^n$ consisting of eigenvectors of $\mathbf{S}$.
4. $\mathbf{S}$ admits the **spectral decomposition**:
   $$\mathbf{S} = \mathbf{Q} \boldsymbol{\Lambda} \mathbf{Q}^\top = \sum_{i=1}^n \lambda_i \mathbf{q}_i \mathbf{q}_i^\top$$
   where $\mathbf{Q} = [\mathbf{q}_1, \dots, \mathbf{q}_n]$ is orthogonal ($\mathbf{Q}^\top \mathbf{Q} = \mathbf{I}$).

### 3.2 Step-by-Step Proof
#### Step 1: Real Eigenvalues
Let $\lambda \in \mathbb{C}$ be an eigenvalue of $\mathbf{S}$ with non-zero eigenvector $\mathbf{x} \in \mathbb{C}^n$, so $\mathbf{S}\mathbf{x} = \lambda \mathbf{x}$.
Taking the conjugate transpose (Hermitian adjoint $\dagger$):
$$\mathbf{x}^\dagger \mathbf{S} \mathbf{x} = \mathbf{x}^\dagger (\lambda \mathbf{x}) = \lambda (\mathbf{x}^\dagger \mathbf{x}) = \lambda \|\mathbf{x}\|_2^2$$
Now take the conjugate transpose of the left-hand side, noting that $\mathbf{S}$ is real symmetric ($\mathbf{S}^\dagger = \mathbf{S}^\top = \mathbf{S}$):
$$(\mathbf{x}^\dagger \mathbf{S} \mathbf{x})^\dagger = \mathbf{x}^\dagger \mathbf{S}^\dagger \mathbf{x} = \mathbf{x}^\dagger \mathbf{S} \mathbf{x}$$
Therefore, $\mathbf{x}^\dagger \mathbf{S} \mathbf{x}$ is a real scalar. Since $\mathbf{x} \neq \mathbf{0}$, $\|\mathbf{x}\|_2^2 > 0$ is a positive real scalar. Thus:
$$\lambda = \frac{\mathbf{x}^\dagger \mathbf{S} \mathbf{x}}{\|\mathbf{x}\|_2^2} \in \mathbb{R}$$
All eigenvalues of a real symmetric matrix are real.

#### Step 2: Orthogonality of Eigenvectors
Let $\mathbf{S}\mathbf{q}_1 = \lambda_1 \mathbf{q}_1$ and $\mathbf{S}\mathbf{q}_2 = \lambda_2 \mathbf{q}_2$ with $\lambda_1 \neq \lambda_2$.
Compute the scalar $\mathbf{q}_1^\top \mathbf{S} \mathbf{q}_2$ in two ways:
$$\mathbf{q}_1^\top (\mathbf{S} \mathbf{q}_2) = \mathbf{q}_1^\top (\lambda_2 \mathbf{q}_2) = \lambda_2 (\mathbf{q}_1^\top \mathbf{q}_2)$$
$$(\mathbf{q}_1^\top \mathbf{S}) \mathbf{q}_2 = (\mathbf{S}^\top \mathbf{q}_1)^\top \mathbf{q}_2 = (\mathbf{S} \mathbf{q}_1)^\top \mathbf{q}_2 = (\lambda_1 \mathbf{q}_1)^\top \mathbf{q}_2 = \lambda_1 (\mathbf{q}_1^\top \mathbf{q}_2)$$
Subtracting the two equations:
$$(\lambda_1 - \lambda_2)(\mathbf{q}_1^\top \mathbf{q}_2) = 0$$
Since $\lambda_1 \neq \lambda_2$, we must have $\mathbf{q}_1^\top \mathbf{q}_2 = 0$. Hence, eigenvectors of distinct eigenvalues are orthogonal. $\blacksquare$

---

## 4. Singular Value Decomposition (SVD): Geometry & Mechanics

While the Spectral Theorem applies only to square symmetric matrices, **SVD applies to any arbitrary rectangular matrix** $\mathbf{A} \in \mathbb{R}^{m \times n}$.

```
                 Singular Value Decomposition: A = U Σ V^T
                 
  Domain (R^n)                 Diagonal Scaling (R^r)               Codomain (R^m)
  Unit Hypersphere             Stretching by σ_i                   Hyperellipsoid
  
      ▲ v_2                         ▲ σ_2 u_2                          ▲ u_2
      │                             │                                  │     /───────\
   ───┼───▶ v_1   ───▶ V^T ───▶   ──┼──────▶ σ_1 u_1   ───▶ U ───▶   ──┼────│  A(x)  │
      │                             │                                  │     \_______/
      ▼                             ▼                                  ▼              ▶ u_1
  ||x||_2 = 1                  Principal axes:                     Image is an ellipsoid
                               Semi-axes lengths σ_i               with axes σ_i u_i
```

### 4.1 Construction via Normal Matrices $\mathbf{A}^\top \mathbf{A}$ and $\mathbf{A}\mathbf{A}^\top$
Consider the Gram matrix $\mathbf{S} = \mathbf{A}^\top \mathbf{A} \in \mathbb{R}^{n \times n}$.
1. $\mathbf{S}$ is symmetric: $(\mathbf{A}^\top \mathbf{A})^\top = \mathbf{A}^\top \mathbf{A}$.
2. $\mathbf{S}$ is positive semi-definite: for any $\mathbf{x} \in \mathbb{R}^n$:
   $$\mathbf{x}^\top (\mathbf{A}^\top \mathbf{A}) \mathbf{x} = (\mathbf{A}\mathbf{x})^\top (\mathbf{A}\mathbf{x}) = \|\mathbf{A}\mathbf{x}\|_2^2 \ge 0$$
3. Therefore, by the Spectral Theorem, all eigenvalues $\lambda_i \ge 0$ are non-negative real numbers.
4. Define the singular values as the non-negative square roots:
   $$\sigma_i \triangleq \sqrt{\lambda_i(\mathbf{A}^\top \mathbf{A})}, \quad \sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0$$
5. The orthonormal eigenvectors of $\mathbf{A}^\top \mathbf{A}$ form the **right singular vectors** $\mathbf{V} = [\mathbf{v}_1, \dots, \mathbf{v}_n]$.
6. For each non-zero singular value $\sigma_i > 0$, define the **left singular vector**:
   $$\mathbf{u}_i \triangleq \frac{1}{\sigma_i} \mathbf{A} \mathbf{v}_i \in \mathbb{R}^m$$

### 4.2 Orthonormality of Left Singular Vectors $\mathbf{u}_i$
Let us verify that $\mathbf{u}_i$ are indeed orthonormal:
$$\mathbf{u}_i^\top \mathbf{u}_j = \left(\frac{1}{\sigma_i} \mathbf{A} \mathbf{v}_i\right)^\top \left(\frac{1}{\sigma_j} \mathbf{A} \mathbf{v}_j\right) = \frac{1}{\sigma_i \sigma_j} \mathbf{v}_i^\top (\mathbf{A}^\top \mathbf{A} \mathbf{v}_j)$$
Since $\mathbf{v}_j$ is an eigenvector of $\mathbf{A}^\top \mathbf{A}$ with eigenvalue $\sigma_j^2$:
$$\mathbf{u}_i^\top \mathbf{u}_j = \frac{1}{\sigma_i \sigma_j} \mathbf{v}_i^\top (\sigma_j^2 \mathbf{v}_j) = \frac{\sigma_j}{\sigma_i} (\mathbf{v}_i^\top \mathbf{v}_j) = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases}$$
Hence, $\mathbf{U} = [\mathbf{u}_1, \dots, \mathbf{u}_r]$ is an orthonormal set in $\mathbb{R}^m$.

---

## 5. Low-Rank Matrix Approximation & Eckart-Young-Mirsky Theorem

In modern deep learning, models are overparameterized, yet their weight matrices exhibit low intrinsic rank. The **Eckart-Young-Mirsky Theorem** provides the fundamental mathematical guarantee behind LoRA (Low-Rank Adaptation) and matrix compression.

### 5.1 Theorem Statement
Let $\mathbf{A} \in \mathbb{R}^{m \times n}$ have singular value decomposition $\mathbf{A} = \sum_{i=1}^r \sigma_i \mathbf{u}_i \mathbf{v}_i^\top$ with $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0$.
For any integer $k < r$, define the truncated rank-$k$ SVD approximation:
$$\mathbf{A}_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^\top$$
Then for any matrix $\mathbf{B} \in \mathbb{R}^{m \times n}$ with $\text{rank}(\mathbf{B}) \le k$:
1. **Frobenius Norm Optimality**:
   $$\min_{\text{rank}(\mathbf{B}) \le k} \|\mathbf{A} - \mathbf{B}\|_F = \|\mathbf{A} - \mathbf{A}_k\|_F = \sqrt{\sum_{i=k+1}^r \sigma_i^2}$$
2. **Spectral Norm Optimality**:
   $$\min_{\text{rank}(\mathbf{B}) \le k} \|\mathbf{A} - \mathbf{B}\|_2 = \|\mathbf{A} - \mathbf{A}_k\|_2 = \sigma_{k+1}$$

### 5.2 Proof for Spectral Norm
1. For $\mathbf{A}_k$, the error matrix is $\mathbf{A} - \mathbf{A}_k = \sum_{i=k+1}^r \sigma_i \mathbf{u}_i \mathbf{v}_i^\top$. The maximum singular value of this residual is clearly $\sigma_{k+1}$, so $\|\mathbf{A} - \mathbf{A}_k\|_2 = \sigma_{k+1}$.
2. Now let $\mathbf{B}$ be an arbitrary matrix with $\text{rank}(\mathbf{B}) \le k$.
3. The null space $\text{null}(\mathbf{B})$ has dimension $\dim(\text{null}(\mathbf{B})) = n - \text{rank}(\mathbf{B}) \ge n - k$.
4. Consider the subspace $\mathcal{V}_{k+1} = \text{span}\{\mathbf{v}_1, \dots, \mathbf{v}_{k+1}\} \subset \mathbb{R}^n$, which has dimension $k+1$.
5. By the dimension formula for intersecting subspaces:
   $$\dim(\text{null}(\mathbf{B}) \cap \mathcal{V}_{k+1}) \ge (n - k) + (k + 1) - n = 1$$
6. Therefore, there exists a non-zero unit vector $\mathbf{z} \in \text{null}(\mathbf{B}) \cap \mathcal{V}_{k+1}$ with $\|\mathbf{z}\|_2 = 1$.
7. Since $\mathbf{z} \in \text{null}(\mathbf{B})$, $\mathbf{B}\mathbf{z} = \mathbf{0}$.
8. Since $\mathbf{z} \in \mathcal{V}_{k+1}$, we can write $\mathbf{z} = \sum_{i=1}^{k+1} c_i \mathbf{v}_i$ with $\sum_{i=1}^{k+1} c_i^2 = 1$.
9. Now compute:
   $$\|(\mathbf{A} - \mathbf{B})\mathbf{z}\|_2^2 = \|\mathbf{A}\mathbf{z}\|_2^2 = \left\| \sum_{i=1}^{k+1} c_i \sigma_i \mathbf{u}_i \right\|_2^2 = \sum_{i=1}^{k+1} c_i^2 \sigma_i^2 \ge \sigma_{k+1}^2 \sum_{i=1}^{k+1} c_i^2 = \sigma_{k+1}^2$$
10. Taking the square root:
    $$\|\mathbf{A} - \mathbf{B}\|_2 \ge \|(\mathbf{A} - \mathbf{B})\mathbf{z}\|_2 \ge \sigma_{k+1}$$
This proves that no rank-$k$ matrix can have a smaller spectral error than $\mathbf{A}_k$. $\blacksquare$

---

## 6. Condition Numbers & Hypersphere Deformation

### 6.1 The Geometry of Deformation
When a matrix $\mathbf{A}$ acts on the unit sphere $\mathcal{S}^{n-1} = \{\mathbf{x} \in \mathbb{R}^n : \|\mathbf{x}\|_2 = 1\}$, the image is an ellipsoid:
$$\mathcal{E} = \{\mathbf{y} = \mathbf{A}\mathbf{x} : \|\mathbf{x}\|_2 = 1\}$$
- The principal axes of this ellipsoid point along the left singular vectors $\mathbf{u}_i$.
- The semi-axis lengths are precisely the singular values $\sigma_i$.
- The longest radius is $\sigma_{\max} = \sigma_1 = \|\mathbf{A}\|_2$.
- The shortest non-zero radius is $\sigma_{\min} = \sigma_r$.

```
                       Condition Number & Distortion
                       
     Well-Conditioned (κ ≈ 1)                  Ill-Conditioned (κ >> 1)
     Isotropic hypersphere                     Flattened pancake / needle
     
             ┌───┐                                      ┌───────────────┐
            │     │                                    │                 │
             └───┘                                      └───────────────┘
          σ_max ≈ σ_min                              σ_max >> σ_min
     Gradients point to minimum               Gradients oscillate across canyon
```

### 6.2 Condition Number and Optimization
The **condition number** of $\mathbf{A}$ is:
$$\kappa(\mathbf{A}) \triangleq \frac{\sigma_{\max}(\mathbf{A})}{\sigma_{\min}(\mathbf{A})}$$
- **Numerical Sensitivity**: If solving $\mathbf{A}\mathbf{x} = \mathbf{b}$, a relative perturbation $\frac{\|\delta \mathbf{b}\|}{\|\mathbf{b}\|}$ in the input can be magnified by up to $\kappa(\mathbf{A})$ in the solution:
  $$\frac{\|\delta \mathbf{x}\|}{\|\mathbf{x}\|} \le \kappa(\mathbf{A}) \frac{\|\delta \mathbf{b}\|}{\|\mathbf{b}\|}$$
- **Neural Network Training Dynamics**: For gradient descent on quadratic loss $L(\mathbf{w}) = \frac{1}{2} \mathbf{w}^\top \mathbf{H} \mathbf{w}$, the convergence rate is governed by $\kappa(\mathbf{H})$:
  $$\|\mathbf{w}_t - \mathbf{w}^*\| \le \left( \frac{\kappa(\mathbf{H}) - 1}{\kappa(\mathbf{H}) + 1} \right)^t \|\mathbf{w}_0 - \mathbf{w}^*\|$$
  When $\kappa \gg 1$, the factor approaches $1 - \frac{2}{\kappa}$, causing severe gradient oscillation and vanishingly slow progress along the flat directions.

---

## 7. Deep Learning Application: Low-Rank Adaptation (LoRA)

In parameter-efficient fine-tuning (Hu et al., 2021), a pre-trained weight matrix $\mathbf{W}_0 \in \mathbb{R}^{d \times k}$ is frozen, and updates are parameterized as:
$$\mathbf{W} = \mathbf{W}_0 + \Delta \mathbf{W} = \mathbf{W}_0 + \frac{\alpha}{r} \mathbf{B} \mathbf{A}$$
where $\mathbf{B} \in \mathbb{R}^{d \times r}$ and $\mathbf{A} \in \mathbb{R}^{r \times k}$ with rank $r \ll \min(d, k)$.

By the Eckart-Young-Mirsky Theorem, if the true task update $\Delta \mathbf{W}^*$ has singular values that decay exponentially (which is universally observed in deep Transformer representations), a low-rank factorization captures $\approx 90\text{--}99\%$ of the Frobenius energy while saving $99.9\%$ of trainable parameters.
