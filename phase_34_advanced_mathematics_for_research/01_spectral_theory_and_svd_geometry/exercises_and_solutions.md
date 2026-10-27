# Topic 01: Spectral Theory & SVD Geometry — Exercises & Solutions

## Exercise 1: Low-Rank Approximation and LoRA Energy Budget

In fine-tuning a Large Language Model with $d_{\text{model}} = 4096$, the weight update matrix $\Delta \mathbf{W} \in \mathbb{R}^{4096 \times 4096}$ has empirical singular spectrum decaying exponentially:
$$\sigma_i = 10 \cdot 2^{-(i-1)/8}, \quad i = 1, 2, \dots, 4096$$

1. Formulate the Frobenius norm $\|\Delta \mathbf{W}\|_F^2$ as an infinite geometric series and compute its exact closed-form value.
2. Under the Eckart-Young-Mirsky Theorem, what is the exact relative Frobenius approximation error $\frac{\|\Delta \mathbf{W} - \Delta \mathbf{W}_r\|_F^2}{\|\Delta \mathbf{W}\|_F^2}$ achieved by a LoRA rank of $r = 16$?
3. Compute what percentage of the Frobenius energy is retained with $r = 16$.

### Solution 1:
1. **Frobenius Energy Calculation**:
   $$\|\Delta \mathbf{W}\|_F^2 = \sum_{i=1}^{4096} \sigma_i^2 = \sum_{i=1}^{4096} 100 \cdot \left(2^{-(i-1)/8}\right)^2 = 100 \sum_{i=1}^{4096} 2^{-(i-1)/4}$$
   Let $q = 2^{-1/4} = 0.840896$. Since $4096$ is large, $q^{4096} \approx 0$:
   $$\|\Delta \mathbf{W}\|_F^2 \approx \frac{100}{1 - 2^{-1/4}} = \frac{100}{1 - 0.840896} \approx \frac{100}{0.159104} \approx 628.52$$

2. **Residual Energy for Rank $r = 16$**:
   By the Eckart-Young Theorem:
   $$\|\Delta \mathbf{W} - \Delta \mathbf{W}_{16}\|_F^2 = \sum_{i=17}^{4096} \sigma_i^2 = 100 \sum_{i=17}^{4096} 2^{-(i-1)/4}$$
   Factoring out the term at $i = 17$, where $2^{-(16)/4} = 2^{-4} = \frac{1}{16}$:
   $$\|\Delta \mathbf{W} - \Delta \mathbf{W}_{16}\|_F^2 = 2^{-4} \sum_{k=1}^{4080} 100 \cdot 2^{-(k-1)/4} = \frac{1}{16} \|\Delta \mathbf{W}\|_F^2$$
   Therefore, the relative squared Frobenius error is:
   $$\frac{\|\Delta \mathbf{W} - \Delta \mathbf{W}_{16}\|_F^2}{\|\Delta \mathbf{W}\|_F^2} = \frac{1}{16} = 0.0625 = \mathbf{6.25\%}$$

3. **Retained Energy**:
   $$\text{Retained Energy} = 1 - 0.0625 = \mathbf{93.75\%}$$
   A rank-16 factorization retains $93.75\%$ of the total variance/energy while requiring only $2 \times 4096 \times 16 = 131,072$ parameters compared to $4096^2 = 16,777,216$ parameters (a $99.22\%$ reduction in storage).

---

## Exercise 2: Rayleigh Quotient and Courant-Fischer Minimax Theorem

Let $\mathbf{S} \in \mathbb{R}^{n \times n}$ be a symmetric matrix with eigenvalues $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_n$.
The Rayleigh quotient is defined as:
$$R_{\mathbf{S}}(\mathbf{x}) \triangleq \frac{\mathbf{x}^\top \mathbf{S} \mathbf{x}}{\mathbf{x}^\top \mathbf{x}}, \quad \mathbf{x} \neq \mathbf{0}$$

1. Prove that $\max_{\mathbf{x} \neq \mathbf{0}} R_{\mathbf{S}}(\mathbf{x}) = \lambda_1$, and state the vector $\mathbf{x}^*$ achieving this maximum.
2. Prove that if $\mathbf{x} \perp \mathbf{q}_1$ (where $\mathbf{q}_1$ is the principal eigenvector), then $\max_{\mathbf{x} \perp \mathbf{q}_1, \mathbf{x} \neq \mathbf{0}} R_{\mathbf{S}}(\mathbf{x}) = \lambda_2$.
3. Explain how this variational principle directly justifies Power Iteration with deflation.

### Solution 2:
1. **Global Maximum of Rayleigh Quotient**:
   Since $\mathbf{S}$ is symmetric, choose an orthonormal eigenvector basis $\{\mathbf{q}_1, \dots, \mathbf{q}_n\}$ with $\mathbf{S}\mathbf{q}_i = \lambda_i \mathbf{q}_i$. Any non-zero $\mathbf{x} \in \mathbb{R}^n$ can be expanded as $\mathbf{x} = \sum_{i=1}^n c_i \mathbf{q}_i$.
   Then:
   $$\mathbf{x}^\top \mathbf{x} = \sum_{i=1}^n c_i^2, \quad \mathbf{x}^\top \mathbf{S} \mathbf{x} = \sum_{i=1}^n \lambda_i c_i^2$$
   Since $\lambda_i \le \lambda_1$ for all $i$:
   $$\mathbf{x}^\top \mathbf{S} \mathbf{x} = \sum_{i=1}^n \lambda_i c_i^2 \le \lambda_1 \sum_{i=1}^n c_i^2 = \lambda_1 (\mathbf{x}^\top \mathbf{x})$$
   Dividing by $\mathbf{x}^\top \mathbf{x} > 0$ yields $R_{\mathbf{S}}(\mathbf{x}) \le \lambda_1$.
   Evaluating at $\mathbf{x}^* = \mathbf{q}_1$ (where $c_1 = 1, c_i = 0$ for $i > 1$):
   $$R_{\mathbf{S}}(\mathbf{q}_1) = \frac{\lambda_1 (1)}{1} = \lambda_1$$
   Hence $\max_{\mathbf{x} \neq \mathbf{0}} R_{\mathbf{S}}(\mathbf{x}) = \lambda_1$.

2. **Orthogonal Subspace Maximization**:
   If $\mathbf{x} \perp \mathbf{q}_1$, then $c_1 = \mathbf{q}_1^\top \mathbf{x} = 0$. The expansion becomes:
   $$\mathbf{x} = \sum_{i=2}^n c_i \mathbf{q}_i$$
   Applying the same logic:
   $$\mathbf{x}^\top \mathbf{S} \mathbf{x} = \sum_{i=2}^n \lambda_i c_i^2 \le \lambda_2 \sum_{i=2}^n c_i^2 = \lambda_2 (\mathbf{x}^\top \mathbf{x})$$
   Evaluating at $\mathbf{x} = \mathbf{q}_2$ gives $R_{\mathbf{S}}(\mathbf{q}_2) = \lambda_2$.

3. **Justification for Power Iteration with Deflation**:
   When we subtract the principal rank-1 component $\mathbf{S}_{\text{deflated}} = \mathbf{S} - \lambda_1 \mathbf{q}_1 \mathbf{q}_1^\top$, the eigenvalue corresponding to $\mathbf{q}_1$ becomes $0$ while preserving all other eigenpairs $(\lambda_j, \mathbf{q}_j)$ for $j \ge 2$. Power iteration applied to $\mathbf{S}_{\text{deflated}}$ now naturally converges to the dominant eigenvector of the deflated matrix, which is $\mathbf{q}_2$ with eigenvalue $\lambda_2$.

---

## Exercise 3: Condition Number and Gradient Oscillation

Consider the quadratic loss function in $\mathbb{R}^2$:
$$f(x_1, x_2) = \frac{1}{2} \left( \lambda_1 x_1^2 + \lambda_2 x_2^2 \right) = \frac{1}{2} \mathbf{x}^\top \begin{bmatrix} \lambda_1 & 0 \\ 0 & \lambda_2 \end{bmatrix} \mathbf{x}$$
where $\lambda_1 = 100$ and $\lambda_2 = 1$. The condition number is $\kappa = \frac{100}{1} = 100$.

1. What is the maximum learning rate $\eta_{\max}$ for standard Gradient Descent $\mathbf{x}_{t+1} = \mathbf{x}_t - \eta \nabla f(\mathbf{x}_t)$ to remain stable?
2. At the optimal fixed learning rate $\eta^* = \frac{2}{\lambda_1 + \lambda_2}$, compute the contraction factor per iteration:
   $$\rho = \frac{\kappa - 1}{\kappa + 1}$$
3. How many iterations $T$ are required to reduce the initial error by a factor of $10^{-6}$?

### Solution 3:
1. **Stability Limit**:
   The gradient update decouples along each coordinate:
   $$x_{1, t+1} = (1 - \eta \lambda_1) x_{1, t}, \quad x_{2, t+1} = (1 - \eta \lambda_2) x_{2, t}$$
   For stability, we must satisfy $|1 - \eta \lambda_i| < 1$ for all $i$, which requires $0 < \eta < \frac{2}{\lambda_{\max}} = \frac{2}{100} = \mathbf{0.02}$.
   If $\eta \ge 0.02$, oscillations along the steep $x_1$ axis diverge.

2. **Optimal Contraction Factor**:
   $$\eta^* = \frac{2}{100 + 1} = \frac{2}{101} \approx 0.019802$$
   The contraction factor along the slowest direction ($x_2$) is:
   $$\rho = \left|1 - \frac{2 \lambda_2}{\lambda_1 + \lambda_2}\right| = \frac{\lambda_1 - \lambda_2}{\lambda_1 + \lambda_2} = \frac{\kappa - 1}{\kappa + 1} = \frac{100 - 1}{100 + 1} = \frac{99}{101} \approx \mathbf{0.980198}$$

3. **Number of Iterations for $10^{-6}$ Convergence**:
   $$\rho^T \le 10^{-6} \implies T \ln(\rho) \le -6 \ln(10)$$
   $$T \ge \frac{-6 \times 2.302585}{\ln(0.980198)} \approx \frac{-13.8155}{-0.020001} \approx \mathbf{691 \text{ iterations}}$$
   Because $\kappa = 100$, 691 steps are needed for 6 decimal places of precision, demonstrating how ill-conditioned Hessians impede first-order optimization.
