# Topic 02: Matrix Calculus & Hessian-Vector Products — Exercises & Solutions

## Exercise 1: Matrix Calculus of Linear and Ridge Regression

Consider the regularized linear regression loss:
$$\mathcal{L}(\mathbf{w}) = \frac{1}{2N} \|\mathbf{X}\mathbf{w} - \mathbf{y}\|_2^2 + \frac{\lambda}{2} \|\mathbf{w}\|_2^2$$
where $\mathbf{X} \in \mathbb{R}^{N \times d}$, $\mathbf{y} \in \mathbb{R}^N$, and $\mathbf{w} \in \mathbb{R}^d$.

1. Derive the gradient vector $\nabla_{\mathbf{w}} \mathcal{L}(\mathbf{w})$ using matrix calculus (denominator layout).
2. Compute the exact Hessian matrix $\mathbf{H} = \nabla_{\mathbf{w}}^2 \mathcal{L}(\mathbf{w})$.
3. Express the Hessian-Vector Product $\mathbf{H}\mathbf{v}$ for an arbitrary probe vector $\mathbf{v} \in \mathbb{R}^d$ and compute its operational flop complexity compared to forming $\mathbf{H}$.

### Solution 1:
1. **Gradient Derivation**:
   Expand the squared residual:
   $$\mathcal{L}(\mathbf{w}) = \frac{1}{2N} (\mathbf{w}^\top \mathbf{X}^\top \mathbf{X} \mathbf{w} - 2 \mathbf{y}^\top \mathbf{X} \mathbf{w} + \mathbf{y}^\top \mathbf{y}) + \frac{\lambda}{2} \mathbf{w}^\top \mathbf{w}$$
   Taking the derivative with respect to vector $\mathbf{w}$:
   $$\nabla_{\mathbf{w}} \mathcal{L}(\mathbf{w}) = \frac{1}{N} \mathbf{X}^\top (\mathbf{X}\mathbf{w} - \mathbf{y}) + \lambda \mathbf{w}$$

2. **Hessian Matrix**:
   Differentiating the gradient with respect to $\mathbf{w}^\top$:
   $$\mathbf{H} = \nabla_{\mathbf{w}}^2 \mathcal{L}(\mathbf{w}) = \frac{1}{N} \mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I}_d$$
   Note that $\mathbf{H}$ is constant (independent of $\mathbf{w}$) and strictly positive definite for $\lambda > 0$.

3. **Hessian-Vector Product & Complexity**:
   To compute $\mathbf{H}\mathbf{v}$:
   $$\mathbf{H}\mathbf{v} = \left(\frac{1}{N} \mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I}_d\right)\mathbf{v} = \frac{1}{N} \mathbf{X}^\top (\mathbf{X}\mathbf{v}) + \lambda \mathbf{v}$$
   - **Forming $\mathbf{H}$ explicitly**: Computing $\mathbf{X}^\top \mathbf{X}$ requires $\mathcal{O}(N d^2)$ operations, and storing it takes $\mathcal{O}(d^2)$ space. Then multiplying $\mathbf{H}\mathbf{v}$ takes $\mathcal{O}(d^2)$ operations.
   - **Associative HVP**: First compute $\mathbf{u} = \mathbf{X}\mathbf{v}$ in $\mathcal{O}(Nd)$ flops. Then compute $\frac{1}{N} \mathbf{X}^\top \mathbf{u} + \lambda \mathbf{v}$ in $\mathcal{O}(Nd)$ flops.
   Total time is $\mathcal{O}(Nd)$ with $\mathcal{O}(N + d)$ storage. For $d = 100,000$ and $N = 1,000$, associative HVP is $100\times$ faster and requires $0$ matrix memory.

---

## Exercise 2: Finite Difference Truncation Error vs. Dual Arithmetic

To approximate the directional derivative of gradient $\mathbf{g}(\boldsymbol{\theta})$ along $\mathbf{v}$, an engineer tests centered finite difference:
$$\text{HVP}_{\text{FD}}(h) = \frac{\mathbf{g}(\boldsymbol{\theta} + h \mathbf{v}) - \mathbf{g}(\boldsymbol{\theta} - h \mathbf{v})}{2h}$$

1. Using Taylor expansion, derive the leading order error term of $\text{HVP}_{\text{FD}}(h)$ in terms of the third derivative tensor $\nabla^3 f$.
2. In double-precision floating-point arithmetic with machine epsilon $\epsilon_{\text{mach}} \approx 10^{-16}$, explain the trade-off between truncation error and catastrophic cancellation, and derive the optimal step size $h^*$.
3. Contrast this with Pearlmutter's dual-number $\mathcal{R}$-operator.

### Solution 2:
1. **Taylor Expansion of Centered Difference**:
   Expand $\mathbf{g}(\boldsymbol{\theta} \pm h \mathbf{v})$ around $\boldsymbol{\theta}$:
   $$\mathbf{g}(\boldsymbol{\theta} + h \mathbf{v}) = \mathbf{g}(\boldsymbol{\theta}) + h \mathbf{H}\mathbf{v} + \frac{h^2}{2} \nabla^3 f[\mathbf{v}, \mathbf{v}] + \frac{h^3}{6} \nabla^4 f[\mathbf{v}, \mathbf{v}, \mathbf{v}] + \mathcal{O}(h^4)$$
   $$\mathbf{g}(\boldsymbol{\theta} - h \mathbf{v}) = \mathbf{g}(\boldsymbol{\theta}) - h \mathbf{H}\mathbf{v} + \frac{h^2}{2} \nabla^3 f[\mathbf{v}, \mathbf{v}] - \frac{h^3}{6} \nabla^4 f[\mathbf{v}, \mathbf{v}, \mathbf{v}] + \mathcal{O}(h^4)$$
   Subtracting the two equations cancels both even-order terms:
   $$\frac{\mathbf{g}(\boldsymbol{\theta} + h \mathbf{v}) - \mathbf{g}(\boldsymbol{\theta} - h \mathbf{v})}{2h} = \mathbf{H}\mathbf{v} + \frac{h^2}{6} \nabla^4 f[\mathbf{v}, \mathbf{v}, \mathbf{v}] + \mathcal{O}(h^4)$$
   The truncation error is quadratic: $\mathcal{O}(h^2)$.

2. **Optimal Step Size $h^*$**:
   - Roundoff / catastrophic cancellation error scales as $\frac{\epsilon_{\text{mach}}}{h}$.
   - Truncation error scales as $C h^2$.
   Total numerical error: $\mathcal{E}(h) \sim \frac{\epsilon_{\text{mach}}}{h} + C h^2$.
   Differentiating with respect to $h$ and setting to zero:
   $$-\frac{\epsilon_{\text{mach}}}{h^2} + 2 C h = 0 \implies h^* \sim \epsilon_{\text{mach}}^{1/3} \approx (10^{-16})^{1/3} \approx 10^{-5.3} \approx 10^{-6}$$
   At $h = 10^{-6}$, the achieved accuracy is at best $\approx 10^{-8}$ (half of available precision is lost).

3. **Comparison with Pearlmutter's $\mathcal{R}$-operator**:
   Pearlmutter's method uses nilpotent algebra ($\varepsilon^2 = 0$). There is no finite step $h$, no subtraction of nearly identical floats, and no truncation error. It achieves full 64-bit machine precision ($\sim 10^{-15}$) without parameter tuning.

---

## Exercise 3: Krylov Subspace Conjugate Gradient with Matrix-Free HVP

In Trust Region Policy Optimization (TRPO), we solve the linear system:
$$\mathbf{H}\mathbf{x} = \mathbf{g}$$
using Conjugate Gradient (CG), where $\mathbf{H}$ is accessed strictly via a black-box function `hvp(v)`.

Given initial guess $\mathbf{x}_0 = \mathbf{0}$, residual $\mathbf{r}_0 = \mathbf{g}$, search direction $\mathbf{p}_0 = \mathbf{r}_0$:
1. State the update equations for step size $\alpha_k$, iterate $\mathbf{x}_{k+1}$, and new search direction $\mathbf{p}_{k+1}$.
2. If the dimension is $n = 1,000,000$ and we terminate CG after $K = 20$ iterations, compute total memory and HVP calls required.

### Solution 3:
1. **Conjugate Gradient Iteration**:
   At each step $k$:
   - Compute matrix-free product: $\mathbf{q}_k = \text{hvp}(\mathbf{p}_k)$.
   - Curvature in direction $\mathbf{p}_k$: $\kappa_k = \mathbf{p}_k^\top \mathbf{q}_k$.
   - Step size: $\alpha_k = \frac{\mathbf{r}_k^\top \mathbf{r}_k}{\kappa_k}$.
   - Iterate update: $\mathbf{x}_{k+1} = \mathbf{x}_k + \alpha_k \mathbf{p}_k$.
   - Residual update: $\mathbf{r}_{k+1} = \mathbf{r}_k - \alpha_k \mathbf{q}_k$.
   - Conjugacy coefficient: $\beta_k = \frac{\mathbf{r}_{k+1}^\top \mathbf{r}_{k+1}}{\mathbf{r}_k^\top \mathbf{r}_k}$.
   - Next direction: $\mathbf{p}_{k+1} = \mathbf{r}_{k+1} + \beta_k \mathbf{p}_k$.

2. **Resource Consumption**:
   - Total HVP calls: exactly $K = 20$ calls.
   - Total memory: four $n$-dimensional vectors ($\mathbf{x}, \mathbf{r}, \mathbf{p}, \mathbf{q}$), totaling $4 \times 10^6 \times 8\text{ bytes} \approx 32\text{ MB}$.
   - If $\mathbf{H}$ were formed explicitly, it would require $(10^6)^2 \times 8\text{ bytes} = 8\text{ Terabytes}$.
   Matrix-free CG makes second-order optimization feasible for massive parameter spaces.
