# Topic 7.2: Exercises & Solutions — Momentum & Nesterov

---

### Problem 1: Steady-State Velocity & Effective Learning Rate (Easy)

Consider Polyak Heavy-Ball momentum:
$$v_{t+1} = \beta v_t - \eta g_t$$
$$\theta_{t+1} = \theta_t + v_{t+1}$$

1. Suppose the gradient is constant: $g_t = g$ for all $t \ge 0$, and the initial velocity is $v_0 = 0$. Write an explicit formula for $v_t$ as a geometric series.
2. Evaluate the limit $v_\infty = \lim_{t \to \infty} v_t$ for $|\beta| < 1$.
3. What is the effective learning rate scaling factor? If $\eta = 0.001$ and $\beta = 0.99$, what is the effective step size?
4. Now assume an alternating gradient $g_t = (-1)^t g$ for $t \ge 0$ with $v_0 = 0$. Compute $v_\infty$ and explain why momentum damps oscillations.

#### Solution
1. **Geometric Series for $v_t$:**
   Unrolling the recurrence:
   $$v_1 = -\eta g$$
   $$v_2 = \beta v_1 - \eta g = -\eta g (1 + \beta)$$
   $$v_3 = \beta v_2 - \eta g = -\eta g (1 + \beta + \beta^2)$$
   By induction:
   $$v_t = -\eta g \sum_{k=0}^{t-1} \beta^k = -\eta g \frac{1 - \beta^t}{1 - \beta}$$

2. **Limit $v_\infty$:**
   Since $|\beta| < 1$, $\lim_{t \to \infty} \beta^t = 0$:
   $$v_\infty = -\frac{\eta}{1 - \beta} g$$

3. **Effective Learning Rate:**
   The effective learning rate in a steady direction is:
   $$\eta_{\text{eff}} = \frac{\eta}{1 - \beta}$$
   With $\eta = 0.001$ and $\beta = 0.99$:
   $$\eta_{\text{eff}} = \frac{0.001}{1 - 0.99} = \frac{0.001}{0.01} = 0.1$$
   The effective step size is amplified by **$100\times$**!

4. **Alternating Gradient:**
   When $g_t = (-1)^t g$:
   $$v_t = -\eta g \sum_{k=0}^{t-1} \beta^k (-1)^{t-1-k}$$
   For large $t$, the steady magnitude of the alternating velocity is:
   $$|v_\infty| = \frac{\eta |g|}{1 + \beta}$$
   With $\beta = 0.9$, $|v_\infty| = \frac{\eta |g|}{1.9} \approx 0.526 \eta |g|$.
   While consistent gradients are boosted by $\frac{1}{1-\beta} = 10\times$, alternating gradients are reduced to $\frac{1}{1+\beta} \approx 0.53\times$. This differential filtering cancels high-frequency noise while accelerating low-frequency directional motion.

---

### Problem 2: 2D Ravine Simulation by Hand (Medium)

Consider the anisotropic ravine:
$$f(x, y) = 50 x^2 + \frac{1}{2} y^2$$
At any point, $\nabla f(x, y) = [100x, \, y]^T$.
Let the starting point be $(x_0, y_0) = (1.0, 10.0)$, with learning rate $\eta = 0.01$ and momentum coefficient $\beta = 0.8$. Initial velocity $v_0 = (0, 0)$.

1. Compute step 1 $(x_1, y_1)$ for:
   - Standard Gradient Descent
   - Polyak Momentum
   - Nesterov Accelerated Gradient
2. Compute step 2 $(x_2, y_2)$ for each of the three methods.
3. Compare the progression in the steep $x$-direction vs the shallow $y$-direction.

#### Solution
1. **Step 1 ($t = 0 \to 1$):**
   Initial gradient:
   $$\nabla f(1.0, 10.0) = [100(1.0), \, 1.0(10.0)]^T = [100.0, \, 10.0]^T$$

   - **Standard GD:**
     $$x_1 = 1.0 - 0.01(100.0) = 1.0 - 1.0 = 0.0$$
     $$y_1 = 10.0 - 0.01(10.0) = 10.0 - 0.1 = 9.9$$
     Result: $(x_1, y_1) = (0.0, 9.9)$.

   - **Polyak Momentum:**
     $v_1 = 0.8(0) - 0.01 [100.0, 10.0]^T = [-1.0, -0.1]^T$
     $(x_1, y_1) = (1.0, 10.0) + (-1.0, -0.1) = (0.0, 9.9)$.
     (Matches GD on step 1 because $v_0 = 0$).

   - **Nesterov (NAG):**
     Lookahead position: $\theta_0 + \beta v_0 = (1.0, 10.0) + (0, 0) = (1.0, 10.0)$.
     Lookahead gradient: $\nabla f(1.0, 10.0) = [100.0, 10.0]^T$.
     $v_1 = 0.8(0) - 0.01 [100.0, 10.0]^T = [-1.0, -0.1]^T$.
     $(x_1, y_1) = (0.0, 9.9)$.

2. **Step 2 ($t = 1 \to 2$):**
   Gradient at $(x_1, y_1) = (0.0, 9.9)$:
   $$\nabla f(0.0, 9.9) = [100(0.0), \, 9.9]^T = [0.0, \, 9.9]^T$$

   - **Standard GD:**
     $$x_2 = 0.0 - 0.01(0.0) = 0.0$$
     $$y_2 = 9.9 - 0.01(9.9) = 9.9 - 0.099 = 9.801$$
     GD reaches $(0.0, 9.801)$. Progress on $y$ is only $0.199$ total.

   - **Polyak Momentum:**
     $$v_2 = 0.8 v_1 - 0.01 \nabla f(x_1, y_1) = 0.8 \begin{bmatrix} -1.0 \\ -0.1 \end{bmatrix} - 0.01 \begin{bmatrix} 0.0 \\ 9.9 \end{bmatrix} = \begin{bmatrix} -0.8 \\ -0.08 - 0.099 \end{bmatrix} = \begin{bmatrix} -0.8 \\ -0.179 \end{bmatrix}$$
     $$(x_2, y_2) = (0.0, 9.9) + (-0.8, -0.179) = (-0.8, 9.721)$$
     Notice: Polyak overshot the $x$-axis to $-0.8$ because velocity along $x$ was $-1.0$! However, progress on $y$ was faster ($y_2 = 9.721$).

   - **Nesterov (NAG):**
     Lookahead position:
     $$\theta_{\text{lookahead}} = (x_1, y_1) + \beta v_1 = (0.0, 9.9) + 0.8(-1.0, -0.1) = (-0.8, 9.82)$$
     Lookahead gradient:
     $$\nabla f(-0.8, 9.82) = [100(-0.8), \, 9.82]^T = [-80.0, \, 9.82]^T$$
     Velocity update:
     $$v_2 = 0.8 \begin{bmatrix} -1.0 \\ -0.1 \end{bmatrix} - 0.01 \begin{bmatrix} -80.0 \\ 9.82 \end{bmatrix} = \begin{bmatrix} -0.8 + 0.8 \\ -0.08 - 0.0982 \end{bmatrix} = \begin{bmatrix} 0.0 \\ -0.1782 \end{bmatrix}$$
     Position update:
     $$(x_2, y_2) = (0.0, 9.9) + (0.0, -0.1782) = (0.0, 9.7218)$$
     **Remarkable Result:** While Polyak overshot to $x_2 = -0.8$, Nesterov evaluated the lookahead gradient at $x=-0.8$, sensed the uphill climb, and cancelled the $x$-velocity completely ($v_{2,x} = 0.0$), keeping the particle perfectly anchored at $x_2 = 0.0$ while accelerating down $y$!

---

### Problem 3: Spectral Radius & Optimal Damping (Hard)

Consider a 1D quadratic objective $f(\theta) = \frac{1}{2} a \theta^2$ with $a > 0$.
The Polyak Heavy-Ball update can be expressed in state-space form:
$$\begin{bmatrix} \theta_{t+1} \\ \theta_t \end{bmatrix} = T \begin{bmatrix} \theta_t \\ \theta_{t-1} \end{bmatrix}$$

1. Derive the transition matrix $T \in \mathbb{R}^{2 \times 2}$.
2. Write down the characteristic equation $\det(T - \lambda I) = 0$.
3. Show that when roots are complex conjugates, the spectral radius is $\rho(T) = \sqrt{\beta}$.
4. Explain why this leads to the optimal convergence rate $\rho = \frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}$.

#### Solution
1. **Transition Matrix $T$:**
   Recall the Heavy Ball update:
   $$\theta_{t+1} = \theta_t + \beta(\theta_t - \theta_{t-1}) - \eta a \theta_t = (1 + \beta - \eta a)\theta_t - \beta \theta_{t-1}$$
   Together with the identity $\theta_t = \theta_t$, in matrix form:
   $$\begin{bmatrix} \theta_{t+1} \\ \theta_t \end{bmatrix} = \begin{bmatrix} 1 + \beta - \eta a & -\beta \\ 1 & 0 \end{bmatrix} \begin{bmatrix} \theta_t \\ \theta_{t-1} \end{bmatrix} \implies T = \begin{bmatrix} 1 + \beta - \eta a & -\beta \\ 1 & 0 \end{bmatrix}$$

2. **Characteristic Equation:**
   $$\det(T - \lambda I) = \det \begin{bmatrix} 1 + \beta - \eta a - \lambda & -\beta \\ 1 & -\lambda \end{bmatrix} = -\lambda(1 + \beta - \eta a - \lambda) + \beta = 0$$
   $$\lambda^2 - (1 + \beta - \eta a)\lambda + \beta = 0$$

3. **Complex Conjugate Roots & Spectral Radius:**
   By Vieta's formulas, the product of the two eigenvalues is the determinant:
   $$\lambda_1 \lambda_2 = \beta$$
   When the discriminant is negative: $(1 + \beta - \eta a)^2 - 4\beta < 0$, the roots form a complex conjugate pair $\lambda_{1,2} = r e^{\pm i \phi}$.
   Then $|\lambda_1|^2 = \lambda_1 \overline{\lambda_1} = \lambda_1 \lambda_2 = \beta$.
   Therefore, the spectral radius is:
   $$\rho(T) = |\lambda_{1,2}| = \sqrt{\beta}$$
   Remarkably, the convergence rate is completely independent of the eigenvalue $a$, provided the roots remain complex!

4. **Optimal Tuning for Condition Number $\kappa$:**
   To make roots complex for all eigenvalues in $[\mu, L]$, the transition point must be placed at the boundaries $\mu$ and $L$. Polyak showed that equating the spectral radii at both extremes yields:
   $$\beta^* = \left( \frac{\sqrt{L} - \sqrt{\mu}}{\sqrt{L} + \sqrt{\mu}} \right)^2 = \left( \frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1} \right)^2$$
   Taking the square root gives the optimal contraction factor:
   $$\rho^* = \sqrt{\beta^*} = \frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}$$
   This proves the theoretical acceleration over standard gradient descent.

---

### Problem 4: Nesterov Lookahead Equivalence in PyTorch (Medium)

In PyTorch, `torch.optim.SGD(..., momentum=beta, nesterov=True)` implements:
$$v_{t+1} = \beta v_t + g_t$$
$$\theta_{t+1} = \theta_t - \eta (g_t + \beta v_{t+1})$$
where $g_t = \nabla f(\theta_t)$.

Show that this update is mathematically equivalent to the classical Nesterov formulation:
$$y_t = \theta_t - \eta \nabla f(\theta_t)$$
$$\theta_{t+1} = y_t + \beta (y_t - y_{t-1})$$
under a change of variables.

#### Solution
Let us expand PyTorch's update:
Substitute $v_{t+1} = \beta v_t + g_t$ into the position update:
$$\theta_{t+1} = \theta_t - \eta g_t - \eta \beta (\beta v_t + g_t) = \theta_t - \eta (1 + \beta) g_t - \eta \beta^2 v_t$$

Now consider classical Nesterov:
$$y_t = \theta_t - \eta \nabla f(\theta_t)$$
$$\theta_{t+1} = (1 + \beta) y_t - \beta y_{t-1}$$
Notice that $y_t$ is the intermediate gradient descent step, and $\theta_{t+1}$ performs momentum extrapolation on the sequence $\{y_t\}$.
Define the velocity sequence in the $y$-space as $u_{t+1} = y_t - y_{t-1}$.
Then:
$$\theta_{t+1} = y_t + \beta u_{t+1}$$
Evaluating $\nabla f$ at $\theta_{t+1}$ is evaluating at $y_t + \beta u_{t+1}$, which is exactly the lookahead position.
By identifying $v_t = \frac{1}{\eta}(y_{t-1} - \theta_{t-1})$, the two sequences of iterates $\{\theta_t\}$ trace identical mathematical trajectories up to a single step index shift. PyTorch's formulation has the massive software advantage of requiring only the standard gradient $\nabla f(\theta_t)$ without an extra forward/backward pass.
