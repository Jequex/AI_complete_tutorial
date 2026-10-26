# Topic 3.5: Exercises & Solutions — Hessians, Curvature & Optimization

---

## Problem Set Overview

| Problem | Focus | Difficulty |
| :--- | :--- | :--- |
| **1** | **Multivariable Critical Point Discovery & Hessian Classification** | Medium |
| **2** | **Hand-Crafted Newton Optimization Step on a Quadratic Bowl** | Medium |
| **3** | **Condition Number $\kappa(H)$ & The Gradient Descent Stability Limit** | Challenging |

---

## Problem 1: Multivariable Critical Point Discovery & Hessian Classification

### The Context
Consider the non-linear multivariable function:
$$f(x, y) = x^3 - 3x + y^2$$

### Task
1. Calculate the gradient vector $\nabla f(x, y)$.
2. Set $\nabla f(x, y) = \mathbf{0}$ to find all critical points $(x^*, y^*)$ in the 2D plane.
3. Compute the general $2 \times 2$ Hessian matrix $H(x, y)$.
4. Evaluate $H$ at each critical point, compute its eigenvalues, and classify each critical point as a **local minimum**, **local maximum**, or **saddle point**.

---

### Solution 1

#### 1. Gradient Vector
$$\nabla f(x, y) = \begin{bmatrix} \frac{\partial f}{\partial x} \\ \frac{\partial f}{\partial y} \end{bmatrix} = \begin{bmatrix} 3x^2 - 3 \\ 2y \end{bmatrix}$$

#### 2. Critical Points
Setting $\nabla f(x, y) = \mathbf{0}$:
$$3x^2 - 3 = 0 \implies x^2 = 1 \implies x = \pm 1$$
$$2y = 0 \implies y = 0$$
Thus, there are **two critical points**:
- **Point A:** $(+1, 0)$
- **Point B:** $(-1, 0)$

#### 3. General Hessian Matrix
$$H(x, y) = \begin{bmatrix} \frac{\partial^2 f}{\partial x^2} & \frac{\partial^2 f}{\partial x \partial y} \\ \frac{\partial^2 f}{\partial y \partial x} & \frac{\partial^2 f}{\partial y^2} \end{bmatrix} = \begin{bmatrix} 6x & 0 \\ 0 & 2 \end{bmatrix}$$

#### 4. Classification of Critical Points

**At Point A $(+1, 0)$:**
$$H(1, 0) = \begin{bmatrix} 6(1) & 0 \\ 0 & 2 \end{bmatrix} = \begin{bmatrix} 6 & 0 \\ 0 & 2 \end{bmatrix}$$
- The eigenvalues are the diagonal entries: $\lambda_1 = +6, \quad \lambda_2 = +2$.
- Since both eigenvalues are strictly positive ($\lambda_1 > 0, \lambda_2 > 0$), $H \succ 0$ is **Positive Definite**.
- **Conclusion:** Point A $(+1, 0)$ is a **Local Minimum** (bowl shape).

**At Point B $(-1, 0)$:**
$$H(-1, 0) = \begin{bmatrix} 6(-1) & 0 \\ 0 & 2 \end{bmatrix} = \begin{bmatrix} -6 & 0 \\ 0 & 2 \end{bmatrix}$$
- The eigenvalues are: $\lambda_1 = -6, \quad \lambda_2 = +2$.
- The eigenvalues have opposite signs ($\lambda_1 < 0$ while $\lambda_2 > 0$). Thus, $H$ is **Indefinite**.
- **Conclusion:** Point B $(-1, 0)$ is a **Saddle Point** (Pringle chip shape: curves down along $x$, curves up along $y$).

---

## Problem 2: Hand-Crafted Newton Optimization Step on a Quadratic Bowl

### The Context
Consider the quadratic objective function:
$$f(x, y) = 2x^2 + 4xy + 5y^2 - 6x - 8y$$

### Task
1. Calculate the gradient vector $\nabla f(x, y)$.
2. Calculate the Hessian matrix $H$. Verify that $H$ is constant everywhere.
3. Compute the matrix inverse $H^{-1}$.
4. Start at the origin $\mathbf{w}_0 = [0, 0]^T$. Compute the gradient $\nabla f(\mathbf{w}_0)$ and execute a single Newton step:
   $$\mathbf{w}_1 = \mathbf{w}_0 - H^{-1} \nabla f(\mathbf{w}_0)$$
5. Verify that $\nabla f(\mathbf{w}_1) = \mathbf{0}$, proving that Newton's method converged to the exact global minimum in a single jump!

---

### Solution 2

#### 1. Gradient Vector
$$\frac{\partial f}{\partial x} = 4x + 4y - 6$$
$$\frac{\partial f}{\partial y} = 4x + 10y - 8$$
$$\nabla f(x, y) = \begin{bmatrix} 4x + 4y - 6 \\ 4x + 10y - 8 \end{bmatrix}$$

#### 2. Hessian Matrix
$$H = \begin{bmatrix} 
\frac{\partial^2 f}{\partial x^2} & \frac{\partial^2 f}{\partial x \partial y} \\ 
\frac{\partial^2 f}{\partial y \partial x} & \frac{\partial^2 f}{\partial y^2} 
\end{bmatrix} = \begin{bmatrix} 4 & 4 \\ 4 & 10 \end{bmatrix}$$
Notice that $H$ contains only constants and is symmetric ($H_{12} = H_{21} = 4$).

#### 3. Inverting $H$
For a $2 \times 2$ matrix $M = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$:
$$\det(H) = ad - bc = (4)(10) - (4)(4) = 40 - 16 = \mathbf{24}$$
$$H^{-1} = \frac{1}{\det(H)} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix} = \frac{1}{24} \begin{bmatrix} 10 & -4 \\ -4 & 4 \end{bmatrix} = \begin{bmatrix} \frac{5}{12} & -\frac{1}{6} \\ -\frac{1}{6} & \frac{1}{6} \end{bmatrix}$$

#### 4. Executing the Newton Step from $\mathbf{w}_0 = [0, 0]^T$
At $\mathbf{w}_0 = [0, 0]^T$:
$$\nabla f(0, 0) = \begin{bmatrix} -6 \\ -8 \end{bmatrix}$$
The Newton step is:
$$\Delta\mathbf{w} = -H^{-1} \nabla f(\mathbf{w}_0) = - \frac{1}{24} \begin{bmatrix} 10 & -4 \\ -4 & 4 \end{bmatrix} \begin{bmatrix} -6 \\ -8 \end{bmatrix}$$
$$\Delta\mathbf{w} = - \frac{1}{24} \begin{bmatrix} (10)(-6) + (-4)(-8) \\ (-4)(-6) + (4)(-8) \end{bmatrix} = - \frac{1}{24} \begin{bmatrix} -60 + 32 \\ 24 - 32 \end{bmatrix} = - \frac{1}{24} \begin{bmatrix} -28 \\ -8 \end{bmatrix} = \begin{bmatrix} \frac{28}{24} \\ \frac{8}{24} \end{bmatrix} = \mathbf{\begin{bmatrix} \frac{7}{6} \\ \frac{1}{3} \end{bmatrix}}$$

Thus:
$$\mathbf{w}_1 = \mathbf{w}_0 + \Delta\mathbf{w} = \mathbf{\begin{bmatrix} 7/6 \\ 1/3 \end{bmatrix}} \approx \begin{bmatrix} 1.1667 \\ 0.3333 \end{bmatrix}$$

#### 5. Verification
Check the gradient at $\mathbf{w}_1$:
$$\frac{\partial f}{\partial x}\left(\frac{7}{6}, \frac{1}{3}\right) = 4\left(\frac{7}{6}\right) + 4\left(\frac{1}{3}\right) - 6 = \frac{14}{3} + \frac{4}{3} - 6 = \frac{18}{3} - 6 = 6 - 6 = \mathbf{0}$$
$$\frac{\partial f}{\partial y}\left(\frac{7}{6}, \frac{1}{3}\right) = 4\left(\frac{7}{6}\right) + 10\left(\frac{1}{3}\right) - 8 = \frac{14}{3} + \frac{10}{3} - 8 = \frac{24}{3} - 8 = 8 - 8 = \mathbf{0}$$
$\nabla f(\mathbf{w}_1) = \mathbf{0}$. The global minimum was reached in exactly **one step**!

---

## Problem 3: Condition Number $\kappa(H)$ & The Gradient Descent Stability Limit

### The Context
Consider the anisotropic canyon loss function:
$$f(x, y) = \frac{1}{2} \left( 1000 x^2 + y^2 \right)$$

### Task
1. Compute the Hessian matrix $H$, find its eigenvalues $\lambda_1, \lambda_2$, and calculate the condition number $\kappa(H) = \frac{\lambda_{\max}}{\lambda_{\min}}$.
2. Write the 1-step Gradient Descent update rule along each coordinate axis with learning rate $\eta$:
   $$x_{t+1} = (1 - \eta \lambda_1) x_t, \quad y_{t+1} = (1 - \eta \lambda_2) y_t$$
3. What is the maximum learning rate $\eta_{\text{max}}$ before Gradient Descent diverges (explodes to infinity) along the $x$-axis?
4. If you set $\eta = \frac{1}{1000} = 0.001$ to ensure stability along the $x$-axis, by what multiplier does the error along the $y$-axis shrink at each step? How many steps are needed to reduce $y$ to $1\%$ of its initial value?

---

### Solution 3

#### 1. Hessian, Eigenvalues & Condition Number
$$\frac{\partial f}{\partial x} = 1000x, \quad \frac{\partial f}{\partial y} = y$$
$$H = \begin{bmatrix} 1000 & 0 \\ 0 & 1 \end{bmatrix}$$
- Eigenvalues: $\lambda_1 = 1000$ (steep wall), $\lambda_2 = 1$ (flat floor).
- Condition number:
  $$\kappa(H) = \frac{\lambda_{\max}}{\lambda_{\min}} = \frac{1000}{1} = \mathbf{1000}$$

#### 2. Gradient Descent Coordinate Recurrence
Using $\nabla f = [1000x, y]^T$:
$$x_{t+1} = x_t - \eta (1000 x_t) = (1 - 1000\eta) x_t = (1 - \eta \lambda_1) x_t$$
$$y_{t+1} = y_t - \eta (1 y_t) = (1 - 1\eta) y_t = (1 - \eta \lambda_2) y_t$$

#### 3. Maximum Learning Rate Stability Limit
For a geometric sequence $x_{t+1} = c \cdot x_t$ to converge to 0 without exploding, we must have $|c| < 1$:
$$|1 - \eta \lambda_1| < 1 \implies -1 < 1 - 1000\eta < 1$$
$$-2 < -1000\eta < 0 \implies 0 < \eta < \frac{2}{1000} = \mathbf{0.002}$$
If $\eta \ge 0.002$, Gradient Descent will oscillate with ever-increasing amplitude and blow up!

#### 4. Sluggish Convergence along Flat Coordinate
With $\eta = 0.001$:
- For $x$: $x_{t+1} = (1 - 0.001 \times 1000) x_t = (1 - 1) x_t = 0$ (converges in 1 step!).
- For $y$: $y_{t+1} = (1 - 0.001 \times 1) y_t = \mathbf{0.999 \, y_t}$.

At each step, the error along the $y$-axis only shrinks by $0.1\%$ ($0.999$).
To reach $1\%$ ($0.01$) of initial error:
$$0.999^t = 0.01$$
$$t \cdot \ln(0.999) = \ln(0.01)$$
$$t \approx \frac{-4.60517}{-0.0010005} \approx \mathbf{4603 \text{ steps!}}$$

<div class="takeaway">
    <strong>Key Takeaway for AI Optimizers:</strong>
    Because the condition number is $\kappa = 1000$, standard SGD is trapped: it cannot use a larger learning rate without exploding on $x$, but the small learning rate forces it to take 4,600 steps on $y$!
    This is why optimizers like <strong>Adam</strong> adapt the step size per coordinate: Adam scales the effective step size by $1/\sqrt{g^2}$, moving aggressively along $y$ and safely along $x$ simultaneously!
</div>
