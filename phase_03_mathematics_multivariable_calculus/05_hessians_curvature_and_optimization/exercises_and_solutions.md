# Topic 3.5: Exercises & Solutions — Hessians, Curvature & Optimization

Welcome to the problem set for **Topic 3.5: Hessians, Curvature & Optimization in Deep Learning**.

Grab paper and pencil. Solve each problem before revealing the step-by-step solutions below.

---

## 📝 Problem Set Overview

| Problem | Focus | Difficulty | Practical AI Connection |
| :---: | :--- | :---: | :--- |
| **1** | **Critical Point Discovery & Hessian Eigenvalue Classification** | Medium | Identifying saddle points vs true minima in deep learning landscapes |
| **2** | **Hand-Crafted Newton Optimization Step on a Quadratic Bowl** | Medium | The theoretical foundation of natural gradient and quasi-Newton optimizers |
| **3** | **Condition Number $\kappa(H)$ & Gradient Descent Stability Limit** | Challenging | Explaining why loss explodes when learning rates exceed $\eta = 2/\lambda_{\max}$ |

---

## Problem 1: Critical Point Discovery & Hessian Eigenvalue Classification

### 🧒 The Context: Maya & Barnaby's Concrete Skatepark
Maya and her puppy Barnaby are exploring a terrain surface given by:
$$f(x, y) = x^3 - 3x + y^2$$

### Tasks:
1. Compute the gradient vector $\nabla f(x, y)$.
2. Set $\nabla f(x, y) = \mathbf{0}$ to find all critical points $(x^*, y^*)$ in the 2D plane.
3. Compute the general $2 \times 2$ Hessian matrix $H(x, y)$.
4. Evaluate $H$ at each critical point, compute its eigenvalues, and classify each point as a **local minimum**, **local maximum**, or **saddle point**.

---

<details>
<summary>Click to view solution for Problem 1</summary>

### Detailed Step-by-Step Solution:

#### Step 1: Compute Gradient Vector
$$\nabla f(x, y) = \begin{bmatrix} \frac{\partial f}{\partial x} \\ \frac{\partial f}{\partial y} \end{bmatrix} = \begin{bmatrix} 3x^2 - 3 \\ 2y \end{bmatrix}$$

#### Step 2: Find Critical Points
Set both components to zero:
$$3x^2 - 3 = 0 \implies 3(x^2 - 1) = 0 \implies x = +1 \text{ or } x = -1$$
$$2y = 0 \implies y = 0$$

There are two critical points:
- **Point A:** $(+1, 0)$
- **Point B:** $(-1, 0)$

#### Step 3: Compute General Hessian Matrix
$$H(x, y) = \begin{bmatrix}
\frac{\partial^2 f}{\partial x^2} & \frac{\partial^2 f}{\partial x \partial y} \\
\frac{\partial^2 f}{\partial y \partial x} & \frac{\partial^2 f}{\partial y^2}
\end{bmatrix} = \begin{bmatrix} 6x & 0 \\ 0 & 2 \end{bmatrix}$$

#### Step 4: Evaluate and Classify Each Critical Point

**At Point A $(+1, 0)$:**
$$H(1, 0) = \begin{bmatrix} 6(1) & 0 \\ 0 & 2 \end{bmatrix} = \begin{bmatrix} 6 & 0 \\ 0 & 2 \end{bmatrix}$$
- Eigenvalues: $\lambda_1 = +6, \quad \lambda_2 = +2$.
- Since both eigenvalues are strictly positive ($\lambda_1 > 0, \lambda_2 > 0$), $H \succ 0$ is **Positive Definite**.
- **Conclusion:** Point A is a **Local Minimum** (a soup bowl where Barnaby's ball settles)!

**At Point B $(-1, 0)$:**
$$H(-1, 0) = \begin{bmatrix} 6(-1) & 0 \\ 0 & 2 \end{bmatrix} = \begin{bmatrix} -6 & 0 \\ 0 & 2 \end{bmatrix}$$
- Eigenvalues: $\lambda_1 = -6, \quad \lambda_2 = +2$.
- The eigenvalues have opposite signs (curves downward along $x$, curves upward along $y$).
- **Conclusion:** Point B is a **Saddle Point** (a Pringle potato chip)!
</details>

---

## Problem 2: Hand-Crafted Newton Optimization Step on a Quadratic Bowl

### 🧒 The Context: Maya's One-Shot Optimization Magic
Consider the quadratic loss surface:
$$f(x, y) = 2x^2 + 8y^2$$
Notice that the $y$-axis has $4\times$ greater curvature than the $x$-axis.
We start at point $\mathbf{w}_0 = \begin{bmatrix} 4.0 \\ 2.0 \end{bmatrix}$.

### Tasks:
1. Compute the gradient vector $\nabla f(\mathbf{w}_0)$.
2. Compute the constant Hessian matrix $H$ and its matrix inverse $H^{-1}$.
3. Perform **one single Newton's optimization step**:
   $$\mathbf{w}_1 = \mathbf{w}_0 - H^{-1} \nabla f(\mathbf{w}_0)$$
4. Show that Newton's method reaches the global minimum $[0, 0]^T$ in **exactly one step**, whereas Gradient Descent would require dozens of zig-zagging steps!

---

<details>
<summary>Click to view solution for Problem 2</summary>

### Detailed Step-by-Step Solution:

#### Step 1: Gradient Vector at $\mathbf{w}_0$
$$\nabla f(x, y) = \begin{bmatrix} 4x \\ 16y \end{bmatrix} \implies \nabla f(4.0, 2.0) = \begin{bmatrix} 4(4.0) \\ 16(2.0) \end{bmatrix} = \begin{bmatrix} 16.0 \\ 32.0 \end{bmatrix}$$

#### Step 2: Compute Hessian and Inverse
$$H = \begin{bmatrix} \frac{\partial^2 f}{\partial x^2} & \frac{\partial^2 f}{\partial x \partial y} \\ \frac{\partial^2 f}{\partial y \partial x} & \frac{\partial^2 f}{\partial y^2} \end{bmatrix} = \begin{bmatrix} 4 & 0 \\ 0 & 16 \end{bmatrix}$$

For a diagonal matrix, the inverse simply inverts each diagonal entry:
$$H^{-1} = \begin{bmatrix} 1/4 & 0 \\ 0 & 1/16 \end{bmatrix} = \begin{bmatrix} 0.25 & 0 \\ 0 & 0.0625 \end{bmatrix}$$

#### Step 3: Newton's Optimization Step
$$\mathbf{w}_1 = \mathbf{w}_0 - H^{-1} \nabla f(\mathbf{w}_0)$$
$$\mathbf{w}_1 = \begin{bmatrix} 4.0 \\ 2.0 \end{bmatrix} - \begin{bmatrix} 0.25 & 0 \\ 0 & 0.0625 \end{bmatrix} \begin{bmatrix} 16.0 \\ 32.0 \end{bmatrix}$$

Multiplying $H^{-1} \nabla f$:
$$H^{-1} \nabla f = \begin{bmatrix} 0.25 \times 16.0 \\ 0.0625 \times 32.0 \end{bmatrix} = \begin{bmatrix} 4.0 \\ 2.0 \end{bmatrix}$$

Now subtract:
$$\mathbf{w}_1 = \begin{bmatrix} 4.0 \\ 2.0 \end{bmatrix} - \begin{bmatrix} 4.0 \\ 2.0 \end{bmatrix} = \mathbf{\begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix}}$$

#### Step 4: The Takeaway
The global minimum is $[0, 0]^T$. Newton's method converged to the exact optimal solution in **a single step**!
By multiplying by $H^{-1}$, the step along the shallow $x$-axis was stretched ($0.25 \times 16 = 4.0$) and the step along the steep $y$-axis was compressed ($0.0625 \times 32 = 2.0$), eliminating all ravine distortion!
</details>

---

## Problem 3: Condition Number $\kappa(H)$ & Gradient Descent Stability Limit

### 🧒 The Context: Timmy's Exploding Learning Rate
Consider an elliptical loss ravine:
$$f(x, y) = \frac{1}{2} x^2 + 50 y^2$$

### Tasks:
1. Compute the Hessian matrix $H$ and its condition number $\kappa(H) = \frac{\lambda_{\max}}{\lambda_{\min}}$.
2. The mathematical stability criterion for Gradient Descent states that the learning rate must satisfy:
   $$\eta < \frac{2}{\lambda_{\max}}$$
   What is the maximum stable learning rate $\eta_{\max}$ for this function?
3. If Timmy chooses learning rate $\eta = 0.025$:
   - Is the update stable along the $y$-axis?
   - How much progress is made per step along the $x$-axis ($x_{t+1} = (1 - \eta \lambda_{\min}) x_t$)?
   - What happens if Timmy sets $\eta = 0.030$?

---

<details>
<summary>Click to view solution for Problem 3</summary>

### Detailed Step-by-Step Solution:

#### Step 1: Hessian and Condition Number
$$\frac{\partial^2 f}{\partial x^2} = 1.0, \quad \frac{\partial^2 f}{\partial y^2} = 100.0, \quad \frac{\partial^2 f}{\partial x \partial y} = 0$$
$$H = \begin{bmatrix} 1 & 0 \\ 0 & 100 \end{bmatrix}$$
- Eigenvalues: $\lambda_{\min} = 1.0, \quad \lambda_{\max} = 100.0$.
- Condition Number:
  $$\kappa(H) = \frac{\lambda_{\max}}{\lambda_{\min}} = \frac{100.0}{1.0} = \mathbf{100.0}$$
The ravine walls are $100\times$ steeper than the riverbed!

#### Step 2: Maximum Stable Learning Rate
$$\eta_{\max} = \frac{2}{\lambda_{\max}} = \frac{2}{100.0} = \mathbf{0.020}$$

#### Step 3: Analysis with $\eta = 0.025$
- **Along the $y$-axis:**
  The update factor is $1 - \eta \lambda_y = 1 - 0.025(100) = 1 - 2.5 = \mathbf{-1.5}$.
  Since $|-1.5| > 1$, the $y$-coordinate alternates signs and grows exponentially ($y_0 \to -1.5 y_0 \to +2.25 y_0 \to -3.375 y_0 \dots$)!
  **The loss explodes to `NaN` / `inf`!**
- **Along the $x$-axis:**
  The update factor is $1 - \eta \lambda_x = 1 - 0.025(1) = 0.975$.
  The model shrinks $x$ by only $2.5\%$ per step, requiring hundreds of iterations to reach zero!
- **If Timmy sets $\eta = 0.030$:**
  Update factor is $1 - 0.03(100) = -2.0$. Oscillations double in magnitude every step, resulting in catastrophic numerical blowout within 15 steps.

**Deep Learning Connection:** This is why learning rate schedules, gradient clipping, and adaptive optimizers (Adam) are mandatory for training neural networks—they prevent high-curvature directions from blowing up!
</details>
