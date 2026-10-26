# Topic 3.2: Exercises & Solutions — Partial Derivatives, Gradients & Steepest Descent

---

## Problem Set Overview

| Problem | Focus | Difficulty |
| :--- | :--- | :--- |
| **1** | **Gradient Derivation of Linear Regression MSE Loss** | Medium |
| **2** | **Directional Derivatives & Orthogonal Contour Tangents** | Medium |
| **3** | **Manual Gradient Descent Trajectory on an Elliptical Bowl** | Challenging |

---

## Problem 1: Gradient Derivation of Linear Regression MSE Loss

### The Context
Consider a simple linear model $\hat{y}_i = w x_i + b$ with weight $w$ and bias $b$ trained on $N$ data points $(x_1, y_1), \dots, (x_N, y_N)$.
The Mean Squared Error (MSE) loss function is:
$$\mathcal{L}(w, b) = \frac{1}{N} \sum_{i=1}^N (w x_i + b - y_i)^2$$

### Task
1. Compute the partial derivative $\frac{\partial \mathcal{L}}{\partial w}$.
2. Compute the partial derivative $\frac{\partial \mathcal{L}}{\partial b}$.
3. Express the gradient vector $\nabla \mathcal{L}(w, b)$ in terms of residual errors $e_i = \hat{y}_i - y_i$.
4. For a single sample $(x = 2.0, y = 5.0)$ with current parameters $w = 1.0, b = 0.0$, compute the exact gradient vector $\nabla \mathcal{L}$.

---

### Solution 1

#### 1. Partial Derivative with respect to $w$
Using the chain rule and power rule on $(w x_i + b - y_i)^2$:
$$\frac{\partial}{\partial w} (w x_i + b - y_i)^2 = 2(w x_i + b - y_i) \cdot \frac{\partial}{\partial w}(w x_i + b - y_i) = 2(w x_i + b - y_i) \cdot x_i$$
Therefore:
$$\mathbf{\frac{\partial \mathcal{L}}{\partial w} = \frac{2}{N} \sum_{i=1}^N (w x_i + b - y_i) x_i}$$

#### 2. Partial Derivative with respect to $b$
$$\frac{\partial}{\partial b} (w x_i + b - y_i)^2 = 2(w x_i + b - y_i) \cdot \frac{\partial}{\partial b}(w x_i + b - y_i) = 2(w x_i + b - y_i) \cdot 1$$
Therefore:
$$\mathbf{\frac{\partial \mathcal{L}}{\partial b} = \frac{2}{N} \sum_{i=1}^N (w x_i + b - y_i)}$$

#### 3. Gradient in Terms of Residual Error $e_i = \hat{y}_i - y_i$
$$\nabla \mathcal{L}(w, b) = \begin{bmatrix} \frac{\partial \mathcal{L}}{\partial w} \\ \frac{\partial \mathcal{L}}{\partial b} \end{bmatrix} = \frac{2}{N} \begin{bmatrix} \sum_{i=1}^N e_i x_i \\ \sum_{i=1}^N e_i \end{bmatrix}$$

#### 4. Numerical Evaluation for $(x = 2.0, y = 5.0, w = 1.0, b = 0.0)$:
- Prediction: $\hat{y} = 1.0(2.0) + 0.0 = 2.0$
- Error: $e = \hat{y} - y = 2.0 - 5.0 = -3.0$
- Gradient w.r.t $w$: $2 \cdot e \cdot x = 2(-3.0)(2.0) = \mathbf{-12.0}$
- Gradient w.r.t $b$: $2 \cdot e = 2(-3.0) = \mathbf{-6.0}$
$$\nabla \mathcal{L} = \begin{bmatrix} -12.0 \\ -6.0 \end{bmatrix}$$
*Interpretation: To decrease loss, gradient descent moves along $-\nabla \mathcal{L} = [+12.0, +6.0]^T$, increasing both $w$ and $b$ to raise the prediction closer to 5.0!*

---

## Problem 2: Directional Derivatives & Orthogonal Contour Tangents

### The Context
Consider the 2D surface:
$$f(x, y) = 2x^2 + y^2$$

At the point $P = (1, 2)$:
1. Compute the gradient vector $\nabla f(1, 2)$.
2. Compute the unit vector $\mathbf{u}$ pointing in the direction of vector $\mathbf{v} = \begin{bmatrix} 3 \\ 4 \end{bmatrix}$.
3. Compute the directional derivative $D_{\mathbf{u}} f(1, 2) = \nabla f(1, 2) \cdot \mathbf{u}$.
4. Find a unit vector $\mathbf{u}_{\text{flat}}$ along which the altitude does not change ($D_{\mathbf{u}} f = 0$). Prove it is orthogonal to $\nabla f$.

---

### Solution 2

#### 1. Gradient Vector at $(1, 2)$
$$\frac{\partial f}{\partial x} = 4x \implies \frac{\partial f}{\partial x}(1, 2) = 4(1) = 4$$
$$\frac{\partial f}{\partial y} = 2y \implies \frac{\partial f}{\partial y}(1, 2) = 2(2) = 4$$
$$\nabla f(1, 2) = \begin{bmatrix} 4 \\ 4 \end{bmatrix}$$

#### 2. Unit Vector in Direction of $\mathbf{v} = [3, 4]^T$
$$\|\mathbf{v}\| = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = 5$$
$$\mathbf{u} = \frac{1}{5} \begin{bmatrix} 3 \\ 4 \end{bmatrix} = \begin{bmatrix} 0.6 \\ 0.8 \end{bmatrix}$$

#### 3. Directional Derivative $D_{\mathbf{u}} f$
$$D_{\mathbf{u}} f(1, 2) = \nabla f(1, 2) \cdot \mathbf{u} = \begin{bmatrix} 4 \\ 4 \end{bmatrix} \cdot \begin{bmatrix} 0.6 \\ 0.8 \end{bmatrix} = 4(0.6) + 4(0.8) = 2.4 + 3.2 = \mathbf{5.6}$$

#### 4. Orthogonal Flat Contour Direction
We want $\nabla f \cdot \mathbf{u}_{\text{flat}} = 0$:
$$\begin{bmatrix} 4 \\ 4 \end{bmatrix} \cdot \begin{bmatrix} u_x \\ u_y \end{bmatrix} = 4 u_x + 4 u_y = 0 \implies u_x = -u_y$$
Normalizing to unit length ($u_x^2 + u_y^2 = 1$):
$$\mathbf{u}_{\text{flat}} = \begin{bmatrix} \frac{1}{\sqrt{2}} \\ -\frac{1}{\sqrt{2}} \end{bmatrix} \approx \begin{bmatrix} 0.7071 \\ -0.7071 \end{bmatrix}$$
*Check:* $\begin{bmatrix} 4 \\ 4 \end{bmatrix} \cdot \begin{bmatrix} 1/\sqrt{2} \\ -1/\sqrt{2} \end{bmatrix} = \frac{4}{\sqrt{2}} - \frac{4}{\sqrt{2}} = 0$.
Walking along $\mathbf{u}_{\text{flat}}$ moves along the tangent line of the contour ellipse $2x^2 + y^2 = 6$, keeping loss constant!

---

## Problem 3: Manual Gradient Descent Trajectory

### The Context
Consider optimizing the convex quadratic loss:
$$f(x, y) = x^2 + 4y^2$$
Starting at initial point $\mathbf{w}_0 = \begin{bmatrix} 4 \\ 2 \end{bmatrix}$ with learning rate $\eta = 0.1$.

### Task
1. Write the general update equations for $x_{t+1}$ and $y_{t+1}$.
2. Perform **two full manual iterations** of gradient descent: compute $\mathbf{w}_1$ and $\mathbf{w}_2$.
3. Compute the loss values $f(\mathbf{w}_0)$, $f(\mathbf{w}_1)$, and $f(\mathbf{w}_2)$ to verify strict decrease.

---

### Solution 3

#### 1. General Update Equations
$$\nabla f(x, y) = \begin{bmatrix} 2x \\ 8y \end{bmatrix}$$
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla f(\mathbf{w}_t)$$
$$x_{t+1} = x_t - 0.1(2x_t) = x_t - 0.2x_t = 0.8 x_t$$
$$y_{t+1} = y_t - 0.1(8y_t) = y_t - 0.8y_t = 0.2 y_t$$

#### 2. Iteration Calculations
- **Initial State $\mathbf{w}_0$:**
  $$x_0 = 4.0, \quad y_0 = 2.0$$
  $$f(\mathbf{w}_0) = 4^2 + 4(2^2) = 16 + 16 = \mathbf{32.0}$$

- **Step 1 ($\mathbf{w}_1$):**
  $$x_1 = 0.8(4.0) = \mathbf{3.2}$$
  $$y_1 = 0.2(2.0) = \mathbf{0.4}$$
  $$\mathbf{w}_1 = \begin{bmatrix} 3.2 \\ 0.4 \end{bmatrix}$$
  $$f(\mathbf{w}_1) = (3.2)^2 + 4(0.4)^2 = 10.24 + 4(0.16) = 10.24 + 0.64 = \mathbf{10.88}$$

- **Step 2 ($\mathbf{w}_2$):**
  $$x_2 = 0.8(3.2) = \mathbf{2.56}$$
  $$y_2 = 0.2(0.4) = \mathbf{0.08}$$
  $$\mathbf{w}_2 = \begin{bmatrix} 2.56 \\ 0.08 \end{bmatrix}$$
  $$f(\mathbf{w}_2) = (2.56)^2 + 4(0.08)^2 = 6.5536 + 4(0.0064) = 6.5536 + 0.0256 = \mathbf{6.5792}$$

#### 3. Loss Trajectory Verification:
- $f(\mathbf{w}_0) = 32.0000$
- $f(\mathbf{w}_1) = 10.8800$
- $f(\mathbf{w}_2) = 6.5792$
The loss drops by nearly $80\%$ in just 2 steps!
*(Notice also that the steeper $y$-axis shrinks by factor $0.2$, while the flatter $x$-axis shrinks by factor $0.8$—an illustration of why ravines cause anisotropic convergence).*
