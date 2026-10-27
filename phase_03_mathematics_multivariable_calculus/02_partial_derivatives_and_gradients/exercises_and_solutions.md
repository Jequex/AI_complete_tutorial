# Topic 3.2: Exercises & Solutions — Partial Derivatives, Gradients & Steepest Descent

Welcome to the problem set for **Topic 3.2: Partial Derivatives, Gradients & Steepest Descent**.

Grab a pencil and notebook. Work through the calculations before revealing the collapsible solutions below.

---

## 📝 Problem Set Overview

| Problem | Focus | Difficulty | Practical AI Connection |
| :---: | :--- | :---: | :--- |
| **1** | **Gradient Derivation of Linear Regression MSE Loss** | Medium | The core training update of supervised learning models |
| **2** | **Directional Derivatives & Orthogonal Contour Tangents** | Medium | Navigating curved loss landscapes and understanding loss plateaus |
| **3** | **Manual Gradient Descent Trajectory on an Elliptical Bowl** | Challenging | Visualizing learning rate oscillations and ravine dynamics |

---

## Problem 1: Gradient Derivation of Linear Regression MSE Loss

### 🧒 The Context: Maya & Timmy's Lemonade Stand Predictor
Maya and Timmy are predicting lemonade sales $\hat{y}$ from daily outdoor temperature $x$ using a linear equation:
$$\hat{y}_i = w x_i + b$$
They collect $N$ days of data $(x_1, y_1), \dots, (x_N, y_N)$.
To measure prediction error across all days, they use the Mean Squared Error (MSE) loss:
$$\mathcal{L}(w, b) = \frac{1}{N} \sum_{i=1}^N (w x_i + b - y_i)^2$$

### Tasks:
1. Using the "Freeze" Rule, calculate the partial derivative with respect to weight: $\frac{\partial \mathcal{L}}{\partial w}$.
2. Calculate the partial derivative with respect to bias: $\frac{\partial \mathcal{L}}{\partial b}$.
3. Express the gradient vector $\nabla \mathcal{L}(w, b)$ in terms of residual error $e_i = \hat{y}_i - y_i$.
4. On day 1, temperature is $x = 2.0$ and actual cups sold is $y = 5.0$. If current parameters are $w = 1.0$ and $b = 0.0$, compute the exact gradient vector $\nabla \mathcal{L}$.

---

<details>
<summary>Click to view solution for Problem 1</summary>

### Detailed Step-by-Step Solution:

#### Step 1: Differentiate with respect to $w$ (Freeze $b$)
Using the Chain Rule on each term $(w x_i + b - y_i)^2$:
$$\frac{\partial}{\partial w} (w x_i + b - y_i)^2 = 2(w x_i + b - y_i) \cdot \frac{\partial}{\partial w}(w x_i + b - y_i)$$
Since $b$ and $y_i$ are frozen constants, $\frac{\partial}{\partial w}(w x_i + b - y_i) = x_i$.
Therefore:
$$\mathbf{\frac{\partial \mathcal{L}}{\partial w} = \frac{2}{N} \sum_{i=1}^N (w x_i + b - y_i) x_i}$$

#### Step 2: Differentiate with respect to $b$ (Freeze $w$)
$$\frac{\partial}{\partial b} (w x_i + b - y_i)^2 = 2(w x_i + b - y_i) \cdot \frac{\partial}{\partial b}(w x_i + b - y_i)$$
Since $w x_i$ and $y_i$ are frozen constants, $\frac{\partial}{\partial b}(w x_i + b - y_i) = 1$.
Therefore:
$$\mathbf{\frac{\partial \mathcal{L}}{\partial b} = \frac{2}{N} \sum_{i=1}^N (w x_i + b - y_i)}$$

#### Step 3: Gradient Vector in Terms of Error $e_i = \hat{y}_i - y_i$
$$\nabla \mathcal{L}(w, b) = \begin{bmatrix} \frac{\partial \mathcal{L}}{\partial w} \\ \frac{\partial \mathcal{L}}{\partial b} \end{bmatrix} = \frac{2}{N} \begin{bmatrix} \sum_{i=1}^N e_i x_i \\ \sum_{i=1}^N e_i \end{bmatrix}$$

#### Step 4: Concrete Numerical Calculation
For single sample $(x = 2.0, y = 5.0)$ with $w = 1.0, b = 0.0$ and $N = 1$:
1. Current model prediction:
   $$\hat{y} = w x + b = 1.0(2.0) + 0.0 = 2.0$$
2. Residual error:
   $$e = \hat{y} - y = 2.0 - 5.0 = -3.0$$
3. Gradient components:
   $$\frac{\partial \mathcal{L}}{\partial w} = 2(-3.0)(2.0) = \mathbf{-12.0}$$
   $$\frac{\partial \mathcal{L}}{\partial b} = 2(-3.0)(1.0) = \mathbf{-6.0}$$
   $$\nabla \mathcal{L} = \begin{bmatrix} -12.0 \\ -6.0 \end{bmatrix}$$

**🧒 What the math is actually doing:**
To decrease error, gradient descent steps along the negative gradient:
$$-\nabla \mathcal{L} = \begin{bmatrix} +12.0 \\ +6.0 \end{bmatrix}$$
It will increase both $w$ and $b$, boosting the model's prediction from $2.0$ up toward the true value $5.0$!
</details>

---

## Problem 2: Directional Derivatives & Orthogonal Contour Tangents

### 🧒 The Context: Hiking on Elliptical Crater Lake
Maya is exploring a crater landscape defined by the equation:
$$f(x, y) = 2x^2 + y^2$$
She is currently standing at coordinates $P = (1, 2)$.

### Tasks:
1. Compute the gradient vector $\nabla f(1, 2)$.
2. Convert direction vector $\mathbf{v} = \begin{bmatrix} 3 \\ 4 \end{bmatrix}$ into a unit vector $\mathbf{u}$.
3. Compute the directional derivative $D_{\mathbf{u}} f(1, 2) = \nabla f(1, 2) \cdot \mathbf{u}$.
4. Find a direction $\mathbf{u}_{\text{flat}}$ where altitude stays completely flat ($D_{\mathbf{u}} f = 0$). Prove that $\mathbf{u}_{\text{flat}}$ is perpendicular to the gradient $\nabla f$.

---

<details>
<summary>Click to view solution for Problem 2</summary>

### Detailed Step-by-Step Solution:

#### Step 1: Gradient at Point $(1, 2)$
$$\frac{\partial f}{\partial x} = 4x \implies \frac{\partial f}{\partial x}(1, 2) = 4(1) = 4$$
$$\frac{\partial f}{\partial y} = 2y \implies \frac{\partial f}{\partial y}(1, 2) = 2(2) = 4$$
$$\nabla f(1, 2) = \begin{bmatrix} 4 \\ 4 \end{bmatrix}$$

#### Step 2: Normalize $\mathbf{v} = [3, 4]^T$ to Unit Length
$$\|\mathbf{v}\| = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5$$
$$\mathbf{u} = \frac{\mathbf{v}}{\|\mathbf{v}\|} = \begin{bmatrix} 3/5 \\ 4/5 \end{bmatrix} = \begin{bmatrix} 0.6 \\ 0.8 \end{bmatrix}$$

#### Step 3: Directional Derivative Calculation
$$D_{\mathbf{u}} f(1, 2) = \nabla f(1, 2) \cdot \mathbf{u} = \begin{bmatrix} 4 \\ 4 \end{bmatrix} \cdot \begin{bmatrix} 0.6 \\ 0.8 \end{bmatrix} = 4(0.6) + 4(0.8) = 2.4 + 3.2 = \mathbf{5.6}$$
If Maya walks in direction $[0.6, 0.8]^T$, she climbs $5.6$ feet upward per foot of forward distance.

#### Step 4: Flat Walking Direction (Contour Line)
We require $D_{\mathbf{u}} f = \nabla f \cdot \mathbf{u}_{\text{flat}} = 0$:
$$4 u_x + 4 u_y = 0 \implies u_x = -u_y$$
Normalizing so that $u_x^2 + u_y^2 = 1$:
$$(-u_y)^2 + u_y^2 = 1 \implies 2 u_y^2 = 1 \implies u_y = \frac{1}{\sqrt{2}}, \quad u_x = -\frac{1}{\sqrt{2}}$$
$$\mathbf{u}_{\text{flat}} = \begin{bmatrix} -1/\sqrt{2} \\ 1/\sqrt{2} \end{bmatrix} \approx \begin{bmatrix} -0.7071 \\ 0.7071 \end{bmatrix}$$

**Dot Product Check:**
$$\nabla f \cdot \mathbf{u}_{\text{flat}} = 4(-1/\sqrt{2}) + 4(1/\sqrt{2}) = 0$$
Since their dot product is zero, walking along $\mathbf{u}_{\text{flat}}$ traces the ellipse contour line $2x^2 + y^2 = 6$, keeping height constant!
</details>

---

## Problem 3: Manual Gradient Descent Trajectory on an Elliptical Bowl

### 🧒 The Context: Rolling Down a Stretched Bowl
Consider minimizing the asymmetric loss surface:
$$f(x, y) = x^2 + 4y^2$$
Notice that the $y$-direction is $4\times$ steeper than the $x$-direction!
We start at point $\mathbf{w}_0 = \begin{bmatrix} 4.0 \\ 2.0 \end{bmatrix}$ with learning rate $\eta = 0.1$.

### Tasks:
1. Derive the coordinate update rules for $x_{t+1}$ and $y_{t+1}$.
2. Perform two full iterations of gradient descent by hand to calculate $\mathbf{w}_1$ and $\mathbf{w}_2$.
3. Compute the loss $f(\mathbf{w}_0)$, $f(\mathbf{w}_1)$, and $f(\mathbf{w}_2)$ to verify that the loss strictly decreases.

---

<details>
<summary>Click to view solution for Problem 3</summary>

### Detailed Step-by-Step Solution:

#### Step 1: Coordinate Update Formulas
The gradient is:
$$\nabla f(x, y) = \begin{bmatrix} 2x \\ 8y \end{bmatrix}$$
Applying the gradient descent update $\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla f(\mathbf{w}_t)$ with $\eta = 0.1$:
$$x_{t+1} = x_t - 0.1(2x_t) = x_t - 0.2 x_t = \mathbf{0.8 x_t}$$
$$y_{t+1} = y_t - 0.1(8y_t) = y_t - 0.8 y_t = \mathbf{0.2 y_t}$$

#### Step 2: Step-by-Step Iterations
- **Starting Point $\mathbf{w}_0$:**
  $$x_0 = 4.0, \quad y_0 = 2.0$$
  $$f(\mathbf{w}_0) = 4.0^2 + 4(2.0^2) = 16.0 + 16.0 = \mathbf{32.0}$$

- **Iteration 1 ($\mathbf{w}_1$):**
  $$x_1 = 0.8(4.0) = \mathbf{3.2}$$
  $$y_1 = 0.2(2.0) = \mathbf{0.4}$$
  $$\mathbf{w}_1 = \begin{bmatrix} 3.2 \\ 0.4 \end{bmatrix}$$
  $$f(\mathbf{w}_1) = 3.2^2 + 4(0.4^2) = 10.24 + 4(0.16) = 10.24 + 0.64 = \mathbf{10.88}$$

- **Iteration 2 ($\mathbf{w}_2$):**
  $$x_2 = 0.8(3.2) = \mathbf{2.56}$$
  $$y_2 = 0.2(0.4) = \mathbf{0.08}$$
  $$\mathbf{w}_2 = \begin{bmatrix} 2.56 \\ 0.08 \end{bmatrix}$$
  $$f(\mathbf{w}_2) = 2.56^2 + 4(0.08^2) = 6.5536 + 4(0.0064) = 6.5536 + 0.0256 = \mathbf{6.5792}$$

#### Step 3: Trajectory Evaluation & Geometric Insight
- Initial Loss: $32.0000$
- Step 1 Loss: $10.8800$ (drops by $66\%$)
- Step 2 Loss: $6.5792$ (drops by another $40\%$)

**🧒 Geometric Insight:**
Notice that the $y$-coordinate collapsed by $80\%$ per step ($2.0 \to 0.4 \to 0.08$) because the $y$-slope was steep, while the $x$-coordinate shrunk by only $20\%$ per step ($4.0 \to 3.2 \to 2.56$).
This uneven rate of shrinking along different axes is called **anisotropy**, and is the primary reason advanced optimizers like Adam and RMSprop were invented!
</details>
