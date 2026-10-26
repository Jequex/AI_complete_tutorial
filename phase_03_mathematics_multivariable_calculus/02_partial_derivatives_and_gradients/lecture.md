# Topic 3.2: Partial Derivatives, Gradients & Steepest Descent

Welcome to **Topic 3.2: Partial Derivatives, Gradients & Steepest Descent**. In deep learning, neural networks do not optimize functions of a single variable; they optimize loss functions that depend on **millions or billions of parameters** simultaneously:
$$\mathcal{L}(w_1, w_2, \dots, w_D)$$
To navigate this unimaginably vast mathematical landscape, we use **Partial Derivatives** and the **Gradient Vector $\nabla f$**.

---

## 1. Symbol Decoder Table

| Symbol | Mathematical Name | Plain-English Meaning | Child-Friendly Analogy |
| :---: | :--- | :--- | :--- |
| $f(x, y)$ | Surface / Multivariable Function | A landscape where height $z$ depends on coordinates $(x, y)$ | The altitude of a mountain range at latitude $x$ and longitude $y$ |
| $\frac{\partial f}{\partial x}$ ("del f del x") | Partial Derivative | Slope in the $x$-direction while freezing $y$ | Walking strictly East-West along a mountain path |
| $\frac{\partial f}{\partial y}$ | Partial Derivative | Slope in the $y$-direction while freezing $x$ | Walking strictly North-South along a mountain path |
| $\nabla f$ ("grad f" / "nabla") | Gradient Vector | Vector of all partial derivatives $[\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}]^T$ | A 3D compass pointing directly uphill toward the mountain peak |
| $-\nabla f$ | Negative Gradient | Direction of steepest descent | The exact path a bowling ball rolls downhill |
| $\eta$ ("eta") | Learning Rate | Step size taken along the negative gradient | The stride length of your walking boots |
| $D_{\mathbf{u}} f$ | Directional Derivative | Slope when walking in an arbitrary direction $\mathbf{u}$ | Measuring how steep the slope feels when hiking Northwest |

---

## 2. Partial Derivatives: The "Freeze" Rule

When dealing with a multivariable function $f(x, y)$, you cannot ask *"what is the slope?"* because the slope depends on **which direction you look**.

### The 12-Year-Old Rule: Freeze All Other Variables!
To compute the partial derivative with respect to $x$ ($\frac{\partial f}{\partial x}$):
1. **Freeze $y$:** Treat $y$ as if it were a fixed, harmless constant number like $5$ or $\pi$.
2. **Differentiate with respect to $x$** using standard single-variable calculus rules!

### Formal Limit Definition:
$$\frac{\partial f}{\partial x}(x_0, y_0) = \lim_{h \to 0} \frac{f(x_0 + h, y_0) - f(x_0, y_0)}{h}$$
$$\frac{\partial f}{\partial y}(x_0, y_0) = \lim_{h \to 0} \frac{f(x_0, y_0 + h) - f(x_0, y_0)}{h}$$

---

### Step-by-Step Example:
Let $f(x, y) = 3x^2 y + 5y^3 - 4x$.

1. **Compute $\frac{\partial f}{\partial x}$:**
   - Treat $y$ as a constant.
   - $\frac{\partial}{\partial x}(3x^2 y) = 3y \cdot (2x) = 6xy$.
   - $\frac{\partial}{\partial x}(5y^3) = 0$ (since $5y^3$ has no $x$, its derivative is zero!).
   - $\frac{\partial}{\partial x}(-4x) = -4$.
   $$\mathbf{\frac{\partial f}{\partial x} = 6xy - 4}$$

2. **Compute $\frac{\partial f}{\partial y}$:**
   - Treat $x$ as a constant.
   - $\frac{\partial}{\partial y}(3x^2 y) = 3x^2 \cdot (1) = 3x^2$.
   - $\frac{\partial}{\partial y}(5y^3) = 15y^2$.
   - $\frac{\partial}{\partial y}(-4x) = 0$.
   $$\mathbf{\frac{\partial f}{\partial y} = 3x^2 + 15y^2}$$

---

## 3. The Gradient Vector $\nabla f$

The **Gradient** packs every partial derivative into a single column vector:

$$\nabla f(\mathbf{x}) = \begin{bmatrix}
\frac{\partial f}{\partial x_1} \\
\frac{\partial f}{\partial x_2} \\
\vdots \\
\frac{\partial f}{\partial x_D}
\end{bmatrix} \in \mathbb{R}^D$$

If a neural network has $D = 100{,}000$ parameters, its gradient $\nabla \mathcal{L}$ is a 100,000-dimensional vector containing the individual sensitivity of every single parameter!

---

## 4. Directional Derivatives & Mathematical Proof of Steepest Ascent

Suppose you are standing on a hill at position $\mathbf{x}$, and you choose to walk in an arbitrary direction specified by unit vector $\mathbf{u}$ ($\|\mathbf{u}\| = 1$).
What is the slope you experience under your feet?
This is the **Directional Derivative $D_{\mathbf{u}} f$**:

$$D_{\mathbf{u}} f(\mathbf{x}) = \nabla f(\mathbf{x}) \cdot \mathbf{u}$$

### The Beautiful Proof: Why the Gradient Points Steepest Uphill
Using the geometric definition of the dot product:
$$\nabla f \cdot \mathbf{u} = \|\nabla f\| \cdot \|\mathbf{u}\| \cdot \cos \theta$$
Since $\mathbf{u}$ is a unit vector ($\|\mathbf{u}\| = 1$):
$$D_{\mathbf{u}} f = \|\nabla f\| \cos \theta$$
where $\theta$ is the angle between the gradient vector $\nabla f$ and your chosen walking direction $\mathbf{u}$.

```
                 ▲ ∇f (Gradient)
                 │
                 │ θ = 0°  ==> cos(0°) = +1 (Maximum uphill slope: +||∇f||)
                 │
  ◀──────────────┼──────────────▶ u
  θ = 180°       │
  cos(180°) = -1 │
  (Steepest      ▼ -∇f (Negative Gradient: Maximum downhill slope: -||∇f||)
   Descent)
```

1. **Maximum Uphill Slope (Steepest Ascent):**
   The maximum possible value of $\cos \theta$ is $+1$, occurring when $\theta = 0^\circ$.
   This means **$\mathbf{u}$ must point in the exact same direction as $\nabla f$**!
   The maximum rate of increase is $+\|\nabla f\|$.

2. **Maximum Downhill Slope (Steepest Descent):**
   The minimum possible value of $\cos \theta$ is $-1$, occurring when $\theta = 180^\circ$.
   This means **$\mathbf{u}$ must point in the opposite direction: $-\nabla f$**!
   The greatest rate of decrease is $-\|\nabla f\|$.

3. **Zero Change (Contour / Level Set):**
   When $\theta = 90^\circ$, $\cos(90^\circ) = 0 \implies D_{\mathbf{u}} f = 0$.
   Walking perpendicular to the gradient keeps your altitude completely flat!

> [!IMPORTANT] Core Geometric Theorem
> **The gradient vector $\nabla f$ is always perpendicular (orthogonal) to the contour lines (level curves) of the function.**

---

## 5. Gradient Descent: The Master Algorithm of Machine Learning

Armed with the proof of steepest descent, we can now state the fundamental algorithm that trains almost all modern AI:

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla \mathcal{L}(\mathbf{w}_t)$$

```
     Loss Surface ℒ(w₁, w₂)
       ▲
       │        (High Loss: Random Init)
       │        • w₀
       │         ╲  -η ∇ℒ
       │          • w₁
       │           ╲
       │            • w₂
       │             ╲
       │              ★ w* (Minimum Loss / Optimal Model)
       └─────────────────────────▶ Parameters
```

### The Three Scenarios of Learning Rate $\eta$:
1. **$\eta$ too small ($10^{-6}$):** The model takes microscopic baby steps; training takes weeks and gets stuck.
2. **$\eta$ too large ($10.0$):** Steps overshoot the valley and catapult up the opposite cliff; loss explodes to `NaN` or `inf`.
3. **$\eta$ well-tuned ($10^{-3}$):** Smooth, rapid convergence toward the optimal parameter weights $\mathbf{w}^*$.
