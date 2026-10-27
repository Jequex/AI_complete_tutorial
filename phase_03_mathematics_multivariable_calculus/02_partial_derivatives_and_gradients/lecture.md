# Topic 3.2: Partial Derivatives, Gradients & Steepest Descent

Welcome to **Topic 3.2: Partial Derivatives, Gradients & Steepest Descent**. In deep learning, neural networks do not optimize functions of a single variable; they optimize loss functions that depend on **millions or billions of parameters** simultaneously:
$$\mathcal{L}(w_1, w_2, \dots, w_D)$$

To navigate this unimaginably vast mathematical landscape, we use **Partial Derivatives** and the **Gradient Vector $\nabla f$**.

---

## 1. 🧒 The 12-Year-Old Analogy: Maya Blindfolded on Foggy Mount Mist

Imagine Maya is hiking on **Mount Mist** when an ultra-thick fog rolls in. She can't see her hands in front of her face, and her phone has zero reception.
She needs to get down the mountain to the cozy valley cabin at the bottom where Timmy is making hot chocolate.

```text
                     Maya at Peak of Foggy Mountain
                                    ▲
                                   / \  ▲ ∇f (Gradient: Points Steepest Uphill!)
                                  /   \ │
                                 /  ●  \│
               Maya feels ground ╲     ╱
               with her hiking boots: ╲
                                       ╲ -∇f (Negative Gradient: Roll downhill!)
                                        ▼
                                   [ Cozy Valley Cabin ★ ]
```

1. **Feeling with Her Boots (Partial Derivatives):**
   Maya can't see the whole mountain, but she can tap the ground with her hiking boots:
   - She nudges her right foot **strictly East** by one inch and feels if the dirt rises or falls. That is the **partial derivative with respect to $x$** ($\frac{\partial f}{\partial x}$).
   - She nudges her left foot **strictly North** by one inch and feels if the dirt rises or falls. That is the **partial derivative with respect to $y$** ($\frac{\partial f}{\partial y}$).
2. **The Magic Compass (The Gradient $\nabla f$):**
   Maya combines those two boot-taps into an arrow: $\nabla f = [\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}]^T$. 
   This vector points **straight uphill** along the steepest incline on the entire mountain!
3. **The Escape Strategy (Negative Gradient $-\nabla f$):**
   Because Maya wants to reach the warm cabin at the bottom of the valley, she does the exact opposite: she takes a confident step in the direction of **$-\nabla f$** (Steepest Descent). If she repeats this boot-tap step over and over, she is guaranteed to slide directly down to the bottom of the bowl!

---

## 2. 📋 Mathematical Symbol Decoder Table

| Symbol | Pronunciation | Plain English Meaning | Concrete Real-World Analogy / Flashcard Example |
| :---: | :---: | :--- | :--- |
| $f(x, y)$ | "f of x, y" | A landscape or surface where altitude depends on two coordinates | The height of a mountain peak at longitude $x$ and latitude $y$ |
| $\partial$ | "del" or "partial" | Special curly d denoting a partial derivative | A frozen camera lens locking all other variables in place |
| $\frac{\partial f}{\partial x}$ | "del f del x" | Slope in the $x$-direction while freezing $y$ | Walking strictly East-West along a hiking trail |
| $\frac{\partial f}{\partial y}$ | "del f del y" | Slope in the $y$-direction while freezing $x$ | Walking strictly North-South along a hiking trail |
| $\nabla f$ | "grad f" or "nabla f" | Gradient vector containing all partial derivatives | A 3D compass needle pointing directly uphill toward the peak |
| $-\nabla f$ | "negative grad f" | Direction of steepest downhill slope | The exact path a bowling ball rolls when dropped on a slope |
| $\eta$ | "eta" | Learning rate (step size multiplier) | The stride length of Maya's hiking boots (baby steps vs giant leaps) |
| $D_{\mathbf{u}} f$ | "D sub u of f" | Directional derivative along arbitrary unit direction $\mathbf{u}$ | How steep the hill feels when you hike diagonally Northwest |
| $\|\nabla f\|$ | "norm of nabla f" | The steepness (magnitude) of the gradient | How severely your calves burn when walking straight up the hill |

---

## 3. Partial Derivatives: The "Freeze" Rule

When standing on a 3D landscape, you cannot ask *"what is the slope?"* because the slope depends entirely on which direction you face!

### The Golden Rule: Freeze All Other Variables!
To compute the partial derivative with respect to $x$ ($\frac{\partial f}{\partial x}$):
1. **Freeze $y$:** Treat $y$ as if it were a fixed, harmless constant number like $5$ or $\pi$.
2. **Differentiate with respect to $x$** using standard single-variable calculus rules.

#### Formal Limit Definitions:
$$\frac{\partial f}{\partial x}(x_0, y_0) = \lim_{h \to 0} \frac{f(x_0 + h, y_0) - f(x_0, y_0)}{h}$$
$$\frac{\partial f}{\partial y}(x_0, y_0) = \lim_{h \to 0} \frac{f(x_0, y_0 + h) - f(x_0, y_0)}{h}$$

---

### Step-by-Step Numerical Walkthrough:
Let $f(x, y) = 3x^2 y + 5y^3 - 4x$.

1. **Compute $\frac{\partial f}{\partial x}$:**
   - Treat $y$ as a constant.
   - $\frac{\partial}{\partial x}(3x^2 y) = 3y \cdot \frac{d}{dx}(x^2) = 3y(2x) = 6xy$.
   - $\frac{\partial}{\partial x}(5y^3) = 0$ (no $x$ appears, so it acts like a static constant number!).
   - $\frac{\partial}{\partial x}(-4x) = -4$.
   $$\mathbf{\frac{\partial f}{\partial x} = 6xy - 4}$$

2. **Compute $\frac{\partial f}{\partial y}$:**
   - Treat $x$ as a constant.
   - $\frac{\partial}{\partial y}(3x^2 y) = 3x^2 \cdot \frac{d}{dy}(y) = 3x^2(1) = 3x^2$.
   - $\frac{\partial}{\partial y}(5y^3) = 5(3y^2) = 15y^2$.
   - $\frac{\partial}{\partial y}(-4x) = 0$.
   $$\mathbf{\frac{\partial f}{\partial y} = 3x^2 + 15y^2}$$

3. **Evaluate at point $(x = 2, y = 1)$:**
   $$\frac{\partial f}{\partial x}(2, 1) = 6(2)(1) - 4 = 12 - 4 = \mathbf{8}$$
   $$\frac{\partial f}{\partial y}(2, 1) = 3(2^2) + 15(1^2) = 12 + 15 = \mathbf{27}$$
   Moving 1 unit East rises by 8 feet; moving 1 unit North rises by 27 feet!

---

## 4. The Gradient Vector $\nabla f$

The **Gradient** collects every individual partial derivative into a single column vector:

$$\nabla f(\mathbf{x}) = \begin{bmatrix}
\frac{\partial f}{\partial x_1} \\
\frac{\partial f}{\partial x_2} \\
\vdots \\
\frac{\partial f}{\partial x_D}
\end{bmatrix} \in \mathbb{R}^D$$

In modern Large Language Models (like Llama-3 or GPT-4), $D = 70{,}000{,}000{,}000$ (70 billion parameters).
The gradient $\nabla \mathcal{L}$ is a 70-billion-dimensional arrow pointing in the direction that maximizes error!

---

## 5. Directional Derivatives & Proof of Steepest Ascent

What if Maya walks diagonally in a direction $\mathbf{u}$ that is neither purely East nor purely North?
If $\mathbf{u} = [u_x, u_y]^T$ is a unit vector ($\|\mathbf{u}\| = 1$), the slope she feels is the **Directional Derivative $D_{\mathbf{u}} f$**:

$$D_{\mathbf{u}} f(\mathbf{x}) = \nabla f(\mathbf{x}) \cdot \mathbf{u}$$

### Line-by-Line Proof of Steepest Ascent:
By the geometric definition of the dot product:
$$D_{\mathbf{u}} f = \nabla f \cdot \mathbf{u} = \|\nabla f\| \cdot \underbrace{\|\mathbf{u}\|}_{= 1} \cdot \cos \theta = \|\nabla f\| \cos \theta$$
where $\theta$ is the angle between the gradient vector $\nabla f$ and your walking direction $\mathbf{u}$.

```text
                  ▲ ∇f (Gradient: Maximum Uphill!)
                  │
                  │ θ = 0°  ==> cos(0°) = +1  (Slope = +||∇f||)
                  │
   ◀──────────────┼──────────────▶ u (Walking direction)
   θ = 180°       │ θ = 90° ==> cos(90°) = 0  (Contour line: Flat!)
   cos(180°) = -1 │
   (Slope = -||∇f||)
                  ▼ -∇f (Negative Gradient: Steepest Descent!)
```

1. **Steepest Ascent ($\theta = 0^\circ$):**
   The maximum value of $\cos\theta$ is $+1$. This occurs when $\mathbf{u}$ points in the **identical direction as $\nabla f$**.
   The steepest uphill slope is $+\|\nabla f\|$.
2. **Steepest Descent ($\theta = 180^\circ$):**
   The minimum value of $\cos\theta$ is $-1$. This occurs when $\mathbf{u}$ points in the **exact opposite direction: $-\nabla f$**.
   The steepest downhill slope is $-\|\nabla f\|$.
3. **Flat Contour Line ($\theta = 90^\circ$):**
   When walking perpendicular to the gradient, $\cos(90^\circ) = 0$. Altitude does not change at all!

> [!IMPORTANT] Fundamental Geometric Law
> **The gradient vector $\nabla f$ is always perpendicular (orthogonal) to the contour lines (level curves) of the function.**

---

## 6. Gradient Descent: The Master Algorithm of Machine Learning

Armed with the proof of steepest descent, we can state the master formula that trains almost every AI model on Earth:

$$\underbrace{\mathbf{w}_{t+1}}_{\text{New parameter position}} = \underbrace{\mathbf{w}_t}_{\text{Current position}} - \underbrace{\eta}_{\text{Learning rate}} \cdot \underbrace{\nabla \mathcal{L}(\mathbf{w}_t)}_{\text{Compass pointing straight uphill}}$$

- 🧒 **What the math is actually doing:**
  1. We calculate which direction points most aggressively uphill ($\nabla \mathcal{L}$).
  2. We multiply by our boot stride length $\eta$ (e.g., $0.01$).
  3. We subtract it from our current position, taking a purposeful step downhill into the valley of lowest error!

---

## 7. Concrete Numerical Walkthrough

Let us optimize the bowl-shaped loss function:
$$f(x, y) = x^2 + 3y^2$$
We start at point $\mathbf{w}_0 = [3.0, 2.0]^T$ with learning rate $\eta = 0.1$.

1. **Calculate Gradient:**
   $$\frac{\partial f}{\partial x} = 2x, \quad \frac{\partial f}{\partial y} = 6y \implies \nabla f(x, y) = \begin{bmatrix} 2x \\ 6y \end{bmatrix}$$
2. **Evaluate at $\mathbf{w}_0 = [3.0, 2.0]^T$:**
   $$\nabla f(3.0, 2.0) = \begin{bmatrix} 2(3.0) \\ 6(2.0) \end{bmatrix} = \begin{bmatrix} 6.0 \\ 12.0 \end{bmatrix}$$
3. **Compute Next Position $\mathbf{w}_1$:**
   $$\mathbf{w}_1 = \begin{bmatrix} 3.0 \\ 2.0 \end{bmatrix} - 0.1 \begin{bmatrix} 6.0 \\ 12.0 \end{bmatrix} = \begin{bmatrix} 3.0 - 0.6 \\ 2.0 - 1.2 \end{bmatrix} = \begin{bmatrix} 2.4 \\ 0.8 \end{bmatrix}$$
4. **Compare Losses:**
   - Initial Loss: $f(3.0, 2.0) = 3.0^2 + 3(2.0^2) = 9 + 12 = \mathbf{21.0}$
   - Step 1 Loss: $f(2.4, 0.8) = 2.4^2 + 3(0.8^2) = 5.76 + 3(0.64) = 5.76 + 1.92 = \mathbf{7.68}$
   Loss dropped from $21.0$ down to $7.68$ in a single step!
