# Topic 3.5: Hessians, Curvature & Optimization in Deep Learning

Welcome to **Topic 3.5: Hessians, Curvature & Optimization in Deep Learning**. 

In Topic 3.2, we discovered the **gradient vector** $\nabla f(\mathbf{w})$, which reveals the direction of steepest uphill slope. Gradient Descent takes small steps downhill:
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla f(\mathbf{w}_t)$$

However, the gradient has a dangerous blind spot: **it only knows the instantaneous slope, not how the slope is curving!**
- Is the hill flattening out into a safe meadow?
- Is it curving sharply upward into a vertical cliff?
- Or is it a narrow, steep mountain ravine where taking a normal step catapults you into the opposite wall?

To measure and tame curvature, we need the **second derivative** of multivariable calculus: the **Hessian Matrix**.

---

## 1. 🧒 The 12-Year-Old Analogy: Maya & Barnaby's Marble Skatepark

Imagine 12-year-old Maya and her puppy Barnaby are exploring a giant skatepark filled with four strange concrete structures:

```text
       1. THE SOUP BOWL                     2. THE IGLOO DOME
       (Local Minimum: H ≻ 0)               (Local Maximum: H ≺ 0)
              ▲                                    ▲
             │ \          / │                     │   ╭──────╮   │
             │  \        /  │                     │  ╱        ╲  │
             │   ╰──●───╯   │                     │ ╱    ●     ╲ │
             └──────────────▶ x                   └──────────────▶ x
           All curves point UP                  All curves point DOWN
         (Marble settles at center)            (Marble rolls off in any way)

       3. THE PRINGLE CHIP                  4. THE NARROW CANYON
       (Saddle Point: Indefinite)           (Ill-Conditioned Ravine: κ ≫ 1)
              ▲                                    ▲
             │      ╱╲      │                     │ |╲        ╱| │  Super steep walls!
             │    ╱    ╲    │                     │ | ╲  ●   ╱ | │
             │  ●────────●  │                     │ |  ╰────╯  | │  Very gentle riverbed
             └──────────────▶ x                   └──────────────▶ x
          Curves UP one way,                   Ping-pongs back & forth
          curves DOWN the other!               unless curvature is normalized
```

1. **The Soup Bowl (Local Minimum, Positive Definite):**
   The ground curves upward in *every single direction*. If Maya drops a marble anywhere, it rolls down and settles comfortably at the bottom ($H \succ 0$, all eigenvalues positive).
2. **The Igloo Dome (Local Maximum, Negative Definite):**
   The ground curves downward in every direction. If Barnaby balances a marble at the very top, the slightest breeze sends it rolling down ($H \prec 0$, all eigenvalues negative).
3. **The Pringle Potato Chip (Saddle Point, Indefinite):**
   Along the front-to-back axis, the chip curves *upward* like a bowl. Along the left-to-right axis, the chip curves *downward* like a slide! A marble dropped here can get stuck on a flat plateau even though it's not a true minimum.
4. **The Narrow Canyon / Ravine (High Condition Number $\kappa \gg 1$):**
   The canyon walls are nearly vertical (huge curvature), but the riverbed slopes gently forward (tiny curvature). Standard gradient descent ping-pongs wildly between the walls instead of walking down the riverbed!
5. **The Hessian ($H$):**
   The Hessian is the mathematical radar that maps out which of these four shapes you are standing on!

---

## 2. 📋 Mathematical Symbol Decoder Table

| Symbol | Pronunciation | Plain English Meaning | Concrete Real-World Analogy / Flashcard Example |
| :---: | :---: | :--- | :--- |
| $\nabla f(\mathbf{x})$ | "grad f" or "nabla f" | First derivatives: vector of slopes along each axis | A compass showing which way your skateboard rolls |
| $H \text{ or } \nabla^2 f$ | "Hessian of f" | Second derivatives: square matrix of curvatures | A 3D scanner measuring whether the ground is a bowl, dome, or saddle |
| $H_{ij}$ | "H i j" | Second partial derivative $\frac{\partial^2 f}{\partial x_i \partial x_j}$ | How much the East-West slope twists when you step North |
| $\lambda_i$ | "lambda i" | Eigenvalue of the Hessian matrix | The pure steepness curvature along a principal axis of the bowl |
| $H \succ 0$ | "H is positive definite" | All eigenvalues are positive ($\lambda_i > 0$) | A perfect soup bowl: curves up in every single direction |
| $H \prec 0$ | "H is negative definite" | All eigenvalues are negative ($\lambda_i < 0$) | An umbrella or igloo: curves down in every single direction |
| $H^{-1}$ | "H inverse" | Inverse Hessian matrix | The magic lens that stretches an oval ravine into a round, easy bowl |
| $\kappa(H)$ | "kappa of H" | Condition number: $\frac{\lambda_{\max}}{\lambda_{\min}}$ | Ratio of canyon wall steepness to riverbed slope ($1 = \text{circle}$, $1000 = \text{canyon}$) |
| $\text{HVP}$ | "H-V-P" | Hessian-Vector Product: $H \mathbf{v}$ | Measuring curvature along one direction without computing the full matrix |

---

## 3. Mathematical Definition of the Hessian Matrix

Let $f: \mathbb{R}^n \to \mathbb{R}$ be a twice-differentiable function. The **Hessian matrix** $H \in \mathbb{R}^{n \times n}$ collects all second-order partial derivatives:

$$H = \nabla^2 f(\mathbf{x}) = \begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
\end{bmatrix}$$

### Structural Properties:
1. **Symmetry (Clairaut's / Schwarz's Theorem):**
   If second partial derivatives are continuous, the order of differentiation does not matter:
   $$\frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial^2 f}{\partial x_j \partial x_i} \implies \mathbf{H = H^T}$$
2. **Jacobian of the Gradient:**
   The Hessian is simply the Jacobian of the gradient vector: $H = J(\nabla f)$.

---

## 4. Classifying Critical Points with Hessian Eigenvalues

At a critical point $\mathbf{x}^*$, the gradient is zero: $\nabla f(\mathbf{x}^*) = \mathbf{0}$.
To determine whether it is a Minimum, Maximum, or Saddle Point, we examine the eigenvalues $\lambda_1, \dots, \lambda_n$ of the Hessian:

| Eigenvalue Signs | Matrix Type | 3D Geometry | Optimization Status |
| :--- | :---: | :---: | :--- |
| **All $\lambda_i > 0$** | Positive Definite ($H \succ 0$) | Upward Soup Bowl | **Local Minimum** (Desired Goal!) |
| **All $\lambda_i < 0$** | Negative Definite ($H \prec 0$) | Downward Igloo Dome | **Local Maximum** (Worst Spot) |
| **Mixed signs ($+$ and $-$)** | Indefinite | Pringle Saddle Point | **Saddle Point** (Escapable Plateau) |
| **Some $\lambda_i = 0$** | Semi-definite | Flat Trench / Valley | Inconclusive (Higher orders needed) |

---

## 5. Second-Order Optimization: Newton's Method

While Gradient Descent uses a flat linear approximation, **Newton's Method** fits a local quadratic bowl to the loss landscape using Taylor series:
$$f(\mathbf{w} + \Delta\mathbf{w}) \approx f(\mathbf{w}) + \nabla f(\mathbf{w})^T \Delta\mathbf{w} + \frac{1}{2} \Delta\mathbf{w}^T H(\mathbf{w}) \Delta\mathbf{w}$$

Differentiating with respect to $\Delta\mathbf{w}$ and setting to zero yields **Newton's Optimization Step**:

$$\mathbf{\mathbf{w}_{t+1} = \mathbf{w}_t - H^{-1}(\mathbf{w}_t) \nabla f(\mathbf{w}_t)}$$

#### Line-by-Line Formula Breakdown:
$$\underbrace{\mathbf{w}_{t+1}}_{\text{Next position}} = \underbrace{\mathbf{w}_t}_{\text{Current point}} - \underbrace{H^{-1}(\mathbf{w}_t)}_{\text{Curvature normalizer}} \cdot \underbrace{\nabla f(\mathbf{w}_t)}_{\text{Steepest ascent slope}}$$

- 🧒 **What the math is actually doing:**
  Gradient descent takes a fixed step along the steepest slope. 
  Newton's method multiplies by $H^{-1}$, which automatically **shrinks** steps along steep canyon walls and **stretches** steps along flat riverbeds! On a pure quadratic bowl, Newton's method lands on the exact minimum in **a single step**!

---

## 6. The Trillion-Dollar Trap: Why Deep Learning Cannot Invert the Hessian

If Newton's method is so fast, why doesn't PyTorch use it to train LLMs?
Because of **The Curse of Dimension**:

1. **Memory Storage:**
   For an LLM with $n = 70 \text{ Billion parameters}$, the Hessian has $n \times n$ numbers:
   $$(70 \times 10^9)^2 = 4.9 \times 10^{21} \text{ floats} \approx \mathbf{20 \text{ Billion Gigabytes of RAM!}}$$
2. **Matrix Inversion:**
   Inverting an $n \times n$ matrix takes $\mathcal{O}(n^3)$ operations. For $n = 70 \times 10^9$, computing $H^{-1}$ would take longer than the age of the universe!

> [!TIP] How Modern AI Solves This
> 1. **First-order adaptive optimizers (AdamW, RMSprop):** Use running averages of gradients to approximate the diagonal elements of the Hessian ($\mathcal{O}(n)$ memory).
> 2. **Hessian-Vector Products (HVP):** Using Pearlmutter's trick, Pearlmutter showed that $H \mathbf{v} = \left. \frac{d}{d\epsilon} \nabla f(\mathbf{w} + \epsilon \mathbf{v}) \right|_{\epsilon = 0}$ can be computed with two backward passes, enabling **Hessian-Free Optimization** without ever forming $H$!

---

## 7. Concrete Numerical Walkthrough

Let us optimize $f(x, y) = x^3 - 3x + y^2$.

1. **Find Gradient:**
   $$\nabla f(x, y) = \begin{bmatrix} 3x^2 - 3 \\ 2y \end{bmatrix}$$
2. **Find Critical Points ($\nabla f = \mathbf{0}$):**
   $$3x^2 - 3 = 0 \implies x = \pm 1, \quad 2y = 0 \implies y = 0$$
   - Critical Point A: $(+1, 0)$
   - Critical Point B: $(-1, 0)$
3. **Compute Hessian Matrix:**
   $$H(x, y) = \begin{bmatrix} 6x & 0 \\ 0 & 2 \end{bmatrix}$$
4. **Classify Point A $(+1, 0)$:**
   $$H(1, 0) = \begin{bmatrix} 6 & 0 \\ 0 & 2 \end{bmatrix} \implies \lambda_1 = +6, \quad \lambda_2 = +2$$
   Both eigenvalues positive $\implies \mathbf{H \succ 0 \text{ (Local Minimum)}}$!
5. **Classify Point B $(-1, 0)$:**
   $$H(-1, 0) = \begin{bmatrix} -6 & 0 \\ 0 & 2 \end{bmatrix} \implies \lambda_1 = -6, \quad \lambda_2 = +2$$
   One negative, one positive $\implies \mathbf{\text{Saddle Point}}$!
