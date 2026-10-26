# Topic 3.1: Functions, Limits, Continuity, and Derivatives

Welcome to **Topic 3.1: Functions, Limits, Continuity, and Derivatives**. Calculus is the mathematical foundation of how artificial neural networks learn. When an AI makes an error on an image classification or text prediction, it uses derivatives to determine: *"If I tweak this weight by a tiny hair, will the error go up or down, and by how much?"*

---

## 1. Symbol Decoder Table

| Symbol | Mathematical Name | Plain-English Meaning | Child-Friendly Analogy |
| :---: | :--- | :--- | :--- |
| $f(x)$ | Function | A rule that transforms input $x$ into output $y$ | A vending machine that drops a snack when you put in coins |
| $\lim_{x \to c} f(x)$ | Limit | What value $f(x)$ approaches as $x$ gets infinitely close to $c$ | Walking toward a doorway to see where it leads without stepping inside |
| $\frac{df}{dx}$ or $f'(x)$ | Derivative | The instantaneous rate of change of $f$ with respect to $x$ | The speedometer needle showing your exact speed at this split second |
| $h$ or $\Delta x$ | Infinitesimal Nudge | An ultra-tiny step along the $x$-axis | Tiptoeing forward by a single millimeter |
| $\sigma(x)$ | Sigmoid Activation | S-shaped smooth curve squashing numbers into $(0, 1)$ | A dimmer switch turning light from off (0) to on (1) |
| $\sigma'(x)$ | Sigmoid Derivative | Rate of change of the sigmoid curve | How sensitive the light brightness is at the current switch position |

---

## 2. Limits and Continuity

### The 12-Year-Old Analogy: The Zoom Microscope
Imagine drawing a curvy roller-coaster track on a piece of paper.
If you look at the whole track from across the room, it looks bent, looping, and complicated.
Now, put your eye right up to a high-powered **microscope** and zoom in $10{,}000\times$ on any smooth point along the track.
At that extreme zoom level, the curve looks like a **completely straight flat line**!
A **derivative** is simply the slope of that straight line under the microscope!

```
     Macro View (Full Curve)               Micro View (Zoomed In 10,000x)
            ▲                                      ▲
            │       ╭────╮                         │       ╱ Tangent
            │     ╭─╯    ╰╮                        │      ╱  Slope: m = df/dx
            │   ╭─╯       ╰─╮                      │     ╱
            └───┴───────────┴──▶ x                 └────┴────────▶ x
```

### 2.1 The Concept of a Limit
A limit asks: *"As input $x$ creeps arbitrarily close to a target number $c$, what number does output $f(x)$ get close to?"*

$$\lim_{x \to c} f(x) = L$$

*Why Limits Matter in AI:*
When computing rates of change, we want to know what happens when an interval size $h$ shrinks to **zero**. We cannot simply divide by $h = 0$ (which causes a divide-by-zero crash!). The limit allows us to let $h$ get infinitely close to zero without breaking arithmetic laws.

### 2.2 Continuity
A function $f(x)$ is **continuous** at point $c$ if:
1. $f(c)$ exists (no missing holes).
2. $\lim_{x \to c} f(x)$ exists (left and right approaches match).
3. $\lim_{x \to c} f(x) = f(c)$ (no sudden teleports or vertical jumps).

*Why Continuity Matters in AI:*
Neural network loss functions must be continuous so that gradient descent can walk smoothly down the loss landscape without encountering bottomless pits or discontinuous walls.

---

## 3. The Derivative: Geometric, Physical, and Computational

### 3.1 The Formal Limit Definition
To find the slope at point $x$, pick a neighboring point $x + h$. The slope of the secant line connecting them is:
$$\text{Average Rate of Change} = \frac{\Delta y}{\Delta x} = \frac{f(x + h) - f(x)}{h}$$

Shrinking the step size $h \to 0$ turns the secant line into the **tangent line**:

$$\frac{df}{dx} = f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}$$

### 3.2 The Three Lenses of the Derivative:
1. **Geometric Lens:** The derivative $f'(x)$ is the **slope of the tangent line** touching the curve at $x$.
2. **Physical Lens:** The derivative is the **instantaneous rate of change** (e.g., velocity is the derivative of position with respect to time).
3. **Computational / AI Lens (Sensitivity):** The derivative measures how sensitive the output is to a tiny change in the input:
   $$\Delta f \approx f'(x) \cdot \Delta x$$
   If $f'(x) = 3.0$, increasing $x$ by $0.01$ will increase the loss by approximately $0.03$.

---

## 4. Fundamental Rules of Differentiation

### 4.1 The Power Rule
For any real power $n$:
$$\frac{d}{dx} \left( x^n \right) = n x^{n-1}$$

#### Step-by-Step Derivation for $n = 2$ ($f(x) = x^2$):
Using the limit definition:
$$f'(x) = \lim_{h \to 0} \frac{(x + h)^2 - x^2}{h}$$
Expand the numerator:
$$(x + h)^2 - x^2 = x^2 + 2xh + h^2 - x^2 = 2xh + h^2$$
Divide by $h$:
$$\frac{2xh + h^2}{h} = 2x + h$$
Take the limit as $h \to 0$:
$$f'(x) = \lim_{h \to 0} (2x + h) = \mathbf{2x}$$
*(Matches the power rule: $2 x^{2-1} = 2x$!)*

---

### 4.2 The Product Rule
When two functions are multiplied together:
$$\frac{d}{dx} [f(x) \cdot g(x)] = f'(x) g(x) + f(x) g'(x)$$

#### The 12-Year-Old Rectangle Analogy:
Imagine a rectangular carpet with width $u$ and height $v$. Its area is $A = u \cdot v$.
If you expand the width by a tiny nudge $\Delta u$ and height by $\Delta v$:

```
               u             Δu
         ┌───────────────┬──────┐
         │               │      │
       v │   Old Area    │ u·Δv │
         │    (u · v)    │      │
         ├───────────────┼──────┤
      Δv │     v · Δu    │Δu·Δv │ ◀ (Microscopic corner ≈ 0)
         └───────────────┴──────┘
```
The change in area is:
$$\Delta A = v \cdot \Delta u + u \cdot \Delta v + \Delta u \Delta v$$
Dividing by $\Delta x$ and letting $\Delta x \to 0$, the tiny corner $\Delta u \Delta v$ vanishes to zero, leaving:
$$\frac{dA}{dx} = v \frac{du}{dx} + u \frac{dv}{dx}$$

---

### 4.3 The Quotient Rule
$$\frac{d}{dx} \left[ \frac{f(x)}{g(x)} \right] = \frac{f'(x) g(x) - f(x) g'(x)}{[g(x)]^2}$$

---

## 5. Activation Function Derivatives in Deep Learning

### 5.1 The Sigmoid Activation Function
$$\sigma(x) = \frac{1}{1 + e^{-x}} = (1 + e^{-x})^{-1}$$

#### Complete Mathematical Derivation of $\sigma'(x)$:
Apply the chain rule and power rule:
$$\frac{d\sigma}{dx} = -1 \cdot (1 + e^{-x})^{-2} \cdot \frac{d}{dx}(1 + e^{-x})$$
$$\frac{d\sigma}{dx} = -(1 + e^{-x})^{-2} \cdot (-e^{-x}) = \frac{e^{-x}}{(1 + e^{-x})^2}$$

Now, rewrite algebraically using a clever substitution:
$$\frac{e^{-x}}{(1 + e^{-x})^2} = \left( \frac{1}{1 + e^{-x}} \right) \cdot \left( \frac{e^{-x}}{1 + e^{-x}} \right)$$
Notice that:
$$\frac{e^{-x}}{1 + e^{-x}} = \frac{1 + e^{-x} - 1}{1 + e^{-x}} = 1 - \frac{1}{1 + e^{-x}} = 1 - \sigma(x)$$

Therefore:
$$\mathbf{\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))}$$

> [!TIP] Beautiful Computational Efficiency
> To compute the derivative of a Sigmoid neuron during backpropagation, you do **not** need to re-evaluate expensive exponentials! If you already cached the forward output $y = \sigma(x)$, the derivative is simply:
> $$dy = y \cdot (1.0 - y)$$

---

### 5.2 The ReLU Activation Function
$$\text{ReLU}(x) = \max(0, x)$$
$$\text{ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x < 0 \end{cases}$$
*(At $x = 0$, the derivative is mathematically undefined due to a sharp corner. In deep learning software, we adopt the subgradient convention $\text{ReLU}'(0) = 0$).*

---

## 6. Numerical Differentiation: Finite Differences

How do deep learning frameworks test whether their analytical backpropagation code is bug-free?
Through **Numerical Gradient Checking**!

### Two Numerical Approximation Formulas:
1. **Forward Difference:**
   $$f'(x) \approx \frac{f(x + h) - f(x)}{h} \quad (\text{Error: } \mathcal{O}(h))$$
2. **Central Difference (Far more accurate!):**
   $$f'(x) \approx \frac{f(x + h) - f(x - h)}{2h} \quad (\text{Error: } \mathcal{O}(h^2))$$

#### Taylor Series Proof of Central Difference $\mathcal{O}(h^2)$ Accuracy:
Taylor expansion of $f(x + h)$ and $f(x - h)$:
$$f(x + h) = f(x) + h f'(x) + \frac{h^2}{2} f''(x) + \frac{h^3}{6} f'''(x) + \dots$$
$$f(x - h) = f(x) - h f'(x) + \frac{h^2}{2} f''(x) - \frac{h^3}{6} f'''(x) + \dots$$

Subtracting the two equations cancels all even-powered terms ($f(x)$ and $f''(x)$):
$$f(x + h) - f(x - h) = 2h f'(x) + \frac{2h^3}{6} f'''(x) + \dots$$
Dividing by $2h$:
$$\frac{f(x + h) - f(x - h)}{2h} = f'(x) + \frac{h^2}{6} f'''(x) + \dots = f'(x) + \mathcal{O}(h^2)$$

For $h = 10^{-5}$, forward difference has error $\approx 10^{-5}$, whereas central difference has error $\approx 10^{-10}$—**100,000 times more accurate!**
