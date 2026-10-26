# Topic 3.1: Exercises & Solutions — Functions, Limits & Derivatives

---

## Problem Set Overview

| Problem | Focus | Difficulty |
| :--- | :--- | :--- |
| **1** | **First-Principles Limit Derivation of $f(x) = 1/x$** | Medium |
| **2** | **Step-by-Step Derivation of the $\tanh(x)$ Activation Derivative** | Medium |
| **3** | **Finite Difference Error Analysis & Optimal Step Size $h$** | Challenging |

---

## Problem 1: First-Principles Limit Derivation of $f(x) = 1/x$

### The Context
The limit definition of the derivative is the foundational root from which all differentiation rules arise:
$$f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}$$

### Task
1. Using **only** the limit definition above, derive the derivative of $f(x) = \frac{1}{x}$ for $x \neq 0$.
2. Verify that your result matches the Power Rule formula $\frac{d}{dx} x^n = n x^{n-1}$ for $n = -1$.
3. Compute the equation of the tangent line to $f(x) = \frac{1}{x}$ at $x = 2$.

---

### Solution 1

#### 1. Limit Derivation
Substitute $f(x) = \frac{1}{x}$:
$$f'(x) = \lim_{h \to 0} \frac{\frac{1}{x + h} - \frac{1}{x}}{h}$$
Find a common denominator for the numerator:
$$\frac{1}{x + h} - \frac{1}{x} = \frac{x - (x + h)}{x(x + h)} = \frac{-h}{x(x + h)}$$
Now divide by $h$:
$$\frac{\frac{-h}{x(x + h)}}{h} = \frac{-1}{x(x + h)}$$
Taking the limit as $h \to 0$:
$$f'(x) = \lim_{h \to 0} \frac{-1}{x(x + h)} = \frac{-1}{x(x + 0)} = \mathbf{-\frac{1}{x^2}}$$

#### 2. Verification with Power Rule
Setting $n = -1$:
$$\frac{d}{dx} \left( x^{-1} \right) = (-1) x^{-1 - 1} = -1 \cdot x^{-2} = -\frac{1}{x^2} \quad \text{(Exact match!)}$$

#### 3. Tangent Line at $x = 2$
- Point: $(x_0, y_0) = (2, f(2)) = (2, 0.5)$
- Slope: $m = f'(2) = -\frac{1}{2^2} = -0.25$
- Point-slope line equation: $y - y_0 = m(x - x_0)$
  $$y - 0.5 = -0.25(x - 2) \implies y = -0.25x + 0.5 + 0.5 \implies \mathbf{y = -0.25x + 1.0}$$

---

## Problem 2: Derivation of the $\tanh(x)$ Activation Derivative

### The Context
The Hyperbolic Tangent ($\tanh$) activation function squashes any real input into $(-1, +1)$, and was the historical standard activation in Recurrent Neural Networks (RNNs) and LSTMs:
$$\tanh(x) = \frac{\sinh(x)}{\cosh(x)} = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

### Task
1. Using the Quotient Rule $\left( \frac{u}{v} \right)' = \frac{u'v - uv'}{v^2}$, derive $\frac{d}{dx} \tanh(x)$.
2. Show that the derivative simplifies into the elegant form:
   $$\tanh'(x) = 1 - \tanh^2(x)$$
3. If a hidden neuron has forward activation output $a = 0.8$, compute its backpropagation derivative $da$ with zero exponential evaluations.

---

### Solution 2

#### 1. Applying the Quotient Rule
Let $u(x) = e^x - e^{-x} \implies u'(x) = e^x + e^{-x}$
Let $v(x) = e^x + e^{-x} \implies v'(x) = e^x - e^{-x}$

$$\frac{d}{dx} \tanh(x) = \frac{u' v - u v'}{v^2} = \frac{(e^x + e^{-x})(e^x + e^{-x}) - (e^x - e^{-x})(e^x - e^{-x})}{(e^x + e^{-x})^2}$$
$$\frac{d}{dx} \tanh(x) = \frac{(e^x + e^{-x})^2 - (e^x - e^{-x})^2}{(e^x + e^{-x})^2}$$

#### 2. Algebraic Simplification
Split the fraction:
$$\frac{d}{dx} \tanh(x) = \frac{(e^x + e^{-x})^2}{(e^x + e^{-x})^2} - \frac{(e^x - e^{-x})^2}{(e^x + e^{-x})^2}$$
$$= 1 - \left( \frac{e^x - e^{-x}}{e^x + e^{-x}} \right)^2$$
Since $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$, we obtain:
$$\mathbf{\tanh'(x) = 1 - \tanh^2(x)} \quad \blacksquare$$

#### 3. Efficient Backprop Gradient Calculation
Given cached forward activation $a = \tanh(x) = 0.8$:
$$da = 1 - a^2 = 1 - (0.8)^2 = 1 - 0.64 = \mathbf{0.36}$$

---

## Problem 3: Finite Difference Error Analysis & Optimal Step Size $h$

### The Context
When testing neural network autograd code, we approximate derivatives numerically using finite differences.
In double-precision IEEE-754 arithmetic, machine epsilon is $\epsilon_{\text{mach}} \approx 10^{-16}$.

### Theoretical Analysis
The total numerical error consists of two opposing forces:
1. **Truncation Error (Math):** The higher-order terms discarded from Taylor series.
   - For Central Difference: $\text{Truncation Error} \approx c_1 h^2$.
2. **Roundoff Error (Computer):** Subtracting two nearly identical floating point numbers loses significant digits:
   - Roundoff Error $\approx \frac{\epsilon_{\text{mach}}}{h}$.

$$\text{Total Error}(h) \approx c_1 h^2 + \frac{\epsilon_{\text{mach}}}{h}$$

1. Differentiate the total error with respect to $h$, set it to zero, and find the optimal step size $h^*$.
2. If $\epsilon_{\text{mach}} \approx 10^{-16}$, what is the order of magnitude of $h^*$ for central difference?
3. What is the optimal step size for Forward Difference ($\text{Error} \approx c_1 h + \frac{\epsilon_{\text{mach}}}{h}$)?

---

### Solution 3

#### 1. Minimizing Total Error for Central Difference
$$\frac{d}{dh} \left( c_1 h^2 + \epsilon_{\text{mach}} h^{-1} \right) = 2 c_1 h - \frac{\epsilon_{\text{mach}}}{h^2} = 0$$
$$2 c_1 h = \frac{\epsilon_{\text{mach}}}{h^2} \implies h^3 = \frac{\epsilon_{\text{mach}}}{2 c_1}$$
$$h^* \propto \epsilon_{\text{mach}}^{1/3}$$

#### 2. Order of Magnitude for Central Difference
$$h^* \approx (10^{-16})^{1/3} \approx \mathbf{10^{-5.3} \approx 10^{-5} \text{ to } 10^{-6}}$$
*Result: This mathematically proves why deep learning frameworks universally set `eps = 1e-5` or `1e-6` during gradient checking! If you pick $h = 10^{-12}$, computer roundoff error explodes!*

#### 3. Optimal Step Size for Forward Difference
$$\frac{d}{dh} \left( c_1 h + \frac{\epsilon_{\text{mach}}}{h} \right) = c_1 - \frac{\epsilon_{\text{mach}}}{h^2} = 0 \implies h^2 \propto \epsilon_{\text{mach}}$$
$$h^*_{\text{forward}} \propto \sqrt{\epsilon_{\text{mach}}} \approx \sqrt{10^{-16}} = \mathbf{10^{-8}}$$
However, even at its best $h$, forward difference error is $O(10^{-8})$, whereas central difference achieves $O(h^2) = O((10^{-5.3})^2) \approx O(10^{-11})$—three orders of magnitude more precise!
