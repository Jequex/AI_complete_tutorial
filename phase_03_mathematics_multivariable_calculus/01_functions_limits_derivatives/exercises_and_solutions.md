# Topic 3.1: Exercises & Solutions — Functions, Limits & Derivatives

Welcome to the problem set for **Topic 3.1: Functions, Limits, Continuity, and Derivatives**.

Follow along with pen and paper. Attempt each problem independently before clicking the collapsible solution tabs below.

---

## 📝 Problem Set Overview

| Problem | Focus | Difficulty | Practical AI Connection |
| :---: | :--- | :---: | :--- |
| **1** | **First-Principles Limit Derivation of $f(x) = 1/x$** | Medium | Verifying fundamental rate laws without black-box rules |
| **2** | **Step-by-Step Derivation of the $\tanh(x)$ Activation Derivative** | Medium | Speeding up backpropagation in RNNs and LSTMs |
| **3** | **Finite Difference Error Analysis & Optimal Step Size $h$** | Challenging | Production-grade gradient checking in autograd engines |

---

## Problem 1: First-Principles Limit Derivation of $f(x) = 1/x$

### 🧒 The Context: Maya's Reciprocal Energy Meter
Maya is designing a power meter for a toy robot where energy efficiency decreases with distance: $f(x) = \frac{1}{x}$.
She needs to know how sensitive the power meter is when $x = 2$ meters.
The formal limit definition of the instantaneous derivative is:
$$f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}$$

### Tasks:
1. Using **only** the limit definition above (no shortcut rules!), derive the formula for $f'(x)$.
2. Verify that your derivation matches the Power Rule formula $\frac{d}{dx}(x^n) = n x^{n-1}$ for $n = -1$.
3. Compute the exact slope and the linear tangent line equation at $x = 2$.

---

<details>
<summary>Click to view solution for Problem 1</summary>

### Detailed Step-by-Step Solution:

#### Step 1: Set Up the Difference Quotient
Substitute $f(x) = \frac{1}{x}$ into the limit definition:
$$f'(x) = \lim_{h \to 0} \frac{\frac{1}{x + h} - \frac{1}{x}}{h}$$

#### Step 2: Combine the Fractions in the Numerator
Find a common denominator for the top terms:
$$\frac{1}{x + h} - \frac{1}{x} = \frac{x - (x + h)}{x(x + h)} = \frac{x - x - h}{x(x + h)} = \frac{-h}{x(x + h)}$$

#### Step 3: Divide by $h$
$$\frac{\frac{-h}{x(x + h)}}{h} = \frac{-h}{x(x + h) \cdot h} = \frac{-1}{x(x + h)}$$

#### Step 4: Evaluate the Limit as $h \to 0$
As $h$ creeps to zero:
$$f'(x) = \lim_{h \to 0} \frac{-1}{x(x + h)} = \frac{-1}{x(x + 0)} = \mathbf{-\frac{1}{x^2}}$$

#### Step 5: Power Rule Verification
Rewrite $f(x) = \frac{1}{x} = x^{-1}$.
Using $\frac{d}{dx}(x^n) = n x^{n-1}$ with $n = -1$:
$$\frac{d}{dx}(x^{-1}) = (-1) x^{-1 - 1} = -1 \cdot x^{-2} = \mathbf{-\frac{1}{x^2}} \quad \text{✅ Exact Match!}$$

#### Step 6: Tangent Line Equation at $x = 2$
1. **Find point coordinates $(x_0, y_0)$:**
   $$x_0 = 2 \implies y_0 = f(2) = \frac{1}{2} = 0.5$$
2. **Find instantaneous slope $m$:**
   $$m = f'(2) = -\frac{1}{2^2} = -\frac{1}{4} = -0.25$$
3. **Write the point-slope line equation:**
   $$y - y_0 = m(x - x_0)$$
   $$y - 0.5 = -0.25(x - 2)$$
   $$y - 0.5 = -0.25x + 0.50 \implies \mathbf{y = -0.25x + 1.0}$$

At $x = 2$, Maya's robot power reading drops by $0.25$ units for every $1$ meter moved forward!
</details>

---

## Problem 2: Derivation of the $\tanh(x)$ Activation Derivative

### 🧒 The Context: Timmy's Memory Gate
In Recurrent Neural Networks (RNNs) and Long Short-Term Memory cells (LSTMs), signals are bounded between $-1$ and $+1$ using the Hyperbolic Tangent ($\tanh$):
$$\tanh(x) = \frac{\sinh(x)}{\cosh(x)} = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

### Tasks:
1. Using the Quotient Rule:
   $$\left( \frac{u}{v} \right)' = \frac{u' v - u v'}{v^2}$$
   derive the derivative $\frac{d}{dx} \tanh(x)$.
2. Show that it simplifies to the elegant backpropagation shortcut:
   $$\tanh'(x) = 1 - \tanh^2(x)$$
3. If a hidden neuron caches a forward activation value of $a = \tanh(x) = 0.8$, compute its backpropagation gradient $da$ using mental arithmetic (no exponential calculations!).

---

<details>
<summary>Click to view solution for Problem 2</summary>

### Detailed Step-by-Step Solution:

#### Step 1: Identify Functions and Differentiate
Let the numerator be:
$$u(x) = e^x - e^{-x} \implies u'(x) = e^x - (-e^{-x}) = e^x + e^{-x}$$
Let the denominator be:
$$v(x) = e^x + e^{-x} \implies v'(x) = e^x + (-e^{-x}) = e^x - e^{-x}$$

#### Step 2: Apply the Quotient Rule Formula
$$\frac{d}{dx} \tanh(x) = \frac{(e^x + e^{-x})(e^x + e^{-x}) - (e^x - e^{-x})(e^x - e^{-x})}{(e^x + e^{-x})^2}$$
$$\frac{d}{dx} \tanh(x) = \frac{(e^x + e^{-x})^2 - (e^x - e^{-x})^2}{(e^x + e^{-x})^2}$$

#### Step 3: Split into Two Separate Terms
$$\frac{d}{dx} \tanh(x) = \frac{(e^x + e^{-x})^2}{(e^x + e^{-x})^2} - \frac{(e^x - e^{-x})^2}{(e^x + e^{-x})^2}$$
$$= 1 - \left( \frac{e^x - e^{-x}}{e^x + e^{-x}} \right)^2$$

Since the term inside parentheses is the definition of $\tanh(x)$, we have:
$$\mathbf{\tanh'(x) = 1 - \tanh^2(x)} \quad \blacksquare$$

#### Step 4: Backpropagation Gradient Computation
Given cached forward activation $a = \tanh(x) = 0.8$:
$$da = 1 - a^2 = 1 - (0.8)^2 = 1 - 0.64 = \mathbf{0.36}$$

**Key Engineering Takeaway:** The neural network doesn't need to compute costly $e^x$ during backpropagation; a simple multiplication and subtraction yields the exact derivative!
</details>

---

## Problem 3: Finite Difference Error Analysis & Optimal Step Size $h$

### 🧒 The Context: The Tug-of-War Between Math and Silicon
When building a custom neural network library in C++ or CUDA, you must run gradient checks using finite differences.
In computer floating-point math (IEEE-754 double precision), machine precision is $\epsilon_{\text{mach}} \approx 10^{-16}$.
Two opposing errors battle against each other:
1. **Truncation Error (Math):** As $h$ gets bigger, Taylor series terms drop out. For Central Difference:
   $$\text{Truncation Error} \approx c_1 h^2$$
2. **Roundoff Error (Silicon):** As $h$ gets too small, subtracting two nearly identical numbers loses precision:
   $$\text{Roundoff Error} \approx \frac{\epsilon_{\text{mach}}}{h}$$

Total Error is modeled as:
$$E(h) = c_1 h^2 + \frac{\epsilon_{\text{mach}}}{h}$$

### Tasks:
1. Differentiate $E(h)$ with respect to $h$, set the derivative to $0$, and find the optimal step size formula $h^*$.
2. For $\epsilon_{\text{mach}} \approx 10^{-16}$, calculate the order of magnitude of $h^*$ for Central Difference.
3. Compare this with Forward Difference, where total error is $E_{\text{fwd}}(h) = c_1 h + \frac{\epsilon_{\text{mach}}}{h}$.

---

<details>
<summary>Click to view solution for Problem 3</summary>

### Detailed Step-by-Step Solution:

#### Step 1: Differentiate Total Error $E(h)$
Rewrite:
$$E(h) = c_1 h^2 + \epsilon_{\text{mach}} h^{-1}$$
Take the derivative with respect to $h$:
$$\frac{dE}{dh} = 2 c_1 h - \epsilon_{\text{mach}} h^{-2} = 2 c_1 h - \frac{\epsilon_{\text{mach}}}{h^2}$$

#### Step 2: Set Derivative to Zero and Solve for $h^*$
$$2 c_1 h - \frac{\epsilon_{\text{mach}}}{h^2} = 0 \implies 2 c_1 h = \frac{\epsilon_{\text{mach}}}{h^2}$$
$$h^3 = \frac{\epsilon_{\text{mach}}}{2 c_1} \implies h^* \propto \epsilon_{\text{mach}}^{1/3}$$

#### Step 3: Calculate Numerical Order of Magnitude
For standard double-precision float where $\epsilon_{\text{mach}} \approx 10^{-16}$:
$$h^* \approx (10^{-16})^{1/3} = 10^{-16/3} = 10^{-5.33} \approx \mathbf{10^{-5} \text{ to } 10^{-6}}$$

**Why this matters:** This is the rigorous mathematical explanation why deep learning libraries (like PyTorch and TensorFlow) use `eps = 1e-5` for gradient checking! If you blindly set $h = 10^{-12}$, the computer's roundoff error skyrockets and breaks the test.

#### Step 4: Comparison with Forward Difference
For forward difference:
$$E_{\text{fwd}}(h) = c_1 h + \frac{\epsilon_{\text{mach}}}{h}$$
$$\frac{dE_{\text{fwd}}}{dh} = c_1 - \frac{\epsilon_{\text{mach}}}{h^2} = 0 \implies h^2 \propto \epsilon_{\text{mach}} \implies h^*_{\text{fwd}} \approx \sqrt{\epsilon_{\text{mach}}} \approx 10^{-8}$$

- Best possible error for Forward Difference: $E \approx 10^{-8}$.
- Best possible error for Central Difference: $E \approx (10^{-5.33})^2 \approx \mathbf{10^{-10.7} \approx 10^{-11}}$.

Central Difference is **1,000 times more accurate** than forward difference, making it the industry standard.
</details>
