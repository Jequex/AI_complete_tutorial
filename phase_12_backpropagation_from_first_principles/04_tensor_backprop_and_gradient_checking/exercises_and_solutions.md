# Topic 12.4: Exercises & Solutions — Tensor Backprop & Gradient Checking

---

### Problem 1: Matrix Backprop with Elementwise Nonlinearity (Medium)

Consider a standard neural network layer with an elementwise activation function $\sigma$:
$$Z = X W + b, \quad Y = \sigma(Z)$$
where:
- $X \in \mathbb{R}^{B \times D_{\text{in}}}$
- $W \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}$
- $b \in \mathbb{R}^{1 \times D_{\text{out}}}$
- $Z, Y \in \mathbb{R}^{B \times D_{\text{out}}}$

Let $L$ be a scalar loss function with given upstream adjoint $\bar{Y} \equiv \frac{\partial L}{\partial Y} \in \mathbb{R}^{B \times D_{\text{out}}}$.

1. Compute the intermediate adjoint $\bar{Z} \equiv \frac{\partial L}{\partial Z}$ in terms of $\bar{Y}$ and $\sigma'(Z)$ using the Hadamard (elementwise) product $\odot$.
2. Express the parameter gradient $\frac{\partial L}{\partial W}$ in terms of $X$ and $\bar{Z}$.
3. Express the input gradient $\frac{\partial L}{\partial X}$ in terms of $\bar{Z}$ and $W$.
4. Express the bias gradient $\frac{\partial L}{\partial b}$ in terms of $\bar{Z}$.
5. Verify the matrix dimensional compatibility of all four expressions.

#### Solution

1. **Intermediate Pre-Activation Adjoint $\bar{Z}$:**
   Since $Y_{ij} = \sigma(Z_{ij})$ is applied strictly elementwise, each entry $Z_{ij}$ affects only $Y_{ij}$.
   By the chain rule:
   $$\bar{Z}_{ij} \equiv \frac{\partial L}{\partial Z_{ij}} = \frac{\partial L}{\partial Y_{ij}} \frac{\partial Y_{ij}}{\partial Z_{ij}} = \bar{Y}_{ij} \cdot \sigma'(Z_{ij})$$
   In tensor matrix notation:
   $$\boxed{\bar{Z} = \bar{Y} \odot \sigma'(Z) \in \mathbb{R}^{B \times D_{\text{out}}}}$$

2. **Weight Gradient $\frac{\partial L}{\partial W}$:**
   Since $Z = XW + b$, the derivation from Topic 12.4 applies directly using $\bar{Z}$:
   $$\boxed{\frac{\partial L}{\partial W} = X^T \bar{Z} \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}}$$

3. **Input Gradient $\frac{\partial L}{\partial X}$:**
   $$\boxed{\frac{\partial L}{\partial X} = \bar{Z} W^T \in \mathbb{R}^{B \times D_{\text{in}}}}$$

4. **Bias Gradient $\frac{\partial L}{\partial b}$:**
   $$\boxed{\frac{\partial L}{\partial b} = \sum_{i=1}^B \bar{Z}_{i, :} \in \mathbb{R}^{1 \times D_{\text{out}}}}$$

5. **Dimensional Compatibility Checks:**
   - $\bar{Z}$: $(B \times D_{\text{out}}) \odot (B \times D_{\text{out}}) = (B \times D_{\text{out}})$ $\checkmark$
   - $\frac{\partial L}{\partial W}$: $(D_{\text{in}} \times B) \times (B \times D_{\text{out}}) = (D_{\text{in}} \times D_{\text{out}})$ (Matches $W$) $\checkmark$
   - $\frac{\partial L}{\partial X}$: $(B \times D_{\text{out}}) \times (D_{\text{out}} \times D_{\text{in}}) = (B \times D_{\text{in}})$ (Matches $X$) $\checkmark$
   - $\frac{\partial L}{\partial b}$: Row sum reduces $(B \times D_{\text{out}}) \to (1 \times D_{\text{out}})$ (Matches $b$) $\checkmark$

---

### Problem 2: Taylor Proof of Central Difference Truncation Error (Easy)

Let $f: \mathbb{R} \to \mathbb{R}$ be a three-times continuously differentiable function.

1. Write the Taylor expansion of $f(x + \epsilon)$ around $x$ up to third order with Lagrange remainder.
2. Write the Taylor expansion of $f(x - \epsilon)$ around $x$ up to third order with Lagrange remainder.
3. Subtract the two expansions and divide by $2\epsilon$. Prove that:
   $$\frac{f(x + \epsilon) - f(x - \epsilon)}{2\epsilon} = f'(x) + \frac{\epsilon^2}{6} f'''(x) + \mathcal{O}(\epsilon^4)$$
4. Show why the one-sided forward difference $\frac{f(x + \epsilon) - f(x)}{\epsilon}$ has an error term of order $\mathcal{O}(\epsilon)$ rather than $\mathcal{O}(\epsilon^2)$.

#### Solution

1. **Forward Taylor Expansion:**
   $$f(x + \epsilon) = f(x) + \epsilon f'(x) + \frac{\epsilon^2}{2} f''(x) + \frac{\epsilon^3}{6} f'''(x) + \mathcal{O}(\epsilon^4)$$

2. **Backward Taylor Expansion:**
   Substitute $-\epsilon$ into the Taylor formula:
   $$f(x - \epsilon) = f(x) - \epsilon f'(x) + \frac{(-\epsilon)^2}{2} f''(x) + \frac{(-\epsilon)^3}{6} f'''(x) + \mathcal{O}(\epsilon^4)$$
   $$f(x - \epsilon) = f(x) - \epsilon f'(x) + \frac{\epsilon^2}{2} f''(x) - \frac{\epsilon^3}{6} f'''(x) + \mathcal{O}(\epsilon^4)$$

3. **Subtraction & Cancellation:**
   Subtract the second expansion from the first:
   $$f(x + \epsilon) - f(x - \epsilon) = [f(x) - f(x)] + [\epsilon f'(x) - (-\epsilon f'(x))] + \left[\frac{\epsilon^2}{2} f''(x) - \frac{\epsilon^2}{2} f''(x)\right] + \left[\frac{\epsilon^3}{6} f'''(x) - \left(-\frac{\epsilon^3}{6} f'''(x)\right)\right] + \dots$$
   Notice that all **even powers of $\epsilon$ cancel out identically**:
   $$f(x + \epsilon) - f(x - \epsilon) = 2\epsilon f'(x) + \frac{2\epsilon^3}{6} f'''(x) + \mathcal{O}(\epsilon^5) = 2\epsilon f'(x) + \frac{\epsilon^3}{3} f'''(x) + \mathcal{O}(\epsilon^5)$$
   Dividing both sides by $2\epsilon$:
   $$\frac{f(x + \epsilon) - f(x - \epsilon)}{2\epsilon} = f'(x) + \frac{\epsilon^2}{6} f'''(x) + \mathcal{O}(\epsilon^4)$$
   Thus, the truncation error is strictly of order $\mathcal{O}(\epsilon^2)$.

4. **One-Sided Forward Difference:**
   $$f(x + \epsilon) - f(x) = \epsilon f'(x) + \frac{\epsilon^2}{2} f''(x) + \mathcal{O}(\epsilon^3)$$
   Dividing by $\epsilon$:
   $$\frac{f(x + \epsilon) - f(x)}{\epsilon} = f'(x) + \frac{\epsilon}{2} f''(x) + \mathcal{O}(\epsilon^2)$$
   The second-order derivative term $\frac{\epsilon}{2} f''(x)$ does not cancel out, resulting in a coarse $\mathcal{O}(\epsilon)$ first-order truncation error.

---

### Problem 3: GradCheck Failure at Non-Differentiable Kinks (Hard)

Consider the ReLU activation function $f(x) = \max(0, x)$.

1. What is the subgradient defined in software (e.g. PyTorch) for $f'(0)$?
2. Compute the central finite difference approximation $g_{\text{central}} = \frac{f(0 + \epsilon) - f(0 - \epsilon)}{2\epsilon}$ for $\epsilon = 10^{-4}$.
3. Compute the resulting relative error between software analytical gradient and the numerical gradient at $x = 0$.
4. Explain how modern GradCheck harnesses handle "kinks" (points of non-differentiability like $\text{ReLU}$, $\max(0, x)$, or $|x|$).

#### Solution

1. **Analytical Subgradient in Software:**
   In PyTorch and autograd frameworks, the subgradient at $x = 0$ is conventionally defined as:
   $$f'(0) = 0.0 \quad (\text{or sometimes } 1.0 \text{ depending on convention})$$
   Let us take $f'(0) = 0.0$.

2. **Numerical Central Difference:**
   $$f(0 + \epsilon) = \max(0, \epsilon) = \epsilon$$
   $$f(0 - \epsilon) = \max(0, -\epsilon) = 0$$
   $$g_{\text{central}} = \frac{\epsilon - 0}{2\epsilon} = \frac{1}{2} = 0.5$$

3. **Relative Error Computation:**
   $$\text{RelErr} = \frac{|0.5 - 0.0|}{|0.5| + |0.0|} = \frac{0.5}{0.5} = 1.0 \quad (100\% \text{ relative error!})$$
   Despite both the analytical formula and the numerical implementation being completely bug-free, GradCheck reports catastrophic failure!

4. **Remediation in Practical Testing:**
   - **Avoid evaluating at kinks:** When initializing random test tensors for GradCheck, ensure no parameter or intermediate activation is near $0.0$ (e.g. initialize uniformly in $[0.1, 1.0]$ or $[-1.0, -0.1]$, or add small random offsets away from boundaries).
   - **Check directional consistency:** Recognize that non-smooth functions possess directional subderivatives where $f'(0^+) = 1.0$ and $f'(0^-) = 0.0$.
