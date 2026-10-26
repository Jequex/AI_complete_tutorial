# Exercises & Detailed Solutions: Topic 0.3

Welcome to the problem set for **Topic 0.3: The Central Idea of AI**.

Work through each problem on paper before checking the step-by-step solutions below.

---

## 📝 Problem Set

### Problem 1: Manual One-Step Gradient Descent by Hand
Suppose we have a miniature dataset of $N=2$ samples:
- Sample 1: $x^{(1)} = 2.0, \quad y^{(1)} = 5.0$
- Sample 2: $x^{(2)} = 4.0, \quad y^{(2)} = 9.0$

We choose a linear model:
$$\hat{y} = w x + b$$
and the Mean Squared Error (MSE) objective:
$$\mathcal{L}(w, b) = \frac{1}{N} \sum_{i=1}^N (\hat{y}^{(i)} - y^{(i)})^2$$

We initialize our parameters to:
$$w_0 = 1.0, \quad b_0 = 0.0$$
and set our learning rate to:
$$\eta = 0.05$$

- **Part A:** Compute the predictions $\hat{y}^{(1)}$ and $\hat{y}^{(2)}$, the errors $e^{(1)}$ and $e^{(2)}$, and the initial loss $\mathcal{L}(w_0, b_0)$.
- **Part B:** Analytically compute the gradients:
  $$\frac{\partial \mathcal{L}}{\partial w} \quad \text{and} \quad \frac{\partial \mathcal{L}}{\partial b}$$
- **Part C:** Perform one gradient descent update step to compute the new parameters $w_1$ and $b_1$:
  $$w_1 = w_0 - \eta \frac{\partial \mathcal{L}}{\partial w}$$
  $$b_1 = b_0 - \eta \frac{\partial \mathcal{L}}{\partial b}$$
- **Part D:** Compute the new loss $\mathcal{L}(w_1, b_1)$ using the updated parameters. Did the loss decrease?

---

### Problem 2: The Analytic Closed-Form Solution (The Normal Equation)
Instead of taking iterative gradient steps, can we solve for the perfect parameters in a single mathematical step?
Assume we have the zero-bias model:
$$\hat{y}^{(i)} = w x^{(i)}$$
with MSE loss:
$$\mathcal{L}(w) = \frac{1}{N} \sum_{i=1}^N (w x^{(i)} - y^{(i)})^2$$

- **Part A:** Take the derivative $\frac{d\mathcal{L}}{dw}$.
- **Part B:** Set $\frac{d\mathcal{L}}{dw} = 0$ to find the minimum of the loss parabola and solve for $w^*$ analytically.
- **Part C:** Plug in the numbers from Problem 1 ($x = [2, 4]$, $y = [5, 9]$) to find the exact optimal weight $w^*$ under this zero-bias model.

---

### Problem 3: Learning Rate Intuition & Stability
Consider the loss function:
$$\mathcal{L}(w) = 3w^2$$
We start at $w_0 = 10.0$.
The gradient descent update rule is:
$$w_{t+1} = w_t - \eta \frac{d\mathcal{L}}{dw}$$

- **Part A:** Compute $\frac{d\mathcal{L}}{dw}$ in terms of $w$.
- **Part B:** Write the recurrence relation expressing $w_{t+1}$ in terms of $w_t$ and $\eta$.
- **Part C:** What is the maximum threshold value for the learning rate $\eta_{\text{crit}}$ such that the optimization does **not** diverge to infinity?

---

# 💡 Complete Solutions & Walkthroughs

---

### Solution 1: Manual One-Step Gradient Descent

#### Part A: Predictions, Errors, and Initial Loss
Initial parameters: $w_0 = 1.0, \quad b_0 = 0.0$.

1. **Predictions:**
   $$\hat{y}^{(1)} = w_0 x^{(1)} + b_0 = (1.0)(2.0) + 0.0 = \mathbf{2.0}$$
   $$\hat{y}^{(2)} = w_0 x^{(2)} + b_0 = (1.0)(4.0) + 0.0 = \mathbf{4.0}$$

2. **Errors ($e^{(i)} = \hat{y}^{(i)} - y^{(i)}$):**
   $$e^{(1)} = \hat{y}^{(1)} - y^{(1)} = 2.0 - 5.0 = \mathbf{-3.0}$$
   $$e^{(2)} = \hat{y}^{(2)} - y^{(2)} = 4.0 - 9.0 = \mathbf{-5.0}$$

3. **Initial Loss:**
   $$\mathcal{L}(w_0, b_0) = \frac{1}{2} \left[ (e^{(1)})^2 + (e^{(2)})^2 \right]$$
   $$\mathcal{L}(w_0, b_0) = \frac{1}{2} \left[ (-3.0)^2 + (-5.0)^2 \right] = \frac{1}{2} [9.0 + 25.0] = \frac{34.0}{2} = \mathbf{17.0}$$

---

#### Part B: Computing the Gradients
Recall our derived formulas:
$$\frac{\partial \mathcal{L}}{\partial w} = \frac{2}{N} \sum_{i=1}^N e^{(i)} x^{(i)}$$
$$\frac{\partial \mathcal{L}}{\partial b} = \frac{2}{N} \sum_{i=1}^N e^{(i)}$$

Since $N = 2$, the factor $\frac{2}{N} = \frac{2}{2} = 1.0$.

1. **Gradient with respect to weight $w$:**
   $$\frac{\partial \mathcal{L}}{\partial w} = (e^{(1)} x^{(1)}) + (e^{(2)} x^{(2)})$$
   $$\frac{\partial \mathcal{L}}{\partial w} = (-3.0)(2.0) + (-5.0)(4.0) = -6.0 - 20.0 = \mathbf{-26.0}$$

2. **Gradient with respect to bias $b$:**
   $$\frac{\partial \mathcal{L}}{\partial b} = e^{(1)} + e^{(2)} = -3.0 + (-5.0) = \mathbf{-8.0}$$

---

#### Part C: Parameter Updates
Given learning rate $\eta = 0.05$:

1. **Update $w$:**
   $$w_1 = w_0 - \eta \frac{\partial \mathcal{L}}{\partial w}$$
   $$w_1 = 1.0 - (0.05 \times (-26.0)) = 1.0 - (-1.30) = \mathbf{2.30}$$

2. **Update $b$:**
   $$b_1 = b_0 - \eta \frac{\partial \mathcal{L}}{\partial b}$$
   $$b_1 = 0.0 - (0.05 \times (-8.0)) = 0.0 - (-0.40) = \mathbf{0.40}$$

---

#### Part D: Verification of Loss Reduction
Let us test our updated model with $w_1 = 2.30$ and $b_1 = 0.40$:

1. **New Predictions:**
   $$\hat{y}^{(1)}_{\text{new}} = (2.30)(2.0) + 0.40 = 4.60 + 0.40 = \mathbf{5.00}$$
   $$\hat{y}^{(2)}_{\text{new}} = (2.30)(4.0) + 0.40 = 9.20 + 0.40 = \mathbf{9.60}$$

2. **New Errors:**
   $$e^{(1)}_{\text{new}} = 5.00 - 5.00 = \mathbf{0.00}$$
   $$e^{(2)}_{\text{new}} = 9.60 - 9.00 = \mathbf{+0.60}$$

3. **New Loss:**
   $$\mathcal{L}(w_1, b_1) = \frac{1}{2} \left[ (0.00)^2 + (0.60)^2 \right] = \frac{1}{2} [0.0 + 0.36] = \mathbf{0.18}$$

**Outcome:**
The loss collapsed from **$17.0 \to 0.18$ in a single gradient step!** A massive $98.9\%$ reduction in error.

---

### Solution 2: The Analytic Closed-Form Solution

#### Part A: Derivative with respect to $w$
Using the chain rule:
$$\frac{d\mathcal{L}}{dw} = \frac{1}{N} \sum_{i=1}^N 2 (w x^{(i)} - y^{(i)}) \cdot x^{(i)} = \frac{2}{N} \sum_{i=1}^N \left( w (x^{(i)})^2 - x^{(i)} y^{(i)} \right)$$

---

#### Part B: Solving for Optimal $w^*$
Set the derivative equal to zero:
$$\frac{2}{N} \sum_{i=1}^N \left( w (x^{(i)})^2 - x^{(i)} y^{(i)} \right) = 0$$
Multiply both sides by $\frac{N}{2}$:
$$\sum_{i=1}^N w (x^{(i)})^2 - \sum_{i=1}^N x^{(i)} y^{(i)} = 0$$
Factor $w$ out of the first summation:
$$w \sum_{i=1}^N (x^{(i)})^2 = \sum_{i=1}^N x^{(i)} y^{(i)}$$
Isolate $w^*$:
$$\boxed{w^* = \frac{\sum_{i=1}^N x^{(i)} y^{(i)}}{\sum_{i=1}^N (x^{(i)})^2}}$$
*(This is the scalar 1D version of the famous Normal Equation $\mathbf{w}^* = (X^T X)^{-1} X^T \mathbf{y}$!)*

---

#### Part C: Plug in Sample Numbers
Data: $(x^{(1)}, y^{(1)}) = (2, 5)$ and $(x^{(2)}, y^{(2)}) = (4, 9)$.

1. Compute numerator:
   $$\sum x^{(i)} y^{(i)} = (2 \times 5) + (4 \times 9) = 10 + 36 = \mathbf{46}$$

2. Compute denominator:
   $$\sum (x^{(i)})^2 = (2)^2 + (4)^2 = 4 + 16 = \mathbf{20}$$

3. Compute optimal weight:
   $$w^* = \frac{46}{20} = \mathbf{2.30}$$

Notice how close our 1-step gradient descent approximation in Problem 1 ($w_1 = 2.30$) was to the theoretical optimum!

---

### Solution 3: Learning Rate Intuition & Stability

#### Part A: Computing the Derivative
$$\mathcal{L}(w) = 3w^2$$
Using the power rule:
$$\frac{d\mathcal{L}}{dw} = 3 \cdot 2w = \mathbf{6w}$$

---

#### Part B: The Recurrence Relation
Substitute the derivative into the update equation:
$$w_{t+1} = w_t - \eta (6w_t)$$
Factor out $w_t$:
$$\boxed{w_{t+1} = (1 - 6\eta) w_t}$$

---

#### Part C: Stability Threshold ($\eta_{\text{crit}}$)
For the sequence $w_0, w_1, w_2, \dots$ to converge to the minimum at $0$, the multiplier $(1 - 6\eta)$ must satisfy:
$$|1 - 6\eta| < 1$$
This expands to the compound inequality:
$$-1 < 1 - 6\eta < 1$$

1. Left inequality:
   $$-1 < 1 - 6\eta \implies 6\eta < 2 \implies \eta < \frac{2}{6} = \frac{1}{3}$$
2. Right inequality:
   $$1 - 6\eta < 1 \implies -6\eta < 0 \implies \eta > 0$$

Therefore, for convergence:
$$0 < \eta < \frac{1}{3} \approx 0.3333$$

If $\eta \geq \frac{1}{3}$, $|1 - 6\eta| \geq 1$, which causes the updates to oscillate wildly or explode exponentially to $\pm \infty$. Thus, the critical threshold is:
$$\boxed{\eta_{\text{crit}} = \frac{1}{3}}$$

