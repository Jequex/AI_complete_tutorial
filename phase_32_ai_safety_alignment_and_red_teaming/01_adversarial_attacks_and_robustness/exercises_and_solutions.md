# Exercises & Solutions: Adversarial Attacks & Robustness

## Exercise 1: Closed-Form FGSM Perturbation Calculation
**Problem**: Consider a single linear neuron with weights $w = [3.0, -4.0]^T$ and bias $b = 1.0$. The input point is $x = [1.0, 2.0]^T$ with true label $y = 1$. The model outputs binary prediction $\hat{y} = \sigma(w^T x + b)$ where $\sigma(z) = \frac{1}{1 + e^{-z}}$.
1. Compute the logit $z$, probability $\hat{y}$, and binary cross-entropy loss $\mathcal{L}(x, y)$.
2. Calculate the exact analytical gradient of the loss with respect to input $x$: $\nabla_x \mathcal{L}(x, y)$.
3. For perturbation budget $\epsilon = 0.2$ under $\ell_\infty$ norm, compute the FGSM adversarial example $x_{\text{adv}}$.
4. Calculate the new logit $z_{\text{adv}}$ and new predicted probability $\hat{y}_{\text{adv}}$. Did the attack succeed in reducing the confidence of class 1?

---

### Solution 1:
1. **Clean Logit and Probability**:
   $$z = w_1 x_1 + w_2 x_2 + b = (3.0)(1.0) + (-4.0)(2.0) + 1.0 = 3.0 - 8.0 + 1.0 = -4.0$$
   $$\hat{y} = \sigma(-4.0) = \frac{1}{1 + e^{4.0}} \approx \frac{1}{1 + 54.598} \approx 0.017986$$
   $$\mathcal{L}(x, y=1) = -\ln(\hat{y}) = -\ln(0.017986) \approx 4.0182$$

2. **Input Gradient**:
   $$\frac{\partial \mathcal{L}}{\partial x} = (\hat{y} - y) w = (0.017986 - 1.0) \begin{bmatrix} 3.0 \\ -4.0 \end{bmatrix} = -0.982014 \begin{bmatrix} 3.0 \\ -4.0 \end{bmatrix} = \begin{bmatrix} -2.9460 \\ 3.9281 \end{bmatrix}$$

3. **FGSM Perturbation**:
   $$\text{sign}(\nabla_x \mathcal{L}) = \begin{bmatrix} \text{sign}(-2.9460) \\ \text{sign}(3.9281) \end{bmatrix} = \begin{bmatrix} -1.0 \\ +1.0 \end{bmatrix}$$
   $$x_{\text{adv}} = x + \epsilon \cdot \text{sign}(\nabla_x \mathcal{L}) = \begin{bmatrix} 1.0 \\ 2.0 \end{bmatrix} + 0.2 \begin{bmatrix} -1.0 \\ 1.0 \end{bmatrix} = \begin{bmatrix} 0.8 \\ 2.2 \end{bmatrix}$$

4. **Adversarial Evaluation**:
   $$z_{\text{adv}} = (3.0)(0.8) + (-4.0)(2.2) + 1.0 = 2.4 - 8.8 + 1.0 = -5.4$$
   $$\hat{y}_{\text{adv}} = \sigma(-5.4) = \frac{1}{1 + e^{5.4}} \approx \frac{1}{1 + 221.406} \approx 0.004496$$
   The adversarial loss is $\mathcal{L}_{\text{adv}} = -\ln(0.004496) \approx 5.4045$.
   The loss increased from $4.0182$ to $5.4045$, and predicted probability of true label dropped by $75\%$ (from $1.798\%$ to $0.449\%$).

---

## Exercise 2: Certified Robustness Margin Derivation
**Problem**: For the same linear classifier $w = [3.0, -4.0]^T, b = 1.0$:
1. Determine the exact equation of the decision boundary hyperplane $\mathcal{H}$.
2. Calculate the Lipschitz constant of the linear model with respect to the $\ell_2$ norm.
3. For point $x_0 = [3.0, 1.0]^T$:
   - Verify its predicted class.
   - Compute the certified robustness radius $r(x_0)$.
   - Prove that an attacker with Euclidean budget $\|\delta\|_2 = 1.0$ cannot flip the prediction.

---

### Solution 2:
1. **Decision Boundary**:
   $$3.0 x_1 - 4.0 x_2 + 1.0 = 0$$

2. **Lipschitz Constant**:
   For any linear function $f(x) = w^T x + b$:
   $$|f(x) - f(x')| = |w^T (x - x')| \le \|w\|_2 \|x - x'\|_2$$
   $$\|w\|_2 = \sqrt{3.0^2 + (-4.0)^2} = \sqrt{9 + 16} = 5.0$$
   Therefore, the Lipschitz constant is $L = 5.0$.

3. **Certified Robustness for $x_0$**:
   - Logit: $f(x_0) = 3.0(3.0) - 4.0(1.0) + 1.0 = 9 - 4 + 1 = +6.0$. Since $f(x_0) > 0$, predicted class is $+1$ (or probability $\sigma(6.0) \approx 0.9975$).
   - Certified radius:
     $$r(x_0) = \frac{|f(x_0)|}{\|w\|_2} = \frac{6.0}{5.0} = 1.2$$
   - Since $r(x_0) = 1.2$, the minimum Euclidean distance to the decision boundary is $1.2$.
   - Any attacker with budget $\|\delta\|_2 = 1.0 < 1.2$ can at most reduce the logit to:
     $$\min_{\|\delta\|_2 \le 1.0} f(x_0 + \delta) = f(x_0) - L \|\delta\|_2 = 6.0 - (5.0)(1.0) = +1.0 > 0$$
   - Because the minimum possible logit remains $+1.0 > 0$, the prediction is strictly preserved as class $+1$.

---

## Exercise 3: PGD Projection Boundary Clipping
**Problem**: Suppose a 1D input feature $x = 0.5$ has valid domain $[0.0, 1.0]$. The perturbation budget is $\epsilon = 0.15$.
1. What is the feasible adversarial set $\mathcal{S} = B_\epsilon(x) \cap [0.0, 1.0]$?
2. If after gradient ascent the unconstrained iterate is $z = 0.72$, compute $\Pi_{\mathcal{S}}(z)$.
3. If the original point was $x = 0.95$ with the same $\epsilon = 0.15$, what is $\mathcal{S}$, and what is $\Pi_{\mathcal{S}}(1.12)$?

---

### Solution 3:
1. Feasible set for $x = 0.5$:
   $$\mathcal{S} = [\max(0.0, 0.5 - 0.15), \min(1.0, 0.5 + 0.15)] = [0.35, 0.65]$$
2. For $z = 0.72$:
   $$\Pi_{\mathcal{S}}(0.72) = \text{clip}(0.72, 0.35, 0.65) = 0.65$$
3. For $x = 0.95$:
   $$\mathcal{S} = [\max(0.0, 0.95 - 0.15), \min(1.0, 0.95 + 0.15)] = [0.80, \min(1.0, 1.10)] = [0.80, 1.00]$$
   For $z = 1.12$:
   $$\Pi_{\mathcal{S}}(1.12) = \text{clip}(1.12, 0.80, 1.00) = 1.00$$
   The value is properly clipped by the physical domain ceiling of the input tensor.
