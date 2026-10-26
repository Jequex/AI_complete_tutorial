# Topic 8.2: Exercises & Solutions — Logistic Regression

---

### Exercise 1: Sigmoid, Odds, and Log-Odds Hand Calculations

**Problem:**  
A fitted logistic regression model has parameters $w = [0.8, -1.2]^T$ and bias $b = 0.4$.  
Consider an input instance $x = [2.0, 1.0]^T$.

1. Compute the linear logit score $z = w^T x + b$.
2. Compute the predicted probability $\hat{y} = \sigma(z)$. (Round to 4 decimal places).
3. Compute the odds of the event occurring: $\text{Odds} = \frac{\hat{y}}{1 - \hat{y}}$.
4. Verify that $\ln(\text{Odds})$ exactly recovers the logit $z$.

<details>
<summary>View Complete Solution</summary>

**Step 1: Compute logit $z$**
$$z = w_1 x_1 + w_2 x_2 + b = (0.8)(2.0) + (-1.2)(1.0) + 0.4 = 1.6 - 1.2 + 0.4 = 0.8$$

**Step 2: Compute probability $\hat{y} = \sigma(z)$**
$$\hat{y} = \frac{1}{1 + e^{-0.8}} \approx \frac{1}{1 + 0.44933} = \frac{1}{1.44933} \approx 0.68997 \approx 0.6900$$

**Step 3: Compute Odds**
$$\text{Odds} = \frac{\hat{y}}{1 - \hat{y}} = \frac{0.68997}{1 - 0.68997} = \frac{0.68997}{0.31003} \approx 2.2255$$

**Step 4: Verify Logit identity**
$$\ln(2.2255) \approx 0.8000 = z$$
The log-odds perfectly equals the linear output $z$.

$\blacksquare$
</details>

---

### Exercise 2: Binary Cross-Entropy Loss Calculation

**Problem:**  
You evaluate a binary classifier on a mini-batch of $N = 3$ samples:
- Sample 1: True label $y^{(1)} = 1$, predicted probability $\hat{y}^{(1)} = 0.80$
- Sample 2: True label $y^{(2)} = 0$, predicted probability $\hat{y}^{(2)} = 0.25$
- Sample 3: True label $y^{(3)} = 1$, predicted probability $\hat{y}^{(3)} = 0.10$

1. Compute the per-sample loss $\ell^{(i)} = -\left[ y^{(i)} \ln \hat{y}^{(i)} + (1 - y^{(i)}) \ln(1 - \hat{y}^{(i)}) \right]$ for each instance.
2. Compute the total empirical loss $\mathcal{L}_{\text{BCE}} = \frac{1}{3} \sum_{i=1}^3 \ell^{(i)}$.
3. Which sample contributed the largest penalty, and why?

<details>
<summary>View Complete Solution</summary>

**Step 1: Per-sample loss**
- For Sample 1 ($y=1$):
  $$\ell^{(1)} = -\ln(0.80) \approx -(-0.2231) = 0.2231$$
- For Sample 2 ($y=0$):
  $$\ell^{(2)} = -\ln(1 - 0.25) = -\ln(0.75) \approx -(-0.2877) = 0.2877$$
- For Sample 3 ($y=1$):
  $$\ell^{(3)} = -\ln(0.10) \approx -(-2.3026) = 2.3026$$

**Step 2: Total empirical loss**
$$\mathcal{L}_{\text{BCE}} = \frac{0.2231 + 0.2877 + 2.3026}{3} = \frac{2.8134}{3} \approx 0.9378$$

**Step 3: Analysis**
Sample 3 contributed **$2.3026 / 2.8134 \approx 81.8\%$** of the entire mini-batch loss! Because the model predicted a low probability ($10\%$) for a truly positive sample, the logarithmic penalty severely penalizes high-confidence errors.

$\blacksquare$
</details>

---

### Exercise 3: Gradient Descent Step Hand Calculation

**Problem:**  
Using the 3-sample mini-batch from Exercise 2 with single scalar feature values:
$$x^{(1)} = 1.0, \quad x^{(2)} = -1.0, \quad x^{(3)} = 2.0$$
Suppose the current weight is $w = 0.5$ and bias $b = 0.0$. The predictions from before were:
$$\hat{y}^{(1)} = 0.80, \quad \hat{y}^{(2)} = 0.25, \quad \hat{y}^{(3)} = 0.10$$
True labels: $y = [1, 0, 1]^T$.

1. Compute the prediction errors $e^{(i)} = \hat{y}^{(i)} - y^{(i)}$.
2. Compute the gradient with respect to weight $w$: $\frac{\partial \mathcal{L}}{\partial w} = \frac{1}{3} \sum_{i=1}^3 (\hat{y}^{(i)} - y^{(i)}) x^{(i)}$.
3. Compute the gradient with respect to bias $b$: $\frac{\partial \mathcal{L}}{\partial b} = \frac{1}{3} \sum_{i=1}^3 (\hat{y}^{(i)} - y^{(i)})$.
4. With learning rate $\eta = 0.1$, update $w$ and $b$.

<details>
<summary>View Complete Solution</summary>

**Step 1: Prediction errors**
- $e^{(1)} = 0.80 - 1 = -0.20$
- $e^{(2)} = 0.25 - 0 = +0.25$
- $e^{(3)} = 0.10 - 1 = -0.90$

**Step 2: Gradient w.r.t. $w$**
$$\begin{aligned}
\frac{\partial \mathcal{L}}{\partial w} &= \frac{1}{3} \left[ (-0.20)(1.0) + (0.25)(-1.0) + (-0.90)(2.0) \right] \\
&= \frac{1}{3} \left[ -0.20 - 0.25 - 1.80 \right] = \frac{-2.25}{3} = -0.75
\end{aligned}$$

**Step 3: Gradient w.r.t. $b$**
$$\frac{\partial \mathcal{L}}{\partial b} = \frac{1}{3} \left[ -0.20 + 0.25 - 0.90 \right] = \frac{-0.85}{3} \approx -0.2833$$

**Step 4: Parameter update ($\eta = 0.1$)**
$$w_{t+1} = 0.5 - 0.1(-0.75) = 0.5 + 0.075 = 0.575$$
$$b_{t+1} = 0.0 - 0.1(-0.2833) = 0.0 + 0.02833 = 0.02833$$

$\blacksquare$
</details>

---

### Exercise 4: 2D Linear Decision Boundary Equation

**Problem:**  
A 2D logistic regression classifier has learned the parameters $w_1 = 2$, $w_2 = -4$, and bias $b = 6$.
1. Write the equation of the linear decision boundary $\mathcal{H}$ corresponding to $P(y=1 \mid x) = 0.5$.
2. Express $x_2$ as a function of $x_1$ in slope-intercept form ($x_2 = m x_1 + c$).
3. Which side of the boundary corresponds to the positive class ($y = 1$)? Test the origin $(0, 0)$.

<details>
<summary>View Complete Solution</summary>

**Step 1: Decision boundary equation**
The decision threshold is at $z = 0$:
$$w_1 x_1 + w_2 x_2 + b = 0 \implies 2x_1 - 4x_2 + 6 = 0$$

**Step 2: Slope-intercept form**
$$4x_2 = 2x_1 + 6 \implies x_2 = 0.5 x_1 + 1.5$$
The decision boundary is a straight line in $\mathbb{R}^2$ with slope $m = 0.5$ and vertical intercept $c = 1.5$.

**Step 3: Positive class region & Origin test**
Points satisfy $y=1$ when $z \ge 0 \implies 2x_1 - 4x_2 + 6 \ge 0 \implies x_2 \le 0.5 x_1 + 1.5$.  
Testing the origin $(0, 0)$:
$$z = 2(0) - 4(0) + 6 = +6 > 0 \implies \hat{y} = \sigma(6) \approx 0.9975$$
Since $z > 0$, the origin lies deep within the positive ($y=1$) classification region.

$\blacksquare$
</details>
