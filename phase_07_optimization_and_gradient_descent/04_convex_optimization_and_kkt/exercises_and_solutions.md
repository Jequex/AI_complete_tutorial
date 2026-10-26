# Topic 7.4: Exercises & Solutions — Convex Optimization & KKT

---

### Problem 1: Convexity Proof via Hessian (Easy)

Consider the log-barrier function defined on the strictly positive orthant $\mathbb{R}_{++}^n = \{x \in \mathbb{R}^n : x_i > 0, \forall i\}$:
$$f(x) = -\sum_{i=1}^n \ln(x_i)$$

1. Compute the gradient $\nabla f(x)$.
2. Compute the Hessian matrix $H(x) = \nabla^2 f(x)$.
3. Prove that $H(x)$ is positive definite for all $x \in \mathbb{R}_{++}^n$, and conclude that $f(x)$ is strictly convex.

#### Solution
1. **Gradient:**
   For each coordinate $i \in \{1, \dots, n\}$:
   $$\frac{\partial f}{\partial x_i} = -\frac{1}{x_i} \implies \nabla f(x) = \begin{bmatrix} -1/x_1 \\ -1/x_2 \\ \vdots \\ -1/x_n \end{bmatrix}$$

2. **Hessian Matrix:**
   $$\frac{\partial^2 f}{\partial x_i^2} = \frac{1}{x_i^2}, \quad \frac{\partial^2 f}{\partial x_i \partial x_j} = 0 \quad (i \ne j)$$
   $$H(x) = \text{diag}\left( \frac{1}{x_1^2}, \frac{1}{x_2^2}, \dots, \frac{1}{x_n^2} \right)$$

3. **Positive Definiteness:**
   For any non-zero vector $z \in \mathbb{R}^n$ with $z \ne 0$:
   $$z^T H(x) z = \sum_{i=1}^n z_i^2 \left( \frac{1}{x_i^2} \right)$$
   Since $x_i > 0$, $\frac{1}{x_i^2} > 0$ for all $i$. Because $z \ne 0$, at least one $z_i^2 > 0$, so:
   $$z^T H(x) z > 0, \quad \forall z \ne 0$$
   Therefore, $H(x) \succ 0$ (positive definite) for all $x \in \mathbb{R}_{++}^n$. By the second-order condition, $f(x)$ is **strictly convex**. $\blacksquare$

---

### Problem 2: Hand-Solved Constrained Optimization via KKT (Medium)

Consider the constrained minimization problem:
$$\min_{x_1, x_2} f(x_1, x_2) = x_1^2 + x_2^2$$
$$\text{subject to } g(x_1, x_2) = 2 - (x_1 + x_2) \le 0$$
(In other words, minimize distance to the origin subject to $x_1 + x_2 \ge 2$).

1. Write the Lagrangian function $\mathcal{L}(x_1, x_2, \lambda)$.
2. State the four KKT conditions explicitly for this problem.
3. Test Case A: Assume the constraint is inactive ($\lambda = 0$). Show that this leads to a contradiction.
4. Test Case B: Assume the constraint is active ($\lambda > 0, g(x) = 0$). Solve for $x_1^*, x_2^*$, and $\lambda^*$, and verify all four KKT conditions.

#### Solution
1. **Lagrangian Function:**
   The inequality constraint in standard form is $g(x_1, x_2) = 2 - x_1 - x_2 \le 0$.
   $$\mathcal{L}(x_1, x_2, \lambda) = x_1^2 + x_2^2 + \lambda(2 - x_1 - x_2)$$

2. **The Four KKT Conditions:**
   - **Primal Feasibility:** $2 - x_1^* - x_2^* \le 0 \iff x_1^* + x_2^* \ge 2$
   - **Dual Feasibility:** $\lambda^* \ge 0$
   - **Complementary Slackness:** $\lambda^*(2 - x_1^* - x_2^*) = 0$
   - **Stationarity:**
     $$\frac{\partial \mathcal{L}}{\partial x_1} = 2x_1^* - \lambda^* = 0 \implies x_1^* = \frac{\lambda^*}{2}$$
     $$\frac{\partial \mathcal{L}}{\partial x_2} = 2x_2^* - \lambda^* = 0 \implies x_2^* = \frac{\lambda^*}{2}$$

3. **Case A ($\lambda = 0$):**
   If $\lambda^* = 0$, stationarity implies:
   $$x_1^* = \frac{0}{2} = 0, \quad x_2^* = \frac{0}{2} = 0$$
   Now test primal feasibility:
   $$x_1^* + x_2^* = 0 + 0 = 0 \not\ge 2$$
   This violates primal feasibility! Hence, the unconstrained optimum is infeasible, so the constraint must be active.

4. **Case B ($\lambda > 0$ and $2 - x_1^* - x_2^* = 0$):**
   From stationarity, $x_1^* = x_2^* = \frac{\lambda^*}{2}$.
   Substitute into the active constraint:
   $$\frac{\lambda^*}{2} + \frac{\lambda^*}{2} = 2 \implies \lambda^* = 2$$
   Substitute $\lambda^* = 2$ back into coordinate equations:
   $$x_1^* = \frac{2}{2} = 1.0, \quad x_2^* = \frac{2}{2} = 1.0$$
   - Check Dual Feasibility: $\lambda^* = 2 \ge 0$ (Satisfied)
   - Check Primal Feasibility: $x_1^* + x_2^* = 1 + 1 = 2 \ge 2$ (Satisfied)
   - Check Complementary Slackness: $\lambda^*(2 - 1 - 1) = 2(0) = 0$ (Satisfied)
   - Check Stationarity: $\nabla f(1, 1) = [2, 2]^T = \lambda^* [1, 1]^T = 2[1, 1]^T$ (Satisfied)
   Optimal solution: $x^* = (1, 1)$, minimum value $f(1, 1) = 1^2 + 1^2 = 2$.

---

### Problem 3: SVM Dual Formulation from KKT (Hard)

In a hard-margin Support Vector Machine, we find the hyperplane $w^T x + b = 0$ separating data points $\{(x_i, y_i)\}_{i=1}^N$ with labels $y_i \in \{-1, +1\}$ by solving:
$$\min_{w, b} \frac{1}{2} \|w\|_2^2$$
$$\text{subject to } 1 - y_i(w^T x_i + b) \le 0, \quad \forall i = 1, \dots, N$$

1. Write the Lagrangian $\mathcal{L}(w, b, \alpha)$ using dual multipliers $\alpha_i \ge 0$.
2. Compute the stationarity conditions with respect to $w$ and $b$.
3. Substitute the stationarity expressions back into the Lagrangian to derive the **SVM Dual Optimization Problem**.
4. Use complementary slackness $\alpha_i(1 - y_i(w^T x_i + b)) = 0$ to explain what a "support vector" is.

#### Solution
1. **The Lagrangian:**
   $$\mathcal{L}(w, b, \alpha) = \frac{1}{2} w^T w + \sum_{i=1}^N \alpha_i \left[ 1 - y_i(w^T x_i + b) \right], \quad \alpha_i \ge 0$$

2. **Stationarity Conditions:**
   $$\nabla_w \mathcal{L} = w - \sum_{i=1}^N \alpha_i y_i x_i = 0 \implies w = \sum_{i=1}^N \alpha_i y_i x_i$$
   $$\frac{\partial \mathcal{L}}{\partial b} = -\sum_{i=1}^N \alpha_i y_i = 0 \implies \sum_{i=1}^N \alpha_i y_i = 0$$

3. **Dual Problem Derivation:**
   Substitute $w = \sum_i \alpha_i y_i x_i$ and $\sum_i \alpha_i y_i = 0$ back into $\mathcal{L}$:
   $$\mathcal{L} = \frac{1}{2} \left( \sum_i \alpha_i y_i x_i \right)^T \left( \sum_j \alpha_j y_j x_j \right) + \sum_{i=1}^N \alpha_i - \sum_{i=1}^N \alpha_i y_i w^T x_i - b \underbrace{\sum_{i=1}^N \alpha_i y_i}_{=0}$$
   $$\mathcal{L} = \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j x_i^T x_j + \sum_{i=1}^N \alpha_i - \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j x_i^T x_j$$
   $$\mathcal{L} = \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j (x_i^T x_j)$$
   Thus, the Lagrange Dual Problem is:
   $$\max_\alpha \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j (x_i^T x_j)$$
   $$\text{subject to } \alpha_i \ge 0, \quad \sum_{i=1}^N \alpha_i y_i = 0$$
   This is a convex quadratic program depending only on inner products $x_i^T x_j$, enabling the famous **Kernel Trick** ($x_i^T x_j \to K(x_i, x_j)$)!

4. **Complementary Slackness & Support Vectors:**
   Complementary slackness requires $\alpha_i [1 - y_i(w^T x_i + b)] = 0$.
   - For all training points located safely away from the margin ($y_i(w^T x_i + b) > 1$), their multiplier must be **$\alpha_i = 0$**. These points have zero influence on $w$.
   - Only for the points lying directly on the margin boundary ($y_i(w^T x_i + b) = 1$) can $\alpha_i > 0$. These critical boundary samples are called **Support Vectors**, because they alone "support" and define the decision boundary!

---

### Problem 4: Euclidean Projection Operator Derivation (Medium)

Let $C = \{x \in \mathbb{R}^d : \|x\|_2 \le R\}$ be the closed Euclidean ball of radius $R > 0$.
Derive the analytical solution for the projection operator:
$$\Pi_C(z) = \arg\min_{y \in C} \frac{1}{2} \|y - z\|_2^2$$

#### Solution
If $\|z\|_2 \le R$, $z \in C$, so the minimum distance is 0, giving $\Pi_C(z) = z$.

Now suppose $\|z\|_2 > R$.
The problem is:
$$\min_y \frac{1}{2} \|y - z\|_2^2 \quad \text{subject to } \frac{1}{2} \|y\|_2^2 - \frac{1}{2} R^2 \le 0$$
Lagrangian:
$$\mathcal{L}(y, \lambda) = \frac{1}{2} \|y - z\|_2^2 + \lambda \left( \frac{1}{2} \|y\|_2^2 - \frac{1}{2} R^2 \right), \quad \lambda \ge 0$$
Stationarity:
$$\nabla_y \mathcal{L} = (y - z) + \lambda y = 0 \implies (1 + \lambda) y = z \implies y = \frac{1}{1 + \lambda} z$$
Since $\|z\|_2 > R$, the constraint must be active: $\|y\|_2 = R$.
$$\left\| \frac{1}{1 + \lambda} z \right\|_2 = R \implies \frac{\|z\|_2}{1 + \lambda} = R \implies 1 + \lambda = \frac{\|z\|_2}{R}$$
Substituting this back into $y$:
$$y^* = \frac{1}{\|z\|_2 / R} z = R \frac{z}{\|z\|_2}$$
Combining both cases:
$$\Pi_C(z) = \begin{cases} z & \text{if } \|z\|_2 \le R \\ R \frac{z}{\|z\|_2} & \text{if } \|z\|_2 > R \end{cases}$$
$\blacksquare$
