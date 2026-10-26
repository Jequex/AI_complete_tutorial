# Topic 8.4: Exercises & Solutions — Support Vector Machines

---

### Exercise 1: Geometric Margin and Distance to Hyperplane

**Problem:**  
A linear decision boundary in $\mathbb{R}^2$ is defined by weights $w = [3.0, 4.0]^T$ and bias $b = -5.0$.
Consider three sample points:
- Point $A$: $x_A = [1.0, 3.0]^T$, label $y_A = +1$
- Point $B$: $x_B = [0.0, 1.0]^T$, label $y_B = -1$
- Point $C$: $x_C = [2.0, -0.25]^T$, label $y_C = +1$

1. Compute the Euclidean norm of the weight vector $\|w\|_2$.
2. Compute the total margin width between the canonical boundaries $w^T x + b = \pm 1$.
3. Compute the functional margin $\hat{\gamma}_i = y_i (w^T x_i + b)$ for each point.
4. Compute the geometric margin $\gamma_i = \frac{\hat{\gamma}_i}{\|w\|_2}$ for each point.
5. Are all three points on the correct side of their respective margin boundaries ($\hat{\gamma}_i \ge 1$)?

<details>
<summary>View Complete Solution</summary>

**Step 1: Euclidean norm $\|w\|_2$**
$$\|w\|_2 = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5.0$$

**Step 2: Margin width**
$$\text{Margin Width} = \frac{2}{\|w\|_2} = \frac{2}{5.0} = 0.40$$

**Step 3 & 4: Functional and geometric margins**
- **Point $A$ ($y = +1$):**
  $$z_A = w^T x_A + b = 3(1.0) + 4(3.0) - 5.0 = 3 + 12 - 5 = 10.0$$
  $$\hat{\gamma}_A = (+1)(10.0) = 10.0, \quad \gamma_A = \frac{10.0}{5.0} = 2.0$$
- **Point $B$ ($y = -1$):**
  $$z_B = w^T x_B + b = 3(0.0) + 4(1.0) - 5.0 = 4 - 5 = -1.0$$
  $$\hat{\gamma}_B = (-1)(-1.0) = +1.0, \quad \gamma_B = \frac{1.0}{5.0} = 0.20$$
- **Point $C$ ($y = +1$):**
  $$z_C = w^T x_C + b = 3(2.0) + 4(-0.25) - 5.0 = 6 - 1 - 5 = 0.0$$
  $$\hat{\gamma}_C = (+1)(0.0) = 0.0, \quad \gamma_C = 0.0$$

**Step 5: Assessment**
- Point $A$ has $\hat{\gamma}_A = 10.0 \ge 1$: safely outside the margin.
- Point $B$ has $\hat{\gamma}_B = 1.0 = 1$: sits **directly on the negative margin curb** (a Support Vector!).
- Point $C$ has $\hat{\gamma}_C = 0.0 < 1$: sits directly on the decision boundary itself ($z=0$), violating the margin constraint with slack $\xi_C = 1 - 0 = 1.0$.

$\blacksquare$
</details>

---

### Exercise 2: KKT Conditions and Support Vector Identification

**Problem:**  
After solving the Dual SVM problem for a dataset of 4 points, the optimal Lagrange multipliers $\alpha^*$ with $C = 5.0$ are:
$$\alpha_1^* = 0.0, \quad \alpha_2^* = 3.2, \quad \alpha_3^* = 5.0, \quad \alpha_4^* = 0.0$$

By applying KKT complementary slackness:
1. Which points are Support Vectors?
2. Which point is non-critical and could be deleted from the dataset without altering the learned decision hyperplane?
3. Which point is a bounded support vector ($\alpha_i = C$) actively violating the margin ($\xi_i > 0$)?
4. Express the optimal weight vector $w^*$ in terms of the training instances.

<details>
<summary>View Complete Solution</summary>

**1. Support Vectors:**  
Points 2 and 3 are Support Vectors because their dual multipliers are strictly positive ($\alpha_2^* = 3.2 > 0$ and $\alpha_3^* = 5.0 > 0$).

**2. Non-critical points:**  
Points 1 and 4 have $\alpha^* = 0$. By KKT conditions, they lie strictly outside the margin ($y_i(w^T x_i + b) > 1$). They could be modified or removed completely without changing $w^*$ or $b^*$.

**3. Margin-violating Support Vector:**  
Point 3 has $\alpha_3^* = C = 5.0$. Points with $\alpha_i = C$ are bounded support vectors that sit inside the margin corridor or on the misclassified side ($\xi_3 > 0$).

**4. Weight vector formula:**  
$$w^* = \sum_{i=1}^4 \alpha_i^* y_i x_i = 3.2 y_2 x_2 + 5.0 y_3 x_3$$

$\blacksquare$
</details>

---

### Exercise 3: Hinge Loss Subgradient Descent Step

**Problem:**  
We optimize a linear SVM via unconstrained Hinge Loss:
$$\mathcal{L}(w, b) = \max\left(0, 1 - y(w^T x + b)\right) + \frac{\lambda}{2} \|w\|_2^2$$
Let $\lambda = 0.01$ and learning rate $\eta = 0.1$.  
Current parameters: $w = [1.0, 1.0]^T$, $b = 0.0$.  
Consider a sample $(x, y)$ where $x = [2.0, -1.0]^T$ and $y = +1$.

1. Compute the logit score $z = w^T x + b$.
2. Compute the functional margin $y \cdot z$.
3. Is the sample violating the margin? What is the hinge loss?
4. Compute the subgradients $\nabla_w \mathcal{L}$ and $\frac{\partial \mathcal{L}}{\partial b}$.
5. Perform the gradient update to find $w_{\text{new}}$ and $b_{\text{new}}$.

<details>
<summary>View Complete Solution</summary>

**Step 1: Compute logit**
$$z = (1.0)(2.0) + (1.0)(-1.0) + 0.0 = 2.0 - 1.0 = 1.0$$

**Step 2: Functional margin**
$$y \cdot z = (+1)(1.0) = 1.0$$

**Step 3: Margin evaluation**
Since $y \cdot z = 1.0$, the point lies exactly on the margin boundary! The hinge loss is $\max(0, 1 - 1.0) = 0.0$.  
However, by the standard subgradient convention for boundary points ($y z \le 1$):
$$\nabla_w \ell = -y x = -1 \begin{bmatrix} 2.0 \\ -1.0 \end{bmatrix} = \begin{bmatrix} -2.0 \\ 1.0 \end{bmatrix}, \quad \nabla_b \ell = -y = -1$$

Adding the regularization term $\lambda w$:
$$\nabla_w \mathcal{L} = \begin{bmatrix} -2.0 \\ 1.0 \end{bmatrix} + 0.01 \begin{bmatrix} 1.0 \\ 1.0 \end{bmatrix} = \begin{bmatrix} -1.99 \\ 1.01 \end{bmatrix}$$

**Step 5: Parameter update ($\eta = 0.1$)**
$$w_{\text{new}} = \begin{bmatrix} 1.0 \\ 1.0 \end{bmatrix} - 0.1 \begin{bmatrix} -1.99 \\ 1.01 \end{bmatrix} = \begin{bmatrix} 1.0 + 0.199 \\ 1.0 - 0.101 \end{bmatrix} = \begin{bmatrix} 1.199 \\ 0.899 \end{bmatrix}$$
$$b_{\text{new}} = 0.0 - 0.1(-1.0) = +0.10$$

$\blacksquare$
</details>

---

### Exercise 4: Polynomial Kernel Explicit Feature Map Proof

**Problem:**  
Let $x, z \in \mathbb{R}^2$ be two 2D vectors: $x = [x_1, x_2]^T$ and $z = [z_1, z_2]^T$.  
Consider the homogeneous quadratic polynomial kernel:
$$K(x, z) = (x^T z)^2$$

Prove that evaluating $K(x, z)$ is algebraically identical to taking the standard inner product in a 3-dimensional feature space $\phi(x)^T \phi(z)$, and write down the explicit feature mapping $\phi(x) \in \mathbb{R}^3$.

<details>
<summary>View Complete Solution</summary>

**Algebraic Expansion:**
$$x^T z = x_1 z_1 + x_2 z_2$$

Squaring this scalar product:
$$\begin{aligned}
K(x, z) &= (x_1 z_1 + x_2 z_2)^2 \\
&= (x_1 z_1)^2 + 2(x_1 z_1)(x_2 z_2) + (x_2 z_2)^2 \\
&= x_1^2 z_1^2 + 2 x_1 x_2 z_1 z_2 + x_2^2 z_2^2
\end{aligned}$$

Now, express this as a dot product $\phi(x)^T \phi(z)$ of two 3D vectors:
$$\phi(x) = \begin{bmatrix} x_1^2 \\ \sqrt{2} x_1 x_2 \\ x_2^2 \end{bmatrix}, \quad \phi(z) = \begin{bmatrix} z_1^2 \\ \sqrt{2} z_1 z_2 \\ z_2^2 \end{bmatrix}$$

Computing their dot product:
$$\phi(x)^T \phi(z) = (x_1^2)(z_1^2) + (\sqrt{2} x_1 x_2)(\sqrt{2} z_1 z_2) + (x_2^2)(z_2^2) = x_1^2 z_1^2 + 2 x_1 x_2 z_1 z_2 + x_2^2 z_2^2 = K(x, z)$$

This confirms that computing the kernel $(x^T z)^2$ in 2D space implicitly evaluates the geometry of a 3D parabolic feature space $\phi(x) \in \mathbb{R}^3$!

$\blacksquare$
</details>
