# Topic 8.1: Linear Regression & The Normal Equation

---

## 1. Executive Summary & The 12-Year-Old Analogy

Linear regression is the foundational cornerstone of predictive modeling and supervised learning. At its heart, linear regression answers a fundamental question: **given a set of continuous features, what is the best straight line (or flat hyperplane in higher dimensions) that models the target relationship?**

Despite its conceptual simplicity, linear regression exposes the two principal paradigms for solving machine learning problems:
1. **Analytical Closed-Form Solutions (The Normal Equation):** Solving a system of linear equations in a single mathematical step using linear algebra.
2. **Iterative Numerical Optimization (Gradient Descent):** Incrementally navigating down a convex loss surface toward the optimal weights.

```
 Target (y)
    ^
    |          * (Actual observation)
    |         /|
    |        / | Residual e_i = y_i - ŷ_i
    |       /  v
    |      *---[ŷ = Xw + b] (Best Fit Line)
    |     /   *
    |    *
    |   /  *
    +------------------------> Feature (x)
```

---

### The 12-Year-Old Analogy: The Lemonade Stand Line of Best Fit

Imagine you run a lemonade stand every summer afternoon. You want to predict how many cups of ice-cold lemonade you will sell tomorrow based on the weather forecast temperature.

- **The Scatter Plot:** You plot your sales history on graph paper. The horizontal axis is temperature in degrees Fahrenheit ($x$), and the vertical axis is cups of lemonade sold ($y$). The dots tilt upwards—hotter days clearly mean more thirsty customers—but the dots don't line up perfectly. Some days had a baseball game nearby (extra sales), while other days had a sudden shower (fewer sales).
- **The Spaghetti Stick (The Model):** You take an uncooked strand of spaghetti and lay it across the paper to capture the trend. The spaghetti is your linear model: $\hat{y} = w \cdot x + b$. The slope $w$ says how many extra cups you sell for every 1°F increase. The intercept $b$ is your baseline cups sold if it were 0°F.
- **The Rubber Bands (The Residuals):** Imagine tying tiny elastic rubber bands between each actual data dot and the spaghetti stick. If the stick is too high, rubber bands pull it down; if it is too low, rubber bands pull it up.
- **Squaring the Errors (MSE Loss):** Because rubber bands pull harder the more they stretch, the physical energy in each band grows with the **square** of its stretched length: $(y_i - \hat{y}_i)^2$. Squaring ensures that positive errors and negative errors don't cancel each other out, and it penalizes huge misses severely.
- **The Normal Equation (The Magic Ruler):** Instead of wiggling the stick by hand, a master carpenter uses a mathematical formula: $w = (X^T X)^{-1} X^T y$. In a fraction of a second, it computes the exact balance point where all rubber bands exert zero rotational torque.
- **Gradient Descent (The Blindfolded Nudge):** If you had 10 million data points across 50,000 cities, computing the master formula might crash your computer. Instead, you gently tilt the spaghetti stick one millimeter at a time in whatever direction reduces the rubber band pull the fastest, until it settles into place.

---

## 2. Mathematical Symbol Decoder

| Symbol | Mathematical Type | Description & Dimensions |
| :--- | :--- | :--- |
| $N$ | $\mathbb{N}$ | Number of training samples in the dataset |
| $d$ | $\mathbb{N}$ | Number of input features (dimensions) |
| $x^{(i)} \in \mathbb{R}^d$ | Column vector | Feature vector for the $i$-th sample |
| $y \in \mathbb{R}^N$ | Column vector | Ground-truth continuous target vector |
| $X \in \mathbb{R}^{N \times (d+1)}$ | Real matrix | Design matrix with an augmented column of ones: $[1, x_1, \dots, x_d]$ |
| $w \in \mathbb{R}^{d+1}$ | Column vector | Parameter vector containing bias $w_0 = b$ and feature weights $[w_1, \dots, w_d]^T$ |
| $\hat{y} = Xw \in \mathbb{R}^N$ | Column vector | Vector of model predictions across all $N$ samples |
| $e = y - \hat{y} \in \mathbb{R}^N$ | Column vector | Residual error vector ($e = y - Xw$) |
| $J(w) \in \mathbb{R}$ | Scalar field | Mean Squared Error (MSE) objective function: $\frac{1}{2N} \|Xw - y\|_2^2$ |
| $\nabla_w J(w) \in \mathbb{R}^{d+1}$| Vector | Gradient of the MSE loss with respect to weight vector $w$ |
| $(X^T X)^{-1}$ | Real matrix | Inverse of the Grammian / correlation matrix $\mathbb{R}^{(d+1) \times (d+1)}$ |
| $X^\dagger$ | Real matrix | Moore-Penrose pseudoinverse: $(X^T X)^{-1} X^T \in \mathbb{R}^{(d+1) \times N}$ |
| $\eta \in \mathbb{R}_{>0}$ | Scalar | Learning rate (step size hyperparameter for gradient descent) |

---

## 3. Mathematical Formulation of Linear Regression

### 3.1 Scalar and Matrix Hypothesis

For a single data observation $(x^{(i)}, y^{(i)})$, the linear hypothesis is:
$$\hat{y}^{(i)} = w_0 + w_1 x_1^{(i)} + w_2 x_2^{(i)} + \dots + w_d x_d^{(i)} = \sum_{j=0}^d w_j x_j^{(i)}, \quad \text{where } x_0^{(i)} \equiv 1$$

Aggregating all $N$ samples into matrix form:
$$X = \begin{bmatrix} 1 & x_1^{(1)} & x_2^{(1)} & \cdots & x_d^{(1)} \\ 1 & x_1^{(2)} & x_2^{(2)} & \cdots & x_d^{(2)} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 1 & x_1^{(N)} & x_2^{(N)} & \cdots & x_d^{(N)} \end{bmatrix} \in \mathbb{R}^{N \times (d+1)}, \quad w = \begin{bmatrix} w_0 \\ w_1 \\ \vdots \\ w_d \end{bmatrix} \in \mathbb{R}^{d+1}, \quad y = \begin{bmatrix} y^{(1)} \\ y^{(2)} \\ \vdots \\ y^{(N)} \end{bmatrix} \in \mathbb{R}^N$$

The vector of all predictions is cleanly expressed as:
$$\hat{y} = Xw$$

---

### 3.2 The Loss Function: Mean Squared Error (MSE)

We quantify the discrepancy between predictions $\hat{y}$ and true labels $y$ using the Mean Squared Error:
$$J(w) = \frac{1}{2N} \sum_{i=1}^N \left(\hat{y}^{(i)} - y^{(i)}\right)^2 = \frac{1}{2N} \|Xw - y\|_2^2$$

*(Note: The factor of $\frac{1}{2}$ is an algebraic convenience that cancels the factor of 2 when taking the derivative).*

Expanding the squared $L_2$ norm using the vector inner product identity $\|v\|_2^2 = v^T v$:
$$\begin{aligned}
J(w) &= \frac{1}{2N} (Xw - y)^T (Xw - y) \\
&= \frac{1}{2N} \left((Xw)^T - y^T\right) (Xw - y) \\
&= \frac{1}{2N} \left(w^T X^T X w - w^T X^T y - y^T X w + y^T y\right)
\end{aligned}$$

Because $w^T X^T y$ is a scalar ($1 \times 1$ matrix), it equals its own transpose:
$$(w^T X^T y)^T = y^T (X^T)^T (w^T)^T = y^T X w$$

Therefore, the two middle terms combine:
$$J(w) = \frac{1}{2N} \left(w^T X^T X w - 2 y^T X w + y^T y\right)$$

---

## 4. Closed-Form Derivation: The Normal Equation

To find the weight vector $w^*$ that minimizes $J(w)$, we compute the gradient $\nabla_w J(w)$ and set it to zero:
$$\nabla_w J(w) = \mathbf{0}$$

### 4.1 Step-by-Step Matrix Calculus

We differentiate each term in $J(w)$ with respect to the vector $w$:
1. **First term: $\frac{\partial}{\partial w} \left(w^T (X^T X) w\right)$**  
   Using the standard vector identity $\frac{\partial}{\partial w} (w^T A w) = (A + A^T) w$.  
   Here, $A = X^T X$. Since $A^T = (X^T X)^T = X^T (X^T)^T = X^T X = A$, $A$ is symmetric:
   $$\frac{\partial}{\partial w} \left(w^T X^T X w\right) = 2 X^T X w$$

2. **Second term: $\frac{\partial}{\partial w} \left(2 y^T X w\right)$**  
   Let $a^T = 2 y^T X$, so $a = 2 X^T y$.  
   Using the identity $\frac{\partial}{\partial w} (a^T w) = a$:
   $$\frac{\partial}{\partial w} \left(2 y^T X w\right) = 2 X^T y$$

3. **Third term: $\frac{\partial}{\partial w} \left(y^T y\right)$**  
   Since $y^T y$ does not depend on $w$:
   $$\frac{\partial}{\partial w} \left(y^T y\right) = \mathbf{0}$$

Combining these results:
$$\nabla_w J(w) = \frac{1}{2N} \left(2 X^T X w - 2 X^T y\right) = \frac{1}{N} \left(X^T X w - X^T y\right)$$

### 4.2 Setting Gradient to Zero

Setting $\nabla_w J(w) = \mathbf{0}$:
$$\frac{1}{N} \left(X^T X w - X^T y\right) = \mathbf{0} \implies X^T X w = X^T y$$

This fundamental system of linear equations is known as the **Normal Equations**.

Assuming the matrix $X^T X \in \mathbb{R}^{(d+1) \times (d+1)}$ is invertible (non-singular):
$$w^* = (X^T X)^{-1} X^T y$$

$\blacksquare$

---

### 4.3 Geometric Interpretation: Orthogonal Projection

Why is this called the "Normal" equation? "Normal" means perpendicular.

The target vector $y \in \mathbb{R}^N$ lives in an $N$-dimensional sample space. The columns of design matrix $X$ span a lower-dimensional subspace called the **column space** $\text{Col}(X) \subseteq \mathbb{R}^N$.

```
       y (Target vector in R^N)
       ^
       | \
       |  \  Error vector: e = y - Xw*
       |   \ (Orthogonal to Col(X))
       |    v
       +-----> ŷ = Xw* (Orthogonal projection in Col(X))
   [ Column Space Col(X) ]
```

Any prediction $\hat{y} = Xw$ is a linear combination of the columns of $X$, and therefore lies completely inside $\text{Col}(X)$. 

The prediction that minimizes the Euclidean distance $\|y - \hat{y}\|_2$ is the **orthogonal projection** of $y$ onto $\text{Col}(X)$. At this point, the residual error vector $e = y - \hat{y} = y - Xw^*$ must be perpendicular (normal) to every single column of $X$:
$$X^T e = \mathbf{0} \implies X^T (y - Xw^*) = \mathbf{0} \implies X^T X w^* = X^T y$$

---

## 5. Iterative Numerical Solution: Gradient Descent

While the Normal Equation yields an exact solution in a single step, inverting a $(d+1) \times (d+1)$ matrix has time complexity $\mathcal{O}(d^3)$. When $d$ exceeds 10,000 features, matrix inversion becomes computationally prohibitive or memory-impossible.

Gradient descent provides an iterative $\mathcal{O}(N d)$ alternative:

$$w_{t+1} = w_t - \eta \nabla_w J(w_t)$$

Substituting our derived gradient $\nabla_w J(w) = \frac{1}{N} X^T (Xw - y)$:
$$w_{t+1} = w_t - \frac{\eta}{N} X^T (Xw_t - y)$$

In scalar notation, for each weight parameter $w_j$ ($j = 0, 1, \dots, d$):
$$w_j \leftarrow w_j - \frac{\eta}{N} \sum_{i=1}^N \left(\hat{y}^{(i)} - y^{(i)}\right) x_j^{(i)}$$

Because $J(w)$ is a quadratic form with positive semi-definite Hessian $\nabla^2 J(w) = \frac{1}{N} X^T X \succeq 0$, the loss surface is strictly **convex**. Gradient descent is mathematically guaranteed to converge to the global minimum $w^*$, provided the learning rate satisfies:
$$0 < \eta < \frac{2}{\lambda_{\max}\left(\frac{1}{N} X^T X\right)}$$

---

## 6. Comparison: Normal Equation vs. Gradient Descent

| Criterion | Normal Equation | Gradient Descent |
| :--- | :--- | :--- |
| **Formula** | $w = (X^T X)^{-1} X^T y$ | $w_{t+1} = w_t - \frac{\eta}{N} X^T (Xw_t - y)$ |
| **Hyperparameters** | None (no tuning required) | Requires choosing learning rate $\eta$, epochs, stopping threshold |
| **Computational Complexity** | $\mathcal{O}(d^3 + N d^2)$ | $\mathcal{O}(k \cdot N \cdot d)$ where $k$ is iteration count |
| **Memory Footprint** | $\mathcal{O}(d^2)$ to store $X^T X$ | $\mathcal{O}(d)$ to store gradients and parameters |
| **Feature Scaling** | Not required | Strongly required (unscaled features create elongated ravines) |
| **Scaling with $d$** | Slow if $d > 10^4$ | Efficient even for $d = 10^6$ |
| **Scaling with $N$** | Computes $X^T X$ in $\mathcal{O}(N d^2)$ | Scales linearly $\mathcal{O}(N)$, mini-batch scales with batch size $B$ |
| **Singularity Vulnerability** | Fails if $X^T X$ is singular (collinear features) | Robust; finds a valid minimum even if singular |

---

## 7. Numerical Invertibility & Regularization

What happens when $X^T X$ is singular (non-invertible)?
This occurs if:
1. **Multicollinearity:** Two or more features are linearly dependent (e.g., $x_2 = 2.5 x_1$).
2. **Underdetermined System:** There are fewer samples than features ($N < d$).

To guarantee invertibility and prevent overfitting, we add an $L_2$ regularization penalty (Ridge Regression):
$$J_{\text{ridge}}(w) = \frac{1}{2N} \|Xw - y\|_2^2 + \frac{\lambda}{2} \|w\|_2^2$$

Setting the gradient to zero yields:
$$\nabla_w J_{\text{ridge}}(w) = \frac{1}{N} (X^T X w - X^T y) + \lambda w = \mathbf{0}$$
$$\left(X^T X + N \lambda I\right) w = X^T y \implies w^* = \left(X^T X + N \lambda I\right)^{-1} X^T y$$

Since $X^T X$ is positive semi-definite and $N \lambda I$ is strictly positive definite ($\lambda > 0$), their sum is always strictly positive definite and invertible!
