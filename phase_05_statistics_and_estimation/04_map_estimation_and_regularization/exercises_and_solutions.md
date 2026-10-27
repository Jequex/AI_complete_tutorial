# Topic 5.4: Exercises & Solutions — MAP Estimation & Regularization

---

## Problem Set Overview

| Problem | Focus | Difficulty | Real-World Context |
| :---: | :--- | :---: | :--- |
| **1** | **Precision-Weighted 1D Gaussian Mean MAP** | Easy | Fusing noisy sensor measurements with prior calibration beliefs |
| **2** | **Closed-Form Ridge Regression Matrix MAP** | Medium | Deriving weight decay to stabilize overparameterized neural networks |
| **3** | **Subgradient Calculus of L1 Sparsity (Soft-Thresholding)** | Challenging | Proving why Lasso regularization drives unimportant weights to exact zero |

---

## Problem 1: Precision-Weighted 1D Gaussian Mean MAP Estimation

### The Context
Suppose we collect $n$ independent observations $X_1, \dots, X_n \sim \mathcal{N}(\mu, \sigma^2)$ where noise variance $\sigma^2$ is known, but mean $\mu$ is unknown.
Before seeing the data, our domain knowledge suggests a Gaussian prior belief:
$$\mu \sim \mathcal{N}(\mu_0, \, \sigma_0^2)$$

### Task
1. Write the log-likelihood $\ln P(D \mid \mu)$ and log-prior $\ln P(\mu)$.
2. Write the MAP objective function $J(\mu) = \ln P(D \mid \mu) + \ln P(\mu)$ (ignoring constants).
3. Compute $\frac{dJ}{d\mu}$ and set it to zero to solve for $\hat{\mu}_{\text{MAP}}$.
4. Define precision $\tau = \frac{1}{\text{variance}}$, so data precision is $\tau_{\text{data}} = \frac{n}{\sigma^2}$ and prior precision is $\tau_{\text{prior}} = \frac{1}{\sigma_0^2}$. Show that $\hat{\mu}_{\text{MAP}}$ is a weighted average:
   $$\hat{\mu}_{\text{MAP}} = \frac{\tau_{\text{data}} \bar{X} + \tau_{\text{prior}} \mu_0}{\tau_{\text{data}} + \tau_{\text{prior}}}$$
5. What happens as $n \to \infty$? What happens as prior uncertainty $\sigma_0^2 \to \infty$?

---

<details>
<summary>Click to Reveal Step-by-Step Solution</summary>

### Step-by-Step Solution

#### 1. Log-Likelihood and Log-Prior
$$\ln P(D \mid \mu) = -\frac{1}{2\sigma^2} \sum_{i=1}^n (X_i - \mu)^2 + C_1$$
$$\ln P(\mu) = -\frac{1}{2\sigma_0^2} (\mu - \mu_0)^2 + C_2$$

#### 2. MAP Objective Function
$$J(\mu) = -\frac{1}{2\sigma^2} \sum_{i=1}^n (X_i - \mu)^2 - \frac{1}{2\sigma_0^2} (\mu - \mu_0)^2$$

#### 3. Differentiating and Setting to Zero
$$\frac{dJ}{d\mu} = \frac{1}{\sigma^2}\sum_{i=1}^n (X_i - \mu) - \frac{1}{\sigma_0^2}(\mu - \mu_0) = 0$$
Since $\sum_{i=1}^n (X_i - \mu) = n\bar{X} - n\mu$:
$$\frac{n}{\sigma^2}(\bar{X} - \mu) - \frac{1}{\sigma_0^2}(\mu - \mu_0) = 0$$
$$\frac{n}{\sigma^2}\bar{X} + \frac{1}{\sigma_0^2}\mu_0 = \left( \frac{n}{\sigma^2} + \frac{1}{\sigma_0^2} \right) \mu$$
$$\mathbf{\hat{\mu}_{\text{MAP}} = \frac{\frac{n}{\sigma^2}\bar{X} + \frac{1}{\sigma_0^2}\mu_0}{\frac{n}{\sigma^2} + \frac{1}{\sigma_0^2}}}$$

#### 4. Precision-Weighted Representation
Using $\tau_{\text{data}} = \frac{n}{\sigma^2}$ and $\tau_{\text{prior}} = \frac{1}{\sigma_0^2}$:
$$\mathbf{\hat{\mu}_{\text{MAP}} = \frac{\tau_{\text{data}} \bar{X} + \tau_{\text{prior}} \mu_0}{\tau_{\text{data}} + \tau_{\text{prior}}}} \quad \blacksquare$$

#### 5. Limiting Cases
- **Big Data Regime ($n \to \infty$):** $\tau_{\text{data}} = \frac{n}{\sigma^2} \to \infty$. The data completely overwhelms the prior:
  $$\lim_{n \to \infty} \hat{\mu}_{\text{MAP}} = \bar{X} = \hat{\mu}_{\text{MLE}}$$
- **Uninformative Prior ($\sigma_0^2 \to \infty$):** $\tau_{\text{prior}} \to 0$. Grandma has zero opinion, so we recover pure MLE: $\hat{\mu}_{\text{MAP}} = \bar{X}$.

</details>

---

## Problem 2: Closed-Form Ridge Regression Matrix MAP Derivation

### The Context
In vectorized linear regression, we observe feature matrix $X \in \mathbb{R}^{n \times d}$ and target vector $\mathbf{y} \in \mathbb{R}^n$.
The data likelihood assumes zero-mean spherical Gaussian noise:
$$\mathbf{y} \mid X, \mathbf{w} \sim \mathcal{N}(X\mathbf{w}, \, \sigma^2 I_n)$$
We impose an independent zero-mean Gaussian prior on weights:
$$\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \, \sigma_0^2 I_d)$$

### Task
1. Write the full joint log-posterior $\ln P(\mathbf{w} \mid X, \mathbf{y})$ in matrix vector notation (omitting constants).
2. Compute the vector gradient with respect to $\mathbf{w}$.
3. Set the gradient to zero and derive the closed-form **Ridge Regression** solution:
   $$\hat{\mathbf{w}}_{\text{MAP}} = (X^T X + \lambda I_d)^{-1} X^T \mathbf{y}$$
   Identify $\lambda$ in terms of $\sigma^2$ and $\sigma_0^2$.
4. Explain why $(X^T X + \lambda I_d)$ is strictly invertible even when $d > n$ (more features than samples!).

---

<details>
<summary>Click to Reveal Step-by-Step Solution</summary>

### Step-by-Step Solution

#### 1. Matrix Log-Posterior
$$J(\mathbf{w}) = -\frac{1}{2\sigma^2} (\mathbf{y} - X\mathbf{w})^T (\mathbf{y} - X\mathbf{w}) - \frac{1}{2\sigma_0^2} \mathbf{w}^T \mathbf{w}$$

#### 2. Vector Gradient with respect to $\mathbf{w}$
Expand the quadratic form:
$$(\mathbf{y} - X\mathbf{w})^T (\mathbf{y} - X\mathbf{w}) = \mathbf{y}^T \mathbf{y} - 2\mathbf{w}^T X^T \mathbf{y} + \mathbf{w}^T X^T X \mathbf{w}$$
Differentiate using standard matrix calculus ($\nabla_{\mathbf{w}} (\mathbf{w}^T A \mathbf{w}) = 2 A \mathbf{w}$ for symmetric $A$):
$$\nabla_{\mathbf{w}} J(\mathbf{w}) = -\frac{1}{2\sigma^2} \Big( -2 X^T \mathbf{y} + 2 X^T X \mathbf{w} \Big) - \frac{1}{2\sigma_0^2} (2\mathbf{w})$$
$$\nabla_{\mathbf{w}} J(\mathbf{w}) = \frac{1}{\sigma^2} X^T (\mathbf{y} - X\mathbf{w}) - \frac{1}{\sigma_0^2} \mathbf{w}$$

#### 3. Setting to Zero and Solving
$$\frac{1}{\sigma^2} X^T X \mathbf{w} + \frac{1}{\sigma_0^2} \mathbf{w} = \frac{1}{\sigma^2} X^T \mathbf{y}$$
Multiply across by $\sigma^2$:
$$\left( X^T X + \frac{\sigma^2}{\sigma_0^2} I_d \right) \mathbf{w} = X^T \mathbf{y}$$
Setting $\lambda = \frac{\sigma^2}{\sigma_0^2}$:
$$\mathbf{\hat{\mathbf{w}}_{\text{MAP}} = (X^T X + \lambda I_d)^{-1} X^T \mathbf{y}} \quad \blacksquare$$

#### 4. Guaranteed Numerical Invertibility
When $d > n$ (e.g. 100 features but only 20 samples), $X^T X \in \mathbb{R}^{d \times d}$ has rank at most $n < d$, meaning it has zero eigenvalues and cannot be inverted!
Adding $\lambda I_d$ shifts every single eigenvalue upwards by $+\lambda > 0$:
$$\lambda_i(X^T X + \lambda I_d) = \lambda_i(X^T X) + \lambda \ge \lambda > 0$$
Every eigenvalue is strictly positive, making the matrix strictly positive-definite and guaranteed invertible!

</details>

---

## Problem 3: The Subgradient Calculus of L1 Sparsity (Soft-Thresholding)

### The Context
Consider the 1D objective function representing a single weight $w$ under MSE data loss with target $a$ and an L1 (Laplace) regularization penalty:
$$f(w) = \frac{1}{2}(w - a)^2 + \lambda |w|, \quad \lambda > 0$$

Because $|w|$ has a sharp corner at $w = 0$, its derivative is not defined at zero. We use its **subgradient** set:
$$\partial |w| = \begin{cases} \{+1\} & \text{if } w > 0 \\ [-1, +1] & \text{if } w = 0 \\ \{-1\} & \text{if } w < 0 \end{cases}$$

A point $w^*$ is a global minimum if and only if $0 \in \partial f(w^*)$.

### Task
1. Analyze Case 1 ($w > 0$): Find $w^*$ and the condition on $a$.
2. Analyze Case 2 ($w < 0$): Find $w^*$ and the condition on $a$.
3. Analyze Case 3 ($w = 0$): Using the subgradient interval $[-1, +1]$, find the condition on $a$ such that $w^* = 0$ is the exact solution.
4. Synthesize the result into the closed-form **Soft-Thresholding Operator**:
   $$w^* = \operatorname{sign}(a) \max(0, |a| - \lambda)$$

---

<details>
<summary>Click to Reveal Step-by-Step Solution</summary>

### Step-by-Step Solution

#### 1. Case 1: $w > 0$
Here $\partial |w| = +1$:
$$f'(w) = (w - a) + \lambda = 0 \implies \mathbf{w^* = a - \lambda}$$
Since we assumed $w > 0$, this requires $a - \lambda > 0 \implies \mathbf{a > \lambda}$.

#### 2. Case 2: $w < 0$
Here $\partial |w| = -1$:
$$f'(w) = (w - a) - \lambda = 0 \implies \mathbf{w^* = a + \lambda}$$
Since we assumed $w < 0$, this requires $a + \lambda < 0 \implies \mathbf{a < -\lambda}$.

#### 3. Case 3: $w = 0$
At $w = 0$, the subgradient is:
$$\partial f(0) = (0 - a) + \lambda [-1, +1] = -a + [-\lambda, +\lambda] = [-a - \lambda, \, -a + \lambda]$$
For $w = 0$ to be the minimum, zero must belong to this subgradient set:
$$0 \in [-a - \lambda, \, -a + \lambda] \iff -\lambda \le a \le +\lambda \iff \mathbf{|a| \le \lambda}$$

#### 4. Synthesis into Soft-Thresholding Operator
Combining all three cases:
$$w^* = \begin{cases}
a - \lambda & \text{if } a > \lambda \\
0 & \text{if } |a| \le \lambda \\
a + \lambda & \text{if } a < -\lambda
\end{cases}$$

Using the sign function:
$$\mathbf{w^* = \operatorname{sign}(a) \max(0, \, |a| - \lambda)} \quad \blacksquare$$

> **🧒 12-Year-Old Takeaway (The Mathematical Origin of Sparsity):**
> Notice that whenever the raw unregularized signal $|a|$ is smaller than the leash threshold $\lambda$, the weight is forced to **identically $0.000$**! This is why L1 Lasso prunes useless neural network connections, while L2 Ridge only shrinks weights into tiny fractions without ever setting them to true zero.

</details>
