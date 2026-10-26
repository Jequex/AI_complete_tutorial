# Topic 4.4: Exercises & Solutions — Expectation, Variance, and Covariance

---

## Problem Set Overview

| Problem | Focus | Difficulty |
| :--- | :--- | :--- |
| **1** | **Proof that Zero Covariance Does NOT Imply Independence** | Medium |
| **2** | **Derivation of the He (Kaiming) Weight Initialization Variance** | Medium |
| **3** | **Covariance Matrix & Pearson Correlation by Hand** | Challenging |

---

## Problem 1: Zero Covariance Does NOT Imply Independence

### The Context
In machine learning, it is widely known that **independent variables have zero covariance**:
$$X \perp Y \implies \text{Cov}(X, Y) = 0$$
However, many practitioners falsely believe the converse is true. Covariance measures *linear* relationships only. Non-linear relationships can have exact zero covariance while being 100% dependent!

### Mathematical Task
Let $X$ be a continuous random variable uniformly distributed on $[-1, 1]$:
$$f_X(x) = \frac{1}{2} \quad \text{for } x \in [-1, 1]$$
Let $Y = X^2$.
1. Compute $\mathbb{E}[X]$.
2. Compute $\mathbb{E}[Y]$.
3. Compute $\mathbb{E}[XY]$.
4. Compute the covariance $\text{Cov}(X, Y) = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]$.
5. Explain why $X$ and $Y$ are strongly dependent despite having $\text{Cov}(X, Y) = 0$.

---

### Solution 1

#### 1. Expected Value $\mathbb{E}[X]$
Because $X$ is symmetric about $0$:
$$\mathbb{E}[X] = \int_{-1}^1 x \left( \frac{1}{2} \right) dx = \left[ \frac{x^2}{4} \right]_{-1}^1 = \frac{1}{4} - \frac{1}{4} = 0$$

#### 2. Expected Value $\mathbb{E}[Y]$
Since $Y = X^2$:
$$\mathbb{E}[Y] = \mathbb{E}[X^2] = \int_{-1}^1 x^2 \left( \frac{1}{2} \right) dx = \left[ \frac{x^3}{6} \right]_{-1}^1 = \frac{1}{6} - \left( -\frac{1}{6} \right) = \frac{2}{6} = \frac{1}{3}$$

#### 3. Expected Value $\mathbb{E}[XY]$
$$XY = X(X^2) = X^3$$
$$\mathbb{E}[XY] = \mathbb{E}[X^3] = \int_{-1}^1 x^3 \left( \frac{1}{2} \right) dx = \left[ \frac{x^4}{8} \right]_{-1}^1 = \frac{1}{8} - \frac{1}{8} = 0$$
*(The integral of any odd function over a symmetric interval is always zero!)*

#### 4. Covariance $\text{Cov}(X, Y)$
$$\text{Cov}(X, Y) = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y] = 0 - (0 \times \frac{1}{3}) = \mathbf{0}$$

#### 5. Why They Are Dependent:
$Y$ is completely deterministic given $X$ ($Y = X^2$). If someone tells you $X = 0.5$, you know with $100\%$ certainty that $Y = 0.25$.
They are completely dependent, yet their linear covariance is exactly zero!
*Takeaway for AI:* Linear models (like linear regression) cannot capture quadratic dependencies without feature engineering or deep non-linear layers.

---

## Problem 2: Derivation of the He / Kaiming Weight Initialization

### The Context
In deep neural networks with ReLU activations ($y = \max(0, z)$), poor random weight initialization causes activations to either explode to infinity or vanish to zero as the signal propagates through 100+ layers. Kaiming He et al. (2015) derived the exact weight variance needed to keep signal variance constant.

### Derivation Task
Consider a single neuron computing a linear combination of $d_{\text{in}}$ inputs:
$$z = \sum_{i=1}^{d_{\text{in}}} w_i x_i$$
Assume:
- Inputs $x_i$ and weights $w_i$ are independent.
- Weights $w_i$ have zero mean: $\mathbb{E}[w_i] = 0$ and variance $\text{Var}(w_i) = \sigma_w^2$.
- Inputs $x_i$ have zero mean: $\mathbb{E}[x_i] = 0$ and variance $\text{Var}(x_i) = \sigma_x^2$.

1. Using linearity of expectation and independence, compute $\mathbb{E}[z]$.
2. Compute the variance $\text{Var}(z)$.
3. If $z$ is passed through a ReLU activation $a = \max(0, z)$, and since $z$ is symmetric about $0$, half of the activations are zeroed out: $\text{Var}(a) = \frac{1}{2} \text{Var}(z)$.
   Determine the required weight variance $\sigma_w^2$ such that the output activation has the same variance as the input: $\text{Var}(a) = \text{Var}(x_i) = \sigma_x^2$.

---

### Solution 2

#### 1. Mean $\mathbb{E}[z]$
By linearity of expectation and independence:
$$\mathbb{E}[z] = \sum_{i=1}^{d_{\text{in}}} \mathbb{E}[w_i x_i] = \sum_{i=1}^{d_{\text{in}}} \mathbb{E}[w_i] \mathbb{E}[x_i] = \sum_{i=1}^{d_{\text{in}}} (0)(0) = 0$$

#### 2. Variance $\text{Var}(z)$
Because weights and inputs are independent with mean zero:
$$\text{Var}(w_i x_i) = \mathbb{E}[(w_i x_i)^2] - (\mathbb{E}[w_i x_i])^2 = \mathbb{E}[w_i^2] \mathbb{E}[x_i^2] - 0 = \sigma_w^2 \sigma_x^2$$
Since terms are independent, variances add:
$$\text{Var}(z) = \sum_{i=1}^{d_{\text{in}}} \text{Var}(w_i x_i) = d_{\text{in}} \sigma_w^2 \sigma_x^2$$

#### 3. Solving for He Initialization
Passing $z$ through ReLU zeros out the negative half, cutting variance in half:
$$\text{Var}(a) = \frac{1}{2} \text{Var}(z) = \frac{1}{2} d_{\text{in}} \sigma_w^2 \sigma_x^2$$
We want $\text{Var}(a) = \sigma_x^2$ (variance preservation across layers):
$$\frac{1}{2} d_{\text{in}} \sigma_w^2 \sigma_x^2 = \sigma_x^2$$
Dividing both sides by $\sigma_x^2$:
$$\frac{1}{2} d_{\text{in}} \sigma_w^2 = 1 \implies \mathbf{\sigma_w^2 = \frac{2}{d_{\text{in}}}}$$
*Result: This is the exact mathematical derivation of He / Kaiming Normal Initialization used across all modern ResNets and ConvNets!*

---

## Problem 3: Covariance Matrix & Correlation by Hand

### The Context
Given a tiny dataset of 3 observations in 2D space:
$$x_1 = \begin{bmatrix} 1 \\ 2 \end{bmatrix}, \quad x_2 = \begin{bmatrix} 2 \\ 5 \end{bmatrix}, \quad x_3 = \begin{bmatrix} 3 \\ 5 \end{bmatrix}$$

1. Compute the mean vector $\mu = \begin{bmatrix} \mu_1 \\ \mu_2 \end{bmatrix}$.
2. Compute the sample variances $s_1^2$ and $s_2^2$ (using divisor $N - 1 = 2$).
3. Compute the sample covariance $s_{12} = \text{Cov}(X_1, X_2)$.
4. Construct the $2 \times 2$ Covariance Matrix $\Sigma$.
5. Compute the Pearson correlation coefficient $\rho(X_1, X_2)$.

---

### Solution 3

#### 1. Mean Vector $\mu$
$$\mu_1 = \frac{1 + 2 + 3}{3} = 2.0$$
$$\mu_2 = \frac{2 + 5 + 5}{3} = \frac{12}{3} = 4.0$$
$$\mu = \begin{bmatrix} 2.0 \\ 4.0 \end{bmatrix}$$

#### 2. Sample Variances
Mean-centered deviations $X_1 - \mu_1$: $[-1, 0, +1]$
$$s_1^2 = \frac{(-1)^2 + 0^2 + 1^2}{3 - 1} = \frac{1 + 0 + 1}{2} = \frac{2}{2} = 1.0$$

Mean-centered deviations $X_2 - \mu_2$: $[-2, +1, +1]$
$$s_2^2 = \frac{(-2)^2 + 1^2 + 1^2}{3 - 1} = \frac{4 + 1 + 1}{2} = \frac{6}{2} = 3.0$$

#### 3. Sample Covariance $s_{12}$
Cross-products $(X_1 - \mu_1)(X_2 - \mu_2)$:
- Sample 1: $(-1) \times (-2) = +2$
- Sample 2: $(0) \times (1) = 0$
- Sample 3: $(+1) \times (1) = +1$

$$s_{12} = \frac{2 + 0 + 1}{3 - 1} = \frac{3}{2} = 1.5$$

#### 4. Covariance Matrix $\Sigma$
$$\Sigma = \begin{bmatrix} 1.0 & 1.5 \\ 1.5 & 3.0 \end{bmatrix}$$

#### 5. Pearson Correlation $\rho$
$$\rho = \frac{s_{12}}{\sqrt{s_1^2 \cdot s_2^2}} = \frac{1.5}{\sqrt{1.0 \times 3.0}} = \frac{1.5}{\sqrt{3}} = \frac{1.5}{1.732} \approx \mathbf{0.8660}$$
*Result: A strong positive linear correlation ($\rho \approx 0.866$) exists between feature 1 and feature 2.*

