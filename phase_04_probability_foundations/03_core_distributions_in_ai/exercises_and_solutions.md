# Topic 4.3: Exercises & Solutions — Core Probability Distributions in AI

---

## Problem Set Overview

| Problem | Focus | Difficulty |
| :--- | :--- | :--- |
| **1** | **Derivation of Bernoulli MLE & Binary Cross-Entropy (BCE) Loss** | Medium |
| **2** | **2D Multivariate Gaussian Density & Mahalanobis Distance by Hand** | Medium |
| **3** | **The VAE Reparameterization Trick & Gradient Propagation** | Challenging |

---

## Problem 1: Derivation of Bernoulli MLE & BCE Loss

### The Context
Suppose we collect $N$ independent training samples $y_1, y_2, \dots, y_N \in \{0, 1\}$ from a Bernoulli distribution with unknown parameter $p \in [0, 1]$:
$$P(Y = y_i) = p^{y_i} (1 - p)^{1 - y_i}$$

1. Write the Likelihood function $L(p) = P(y_1, \dots, y_N \mid p)$.
2. Compute the Log-Likelihood $\log L(p)$.
3. Differentiate with respect to $p$, set the derivative to zero, and solve for the Maximum Likelihood Estimator $\hat{p}_{\text{MLE}}$.
4. Explain how minimizing Binary Cross-Entropy loss $\mathcal{L}_{\text{BCE}}$ is mathematically identical to maximizing the Bernoulli log-likelihood.

---

### Solution 1

#### 1. Likelihood Function
Because observations are independent:
$$L(p) = \prod_{i=1}^N P(Y = y_i \mid p) = \prod_{i=1}^N p^{y_i} (1 - p)^{1 - y_i}$$

#### 2. Log-Likelihood
Taking natural logarithm:
$$\ell(p) = \ln L(p) = \sum_{i=1}^N \ln\left( p^{y_i} (1 - p)^{1 - y_i} \right)$$
$$\ell(p) = \sum_{i=1}^N \left[ y_i \ln(p) + (1 - y_i) \ln(1 - p) \right]$$

#### 3. Differentiating and Solving for $\hat{p}$
$$\frac{d\ell}{dp} = \sum_{i=1}^N \left( \frac{y_i}{p} - \frac{1 - y_i}{1 - p} \right) = 0$$
$$\frac{1}{p} \sum_{i=1}^N y_i - \frac{1}{1 - p} \sum_{i=1}^N (1 - y_i) = 0$$

Let $S = \sum_{i=1}^N y_i$. Then $\sum_{i=1}^N (1 - y_i) = N - S$:
$$\frac{S}{p} = \frac{N - S}{1 - p}$$
$$S(1 - p) = p(N - S)$$
$$S - S p = N p - S p$$
$$S = N p \implies \hat{p}_{\text{MLE}} = \frac{S}{N} = \frac{1}{N} \sum_{i=1}^N y_i$$
*Result: The maximum likelihood estimate for a coin flip's bias is simply the fraction of heads in the sample!*

#### 4. Connection to BCE Loss
In machine learning, we frame problems as minimizing loss $\mathcal{L}$ rather than maximizing likelihood. Defining the loss as the negative average log-likelihood:
$$\mathcal{L} = -\frac{1}{N} \ell(p) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln(p) + (1 - y_i) \ln(1 - p) \right]$$
This is the exact definition of **Binary Cross-Entropy (BCE) Loss**! Minimizing BCE is identical to maximizing the likelihood of Bernoulli distributed targets.

---

## Problem 2: 2D Multivariate Gaussian Density by Hand

### The Context
Consider a 2D Gaussian distribution $\mathcal{N}(\mu, \Sigma)$ with:
$$\mu = \begin{bmatrix} 0 \\ 0 \end{bmatrix}, \quad \Sigma = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$$

1. Compute the determinant $|\Sigma|$.
2. Invert $\Sigma$ to find $\Sigma^{-1}$.
3. For point $\mathbf{x} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$, compute the squared Mahalanobis distance:
   $$D_M^2 = (\mathbf{x} - \mu)^T \Sigma^{-1} (\mathbf{x} - \mu)$$
4. Compute the squared Euclidean distance $\|\mathbf{x} - \mu\|^2$. Explain why the Mahalanobis distance is smaller than the Euclidean distance.
5. Compute the exact probability density $f(\mathbf{x})$.

---

### Solution 2

#### 1. Determinant
$$|\Sigma| = (2)(2) - (1)(1) = 4 - 1 = 3$$

#### 2. Inverse $\Sigma^{-1}$
For a $2 \times 2$ matrix $\begin{bmatrix} a & b \\ c & d \end{bmatrix}^{-1} = \frac{1}{ad - bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$:
$$\Sigma^{-1} = \frac{1}{3} \begin{bmatrix} 2 & -1 \\ -1 & 2 \end{bmatrix} = \begin{bmatrix} 2/3 & -1/3 \\ -1/3 & 2/3 \end{bmatrix}$$

#### 3. Squared Mahalanobis Distance $D_M^2$
With $\mu = [0, 0]^T$, $\mathbf{x} - \mu = [1, 1]^T$:
$$\Sigma^{-1} (\mathbf{x} - \mu) = \begin{bmatrix} 2/3 & -1/3 \\ -1/3 & 2/3 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 1/3 \\ 1/3 \end{bmatrix}$$
$$D_M^2 = \begin{bmatrix} 1 & 1 \end{bmatrix} \begin{bmatrix} 1/3 \\ 1/3 \end{bmatrix} = \frac{1}{3} + \frac{1}{3} = \frac{2}{3} \approx 0.6667$$

#### 4. Comparison with Euclidean Distance
$$\text{Euclidean } \|\mathbf{x} - \mu\|^2 = 1^2 + 1^2 = 2.0$$
*Why Mahalanobis is smaller ($0.6667 < 2.0$):*
The off-diagonal covariance is positive ($\text{Cov}(X_1, X_2) = +1$), meaning $X_1$ and $X_2$ naturally tend to increase together along the diagonal $y = x$. Because $[1, 1]^T$ points directly along this high-variance axis of correlation, it is much closer in "statistical distance" than Euclidean geometry suggests!

#### 5. Exact Probability Density $f(\mathbf{x})$
$$f(\mathbf{x}) = \frac{1}{(2\pi)^{D/2} |\Sigma|^{1/2}} \exp\left( -\frac{1}{2} D_M^2 \right)$$
$$f(\mathbf{x}) = \frac{1}{(2\pi)^1 \sqrt{3}} \exp\left( -\frac{1}{2} \times \frac{2}{3} \right) = \frac{1}{2\pi \sqrt{3}} e^{-1/3} \approx \frac{1}{10.8828} \times 0.7165 \approx \mathbf{0.0658}$$

---

## Problem 3: The VAE Reparameterization Trick

### The Context
In Variational Autoencoders, a latent variable $z \sim \mathcal{N}(\mu, \sigma^2)$ is generated from encoder outputs $\mu$ and $\sigma$.
If we sample naively: $z = \text{sample\_gaussian}(\mu, \sigma)$, PyTorch cannot compute derivatives $\frac{\partial z}{\partial \mu}$ or $\frac{\partial z}{\partial \sigma}$ because the sampling node is a black-box non-differentiable stochastic operation.

### The Math
Using the reparameterization trick:
$$z = \mu + \sigma \cdot \epsilon \quad \text{where } \epsilon \sim \mathcal{N}(0, 1)$$

1. Compute the partial derivative $\frac{\partial z}{\partial \mu}$.
2. Compute the partial derivative $\frac{\partial z}{\partial \sigma}$.
3. For a downstream loss function $\mathcal{L}(z)$, apply the multivariate chain rule to write $\frac{\partial \mathcal{L}}{\partial \mu}$ and $\frac{\partial \mathcal{L}}{\partial \sigma}$.

---

### Solution 3

#### 1 & 2. Partial Derivatives
Treating $\epsilon$ as an independent external input constant:
$$\frac{\partial z}{\partial \mu} = \frac{\partial}{\partial \mu} (\mu + \sigma \epsilon) = 1$$
$$\frac{\partial z}{\partial \sigma} = \frac{\partial}{\partial \sigma} (\mu + \sigma \epsilon) = \epsilon$$

#### 3. Chain Rule for Loss Gradients
By the calculus chain rule:
$$\frac{\partial \mathcal{L}}{\partial \mu} = \frac{\partial \mathcal{L}}{\partial z} \frac{\partial z}{\partial \mu} = \frac{\partial \mathcal{L}}{\partial z} \times 1 = \frac{\partial \mathcal{L}}{\partial z}$$
$$\frac{\partial \mathcal{L}}{\partial \sigma} = \frac{\partial \mathcal{L}}{\partial z} \frac{\partial z}{\partial \sigma} = \frac{\partial \mathcal{L}}{\partial z} \times \epsilon$$

*Insight:* By factoring out the stochastic randomness into $\epsilon$, both gradients $\frac{\partial \mathcal{L}}{\partial \mu}$ and $\frac{\partial \mathcal{L}}{\partial \sigma}$ are completely deterministic functions of $\frac{\partial \mathcal{L}}{\partial z}$, enabling standard gradient descent backpropagation!

