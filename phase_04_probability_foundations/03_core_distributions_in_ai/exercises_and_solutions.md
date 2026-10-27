# Topic 4.3: Exercises & Solutions — Core Probability Distributions in AI

Welcome to the problem set for **Topic 4.3: Core Probability Distributions in AI**.

Work through each problem with pen and paper before expanding the collapsible solution panels.

---

## 📝 Problem Set Overview

| Problem | Focus | Difficulty | Practical AI Connection |
| :---: | :--- | :---: | :--- |
| **1** | **Derivation of Bernoulli MLE & Binary Cross-Entropy (BCE) Loss** | Medium | Foundation of logistic regression, Sigmoid heads, and binary losses |
| **2** | **2D Multivariate Gaussian Density & Mahalanobis Distance by Hand** | Medium | Evaluating continuous likelihoods in multi-dimensional latent spaces |
| **3** | **The VAE Reparameterization Trick & Gradient Propagation** | Challenging | Training Variational Autoencoders and Diffusion latent spaces via backprop |

---

## Problem 1: Derivation of Bernoulli MLE & BCE Loss

### 🧒 The Context: Maya's Fair Coin & Red Collar Puppies
Maya counts whether $N$ puppies at the kennel are wearing a red collar: $y_1, y_2, \dots, y_N \in \{0, 1\}$.
Each puppy is modeled as an independent draw from a Bernoulli distribution with parameter $p$:
$$P(Y = y_i) = p^{y_i} (1 - p)^{1 - y_i}$$

### Tasks:
1. Write the Likelihood function $L(p) = \prod_{i=1}^N P(Y = y_i \mid p)$.
2. Take the natural logarithm to obtain the log-likelihood $\ell(p) = \ln L(p)$.
3. Differentiate $\ell(p)$ with respect to $p$, set to zero, and derive the Maximum Likelihood Estimator $\hat{p}_{\text{MLE}}$.
4. Prove that minimizing Binary Cross-Entropy loss is mathematically identical to maximizing the Bernoulli log-likelihood.

---

<details>
<summary>Click to view solution for Problem 1</summary>

### Detailed Step-by-Step Solution:

#### Step 1: Likelihood Function
Since observations are independent:
$$L(p) = \prod_{i=1}^N p^{y_i} (1 - p)^{1 - y_i}$$

#### Step 2: Log-Likelihood Function
$$\ell(p) = \ln L(p) = \sum_{i=1}^N \ln\left[ p^{y_i} (1 - p)^{1 - y_i} \right] = \sum_{i=1}^N \left[ y_i \ln(p) + (1 - y_i) \ln(1 - p) \right]$$

#### Step 3: Differentiating w.r.t $p$
$$\frac{d\ell}{dp} = \sum_{i=1}^N \left( \frac{y_i}{p} - \frac{1 - y_i}{1 - p} \right) = 0$$
$$\frac{1}{p} \sum_{i=1}^N y_i = \frac{1}{1 - p} \sum_{i=1}^N (1 - y_i)$$

Let $k = \sum_{i=1}^N y_i$ be the total count of positive puppies. Then $\sum_{i=1}^N (1 - y_i) = N - k$:
$$\frac{k}{p} = \frac{N - k}{1 - p} \implies k(1 - p) = p(N - k)$$
$$k - k p = N p - k p \implies k = N p \implies \mathbf{\hat{p}_{\text{MLE}} = \frac{k}{N} = \frac{1}{N} \sum_{i=1}^N y_i}$$

**🧒 Intuition:** The optimal estimate of $p$ is simply the fraction of positive puppy collars observed!

#### Step 4: Identity with Binary Cross-Entropy (BCE) Loss
In machine learning, we minimize loss rather than maximizing likelihood. We define loss as the negative average log-likelihood:
$$\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \ell(p) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln(p) + (1 - y_i) \ln(1 - p) \right]$$
Minimizing BCE loss is mathematically identical to maximizing the likelihood of Bernoulli distributed observations!
</details>

---

## Problem 2: 2D Multivariate Gaussian Density & Mahalanobis Distance by Hand

### 🧒 The Context: Maya's 2D Puppy Measurements
Maya measures height and weight for puppy data following a 2D Gaussian $\mathcal{N}(\mu, \Sigma)$ with:
$$\mu = \begin{bmatrix} 2.0 \\ 2.0 \end{bmatrix}, \quad \Sigma = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 4.0 \end{bmatrix}$$
We observe a new puppy at point $\mathbf{x} = \begin{bmatrix} 3.0 \\ 4.0 \end{bmatrix}$.

### Tasks:
1. Compute the difference vector $\mathbf{x} - \mu$.
2. Compute the matrix inverse $\Sigma^{-1}$ and determinant $|\Sigma|$.
3. Compute the squared Mahalanobis distance $d_M^2 = (\mathbf{x} - \mu)^T \Sigma^{-1} (\mathbf{x} - \mu)$.
4. Compute the exact probability density $f(\mathbf{x})$.

---

<details>
<summary>Click to view solution for Problem 2</summary>

### Detailed Step-by-Step Solution:

#### Step 1: Difference Vector
$$\mathbf{x} - \mu = \begin{bmatrix} 3.0 \\ 4.0 \end{bmatrix} - \begin{bmatrix} 2.0 \\ 2.0 \end{bmatrix} = \begin{bmatrix} 1.0 \\ 2.0 \end{bmatrix}$$

#### Step 2: Inverse and Determinant
For diagonal matrix $\Sigma = \begin{bmatrix} 1 & 0 \\ 0 & 4 \end{bmatrix}$:
$$\Sigma^{-1} = \begin{bmatrix} 1/1 & 0 \\ 0 & 1/4 \end{bmatrix} = \begin{bmatrix} 1.0 & 0 \\ 0 & 0.25 \end{bmatrix}$$
$$|\Sigma| = (1.0)(4.0) - (0)(0) = \mathbf{4.0} \implies |\Sigma|^{1/2} = \mathbf{2.0}$$

#### Step 3: Squared Mahalanobis Distance
$$d_M^2 = \begin{bmatrix} 1.0 & 2.0 \end{bmatrix} \begin{bmatrix} 1.0 & 0 \\ 0 & 0.25 \end{bmatrix} \begin{bmatrix} 1.0 \\ 2.0 \end{bmatrix} = \begin{bmatrix} 1.0 & 2.0 \end{bmatrix} \begin{bmatrix} 1.0 \\ 0.5 \end{bmatrix}$$
$$d_M^2 = (1.0)(1.0) + (2.0)(0.5) = 1.0 + 1.0 = \mathbf{2.0}$$

#### Step 4: Probability Density Evaluation
Dimension $D = 2$:
$$\text{Normalizer} = \frac{1}{(2\pi)^{2/2} |\Sigma|^{1/2}} = \frac{1}{2\pi \times 2.0} = \frac{1}{4\pi} \approx \mathbf{0.079577}$$
$$f(\mathbf{x}) = 0.079577 \times \exp\left( -\frac{1}{2}(2.0) \right) = 0.079577 \times e^{-1} \approx 0.079577 \times 0.367879 = \mathbf{0.02927}$$
</details>

---

## Problem 3: The VAE Reparameterization Trick & Gradient Propagation

### 🧒 The Context: Maya's Differentiable Randomness
In a Variational Autoencoder, a latent encoder outputs mean $\mu = 1.5$ and standard deviation $\sigma = 0.8$.
The reparameterized latent code is:
$$z = \mu + \sigma \cdot \epsilon \quad \text{where } \epsilon \sim \mathcal{N}(0, 1)$$

Downstream decoder computes reconstruction loss $\mathcal{L}(z) = (z - 3.0)^2$.

### Tasks:
1. For fixed noise sample $\epsilon = 0.5$, compute forward latent value $z$ and loss $\mathcal{L}$.
2. Compute the local derivative $\frac{\partial \mathcal{L}}{\partial z}$.
3. Using the chain rule, compute $\frac{\partial \mathcal{L}}{\partial \mu}$ and $\frac{\partial \mathcal{L}}{\partial \sigma}$.
4. Explain why sampling $z \sim \mathcal{N}(\mu, \sigma^2)$ directly breaks backpropagation without this trick.

---

<details>
<summary>Click to view solution for Problem 3</summary>

### Detailed Step-by-Step Solution:

#### Step 1: Forward Value of $z$ and Loss $\mathcal{L}$
$$z = \mu + \sigma \cdot \epsilon = 1.5 + 0.8(0.5) = 1.5 + 0.4 = \mathbf{1.9}$$
$$\mathcal{L} = (z - 3.0)^2 = (1.9 - 3.0)^2 = (-1.1)^2 = \mathbf{1.21}$$

#### Step 2: Local Derivative $\frac{\partial \mathcal{L}}{\partial z}$
$$\frac{\partial \mathcal{L}}{\partial z} = 2(z - 3.0) = 2(1.9 - 3.0) = 2(-1.1) = \mathbf{-2.2}$$

#### Step 3: Gradients with respect to Encoder Parameters $\mu$ and $\sigma$
Since $z = \mu + \sigma \epsilon$:
$$\frac{\partial z}{\partial \mu} = 1.0, \quad \frac{\partial z}{\partial \sigma} = \epsilon = 0.5$$

Applying the chain rule:
$$\frac{\partial \mathcal{L}}{\partial \mu} = \frac{\partial \mathcal{L}}{\partial z} \frac{\partial z}{\partial \mu} = (-2.2) \times 1.0 = \mathbf{-2.2}$$
$$\frac{\partial \mathcal{L}}{\partial \sigma} = \frac{\partial \mathcal{L}}{\partial z} \frac{\partial z}{\partial \sigma} = (-2.2) \times 0.5 = \mathbf{-1.1}$$

#### Step 4: Why Direct Sampling Breaks Autograd
If we wrote $z \sim \mathcal{N}(\mu, \sigma^2)$, $z$ would be the direct output of a stochastic sampling operator. 
A random number generator has no analytical derivative ($\frac{\partial}{\partial \mu} \text{RandomDraw}$ is undefined), so gradients cannot flow backwards into the encoder network weights!
By reparameterizing as $z = \mu + \sigma \epsilon$, the random number $\epsilon$ is treated as an external constant input, making $z$ fully differentiable w.r.t $\mu$ and $\sigma$.
</details>
