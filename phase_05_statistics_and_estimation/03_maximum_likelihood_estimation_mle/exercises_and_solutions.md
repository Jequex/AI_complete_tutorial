# Topic 5.3: Exercises & Solutions — Maximum Likelihood Estimation

---

## Problem Set Overview

| Problem | Focus | Difficulty | Real-World Context |
| :---: | :--- | :---: | :--- |
| **1** | **Exponential Distribution MLE** | Easy | Modeling GPU hardware lifetime and failure rates in data centers |
| **2** | **Poisson Distribution MLE** | Medium | Estimating user query arrival rates for LLM cloud endpoints |
| **3** | **Formal Proof: MLE Gaussian Variance is Biased** | Challenging | Uncovering why standard MLE fails to correct for Bessel's penalty |

---

## Problem 1: Exponential Distribution MLE for Server Hardware Lifetime

### The Context
An AI data center monitors GPU failure times (in years). The continuous lifetime $X$ of a GPU follows an **Exponential Distribution**:
$$P(x \mid \lambda) = \lambda e^{-\lambda x}, \quad x \ge 0$$
where $\lambda > 0$ is the failure rate parameter.
Suppose we observe failure times for $n$ independent GPUs: $\{x_1, x_2, \dots, x_n\}$.

### Task
1. Write the joint likelihood function $\mathcal{L}(\lambda; x_1, \dots, x_n)$.
2. Compute the log-likelihood function $\ell(\lambda) = \ln \mathcal{L}(\lambda)$.
3. Compute the first derivative $\frac{d\ell}{d\lambda}$ and set it to zero to find the critical point $\hat{\lambda}$.
4. Compute the second derivative $\frac{d^2\ell}{d\lambda^2}$ to prove that $\hat{\lambda}$ is indeed a local maximum.
5. Express $\hat{\lambda}_{\text{MLE}}$ in terms of the sample mean $\bar{X}$.

---

<details>
<summary>Click to Reveal Step-by-Step Solution</summary>

### Step-by-Step Solution

#### 1. Likelihood Function
Multiply the probabilities of all $n$ independent GPU lifetimes:
$$\mathcal{L}(\lambda) = \prod_{i=1}^n \left( \lambda e^{-\lambda x_i} \right) = \lambda^n \exp\left( -\lambda \sum_{i=1}^n x_i \right)$$

#### 2. Log-Likelihood Function
Take the natural logarithm:
$$\ell(\lambda) = \ln\left( \lambda^n \right) + \ln\left( \exp\left( -\lambda \sum_{i=1}^n x_i \right) \right) = \mathbf{n \ln\lambda - \lambda \sum_{i=1}^n x_i}$$

#### 3. First Derivative & Setting to Zero
$$\frac{d\ell}{d\lambda} = \frac{n}{\lambda} - \sum_{i=1}^n x_i = 0$$
$$\frac{n}{\lambda} = \sum_{i=1}^n x_i \implies \mathbf{\hat{\lambda} = \frac{n}{\sum_{i=1}^n x_i} = \frac{1}{\bar{X}}}$$

#### 4. Second Derivative (Concavity Check)
$$\frac{d^2\ell}{d\lambda^2} = -\frac{n}{\lambda^2}$$
Since $n > 0$ and $\lambda^2 > 0$, $\frac{d^2\ell}{d\lambda^2} < 0$ for all values of $\lambda$!
The curve bends strictly downward like an umbrella, guaranteeing that $\hat{\lambda}$ is the unique global maximum.

#### 5. Final Result & Intuition
$$\mathbf{\hat{\lambda}_{\text{MLE}} = \frac{1}{\bar{X}}}$$

> **🧒 12-Year-Old Takeaway:**
> If GPUs in your cluster last an average of $\bar{X} = 4\text{ years}$, the most likely failure rate is $\hat{\lambda} = \frac{1}{4} = 0.25\text{ failures per year}$. It matches pure common sense!

</details>

---

## Problem 2: Poisson Distribution MLE for AI API Request Traffic

### The Context
An AI company tracks the number of inference requests arriving per minute at an LLM API endpoint.
The count $K$ follows a **Poisson distribution**:
$$P(k \mid \lambda) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k \in \{0, 1, 2, \dots\}$$
Over $n = 5$ separate minutes, the recorded query counts are:
$$D = \{ 5, \, 8, \, 3, \, 4, \, 10 \}$$

### Task
1. Write the general log-likelihood function $\ell(\lambda)$ for $n$ observations $\{k_1, \dots, k_n\}$.
2. Derive the analytical formula for $\hat{\lambda}_{\text{MLE}}$.
3. Compute the numerical estimate $\hat{\lambda}_{\text{MLE}}$ for the observed dataset $D$.

---

<details>
<summary>Click to Reveal Step-by-Step Solution</summary>

### Step-by-Step Solution

#### 1. Log-Likelihood Function
Write the joint likelihood product:
$$\mathcal{L}(\lambda) = \prod_{i=1}^n \frac{\lambda^{k_i} e^{-\lambda}}{k_i!} = \frac{\lambda^{\sum k_i} e^{-n\lambda}}{\prod_{i=1}^n k_i!}$$

Take the natural logarithm:
$$\ell(\lambda) = \ln\left( \lambda^{\sum k_i} \right) + \ln\left( e^{-n\lambda} \right) - \sum_{i=1}^n \ln(k_i!)$$
$$\mathbf{\ell(\lambda) = \left( \sum_{i=1}^n k_i \right) \ln\lambda - n\lambda - \sum_{i=1}^n \ln(k_i!)}$$

#### 2. Deriving $\hat{\lambda}_{\text{MLE}}$
Differentiate with respect to $\lambda$:
$$\frac{d\ell}{d\lambda} = \frac{\sum_{i=1}^n k_i}{\lambda} - n = 0$$
$$\frac{\sum_{i=1}^n k_i}{\lambda} = n \implies \mathbf{\hat{\lambda}_{\text{MLE}} = \frac{1}{n} \sum_{i=1}^n k_i = \bar{K}}$$

#### 3. Numerical Calculation for Dataset $D$
For $D = \{5, 8, 3, 4, 10\}$:
$$\hat{\lambda}_{\text{MLE}} = \frac{5 + 8 + 3 + 4 + 10}{5} = \frac{30}{5} = \mathbf{6.0\text{ requests / minute}}$$

</details>

---

## Problem 3: Formal Proof: The Gaussian MLE Variance is Systematically Biased

### The Context
Let $X_1, \dots, X_n \sim \mathcal{N}(\mu, \sigma^2)$ be i.i.d. Gaussian random variables where both $\mu$ and $\sigma^2$ are unknown.
Recall the Gaussian log-likelihood from Lecture 5.3:
$$\ell(\mu, \sigma^2) = -\frac{n}{2}\ln(2\pi) - \frac{n}{2}\ln(\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^n (X_i - \mu)^2$$

### Task
1. Differentiate $\ell$ with respect to $\mu$ to prove that $\hat{\mu}_{\text{MLE}} = \bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$.
2. Treating $\sigma^2$ as a single variable $v = \sigma^2$, differentiate $\ell$ with respect to $v$, set to zero, and derive $\hat{\sigma}^2_{\text{MLE}}$.
3. Compute the mathematical expectation $\mathbb{E}[\hat{\sigma}^2_{\text{MLE}}]$.
4. Is the Maximum Likelihood Estimator for variance unbiased? What happens as $n \to \infty$?

---

<details>
<summary>Click to Reveal Step-by-Step Solution</summary>

### Step-by-Step Solution

#### 1. Derivation of $\hat{\mu}_{\text{MLE}}$
$$\frac{\partial \ell}{\partial \mu} = -\frac{1}{2\sigma^2} \sum_{i=1}^n 2(X_i - \mu)(-1) = \frac{1}{\sigma^2}\sum_{i=1}^n (X_i - \mu) = 0$$
$$\sum_{i=1}^n X_i - n\mu = 0 \implies \mathbf{\hat{\mu}_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n X_i = \bar{X}} \quad \blacksquare$$

#### 2. Derivation of $\hat{\sigma}^2_{\text{MLE}}$
Let $v = \sigma^2$:
$$\ell(\mu, v) = -\frac{n}{2}\ln(2\pi) - \frac{n}{2}\ln(v) - \frac{1}{2v} \sum_{i=1}^n (X_i - \bar{X})^2$$
Differentiate with respect to $v$:
$$\frac{\partial \ell}{\partial v} = -\frac{n}{2v} + \frac{1}{2v^2} \sum_{i=1}^n (X_i - \bar{X})^2 = 0$$
Multiply both sides by $2v^2$:
$$-nv + \sum_{i=1}^n (X_i - \bar{X})^2 = 0 \implies nv = \sum_{i=1}^n (X_i - \bar{X})^2$$
$$\mathbf{\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^2} \quad \blacksquare$$

#### 3. Mathematical Expectation of $\hat{\sigma}^2_{\text{MLE}}$
From our derivation of Bessel's correction in Topic 5.1:
$$\mathbb{E}\left[ \sum_{i=1}^n (X_i - \bar{X})^2 \right] = (n - 1)\sigma^2$$
Therefore:
$$\mathbb{E}[\hat{\sigma}^2_{\text{MLE}}] = \mathbb{E}\left[ \frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^2 \right] = \frac{1}{n} (n - 1)\sigma^2 = \mathbf{\frac{n - 1}{n} \sigma^2}$$

#### 4. Conclusion on MLE Bias
$$\operatorname{Bias}(\hat{\sigma}^2_{\text{MLE}}) = \frac{n - 1}{n}\sigma^2 - \sigma^2 = -\frac{\sigma^2}{n} \ne 0$$
- **Verdict:** The Maximum Likelihood Estimator for variance is **systematically biased downwards**! MLE does not automatically know about Bessel's correction—it divides by $n$ because it fits the sample mean $\bar{X}$ perfectly.
- **Asymptotic Behavior:** As $n \to \infty$, $\frac{n-1}{n} \to 1$. The bias completely vanishes on big datasets, proving that MLE is **asymptotically unbiased**!

</details>
