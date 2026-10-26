# Topic 5.1: Exercises & Solutions — Populations, Samples & Estimators

---

## Problem Set Overview

| Problem | Focus | Difficulty |
| :--- | :--- | :--- |
| **1** | **Sample Statistics & The Bessel Ratio on Numerical Data** | Easy |
| **2** | **Standard Error Scaling in LLM Benchmark Evaluation** | Medium |
| **3** | **Optimal Weighting Proof & The Gauss-Markov Theorem** | Challenging |

---

## Problem 1: Sample Statistics & The Bessel Ratio on Numerical Data

### The Context
An AI engineer records the inference latency (in milliseconds) of 4 consecutive requests to an optimized neural model:
$$X = \{ 4.0, \, 7.0, \, 10.0, \, 15.0 \}$$

### Task
1. Calculate the sample mean $\bar{X}$.
2. Compute the deviations from the mean $(X_i - \bar{X})$ and the sum of squared deviations $\sum_{i=1}^4 (X_i - \bar{X})^2$.
3. Compute the **naive sample variance** $S_n^2$ (dividing by $n = 4$).
4. Compute the **unbiased sample variance** $S^2$ (dividing by $n - 1 = 3$).
5. Confirm that $\frac{S_n^2}{S^2} = \frac{n - 1}{n}$.

---

### Solution 1

#### 1. Sample Mean
$$\bar{X} = \frac{4.0 + 7.0 + 10.0 + 15.0}{4} = \frac{36.0}{4} = \mathbf{9.0 \text{ ms}}$$

#### 2. Deviations and Squared Deviations
- For $X_1 = 4.0$: $(4.0 - 9.0) = -5.0 \implies (-5.0)^2 = 25.0$
- For $X_2 = 7.0$: $(7.0 - 9.0) = -2.0 \implies (-2.0)^2 = 4.0$
- For $X_3 = 10.0$: $(10.0 - 9.0) = +1.0 \implies (+1.0)^2 = 1.0$
- For $X_4 = 15.0$: $(15.0 - 9.0) = +6.0 \implies (+6.0)^2 = 36.0$

Sum of squared deviations:
$$\sum_{i=1}^4 (X_i - \bar{X})^2 = 25.0 + 4.0 + 1.0 + 36.0 = \mathbf{66.0}$$

#### 3. Naive Sample Variance ($S_n^2$)
$$S_n^2 = \frac{66.0}{4} = \mathbf{16.5 \text{ ms}^2}$$

#### 4. Unbiased Sample Variance ($S^2$)
$$S^2 = \frac{66.0}{4 - 1} = \frac{66.0}{3} = \mathbf{22.0 \text{ ms}^2}$$

#### 5. Verification of the Ratio
$$\frac{S_n^2}{S^2} = \frac{16.5}{22.0} = 0.75 = \mathbf{\frac{3}{4} = \frac{n - 1}{n}} \quad \blacksquare$$

---

## Problem 2: Standard Error Scaling in LLM Benchmark Evaluation

### The Context
An AI evaluation engineer tests an LLM on a reasoning benchmark (MMLU). Each test prompt is scored binary: $1$ if correct, $0$ if wrong.
Suppose the true model accuracy is $p = 0.50$, so the population variance of a single test item is:
$$\sigma^2 = p(1 - p) = (0.50)(0.50) = 0.25 \implies \sigma = 0.50$$

### Task
1. If the engineer tests the model on $n = 100$ prompts, what is the **Standard Error** $\text{SE}(\bar{X})$ of the accuracy score?
2. If the engineer wants to report the model's benchmark accuracy with a precision margin of uncertainty (Standard Error) of no more than $\pm 1\%$ ($\text{SE} \le 0.01$), what is the minimum number of evaluation prompts $n$ required?
3. What is the fundamental practical implication for leaderboards comparing models that differ by only $0.5\%$ on 500 prompts?

---

### Solution 2

#### 1. Standard Error for $n = 100$
$$\text{SE}(\bar{X}) = \frac{\sigma}{\sqrt{n}} = \frac{0.50}{\sqrt{100}} = \frac{0.50}{10} = \mathbf{0.05} \quad (5.0\%)$$
*Interpretation:* With only 100 questions, the measured accuracy can easily fluctuate by $\pm 5\%$ simply due to sampling noise!

#### 2. Minimum Sample Size for $\text{SE} \le 0.01$
$$\text{SE} = \frac{\sigma}{\sqrt{n}} \le 0.01$$
$$\frac{0.50}{\sqrt{n}} \le 0.01 \implies \sqrt{n} \ge \frac{0.50}{0.01} = 50$$
Square both sides:
$$n \ge 50^2 = \mathbf{2{,}500 \text{ prompts}}$$

#### 3. Practical Implication for AI Leaderboards
If two LLMs differ by only $0.5\%$ on a 500-prompt benchmark:
$$\text{SE} = \frac{0.50}{\sqrt{500}} \approx \frac{0.50}{22.36} \approx \mathbf{0.0224} \quad (2.24\%)$$
The standard error of measurement ($2.24\%$) is **over 4 times larger** than the observed difference ($0.5\%$)!
Therefore, a $0.5\%$ difference on 500 prompts is pure random statistical noise, not proof of superior intelligence.

---

## Problem 3: Optimal Weighting Proof & The Gauss-Markov Theorem

### The Context
Let $X_1, X_2, \dots, X_n$ be independent observations from a distribution with mean $\mu$ and variance $\sigma^2$.
Consider any linear estimator for the mean:
$$\hat{\mu} = \sum_{i=1}^n w_i X_i$$
where $w_1, \dots, w_n$ are fixed real-valued weights.

### Task
1. Find the necessary and sufficient condition on the weights $w_i$ such that $\hat{\mu}$ is an **unbiased estimator** of $\mu$.
2. Write the formula for the variance $\operatorname{Var}(\hat{\mu})$ in terms of $w_i$ and $\sigma^2$.
3. Prove that under the unbiasedness constraint, the variance $\operatorname{Var}(\hat{\mu})$ is strictly minimized when all weights are equal:
   $$w_i = \frac{1}{n} \quad \text{for all } i \in \{1, \dots, n\}$$
   *(Hint: Use the Cauchy-Schwarz inequality or Lagrange multipliers).*

---

### Solution 3

#### 1. Unbiasedness Condition
$$\mathbb{E}[\hat{\mu}] = \mathbb{E}\left[ \sum_{i=1}^n w_i X_i \right] = \sum_{i=1}^n w_i \mathbb{E}[X_i] = \sum_{i=1}^n w_i \mu = \mu \left( \sum_{i=1}^n w_i \right)$$
For $\mathbb{E}[\hat{\mu}] = \mu$ for any value of $\mu$:
$$\mathbf{\sum_{i=1}^n w_i = 1}$$

#### 2. Variance Formula
Because the observations are independent:
$$\operatorname{Var}(\hat{\mu}) = \operatorname{Var}\left( \sum_{i=1}^n w_i X_i \right) = \sum_{i=1}^n w_i^2 \operatorname{Var}(X_i) = \mathbf{\sigma^2 \sum_{i=1}^n w_i^2}$$

#### 3. Minimization of Variance (Optimality Proof)
We want to minimize $\sum_{i=1}^n w_i^2$ subject to $\sum_{i=1}^n w_i = 1$.

**Method A: Cauchy-Schwarz Inequality**
For any vectors $\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$:
$$(\mathbf{u} \cdot \mathbf{v})^2 \le \|\mathbf{u}\|^2 \|\mathbf{v}\|^2$$
Let $\mathbf{u} = [w_1, \dots, w_n]^T$ and $\mathbf{v} = [1, 1, \dots, 1]^T$.
$$\left( \sum_{i=1}^n w_i \cdot 1 \right)^2 \le \left( \sum_{i=1}^n w_i^2 \right) \left( \sum_{i=1}^n 1^2 \right)$$
Since $\sum_{i=1}^n w_i = 1$ and $\sum_{i=1}^n 1 = n$:
$$1^2 \le \left( \sum_{i=1}^n w_i^2 \right) n \implies \sum_{i=1}^n w_i^2 \ge \mathbf{\frac{1}{n}}$$

Equality holds if and only if $\mathbf{u}$ is proportional to $\mathbf{v}$, which means:
$$w_1 = w_2 = \dots = w_n = \mathbf{\frac{1}{n}} \quad \blacksquare$$

*Conclusion:* The simple sample mean $\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$ is mathematically the Best Linear Unbiased Estimator (BLUE), possessing the lowest possible variance among all linear unbiased estimators!

