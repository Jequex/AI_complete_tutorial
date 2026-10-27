# Topic 5.1: Exercises & Solutions — Populations, Samples & Estimators

---

## Problem Set Overview

| Problem | Focus | Difficulty | Real-World Context |
| :---: | :--- | :---: | :--- |
| **1** | **Sample Statistics & The Bessel Ratio** | Easy | AI inference latency monitoring on edge devices |
| **2** | **Standard Error Scaling & The $\sqrt{n}$ Law** | Medium | Evaluating LLM accuracy on the MMLU reasoning benchmark |
| **3** | **Optimal Weighting Proof & Gauss-Markov Theorem** | Challenging | Proving why equal weighting is optimal in distributed AI training |

---

## Problem 1: Sample Statistics & The Bessel Ratio on Numerical Data

### The Context
An AI engineer records the inference latency (in milliseconds) of 4 consecutive requests to an optimized neural network on a smartphone:
$$X = \{ 4.0\text{ ms}, \, 7.0\text{ ms}, \, 10.0\text{ ms}, \, 15.0\text{ ms} \}$$

### Task
1. Calculate the sample mean $\bar{X}$.
2. Compute each individual deviation from the mean $(X_i - \bar{X})$ and the sum of squared deviations $\sum_{i=1}^4 (X_i - \bar{X})^2$.
3. Compute the **naive sample variance** $S_n^2$ (dividing by $n = 4$).
4. Compute the **unbiased Bessel-corrected sample variance** $S^2$ (dividing by $n - 1 = 3$).
5. Confirm that the ratio $\frac{S_n^2}{S^2}$ equals exactly $\frac{n - 1}{n}$.

---

<details>
<summary>Click to Reveal Step-by-Step Solution</summary>

### Step-by-Step Solution

#### 1. Sample Mean ($\bar{X}$)
$$\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i = \frac{4.0 + 7.0 + 10.0 + 15.0}{4} = \frac{36.0}{4} = \mathbf{9.0\text{ ms}}$$

#### 2. Deviations and Squared Deviations
Calculate how far each measurement sits from the sample center $\bar{X} = 9.0$:
- Request 1 ($X_1 = 4.0$): $(4.0 - 9.0) = -5.0 \implies (-5.0)^2 = 25.0$
- Request 2 ($X_2 = 7.0$): $(7.0 - 9.0) = -2.0 \implies (-2.0)^2 = 4.0$
- Request 3 ($X_3 = 10.0$): $(10.0 - 9.0) = +1.0 \implies (+1.0)^2 = 1.0$
- Request 4 ($X_4 = 15.0$): $(15.0 - 9.0) = +6.0 \implies (+6.0)^2 = 36.0$

Summing the squared deviations:
$$\sum_{i=1}^4 (X_i - \bar{X})^2 = 25.0 + 4.0 + 1.0 + 36.0 = \mathbf{66.0\text{ ms}^2}$$

#### 3. Naive Sample Variance ($S_n^2$)
Dividing by sample size $n = 4$:
$$S_n^2 = \frac{66.0}{4} = \mathbf{16.5\text{ ms}^2}$$

#### 4. Unbiased Bessel-Corrected Sample Variance ($S^2$)
Dividing by degrees of freedom $n - 1 = 3$:
$$S^2 = \frac{66.0}{4 - 1} = \frac{66.0}{3} = \mathbf{22.0\text{ ms}^2}$$

#### 5. Verification of the Ratio
$$\frac{S_n^2}{S^2} = \frac{16.5}{22.0} = 0.75 = \mathbf{\frac{3}{4} = \frac{n - 1}{n}} \quad \blacksquare$$

> **🧒 12-Year-Old Takeaway:**
> Notice that the naive variance ($16.5$) was $25\%$ smaller than the unbiased variance ($22.0$)! Because our sample points clustered around their own middle point, dividing by $n$ cheated and underestimated the true variation. Dividing by $n-1 = 3$ fixes the cheat.

</details>

---

## Problem 2: Standard Error Scaling in LLM Benchmark Evaluation

### The Context
An AI evaluation engineer tests a new open-weights LLM on the MMLU reasoning benchmark. Each test prompt is scored binary: $1$ if the model answers correctly, and $0$ if it answers incorrectly.
Suppose the model's true population accuracy is $p = 0.50$, so the population variance of a single test item is:
$$\sigma^2 = p(1 - p) = (0.50)(0.50) = 0.25 \implies \sigma = \sqrt{0.25} = 0.50$$

### Task
1. If the engineer tests the model on $n = 100$ prompts, what is the **Standard Error** $\text{SE}(\bar{X})$ of the accuracy score?
2. If the engineer wants to report the benchmark accuracy with a precision margin of uncertainty (Standard Error) of no more than $\pm 1\%$ ($\text{SE} \le 0.01$), what is the minimum number of evaluation prompts $n$ required?
3. What is the fundamental practical implication for public leaderboards that compare models differing by only $0.5\%$ on 500 prompts?

---

<details>
<summary>Click to Reveal Step-by-Step Solution</summary>

### Step-by-Step Solution

#### 1. Standard Error for $n = 100$ Prompts
$$\text{SE}(\bar{X}) = \frac{\sigma}{\sqrt{n}} = \frac{0.50}{\sqrt{100}} = \frac{0.50}{10} = \mathbf{0.05 \quad (5.0\%)}$$

*Interpretation:* With only 100 test questions, the measured score will bounce up and down by $\pm 5\%$ simply due to the luck of the draw! A 50% model might easily score 45% or 55%.

#### 2. Minimum Sample Size for $\text{SE} \le 0.01$ ($\pm 1\%$)
Set up the inequality:
$$\text{SE}(\bar{X}) = \frac{\sigma}{\sqrt{n}} \le 0.01$$
$$\frac{0.50}{\sqrt{n}} \le 0.01 \implies \sqrt{n} \ge \frac{0.50}{0.01} = 50$$
Square both sides:
$$n \ge 50^2 = \mathbf{2{,}500\text{ prompts}}$$

*Interpretation:* To reduce uncertainty from $\pm 5\%$ down to $\pm 1\%$ ($5\times$ smaller wiggle), you need $5^2 = 25\times$ more data: $100 \times 25 = 2{,}500$ prompts!

#### 3. Practical Implication for AI Leaderboards
On a benchmark with $n = 500$ prompts, the standard error is:
$$\text{SE} = \frac{0.50}{\sqrt{500}} \approx \frac{0.50}{22.36} \approx \mathbf{0.0224 \quad (2.24\%)}$$

The measurement noise ($2.24\%$) is **more than 4 times larger** than the observed difference ($0.50\%$)!
Therefore, bragging that Model B scored $72.3\%$ while Model A scored $71.8\%$ on 500 prompts is statistically meaningless—the difference is pure random noise, not true superiority.

</details>

---

## Problem 3: Optimal Weighting Proof & The Gauss-Markov Theorem

### The Context
Suppose we collect $n$ independent sensor measurements $X_1, X_2, \dots, X_n$ from a fleet of robots, each with mean $\mu$ and variance $\sigma^2$.
Instead of a simple average, an engineer considers a general linear weighted estimator:
$$\hat{\mu} = \sum_{i=1}^n w_i X_i = w_1 X_1 + w_2 X_2 + \dots + w_n X_n$$
where $w_1, \dots, w_n$ are fixed real numbers.

### Task
1. Find the condition on the weights $w_i$ such that $\hat{\mu}$ is an **unbiased estimator** of $\mu$ ($\mathbb{E}[\hat{\mu}] = \mu$).
2. Write the formula for the variance $\operatorname{Var}(\hat{\mu})$ in terms of $w_i$ and $\sigma^2$.
3. Prove that under the unbiasedness condition, the variance $\operatorname{Var}(\hat{\mu})$ is strictly minimized when all weights are equal:
   $$w_i = \frac{1}{n} \quad \text{for all } i \in \{1, \dots, n\}$$
   *(Hint: Use the Cauchy-Schwarz inequality).*

---

<details>
<summary>Click to Reveal Step-by-Step Solution</summary>

### Step-by-Step Solution

#### 1. Unbiasedness Condition
Take the expectation of the weighted estimator:
$$\mathbb{E}[\hat{\mu}] = \mathbb{E}\left[ \sum_{i=1}^n w_i X_i \right] = \sum_{i=1}^n w_i \mathbb{E}[X_i] = \sum_{i=1}^n w_i \mu = \mu \left( \sum_{i=1}^n w_i \right)$$

For $\mathbb{E}[\hat{\mu}] = \mu$ to hold for any true mean $\mu$:
$$\mathbf{\sum_{i=1}^n w_i = 1}$$
The weights must sum to exactly $1.0$!

#### 2. Variance of the Linear Combination
Because the observations $X_1, \dots, X_n$ are statistically independent, the variance of the sum is the sum of the variances:
$$\operatorname{Var}(\hat{\mu}) = \operatorname{Var}\left( \sum_{i=1}^n w_i X_i \right) = \sum_{i=1}^n w_i^2 \operatorname{Var}(X_i) = \mathbf{\sigma^2 \sum_{i=1}^n w_i^2}$$

#### 3. Proving That Equal Weights Minimize Variance
We want to minimize the sum of squared weights $\sum_{i=1}^n w_i^2$ subject to $\sum_{i=1}^n w_i = 1$.

**Using the Cauchy-Schwarz Inequality:**
For any two vectors $\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$:
$$(\mathbf{u} \cdot \mathbf{v})^2 \le \|\mathbf{u}\|^2 \|\mathbf{v}\|^2$$

Let vector $\mathbf{u} = [w_1, w_2, \dots, w_n]^T$ and vector $\mathbf{v} = [1, 1, \dots, 1]^T$:
$$\left( \sum_{i=1}^n w_i \cdot 1 \right)^2 \le \left( \sum_{i=1}^n w_i^2 \right) \left( \sum_{i=1}^n 1^2 \right)$$

Substitute $\sum_{i=1}^n w_i = 1$ and $\sum_{i=1}^n 1^2 = n$:
$$1^2 \le \left( \sum_{i=1}^n w_i^2 \right) \cdot n \implies \mathbf{\sum_{i=1}^n w_i^2 \ge \frac{1}{n}}$$

Equality holds if and only if vector $\mathbf{u}$ is a constant multiple of vector $\mathbf{v}$:
$$w_1 = w_2 = \dots = w_n = \mathbf{\frac{1}{n}} \quad \blacksquare$$

> **🌟 The Gauss-Markov Milestone:**
> This famous proof establishes that the ordinary arithmetic mean $\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$ is the **Best Linear Unbiased Estimator (BLUE)**. No other weighting scheme can ever beat equal weighting in stability!

</details>
