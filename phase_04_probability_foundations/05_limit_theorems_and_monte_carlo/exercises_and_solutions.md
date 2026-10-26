# Topic 4.5: Exercises & Solutions — Limit Theorems, Monte Carlo & LLM Sampling

---

## Problem Set Overview

| Problem | Focus | Difficulty |
| :--- | :--- | :--- |
| **1** | **CLT & Benchmark Sample Sizing ($\pm 1\%$ Margin of Error)** | Medium |
| **2** | **Monte Carlo Integration for the Gaussian Error Function** | Medium |
| **3** | **LLM Decoding Arithmetic: Temperature, Top-K & Top-P by Hand** | Challenging |

---

## Problem 1: CLT & Benchmark Sample Sizing

### The Context
An AI lab evaluates a new reasoning model on a multiple-choice benchmark.
Each question outcome is modeled as a Bernoulli random variable:
- $X_i = 1$ if answered correctly, $X_i = 0$ if incorrect.
- True accuracy is $p \approx 0.80$.

By the Central Limit Theorem, the sample accuracy $\hat{p} = \frac{1}{n} \sum_{i=1}^n X_i$ is approximately normal:
$$\hat{p} \sim \mathcal{N}\left( p, \frac{p(1 - p)}{n} \right)$$

### Task
1. What is the standard error of the sample accuracy $\text{SE}(\hat{p})$ in terms of $p$ and $n$?
2. For a $95\%$ confidence interval, the margin of error is $E \approx 1.96 \times \text{SE}$.
   If the lab wants to guarantee that the benchmark report is accurate within a margin of $\pm 1.0\%$ ($E \le 0.01$) with $95\%$ confidence, **how many benchmark questions $n$ must the test set contain?**
3. If $p$ was completely unknown, what is the worst-case variance $p(1 - p)$, and what would the required sample size be?

---

### Solution 1

#### 1. Standard Error
The variance of a single Bernoulli trial is $\sigma^2 = p(1 - p)$.
By the Central Limit Theorem, the variance of the sample average $\hat{p}$ is $\frac{\sigma^2}{n}$.
$$\text{SE}(\hat{p}) = \sqrt{\frac{p(1 - p)}{n}}$$

#### 2. Sample Size for $p = 0.80$
Setting margin of error $E = 1.96 \times \sqrt{\frac{p(1 - p)}{n}} \le 0.01$:
$$\sqrt{n} \ge \frac{1.96 \times \sqrt{p(1 - p)}}{0.01}$$
$$n \ge \frac{1.96^2 \times p(1 - p)}{0.01^2} = \frac{3.8416 \times (0.80 \times 0.20)}{0.0001}$$
$$n \ge \frac{3.8416 \times 0.16}{0.0001} = \frac{0.614656}{0.0001} \approx \mathbf{6{,}147 \text{ questions}}$$
*Result: The benchmark test suite must contain at least 6,147 questions!*

#### 3. Worst-Case Sample Size (Unknown $p$)
The product $p(1 - p)$ reaches its maximum when $p = 0.50$:
$$\max p(1 - p) = 0.50 \times 0.50 = 0.25$$
$$n_{\text{worst}} \ge \frac{3.8416 \times 0.25}{0.0001} = \frac{0.9604}{0.0001} = \mathbf{9{,}604 \text{ questions}}$$
*Takeaway for AI Benchmarking:* Evaluating models on tiny test sets (e.g. 100 questions) yields massive confidence intervals of $\pm 10\%$, rendering minor benchmark gains statistically meaningless!

---

## Problem 2: Monte Carlo Integration for the Gaussian Error Function

### The Context
The Gaussian bell curve $e^{-x^2}$ has no elementary antiderivative. To evaluate:
$$I = \int_0^1 e^{-x^2} \, dx$$
we use Monte Carlo integration.

1. Frame $I$ as an expected value $\mathbb{E}_{X \sim \text{Uniform}(0, 1)}[g(X)]$.
2. State the Monte Carlo estimator $\hat{I}_N$ using $N$ uniform draws $u_i \sim \text{Uniform}(0, 1)$.
3. If $N = 4$ random draws are $u = [0.2, 0.5, 0.8, 0.1]$, compute the estimate $\hat{I}_4$ by hand.
4. Compare with the true value $I \approx 0.746824$.

---

### Solution 2

#### 1. Expectation Formulation
Since the uniform density on $[0, 1]$ is $f_X(x) = 1$:
$$I = \int_0^1 e^{-x^2} \times 1 \, dx = \mathbb{E}_{X \sim \text{Uniform}(0, 1)} \left[ e^{-X^2} \right]$$
Here, $g(x) = e^{-x^2}$.

#### 2. Monte Carlo Estimator
$$\hat{I}_N = \frac{1}{N} \sum_{i=1}^N e^{-u_i^2} \quad \text{where } u_i \sim \text{Uniform}(0, 1)$$

#### 3. Hand Calculation for $N = 4$:
- $u_1 = 0.2 \implies e^{-(0.2)^2} = e^{-0.04} \approx 0.960789$
- $u_2 = 0.5 \implies e^{-(0.5)^2} = e^{-0.25} \approx 0.778801$
- $u_3 = 0.8 \implies e^{-(0.8)^2} = e^{-0.64} \approx 0.527292$
- $u_4 = 0.1 \implies e^{-(0.1)^2} = e^{-0.01} \approx 0.990050$

$$\hat{I}_4 = \frac{0.960789 + 0.778801 + 0.527292 + 0.990050}{4} = \frac{3.256932}{4} \approx \mathbf{0.81423}$$

#### 4. Comparison
True value is $0.7468$. Even with only 4 random draws, the estimate is within $0.067$ of truth. With $N = 100{,}000$, Monte Carlo matches $0.7468$ to 4 decimal places!

---

## Problem 3: LLM Decoding by Hand (Temperature, Top-K & Top-P)

### The Context
A language model outputs 4 candidate tokens with unnormalized logits:
- $z_1 = 3.0$ ("apple")
- $z_2 = 2.0$ ("banana")
- $z_3 = 1.0$ ("carrot")
- $z_4 = 0.0$ ("donut")

### Part A: Standard Softmax ($T = 1.0$)
1. Compute the unnormalized exponentials $e^{z_i}$.
2. Compute the probabilities $p_1, p_2, p_3, p_4$.

### Part B: High Temperature ($T = 2.0$)
1. Scale logits: $z_i / 2.0 = [1.5, 1.0, 0.5, 0.0]$.
2. Compute the new probabilities $p_1, p_2, p_3, p_4$. How did the distribution change?

### Part C: Top-K ($K = 2$, $T = 1.0$)
Apply Top-2 truncation and compute final probabilities.

### Part D: Top-P (Nucleus $P = 0.85$, $T = 1.0$)
Find the nucleus subset of tokens whose cumulative sum $\ge 0.85$. Compute the re-normalized probabilities.

---

### Solution 3

#### Part A: Standard Softmax ($T = 1.0$)
- $e^3 \approx 20.0855$
- $e^2 \approx 7.3891$
- $e^1 \approx 2.7183$
- $e^0 = 1.0000$
- Sum: $20.0855 + 7.3891 + 2.7183 + 1.0000 = 31.1929$

$$p_1 = \frac{20.0855}{31.1929} \approx \mathbf{64.39\%} \quad (\text{apple})$$
$$p_2 = \frac{7.3891}{31.1929} \approx \mathbf{23.69\%} \quad (\text{banana})$$
$$p_3 = \frac{2.7183}{31.1929} \approx \mathbf{8.71\%} \quad (\text{carrot})$$
$$p_4 = \frac{1.0000}{31.1929} \approx \mathbf{3.21\%} \quad (\text{donut})$$

---

#### Part B: High Temperature ($T = 2.0$)
- Scaled logits: $[1.5, 1.0, 0.5, 0.0]$
- $e^{1.5} \approx 4.4817$
- $e^{1.0} \approx 2.7183$
- $e^{0.5} \approx 1.6487$
- $e^0 = 1.0000$
- Sum: $4.4817 + 2.7183 + 1.6487 + 1.0000 = 9.8487$

$$p_1 = \frac{4.4817}{9.8487} \approx \mathbf{45.51\%}$$
$$p_2 = \frac{2.7183}{9.8487} \approx \mathbf{27.60\%}$$
$$p_3 = \frac{1.6487}{9.8487} \approx \mathbf{16.74\%}$$
$$p_4 = \frac{1.0000}{9.8487} \approx \mathbf{10.15\%}$$
*Observation: Increasing $T$ flattens the distribution: top token "apple" dropped from 64.4% down to 45.5%, while rare token "donut" tripled from 3.2% up to 10.2%!*

---

#### Part C: Top-K ($K = 2$, $T = 1.0$)
Keep only top 2: "apple" ($e^3 \approx 20.0855$) and "banana" ($e^2 \approx 7.3891$).
Mask "carrot" and "donut" to $0$.
Sum of kept terms: $20.0855 + 7.3891 = 27.4746$.
$$p_{\text{apple}} = \frac{20.0855}{27.4746} \approx \mathbf{73.11\%}$$
$$p_{\text{banana}} = \frac{7.3891}{27.4746} \approx \mathbf{26.89\%}$$
$$p_{\text{carrot}} = 0\%, \quad p_{\text{donut}} = 0\%$$

---

#### Part D: Top-P ($P = 0.85$, $T = 1.0$)
Sorted standard probabilities:
1. "apple": $0.6439$ (Cumulative: $0.6439 < 0.85$)
2. "banana": $0.2369$ (Cumulative: $0.6439 + 0.2369 = 0.8808 \ge 0.85 \implies \text{STOP!}$)

The nucleus candidate set is strictly **{"apple", "banana"}**.
Tokens "carrot" and "donut" are pruned!
Re-normalizing over {"apple", "banana"}:
$$p_{\text{apple}} = \frac{0.6439}{0.8808} \approx \mathbf{73.10\%}$$
$$p_{\text{banana}} = \frac{0.2369}{0.8808} \approx \mathbf{26.90\%}$$
*(Nucleus sampling dynamically excluded the lower 12% probability tail without needing an arbitrary fixed K!)*

