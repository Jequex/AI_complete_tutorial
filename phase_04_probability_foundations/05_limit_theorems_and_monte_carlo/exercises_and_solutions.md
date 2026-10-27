# Topic 4.5: Exercises & Solutions — Limit Theorems, Monte Carlo & LLM Sampling

Welcome to the problem set for **Topic 4.5: Limit Theorems, Monte Carlo Methods & LLM Sampling**.

Grab a pencil and notebook. Work through each problem before revealing the collapsible solutions below.

---

## 📝 Problem Set Overview

| Problem | Focus | Difficulty | Practical AI Connection |
| :---: | :--- | :---: | :--- |
| **1** | **CLT & Benchmark Sample Sizing ($\pm 1\%$ Margin of Error)** | Medium | Designing statistically sound AI evaluation leaderboards |
| **2** | **Monte Carlo Integration for the Gaussian Error Function** | Medium | Estimating intractable integrals in Diffusion and Physics-Informed ML |
| **3** | **LLM Decoding Arithmetic: Temperature, Top-K & Top-P by Hand** | Challenging | The exact inference generation loop in vLLM, HuggingFace, and Ollama |

---

## Problem 1: CLT & Benchmark Sample Sizing

### 🧒 The Context: Maya's AI Benchmark Leaderboard
An AI research lab evaluates a new reasoning model on a multiple-choice benchmark.
Each question outcome is modeled as a Bernoulli random variable:
- $X_i = 1$ if answered correctly, $X_i = 0$ if incorrect.
- True model accuracy is $p \approx 0.80$.

By the Central Limit Theorem, the sample accuracy $\hat{p} = \frac{1}{n} \sum_{i=1}^n X_i$ is approximately normal:
$$\hat{p} \sim \mathcal{N}\left( p, \frac{p(1 - p)}{n} \right)$$

### Tasks:
1. Write the standard error of the sample accuracy $\text{SE}(\hat{p})$.
2. For a $95\%$ confidence interval, the margin of error is $E \approx 1.96 \times \text{SE}$.
   If Maya wants to guarantee that the reported benchmark accuracy is within $\pm 1.0\%$ ($E \le 0.01$) with $95\%$ confidence, **how many benchmark questions $n$ must the test set contain?**
3. If accuracy $p$ was completely unknown, what is the worst-case variance $p(1 - p)$, and what would the required sample size be?

---

<details>
<summary>Click to view solution for Problem 1</summary>

### Detailed Step-by-Step Solution:

#### Step 1: Standard Error Formula
The variance of a single Bernoulli trial is $\sigma^2 = p(1 - p)$.
By the Central Limit Theorem, the variance of the sample average $\hat{p}$ is $\frac{\sigma^2}{n}$.
$$\text{SE}(\hat{p}) = \sqrt{\frac{p(1 - p)}{n}}$$

#### Step 2: Sample Size for $p = 0.80$
Setting margin of error $E = 1.96 \times \sqrt{\frac{p(1 - p)}{n}} \le 0.01$:
$$\sqrt{n} \ge \frac{1.96 \times \sqrt{p(1 - p)}}{0.01}$$
$$n \ge \frac{1.96^2 \times p(1 - p)}{0.01^2} = \frac{3.8416 \times (0.80 \times 0.20)}{0.0001}$$
$$n \ge \frac{3.8416 \times 0.16}{0.0001} = \frac{0.614656}{0.0001} \approx \mathbf{6{,}147 \text{ questions}}$$
Maya's benchmark test suite must contain at least **6,147 questions**!

#### Step 3: Worst-Case Sample Size (Unknown $p$)
The variance $p(1 - p)$ reaches its maximum when $p = 0.50$:
$$\max p(1 - p) = 0.50 \times 0.50 = 0.25$$
$$n_{\text{worst}} \ge \frac{3.8416 \times 0.25}{0.0001} = \frac{0.9604}{0.0001} = \mathbf{9{,}604 \text{ questions}}$$

**AI Engineering Insight:** Evaluating models on tiny test sets (e.g. 100 questions) yields massive confidence intervals of $\pm 10\%$, rendering small benchmark gains statistically meaningless!
</details>

---

## Problem 2: Monte Carlo Integration for the Gaussian Error Function

### 🧒 The Context: Maya's Popcorn Integral
The Gaussian curve $e^{-x^2}$ has no elementary antiderivative. To evaluate:
$$I = \int_0^1 e^{-x^2} \, dx$$
Maya draws $N = 4$ independent uniform samples from $[0, 1]$:
$$x_1 = 0.2, \quad x_2 = 0.5, \quad x_3 = 0.7, \quad x_4 = 0.9$$

### Tasks:
1. State the Monte Carlo estimator formula for $I$.
2. Compute $g(x_i) = e^{-x_i^2}$ for each of the 4 samples.
3. Compute the Monte Carlo estimate $\hat{I}_4$.
4. Compare your estimate to the true numerical value $I \approx 0.746824$.

---

<details>
<summary>Click to view solution for Problem 2</summary>

### Detailed Step-by-Step Solution:

#### Step 1: Monte Carlo Estimator Formula
For uniform random variable $X \sim \text{Uniform}(0, 1)$, density is $f(x) = 1$. Therefore:
$$\hat{I}_N = \frac{1}{N} \sum_{i=1}^N g(x_i) = \frac{1}{N} \sum_{i=1}^N e^{-x_i^2}$$

#### Step 2: Compute Sample Values
- $g(x_1) = e^{-(0.2)^2} = e^{-0.04} \approx \mathbf{0.960789}$
- $g(x_2) = e^{-(0.5)^2} = e^{-0.25} \approx \mathbf{0.778801}$
- $g(x_3) = e^{-(0.7)^2} = e^{-0.49} \approx \mathbf{0.612626}$
- $g(x_4) = e^{-(0.9)^2} = e^{-0.81} \approx \mathbf{0.444858}$

#### Step 3: Compute Sample Average
$$\hat{I}_4 = \frac{0.960789 + 0.778801 + 0.612626 + 0.444858}{4} = \frac{2.797074}{4} = \mathbf{0.699269}$$

#### Step 4: Comparison
- True Value: $0.746824$
- 4-Sample Monte Carlo Estimate: $0.699269$
- Absolute Error: $|0.699269 - 0.746824| \approx 0.0475$ (Under $6\%$ error with just 4 points!).
With $N = 100{,}000$ points, Monte Carlo matches the true integral to 3 decimal places!
</details>

---

## Problem 3: LLM Decoding Arithmetic: Temperature, Top-K & Top-P by Hand

### 🧒 The Context: Timmy's LLM Vocabulary Selector
A language model outputs raw logits for 5 candidate words:
$$\mathbf{z} = \begin{bmatrix} \text{"apple"} \\ \text{"banana"} \\ \text{"cherry"} \\ \text{"date"} \\ \text{"eggplant"} \end{bmatrix} = \begin{bmatrix} 4.0 \\ 3.0 \\ 2.0 \\ 1.0 \\ 0.0 \end{bmatrix}$$

### Tasks:
1. Compute the baseline Softmax probabilities at Temperature $T = 1.0$.
2. Compute the scaled probabilities at Low Temperature $T = 0.5$. Show how the top word "apple" increases in dominance.
3. Starting from the baseline probabilities ($T = 1.0$), apply **Top-$k = 2$ Truncation**. What are the normalized probabilities?
4. Starting from the baseline probabilities ($T = 1.0$), apply **Top-$p = 0.85$ (Nucleus) Sampling**. Which words survive, and what are their normalized probabilities?

---

<details>
<summary>Click to view solution for Problem 3</summary>

### Detailed Step-by-Step Solution:

#### Step 1: Baseline Softmax at $T = 1.0$
- $e^4 \approx 54.598$
- $e^3 \approx 20.086$
- $e^2 \approx 7.389$
- $e^1 \approx 2.718$
- $e^0 = 1.000$
$$\text{Sum} = 54.598 + 20.086 + 7.389 + 2.718 + 1.000 = \mathbf{85.791}$$

Probabilities:
- $p(\text{"apple"}) = 54.598 / 85.791 \approx \mathbf{63.64\%}$
- $p(\text{"banana"}) = 20.086 / 85.791 \approx \mathbf{23.41\%}$
- $p(\text{"cherry"}) = 7.389 / 85.791 \approx \mathbf{8.61\%}$
- $p(\text{"date"}) = 2.718 / 85.791 \approx \mathbf{3.17\%}$
- $p(\text{"eggplant"}) = 1.000 / 85.791 \approx \mathbf{1.17\%}$

#### Step 2: Low Temperature $T = 0.5$
Divide logits by $0.5$ (multiply by 2): $[8, 6, 4, 2, 0]$.
- $e^8 \approx 2980.96$
- $e^6 \approx 403.43$
- $e^4 \approx 54.60$
- $e^2 \approx 7.39$
- $e^0 = 1.00$
$$\text{Sum} = 2980.96 + 403.43 + 54.60 + 7.39 + 1.00 = \mathbf{3447.38}$$

Probabilities:
- $p(\text{"apple"}) = 2980.96 / 3447.38 \approx \mathbf{86.47\%}$!
- $p(\text{"banana"}) = 403.43 / 3447.38 \approx \mathbf{11.70\%}$
- $p(\text{"cherry"}) \approx 1.58\%$, &nbsp; $p(\text{"date"}) \approx 0.21\%$, &nbsp; $p(\text{"eggplant"}) \approx 0.03\%$.
"Apple" jumped from $63.6\%$ to $86.5\%$!

#### Step 3: Top-$k = 2$ Truncation (from $T = 1.0$)
Keep only top 2: {"apple", "banana"}.
- Unnormalized sum: $0.6364 + 0.2341 = \mathbf{0.8705}$
- Normalized $p(\text{"apple"}) = \frac{0.6364}{0.8705} \approx \mathbf{73.1\%}$
- Normalized $p(\text{"banana"}) = \frac{0.2341}{0.8705} \approx \mathbf{26.9\%}$
All other words are truncated to $0\%$.

#### Step 4: Top-$p = 0.85$ (Nucleus) Sampling (from $T = 1.0$)
Accumulate cumulative probability:
- Word 1: "apple" ($63.64\% < 85\%$)
- Word 2: "banana" ($63.64\% + 23.41\% = 87.05\% \ge 85\%$)
The nucleus threshold $0.85$ is satisfied by the first two words!
- **Surviving Words:** {"apple", "banana"}.
- Normalized probabilities match Top-$k=2$: **$73.1\%$ apple, $26.9\%$ banana**.
The tail tokens ("cherry", "date", "eggplant") are pruned away, eliminating hallucination risks!
</details>
