# Topic 4.2: Exercises & Solutions — Random Variables, PMF, PDF, and CDF

---

## Problem Set Overview

| Problem | Focus | Difficulty |
| :--- | :--- | :--- |
| **1** | **PDF Normalization Constant & CDF Calculus** | Medium |
| **2** | **Inverse Transform Sampling for the Pareto Distribution** | Medium |
| **3** | **Joint Distribution Matrix: Marginals, Conditioning & Independence** | Challenging |

---

## Problem 1: PDF Normalization Constant & CDF Calculus

### The Context
In generative models (such as Normalizing Flows and energy-based models), an unnormalized density function $q(x)$ must be scaled by a normalizing constant $Z$ so that its total integral equals $1$:
$$f(x) = \frac{1}{Z} q(x)$$

### The Problem
Consider a continuous random variable $X$ defined on the support interval $[0, 3]$ with unnormalized density $q(x) = c x^2$:
$$f_X(x) = \begin{cases} c x^2 & \text{if } 0 \le x \le 3 \\ 0 & \text{otherwise} \end{cases}$$

1. Determine the exact value of the constant $c$ such that $f_X(x)$ is a valid Probability Density Function.
2. Derive the Cumulative Distribution Function $F_X(x)$ for all $x \in \mathbb{R}$.
3. Calculate the probability $P(1 \le X \le 2)$.
4. Find the median $m$ of the distribution (the value such that $F_X(m) = 0.5$).

---

### Solution 1

#### 1. Finding Constant $c$
For $f_X(x)$ to be a valid PDF, its total integral must equal $1$:
$$\int_{-\infty}^\infty f_X(x) \, dx = \int_0^3 c x^2 \, dx = 1$$
$$\left[ c \frac{x^3}{3} \right]_0^3 = c \left( \frac{27}{3} - 0 \right) = 9c = 1 \implies c = \frac{1}{9}$$

$$f_X(x) = \frac{1}{9} x^2 \quad \text{for } x \in [0, 3]$$

#### 2. Deriving the CDF $F_X(x)$
For $x < 0$: $F_X(x) = 0$.
For $0 \le x \le 3$:
$$F_X(x) = \int_0^x \frac{1}{9} t^2 \, dt = \left[ \frac{t^3}{27} \right]_0^x = \frac{x^3}{27}$$
For $x > 3$: $F_X(x) = 1$.

$$F_X(x) = \begin{cases} 0 & x < 0 \\ \frac{x^3}{27} & 0 \le x \le 3 \\ 1 & x > 3 \end{cases}$$

#### 3. Interval Probability $P(1 \le X \le 2)$
Using the CDF:
$$P(1 \le X \le 2) = F_X(2) - F_X(1) = \frac{2^3}{27} - \frac{1^3}{27} = \frac{8 - 1}{27} = \frac{7}{27} \approx 0.2593$$

#### 4. Finding the Median $m$
Set $F_X(m) = 0.5$:
$$\frac{m^3}{27} = \frac{1}{2} \implies m^3 = \frac{27}{2} = 13.5$$
$$m = (13.5)^{1/3} \approx 2.3811$$

---

## Problem 2: Inverse Transform Sampling for the Pareto Distribution

### The Context
The **Pareto Distribution** models "power law" phenomena in AI, such as word frequencies in human language (Zipf's Law) and wealth distribution.
Its PDF for $x \ge x_m$ is:
$$f(x) = \frac{\alpha x_m^\alpha}{x^{\alpha + 1}}$$
where $x_m > 0$ is the minimum possible value and $\alpha > 0$ is the shape parameter.

### Task
1. Integrate the PDF to find the closed-form CDF $F(x)$.
2. Derive the inverse CDF $F^{-1}(u)$.
3. Write a pure Python function `sample_pareto(alpha, x_m)` that generates exact Pareto samples from `random.random()`.

---

### Solution 2

#### 1. CDF Derivation
For $x \ge x_m$:
$$F(x) = \int_{x_m}^x \alpha x_m^\alpha t^{-(\alpha + 1)} \, dt = \left[ -x_m^\alpha t^{-\alpha} \right]_{x_m}^x = -x_m^\alpha x^{-\alpha} - (-x_m^\alpha x_m^{-\alpha}) = 1 - \left( \frac{x_m}{x} \right)^\alpha$$

$$F(x) = 1 - \left( \frac{x_m}{x} \right)^\alpha \quad \text{for } x \ge x_m$$

#### 2. Inverse CDF $F^{-1}(u)$
Set $u = 1 - \left( \frac{x_m}{x} \right)^\alpha$:
$$\left( \frac{x_m}{x} \right)^\alpha = 1 - u$$
$$\frac{x_m}{x} = (1 - u)^{1/\alpha}$$
$$x = \frac{x_m}{(1 - u)^{1/\alpha}} = x_m (1 - u)^{-1/\alpha}$$
Since $1 - u \sim \text{Uniform}(0, 1)$, we can simplify to $X = x_m U^{-1/\alpha}$.

#### 3. Python Implementation
```python
import random

def sample_pareto(alpha: float, x_m: float = 1.0) -> float:
    """Samples from Pareto(alpha, x_m) using Inverse Transform Sampling."""
    u = random.random()
    return x_m / (u ** (1.0 / alpha))

# Example draw
samples = [sample_pareto(alpha=2.5, x_m=1.0) for _ in range(10_000)]
print("Pareto Sample Min:", min(samples))  # Always >= 1.0
```

---

## Problem 3: Joint Probability Table, Marginals & Independence

### The Context
A robotics AI has two sensors:
- $X \in \{0, 1\}$ (Proximity sensor: 0 = Clear, 1 = Obstacle)
- $Y \in \{0, 1\}$ (Lidar sensor: 0 = Clear, 1 = Obstacle)

Their joint probability distribution $P(X, Y)$ is given by the matrix:

| | $Y = 0$ | $Y = 1$ |
| :---: | :---: | :---: |
| **$X = 0$** | $0.54$ | $0.06$ |
| **$X = 1$** | $0.16$ | $0.24$ |

1. Find the marginal distribution $P(X)$.
2. Find the marginal distribution $P(Y)$.
3. Compute the conditional probability $P(Y = 1 \mid X = 1)$.
4. Determine whether random variables $X$ and $Y$ are **statistically independent**. Support your answer with explicit calculations.

---

### Solution 3

#### 1. Marginal $P(X)$
Sum across columns:
- $P(X = 0) = 0.54 + 0.06 = 0.60$
- $P(X = 1) = 0.16 + 0.24 = 0.40$
*(Check: $0.60 + 0.40 = 1.00$)*

#### 2. Marginal $P(Y)$
Sum across rows:
- $P(Y = 0) = 0.54 + 0.16 = 0.70$
- $P(Y = 1) = 0.06 + 0.24 = 0.30$
*(Check: $0.70 + 0.30 = 1.00$)*

#### 3. Conditional Probability $P(Y = 1 \mid X = 1)$
$$P(Y = 1 \mid X = 1) = \frac{P(X = 1, Y = 1)}{P(X = 1)} = \frac{0.24}{0.40} = 0.60 = \mathbf{60.0\%}$$

#### 4. Independence Test
Two variables are independent if and only if:
$$P(X = x, Y = y) = P(X = x) \times P(Y = y) \quad \text{for all } x, y$$

Check for $(X = 1, Y = 1)$:
- $P(X = 1) \times P(Y = 1) = 0.40 \times 0.30 = \mathbf{0.12}$
- But observed joint probability is $P(X = 1, Y = 1) = \mathbf{0.24}$!
- Since $0.24 \neq 0.12$, **$X$ and $Y$ are NOT independent**.
- Alternatively, $P(Y = 1 \mid X = 1) = 0.60 \neq P(Y = 1) = 0.30$.
Knowing that the proximity sensor detected an obstacle doubles the likelihood that the Lidar also detected an obstacle!

