# Topic 4.2: Random Variables, PMF, PDF, and CDF

Welcome to **Topic 4.2: Random Variables, PMF, PDF, and CDF**. When training machine learning models, we don't work with abstract English events like "it rained" or "the coin landed heads". Instead, we translate physical events into numerical values through **Random Variables**.

---

## 1. Symbol Decoder Table

| Symbol | Name | Plain-English Meaning | Child-Friendly Analogy |
| :---: | :--- | :--- | :--- |
| $X$ | Random Variable | A rule/function that maps outcomes to numbers ($X: \Omega \to \mathbb{R}$) | The video game scorekeeper who awards $+10$ points for coins and $-5$ for hazards |
| $x$ | Realized Value | The specific number that actually came up | The score displayed on the TV screen at game over |
| $p_X(x)$ | PMF (Probability Mass Function) | Exact probability of discrete value: $P(X = x)$ | Slices of a chocolate cake handed out to kids |
| $f_X(x)$ | PDF (Probability Density Function) | Relative concentration/height of probability for continuous variables | Mountain elevation at coordinate $x$ |
| $F_X(x)$ | CDF (Cumulative Distribution Function) | Total accumulated probability up to $x$: $P(X \le x)$ | A progress bar filling from 0% at $-\infty$ to 100% at $+\infty$ |
| $\int_a^b f(x) dx$ | Definite Integral | The area under the density curve between $a$ and $b$ | Weighing the sand in a sandbox between markers $a$ and $b$ |
| $F^{-1}(u)$ | Inverse CDF / Quantile Function | Returns the threshold $x$ whose cumulative probability is $u$ | Setting the progress bar to 50% and asking which level you are at |

---

## 2. What Is a Random Variable?

### The 12-Year-Old Mental Model: The Scorekeeper
A **Random Variable** is not a single number, and it is not a variable in the algebra sense ($x + 2 = 5$).
A random variable is a **machine or scorekeeper** that takes an outcome from real life and converts it into a number:

$$\text{Coin Toss: } \Omega = \{\text{Heads}, \text{Tails}\}$$
$$X(\text{Heads}) = 1, \quad X(\text{Tails}) = 0$$

Now we can apply mathematics, compute averages, and optimize models!

---

## 3. Discrete Random Variables & The PMF

A random variable is **discrete** if its possible outcomes can be counted ($x \in \{0, 1, 2, \dots\}$).

### The Probability Mass Function (PMF)
The PMF gives the exact probability that $X$ takes on a specific value $x$:
$$p_X(x) = P(X = x)$$

### The Two Golden Rules of PMFs:
1. Every mass is non-negative:
   $$p_X(x) \ge 0 \quad \text{for all } x$$
2. The sum of all probabilities across all possible values equals exactly $1$:
   $$\sum_{x} p_X(x) = 1.0$$

*Example in AI:* A language model generating the next token over a vocabulary of $50{,}000$ words outputs a discrete PMF using the Softmax function: $\sum_{i=1}^{50000} p_i = 1.0$.

---

## 4. Continuous Random Variables & The PDF Paradox

A random variable is **continuous** if its possible values form an unbroken continuum along the real number line (e.g., weights in a neural network, inference latency in milliseconds, human height, or pixel brightness).

### The Infinite Precision Paradox: Why $P(X = x) = 0$
Ask yourself: *What is the probability that a randomly chosen person is **exactly** $1.75000000000000000\dots$ meters tall?*
Because there are an **infinite number of decimal values** between $1.7$ and $1.8$, the chance of hitting any single infinite-precision number is:
$$P(X = 1.75) = \frac{1}{\infty} = 0$$

For any continuous random variable, **the probability of any single exact number is always zero!**

### The Solution: Probability Density Function (PDF)
Instead of measuring the probability at a single microscopic point, we measure the **probability density** $f_X(x)$, and calculate probabilities by finding the **area under the curve** over an interval $[a, b]$:

$$P(a \le X \le b) = \int_a^b f_X(x) \, dx$$

```
    f(x) (Density)
      ▲           ╭─────────╮
      │          ╱           ╲
      │         ╱  Shaded     ╲
      │        ╱    Area       ╲
      │───────┼───────────────┼────────▶ x
              a               b
            P(a ≤ X ≤ b) = Area
```

### Key Properties of PDFs:
1. **Density can exceed 1.0!**
   Unlike probabilities, the height $f_X(x)$ can be $10, 100,$ or $1000$! Only the *integrated area* cannot exceed 1.
   *(For example, Uniform$(0, 0.1)$ has a constant height of $f(x) = 10.0$ because $\int_0^{0.1} 10 \, dx = 1.0$).*
2. **Total Area equals 1:**
   $$\int_{-\infty}^\infty f_X(x) \, dx = 1.0$$
3. **Non-negativity:**
   $$f_X(x) \ge 0 \quad \text{for all } x$$

---

## 5. The Cumulative Distribution Function (CDF)

The **Cumulative Distribution Function (CDF)** unites both discrete and continuous variables. It measures the accumulated probability that $X$ is less than or equal to a threshold $x$:

$$F_X(x) = P(X \le x)$$

### Mathematical Properties of All CDFs:
1. $0 \le F_X(x) \le 1$ for all $x$.
2. It is **monotonically non-decreasing**: if $x_1 \le x_2$, then $F(x_1) \le F(x_2)$.
3. $\lim_{x \to -\infty} F_X(x) = 0$ and $\lim_{x \to +\infty} F_X(x) = 1$.
4. Calculating interval probability:
   $$P(a < X \le b) = F_X(b) - F_X(a)$$

### Connecting CDF to PDF via Calculus:
By the Fundamental Theorem of Calculus:
- The PDF is the derivative (slope) of the CDF:
  $$f_X(x) = \frac{d}{dx} F_X(x)$$
- The CDF is the integral of the PDF:
  $$F_X(x) = \int_{-\infty}^x f_X(t) \, dt$$

---

## 6. Inverse Transform Sampling: Generating Random Data

How does a computer generate random samples from a complex continuous distribution (such as Exponential or Cauchy) using only a standard `random()` generator?
Through **Inverse Transform Sampling**!

### The Theorem
If $U \sim \text{Uniform}(0, 1)$ is a uniform random number between 0 and 1, then:
$$X = F^{-1}(U)$$
follows the exact distribution defined by CDF $F$!

```
     1.0 ▲                     ╭─── CDF F(x)
         │                   ╭─╯
       U ┼──────────────────*
         │                  │
         │                  │
     0.0 └──────────────────┼────────▶ x
                            X = F⁻¹(U)
```

### Derivation for the Exponential Distribution:
1. The Exponential PDF is $f(x) = \lambda e^{-\lambda x}$ for $x \ge 0$.
2. Integrating to find the CDF:
   $$F(x) = \int_0^x \lambda e^{-\lambda t} \, dt = \left[ -e^{-\lambda t} \right]_0^x = 1 - e^{-\lambda x}$$
3. Set $u = 1 - e^{-\lambda x}$ and solve for $x$:
   $$e^{-\lambda x} = 1 - u$$
   $$-\lambda x = \ln(1 - u)$$
   $$x = -\frac{1}{\lambda} \ln(1 - u)$$
Since $1 - u$ is also distributed uniformly on $(0, 1)$, we can generate exponential random numbers with a single line of Python:
```python
x = - (1.0 / lambda_param) * math.log(random.random())
```

---

## 7. Joint, Marginal, and Conditional Distributions

When an AI models relationships between multiple inputs (e.g., token $X$ and next token $Y$):
1. **Joint Distribution $p(x, y)$:** The probability that $X=x$ **and** $Y=y$ simultaneously.
2. **Marginalization (Summing Out Nuisance Variables):**
   To find the distribution of $X$ alone, we sum/integrate over all possible values of $Y$:
   $$p_X(x) = \sum_y p(x, y) \quad \text{(Discrete)}$$
   $$f_X(x) = \int_{-\infty}^\infty f(x, y) \, dy \quad \text{(Continuous)}$$
3. **Conditional Distribution:**
   $$p(y \mid x) = \frac{p(x, y)}{p_X(x)}$$
   This is the exact mathematical formulation of language model generation: predicting next token $Y$ given the context $X$!

