# Topic 5.5: Exercises & Solutions — Bayesian Inference & Conjugate Priors

---

## Problem Set Overview

| Problem | Focus | Difficulty | Real-World Context |
| :---: | :--- | :---: | :--- |
| **1** | **Multi-Day Beta-Binomial Sequential Updating** | Easy | Streaming online CTR estimation for real-time recommendation engines |
| **2** | **Posterior Predictive & Laplace's Rule of Succession** | Medium | Avoiding catastrophic overconfidence in safety-critical model audits |
| **3** | **Completing the Square: Gaussian Conjugate Model** | Challenging | Kalman filtering and sensor fusion in robotics and self-driving vehicles |

---

## Problem 1: Multi-Day Beta-Binomial Sequential Updating in Recommender Systems

### The Context
An AI recommender system tests a newly generated movie thumbnail.
Before deployment, the system initializes an uninformative uniform prior:
$$\theta \sim \text{Beta}(\alpha_0 = 1, \, \beta_0 = 1)$$
where $\theta$ represents the user Click-Through Rate (CTR).

- **Day 1:** The thumbnail is shown to $n_1 = 10$ users, receiving $k_1 = 8$ clicks.
- **Day 2:** The thumbnail is shown to $n_2 = 10$ more users, receiving $k_2 = 4$ clicks.

### Task
1. Using the conjugate update rule, find the posterior parameters $\alpha_1, \beta_1$ after Day 1. Compute the posterior mean and mode.
2. Using Day 1's posterior as the prior for Day 2, compute the updated parameters $\alpha_2, \beta_2$ after Day 2.
3. Compute the final posterior mean $\mathbb{E}[\theta \mid D]$ and posterior variance $\operatorname{Var}(\theta \mid D)$ after Day 2.
4. Show that processing Day 1 and Day 2 together in one batch of 20 users produces the exact same posterior parameters.

---

<details>
<summary>Click to Reveal Step-by-Step Solution</summary>

### Step-by-Step Solution

#### 1. Day 1 Update
Add Day 1's clicks and skips to the prior counts:
- $\alpha_1 = \alpha_0 + k_1 = 1 + 8 = \mathbf{9}$
- $\beta_1 = \beta_0 + (n_1 - k_1) = 1 + (10 - 8) = 1 + 2 = \mathbf{3}$
- Posterior distribution after Day 1: $\theta \mid D_1 \sim \text{Beta}(9, 3)$
- Posterior Mean: $\mathbb{E}[\theta] = \frac{9}{9 + 3} = \frac{9}{12} = \mathbf{0.750 \quad (75.0\%)}$
- Posterior Mode: $\text{Mode}[\theta] = \frac{9 - 1}{9 + 3 - 2} = \frac{8}{10} = \mathbf{0.800 \quad (80.0\%)}$

#### 2. Day 2 Sequential Update
Use Day 1's posterior $\text{Beta}(9, 3)$ as the new prior for Day 2:
- $\alpha_2 = \alpha_1 + k_2 = 9 + 4 = \mathbf{13}$
- $\beta_2 = \beta_1 + (n_2 - k_2) = 3 + (10 - 4) = 3 + 6 = \mathbf{9}$
- Posterior distribution after Day 2: $\mathbf{\theta \mid D_{1,2} \sim \text{Beta}(13, 9)}$

#### 3. Day 2 Final Statistics
$$\mathbb{E}[\theta \mid D_{1,2}] = \frac{\alpha_2}{\alpha_2 + \beta_2} = \frac{13}{13 + 9} = \frac{13}{22} \approx \mathbf{0.5909 \quad (59.09\%)}$$
$$\operatorname{Var}(\theta \mid D_{1,2}) = \frac{\alpha \beta}{(\alpha + \beta)^2 (\alpha + \beta + 1)} = \frac{13 \times 9}{(22)^2 (23)} = \frac{117}{484 \times 23} \approx \mathbf{0.01051}$$
$$\text{Posterior Std Dev} = \sqrt{0.01051} \approx \mathbf{0.1025 \quad (10.25\%)}$$

#### 4. Batch Invariance Check
Total clicks: $k_{\text{total}} = 8 + 4 = 12$. Total users: $n_{\text{total}} = 10 + 10 = 20$.
$$\alpha_{\text{batch}} = 1 + 12 = \mathbf{13}, \quad \beta_{\text{batch}} = 1 + (20 - 12) = 1 + 8 = \mathbf{9}$$
The sequential streaming update and the full batch update are **100% mathematically identical**!

</details>

---

## Problem 2: Posterior Predictive Probability & Laplace's Rule of Succession

### The Context
In 1774, Pierre-Simon Laplace famously asked:
> *"If the sun has risen every single day for the past 5,000 years (say, $n$ days with $n$ successes and 0 failures), what is the probability it will rise tomorrow?"*

In AI, an autonomous agent tests a new API tool and observes $n$ successes in $n$ consecutive calls ($k = n$).
We adopt an uninformative uniform prior $\theta \sim \text{Beta}(1, 1)$.

### Task
1. Write the posterior distribution $\theta \mid D$.
2. Write the integral formula for the **Posterior Predictive Distribution** of the next single event $X_{n+1} = 1$:
   $$P(X_{n+1} = 1 \mid D) = \int_0^1 P(X_{n+1} = 1 \mid \theta) P(\theta \mid D) d\theta$$
3. Evaluate the integral to prove **Laplace's Rule of Succession**:
   $$P(X_{n+1} = 1 \mid D) = \frac{k + 1}{n + 2}$$
4. If a safety model passes 3 consecutive red-team tests ($k = 3, n = 3$), compare the prediction of MLE vs Laplace's Rule of Succession for the 4th test.

---

<details>
<summary>Click to Reveal Step-by-Step Solution</summary>

### Step-by-Step Solution

#### 1. Posterior Distribution
With prior $\text{Beta}(1, 1)$ and observed data $k$ successes in $n$ trials:
$$\theta \mid D \sim \text{Beta}(1 + k, \quad 1 + n - k)$$

#### 2. Posterior Predictive Integral
The event $X_{n+1} = 1$ given parameter $\theta$ has probability $\theta$:
$$P(X_{n+1} = 1 \mid D) = \int_0^1 \theta \cdot P(\theta \mid D) d\theta$$
Notice that $\int_0^1 \theta P(\theta \mid D) d\theta$ is simply the **expected value** of the posterior distribution:
$$P(X_{n+1} = 1 \mid D) = \mathbb{E}[\theta \mid D]$$

#### 3. Derivation of Laplace's Rule of Succession
For $\text{Beta}(\alpha_{\text{post}}, \beta_{\text{post}})$ with $\alpha_{\text{post}} = k + 1$ and $\beta_{\text{post}} = n - k + 1$:
$$\mathbb{E}[\theta \mid D] = \frac{\alpha_{\text{post}}}{\alpha_{\text{post}} + \beta_{\text{post}}} = \frac{k + 1}{(k + 1) + (n - k + 1)} = \mathbf{\frac{k + 1}{n + 2}} \quad \blacksquare$$

#### 4. Comparison for $k = 3, n = 3$
- **Pure MLE:** $\hat{\theta}_{\text{MLE}} = \frac{3}{3} = \mathbf{1.000 \quad (100\%)}$.
  *Danger:* MLE arrogantly claims with $100\%$ certainty that the model can never fail!
- **Laplace's Rule of Succession (Full Bayes):**
  $$P(X_4 = 1 \mid D) = \frac{3 + 1}{3 + 2} = \frac{4}{5} = \mathbf{0.800 \quad (80\%)}$$
  *Wisdom:* Bayesian inference awards an $80\%$ chance of success, while wisely holding a $20\%$ chance of failure because our sample size was only 3 tests!

</details>

---

## Problem 3: Completing the Square: Gaussian-Gaussian Conjugate Derivation

### The Context
Let $X \sim \mathcal{N}(\mu, \sigma^2)$ where measurement noise variance $\sigma^2$ is known.
We observe a single sensor reading $x_1$.
Prior belief on $\mu$ is Gaussian:
$$\mu \sim \mathcal{N}(\mu_0, \, \sigma_0^2)$$

### Task
1. Write the product $P(x_1 \mid \mu) P(\mu)$ in exponential form.
2. Group the terms in the exponent and complete the square with respect to $\mu$.
3. Prove that the posterior distribution $\mu \mid x_1$ is strictly Gaussian $\mathcal{N}(\mu_1, \sigma_1^2)$ with precision addition:
   $$\frac{1}{\sigma_1^2} = \frac{1}{\sigma_0^2} + \frac{1}{\sigma^2}, \quad \mu_1 = \sigma_1^2 \left( \frac{\mu_0}{\sigma_0^2} + \frac{x_1}{\sigma^2} \right)$$

---

<details>
<summary>Click to Reveal Step-by-Step Solution</summary>

### Step-by-Step Solution

#### 1. Joint Exponential Product
$$P(\mu \mid x_1) \propto P(x_1 \mid \mu) P(\mu) \propto \exp\left( -\frac{(x_1 - \mu)^2}{2\sigma^2} \right) \exp\left( -\frac{(\mu - \mu_0)^2}{2\sigma_0^2} \right)$$
$$P(\mu \mid x_1) \propto \exp\left( -\frac{1}{2} \left[ \frac{(\mu - x_1)^2}{\sigma^2} + \frac{(\mu - \mu_0)^2}{\sigma_0^2} \right] \right)$$

#### 2. Expanding and Completing the Square
Expand the quadratic terms:
$$\frac{\mu^2 - 2\mu x_1 + x_1^2}{\sigma^2} + \frac{\mu^2 - 2\mu \mu_0 + \mu_0^2}{\sigma_0^2}$$
Group by powers of $\mu$:
$$= \mu^2 \left( \frac{1}{\sigma_0^2} + \frac{1}{\sigma^2} \right) - 2\mu \left( \frac{\mu_0}{\sigma_0^2} + \frac{x_1}{\sigma^2} \right) + \text{constant}$$

Define posterior precision $\frac{1}{\sigma_1^2}$:
$$\mathbf{\frac{1}{\sigma_1^2} = \frac{1}{\sigma_0^2} + \frac{1}{\sigma^2}}$$

Now the expression inside the exponent becomes:
$$= \frac{1}{\sigma_1^2} \mu^2 - 2\mu \left( \frac{\mu_0}{\sigma_0^2} + \frac{x_1}{\sigma^2} \right) = \frac{1}{\sigma_1^2} \left[ \mu^2 - 2\mu \sigma_1^2 \left( \frac{\mu_0}{\sigma_0^2} + \frac{x_1}{\sigma^2} \right) \right]$$

Define posterior mean $\mu_1$:
$$\mathbf{\mu_1 = \sigma_1^2 \left( \frac{\mu_0}{\sigma_0^2} + \frac{x_1}{\sigma^2} \right)}$$

Completing the square:
$$= \frac{1}{\sigma_1^2} (\mu - \mu_1)^2 + \text{constant}$$

#### 3. Final Posterior Distribution
Substituting back into the exponential:
$$P(\mu \mid x_1) \propto \exp\left( -\frac{(\mu - \mu_1)^2}{2\sigma_1^2} \right) \implies \mathbf{\mu \mid x_1 \sim \mathcal{N}(\mu_1, \, \sigma_1^2)} \quad \blacksquare$$

> **🌟 The Sensor Fusion Takeaway:**
> Posterior precision is literally the sum of prior precision and data precision:
> $$\tau_{\text{post}} = \tau_{\text{prior}} + \tau_{\text{data}}$$
> Every new sensor measurement adds information, guaranteeing that uncertainty strictly shrinks with every reading!

</details>
