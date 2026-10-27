# Topic 05: Exercises & Solutions — Scientific Discovery & Literature Synthesis

## Exercise 1: Asymptotic Consistency of the Non-Parametric Bootstrap

Consider a benchmark dataset $\mathbf{X} = \{X_1, X_2, \dots, X_n\}$ of $n$ independent and identically distributed metric scores with true mean $\mu$ and finite variance $\sigma^2 > 0$.
Let $\bar{X} = \frac{1}{n} \sum_{i=1}^n X_i$ be the sample mean.

In the non-parametric bootstrap, a bootstrap sample $\mathbf{X}^* = \{X_1^*, \dots, X_n^*\}$ is drawn by sampling $n$ times with replacement from the empirical distribution $\hat{F}_n$.
Let $\bar{X}^* = \frac{1}{n} \sum_{i=1}^n X_i^*$ be the bootstrap sample mean.

1. Compute the conditional expectation $\mathbb{E}_*[\bar{X}^* \mid \mathbf{X}]$ and conditional variance $\text{Var}_*(\bar{X}^* \mid \mathbf{X})$.
2. State Efron's Bootstrap Central Limit Theorem and explain why percentile bootstrap confidence intervals provide asymptotically exact coverage as $n \to \infty$.

---

### Solution 1:

#### 1. Conditional Moments of the Bootstrap Estimator
Under sampling with replacement from $\mathbf{X}$, each draw $X_i^*$ is a discrete random variable taking value $X_j$ with probability $\frac{1}{n}$ for all $j \in \{1, \dots, n\}$.
The conditional expectation of an individual bootstrap draw is:
$$\mathbb{E}_*[X_i^* \mid \mathbf{X}] = \sum_{j=1}^n X_j \cdot P(X_i^* = X_j) = \sum_{j=1}^n X_j \cdot \frac{1}{n} = \bar{X}$$
Therefore, the conditional mean of the bootstrap sample average is unbiased relative to the sample mean:
$$\mathbb{E}_*[\bar{X}^* \mid \mathbf{X}] = \frac{1}{n} \sum_{i=1}^n \mathbb{E}_*[X_i^* \mid \mathbf{X}] = \frac{1}{n} \sum_{i=1}^n \bar{X} = \bar{X}$$

For the conditional variance:
$$\text{Var}_*(X_i^* \mid \mathbf{X}) = \mathbb{E}_*[(X_i^* - \bar{X})^2 \mid \mathbf{X}] = \frac{1}{n} \sum_{j=1}^n (X_j - \bar{X})^2 = \hat{\sigma}^2$$
Because bootstrap draws $X_1^*, \dots, X_n^*$ are conditionally independent given $\mathbf{X}$:
$$\text{Var}_*(\bar{X}^* \mid \mathbf{X}) = \frac{1}{n^2} \sum_{i=1}^n \text{Var}_*(X_i^* \mid \mathbf{X}) = \frac{\hat{\sigma}^2}{n} = \frac{\sum_{j=1}^n (X_j - \bar{X})^2}{n^2}$$

#### 2. Efron's Bootstrap Central Limit Theorem
By Bickel & Freedman (1981) and Singh (1981):
$$\sup_t \left| P_*\left( \sqrt{n}(\bar{X}^* - \bar{X}) \le t \right) - P\left( \sqrt{n}(\bar{X} - \mu) \le t \right) \right| \xrightarrow{a.s.} 0 \quad \text{as } n \to \infty$$
As the number of resamples $B \to \infty$ and sample size $n \to \infty$, the empirical percentiles of $(\bar{X}^* - \bar{X})$ converge to the true quantiles of the sampling distribution:
$$P\left( \mu \in [\hat{\theta}^*_{\alpha/2}, \hat{\theta}^*_{1-\alpha/2}] \right) \to 1 - \alpha$$
This allows researchers to construct valid $95\%$ confidence intervals for complex, non-linear machine learning metrics (e.g. BLEU, F1 score, perplexity) without assuming Gaussian errors.

---

## Exercise 2: Small-Sample Bias in Cohen's $d$ & Hedges' $g$ Correction

In empirical evaluations with a small number of training seeds ($n_A = n_B = 5$), the sample standardized effect size is computed as:

$$d = \frac{\bar{X}_B - \bar{X}_A}{s_{\text{pooled}}}, \quad s_{\text{pooled}} = \sqrt{\frac{(n_A - 1)s_A^2 + (n_B - 1)s_B^2}{n_A + n_B - 2}}$$

1. Show that sample Cohen's $d$ is a **biased estimator** that systematically overestimates the true population effect size $\delta = \frac{\mu_B - \mu_A}{\sigma}$.
2. State the **Hedges' $g$ correction factor** $J(\nu)$ where $\nu = n_A + n_B - 2$ is the degrees of freedom, and compute the percentage overestimation for $n_A = n_B = 5$.

---

### Solution 2:

#### 1. Origin of Small-Sample Bias
The numerator $(\bar{X}_B - \bar{X}_A)$ is an unbiased estimator of $(\mu_B - \mu_A)$.
However, the denominator $s_{\text{pooled}}$ is the square root of an unbiased variance estimator:
$$\frac{\nu \cdot s_{\text{pooled}}^2}{\sigma^2} \sim \chi^2(\nu)$$
By Jensen's inequality (or properties of the Chi distribution), $\mathbb{E}[1 / s_{\text{pooled}}] > 1 / \sigma$:
$$\mathbb{E}\left[ \frac{1}{s_{\text{pooled}}} \right] = \frac{1}{\sigma} \sqrt{\frac{\nu}{2}} \frac{\Gamma\left(\frac{\nu - 1}{2}\right)}{\Gamma\left(\frac{\nu}{2}\right)} > \frac{1}{\sigma}$$
Because the numerator and denominator are statistically independent, their expectation factors:
$$\mathbb{E}[d] = \mathbb{E}\left[ \bar{X}_B - \bar{X}_A \right] \cdot \mathbb{E}\left[ \frac{1}{s_{\text{pooled}}} \right] = \delta \cdot \left[ \sqrt{\frac{\nu}{2}} \frac{\Gamma\left(\frac{\nu - 1}{2}\right)}{\Gamma\left(\frac{\nu}{2}\right)} \right] > \delta$$
Thus, Cohen's $d$ systematically overstates the magnitude of experimental improvements when evaluated across few random seeds.

#### 2. Hedges' $g$ Correction Factor
Larry Hedges (1981) introduced the exact correction factor $J(\nu)$:
$$g = J(\nu) \cdot d, \quad J(\nu) = \frac{\Gamma(\nu / 2)}{\sqrt{\nu / 2} \, \Gamma((\nu - 1) / 2)} \approx 1 - \frac{3}{4\nu - 1}$$
For $n_A = n_B = 5$, degrees of freedom $\nu = 5 + 5 - 2 = 8$:
$$J(8) \approx 1 - \frac{3}{4(8) - 1} = 1 - \frac{3}{31} \approx 1 - 0.0968 = 0.9032$$
Percentage overestimation:
$$\text{Overestimation} = \left(\frac{1}{0.9032} - 1\right) \times 100\% \approx 10.7\%$$
Without Hedges' $g$ correction, an author running only 5 seeds exaggerates their effect size by over $10\%$.

---

## Exercise 3: Factorial Interaction Contrasts & Degrees of Freedom

Consider a $2^2$ factorial ablation experiment testing Factor A (Attention Gating) and Factor B (RoPE Positional Encoding).
The four experimental cell means are denoted $\bar{y}_{00}, \bar{y}_{10}, \bar{y}_{01}, \bar{y}_{11}$.

1. Define the contrast vector $\mathbf{c}_A$ for the Main Effect of Factor A, $\mathbf{c}_B$ for Factor B, and $\mathbf{c}_{AB}$ for the Two-Way Interaction Effect.
2. Prove that $\mathbf{c}_A$, $\mathbf{c}_B$, and $\mathbf{c}_{AB}$ are mutually orthogonal vectors.
3. Explain the scientific interpretation of $\mathbf{c}_{AB}^\top \bar{\mathbf{y}} = 0$ versus $\mathbf{c}_{AB}^\top \bar{\mathbf{y}} > 0$.

---

### Solution 3:

#### 1. Contrast Definitions
Arranging cell means as the vector $\bar{\mathbf{y}} = [\bar{y}_{00}, \bar{y}_{10}, \bar{y}_{01}, \bar{y}_{11}]^\top$:
- **Main Effect A**: Compares $A=1$ vs $A=0$ averaged over $B$:
  $$\mathbf{c}_A = \frac{1}{2} [-1, +1, -1, +1]^\top$$
- **Main Effect B**: Compares $B=1$ vs $B=0$ averaged over $A$:
  $$\mathbf{c}_B = \frac{1}{2} [-1, -1, +1, +1]^\top$$
- **Interaction AB**: The difference in effect of Factor A when $B=1$ vs when $B=0$:
  $$\mathbf{c}_{AB} = \frac{1}{2} \left( [0, 0, -1, +1]^\top - [-1, +1, 0, 0]^\top \right) = \frac{1}{2} [+1, -1, -1, +1]^\top$$

#### 2. Proof of Mutual Orthogonality
Evaluate inner products between the contrast vectors:
$$\mathbf{c}_A^\top \mathbf{c}_B = \frac{1}{4} \left( (-1)(-1) + (1)(-1) + (-1)(1) + (1)(1) \right) = \frac{1}{4} (1 - 1 - 1 + 1) = 0$$
$$\mathbf{c}_A^\top \mathbf{c}_{AB} = \frac{1}{4} \left( (-1)(1) + (1)(-1) + (-1)(-1) + (1)(1) \right) = \frac{1}{4} (-1 - 1 + 1 + 1) = 0$$
$$\mathbf{c}_B^\top \mathbf{c}_{AB} = \frac{1}{4} \left( (-1)(1) + (-1)(-1) + (1)(-1) + (1)(1) \right) = \frac{1}{4} (-1 + 1 - 1 + 1) = 0$$
All three vectors are mutually orthogonal in $\mathbb{R}^4$.

#### 3. Scientific Significance of Interaction Contrast
- **$\mathbf{c}_{AB}^\top \bar{\mathbf{y}} = 0$ (Pure Additivity):** The components are independent. The performance gain of combining Gated Attention and RoPE is exactly equal to the sum of their individual isolated gains:
  $$\Delta_{\text{combined}} = \Delta_A + \Delta_B$$
- **$\mathbf{c}_{AB}^\top \bar{\mathbf{y}} > 0$ (Positive Non-Linear Synergy):** The features reinforce each other. RoPE provides relative positional geometry that allows Gated Attention to selectively amplify relevant tokens across distant sequence intervals, producing gains that exceed the sum of parts.
- **$\mathbf{c}_{AB}^\top \bar{\mathbf{y}} < 0$ (Negative Redundancy / Interference):** The features solve the same underlying representational issue or destabilize optimization when combined.
