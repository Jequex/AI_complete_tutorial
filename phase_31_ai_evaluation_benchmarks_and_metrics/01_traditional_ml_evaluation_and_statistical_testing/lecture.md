# Topic 01: Traditional ML Evaluation and Statistical Testing

## 1. Introduction & 12-Year-Old Intuition

Imagine two runners, Alice and Bob, competing across 10 different tracks. 
- Track 1 is a rocky mountain trail covered in mud.
- Track 2 is a smooth Olympic stadium oval.
- Track 3 is an uphill sandy beach in heavy wind.

At the end of the 10 races, Alice's average lap time is **50.2 seconds**, while Bob's average lap time is **50.8 seconds**. 

A naive observer shouts: *"Alice is faster by 0.6 seconds! She is clearly the superior runner!"*

But a statistician asks: *"Did Alice beat Bob on the muddy hill? Did Alice beat Bob on the sandy beach? Or was Bob simply stuck on the rocky trail during a flash thunderstorm while Alice ran on smooth asphalt?"*

If you look at the **paired differences** on the exact same tracks:
- Race 1: Alice beats Bob by 0.5s
- Race 2: Alice beats Bob by 0.7s
- Race 3: Alice beats Bob by 0.4s
- ...
- Race 10: Alice beats Bob by 0.8s

Alice beat Bob on **every single track under identical weather and footing**. The probability that this occurred purely by random luck is less than $(\frac{1}{2})^{10} \approx 0.00098$ ($0.098\%$). We are virtually certain Alice is genuinely better.

Now imagine the reverse: Alice and Bob ran on completely random, unshared days. Alice ran mostly on flat stadiums while Bob ran through mud. Their averages differ by 0.6 seconds, but on shared tracks Bob wins half the time. The 0.6-second gap was pure statistical noise!

In Machine Learning and AI engineering, benchmark leaderboards are flooded with claims: *"Our new model achieves 84.7% accuracy, outperforming the baseline at 84.1%!"* Without paired statistical testing, cross-validation variance modeling, bootstrap confidence intervals, and multiple testing corrections, a 0.6% gain is frequently **pure noise masquerading as scientific progress**.

---

## 2. Mathematical Symbol Decoder

| Symbol | Name | Domain / Units | Description & Engineering Significance |
| :--- | :--- | :--- | :--- |
| $H_0$ | Null Hypothesis | Boolean Proposition | The baseline assumption that no true difference exists between models ($\mu_d = 0$). |
| $H_1$ | Alternative Hypothesis | Boolean Proposition | The research claim that a real performance difference exists ($\mu_d \ne 0$ or $\mu_d > 0$). |
| $\alpha$ | Significance Level | Scalar $\in (0, 1)$ | Maximum acceptable probability of committing a Type I error (false positive). Typically $0.05$ or $0.01$. |
| $\beta$ | Type II Error Rate | Scalar $\in (0, 1)$ | Probability of failing to detect a true difference (false negative). |
| $1 - \beta$ | Statistical Power | Scalar $\in (0, 1)$ | Probability of correctly rejecting $H_0$ when a true effect exists. |
| $p$-value | Probability Value | Scalar $\in [0, 1]$ | $\mathbb{P}(\text{observing data as or more extreme} \mid H_0 \text{ is true})$. |
| $d_i$ | Paired Difference | Real number | $d_i = x_{A,i} - x_{B,i}$, performance metric difference on fold/slice $i$. |
| $\bar{d}$ | Sample Mean Difference | Real number | $\bar{d} = \frac{1}{n} \sum_{i=1}^n d_i$. |
| $s_d$ | Sample Std Deviation | Real number $\ge 0$ | Sample dispersion of paired differences $s_d = \sqrt{\frac{1}{n-1}\sum (d_i - \bar{d})^2}$. |
| $t$ | Student's $t$ Statistic | Real number | $t = \frac{\bar{d}}{s_d / \sqrt{n}}$, ratio of observed effect to standard error. |
| $W$ | Wilcoxon Test Statistic | Integer $\ge 0$ | Sum of signed ranks; non-parametric test statistic resistant to outliers. |
| $T_{\text{perm}}$ | Permutation Statistic | Real number | Difference metric computed under randomized sign assignments of paired folds. |
| $B$ | Resampling Replicates | Integer $\gg 1000$ | Number of Monte Carlo or bootstrap resamples executed. |
| $\text{CI}_{1-\alpha}$ | Confidence Interval | $[L, U] \subset \mathbb{R}$ | Interval estimated to contain the true population parameter with probability $1 - \alpha$. |
| $\text{FWER}$ | Family-Wise Error Rate | Scalar $\in [0, 1]$ | $\mathbb{P}(\text{at least one false discovery among } m \text{ hypothesis tests})$. |
| $\text{FDR}$ | False Discovery Rate | Scalar $\in [0, 1]$ | Expected proportion of rejected hypotheses that are false positives: $\mathbb{E}[V / R]$. |
| $q$-value | Adjusted $p$-value | Scalar $\in [0, 1]$ | Minimum FDR threshold at which a specific hypothesis would be deemed significant. |

---

## 3. The Core Hypothesis Testing Framework

When comparing Model $A$ (candidate) against Model $B$ (baseline) evaluated over $n$ distinct evaluation folds or data splits:

```
                  Ground Truth: H_0 True             Ground Truth: H_1 True
Decision:       ┌──────────────────────────────────┬──────────────────────────────────┐
Reject H_0      │           TYPE I ERROR           │        CORRECT DECISION          │
                │        False Positive (α)        │           Power (1 - β)          │
                ├──────────────────────────────────┼──────────────────────────────────┤
Fail to         │         CORRECT DECISION         │          TYPE II ERROR           │
Reject H_0      │              (1 - α)             │        False Negative (β)        │
                └──────────────────────────────────┴──────────────────────────────────┘
```

A test is statistically valid if and only if:
$$\mathbb{P}(\text{Reject } H_0 \mid H_0) \le \alpha$$

---

## 4. Student's Paired $t$-Test

### 4.1 Derivation & Assumptions
Let $X_{A, 1}, \dots, X_{A, n}$ and $X_{B, 1}, \dots, X_{B, n}$ denote evaluation scores (e.g., test accuracy or macro-F1) across $n$ cross-validation folds. 
Define the paired difference vector:
$$d_i = X_{A, i} - X_{B, i}, \quad i = 1, \dots, n$$

Under the null hypothesis $H_0$, the true mean difference is zero:
$$H_0: \mu_d = 0 \quad \text{vs.} \quad H_1: \mu_d \ne 0$$

Assuming the differences $d_i \overset{\text{i.i.d.}}{\sim} \mathcal{N}(\mu_d, \sigma_d^2)$:
1. The sample mean difference is:
   $$\bar{d} = \frac{1}{n}\sum_{i=1}^n d_i$$
2. The unbiased sample variance of differences is:
   $$s_d^2 = \frac{1}{n-1}\sum_{i=1}^n (d_i - \bar{d})^2$$
3. The standard error of the mean difference ($\text{SE}$) is:
   $$\text{SE}(\bar{d}) = \frac{s_d}{\sqrt{n}}$$

The test statistic $t$ is:
$$\boxed{t = \frac{\bar{d} - 0}{\text{SE}(\bar{d})} = \frac{\bar{d}}{s_d / \sqrt{n}}}$$

Under $H_0$, $t$ follows a Student's $t$-distribution with $\nu = n - 1$ degrees of freedom:
$$f(t; \nu) = \frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\sqrt{\pi \nu} \, \Gamma\left(\frac{\nu}{2}\right)} \left(1 + \frac{t^2}{\nu}\right)^{-\frac{\nu+1}{2}}$$

The two-tailed $p$-value is:
$$p = 2 \cdot \int_{|t_{\text{obs}}|}^{\infty} f(u; n - 1) \, du = 2 \left(1 - F_t(|t_{\text{obs}}|; n - 1)\right)$$

> [!WARNING]
> Standard $k$-fold cross-validation violates the independence assumption because folds share $(k-2)/(k-1)$ training samples! For rigorous ML benchmarking, use **$5 \times 2$-fold cross-validation** with the corrected paired $t$-test (Dietterich, 1998) or non-parametric permutation testing.

---

## 5. Wilcoxon Signed-Rank Test (Non-Parametric)

When evaluation metric differences $d_i$ are skewed, bounded, or violate normality (common in NLP and recommendation benchmarks), Student's $t$-test loses statistical power and can yield invalid $p$-values. The Wilcoxon signed-rank test is a non-parametric alternative based on ordinal rankings.

### 5.1 Algorithm & Exact Formulation
1. Compute differences $d_i = X_{A, i} - X_{B, i}$ for all $n$ pairs.
2. Remove all zero differences ($d_i = 0$), leaving $n_r \le n$ non-zero pairs.
3. Sort the absolute differences $|d_i|$ in ascending order.
4. Assign ranks $R_i \in \{1, 2, \dots, n_r\}$. If absolute values are tied, assign the average rank to each tied entry.
5. Compute the sum of ranks for positive differences ($W^+$) and negative differences ($W^-$):
   $$W^+ = \sum_{d_i > 0} R_i, \qquad W^- = \sum_{d_i < 0} R_i$$
6. Notice that the sum of all ranks is:
   $$W^+ + W^- = \sum_{j=1}^{n_r} j = \frac{n_r(n_r + 1)}{2}$$
7. The test statistic is:
   $$\boxed{W = \min(W^+, W^-)}$$

### 5.2 Asymptotic Normal Approximation ($n_r \ge 20$)
Under the null hypothesis that differences are symmetric around zero:
$$\mathbb{E}[W] = \mu_W = \frac{n_r(n_r + 1)}{4}$$
$$\operatorname{Var}(W) = \sigma_W^2 = \frac{n_r(n_r + 1)(2n_r + 1)}{24} - \sum_{k=1}^T \frac{t_k^3 - t_k}{48}$$
where $t_k$ is the number of tied ranks in tie-group $k$.

The continuity-corrected $z$-statistic is:
$$\boxed{z = \frac{W - \mu_W + 0.5}{\sigma_W}}$$
and $p \approx 2(1 - \Phi(|z|))$ where $\Phi$ is the standard normal cumulative distribution function.

---

## 6. Paired Permutation Test (Exact Resampling)

The permutation test (Pitman, 1937) makes **zero distributional assumptions**. Under $H_0$, there is no difference between Model $A$ and Model $B$; therefore, swapping the labels $(A, B)$ on any fold is equally likely.

### 6.1 Mathematical Formulation
For $n$ paired folds, there are $2^n$ possible sign assignments:
$$s = (s_1, s_2, \dots, s_n), \quad s_i \in \{-1, +1\}$$

The observed test statistic is the mean difference:
$$T_{\text{obs}} = \frac{1}{n} \sum_{i=1}^n d_i$$

Under a permutation $b$, the permuted statistic is:
$$T^{(b)} = \frac{1}{n} \sum_{i=1}^n s_i^{(b)} d_i$$

If $n \le 20$, the exact test computes all $2^n$ combinations. For large $n$, we draw $B$ independent Monte Carlo permutations ($B \ge 10{,}000$):
$$\boxed{p_{\text{perm}} = \frac{1 + \sum_{b=1}^B \mathbb{I}\left(|T^{(b)}| \ge |T_{\text{obs}}|\right)}{B + 1}}$$
The addition of $+1$ to numerator and denominator prevents a $p$-value of exactly $0$ and guarantees an unbiased finite-sample test.

---

## 7. Bootstrap Confidence Intervals

Rather than reporting a single point metric (e.g., Accuracy = $84.2\%$), robust engineering demands reporting a confidence interval: $\text{Acc} = 84.2\% \pm 1.4\%$ at $95\%$ confidence.

Given an evaluation dataset $D = \{(x_1, y_1), \dots, (x_N, y_N)\}$:
1. Draw $B$ bootstrap datasets $D^{*(1)}, \dots, D^{*(B)}$, each of size $N$, by **sampling with replacement** from $D$.
2. For each replicate $b \in \{1, \dots, B\}$, compute the evaluation metric $\hat{\theta}^{*(b)} = \mathcal{M}(D^{*(b)})$.
3. Sort the bootstrap estimates: $\hat{\theta}^{*(1)} \le \hat{\theta}^{*(2)} \le \dots \le \hat{\theta}^{*(B)}$.

### 7.1 Percentile Bootstrap Interval
For significance level $\alpha$ (e.g., $\alpha = 0.05$ for $95\%$ CI):
$$\boxed{\text{CI}_{\text{percentile}} = \left[\hat{\theta}^*_{(\lfloor B \cdot \frac{\alpha}{2} \rfloor)}, \; \hat{\theta}^*_{(\lceil B \cdot (1 - \frac{\alpha}{2}) \rceil)}\right]}$$

### 7.2 Empirical (Basic) Bootstrap Interval
Let $\delta = \hat{\theta}^* - \hat{\theta}$ represent the estimation error. The basic bootstrap adjusts for metric bias:
$$\boxed{\text{CI}_{\text{basic}} = \left[2\hat{\theta} - \hat{\theta}^*_{(1 - \alpha/2)}, \; 2\hat{\theta} - \hat{\theta}^*_{(\alpha/2)}\right]}$$

---

## 8. Multiple Comparisons Corrections

When an AI lab compares a new architecture against $m$ different baselines (or sweeps over $m$ hyperparameters), running independent hypothesis tests at $\alpha = 0.05$ leads to rampant false discoveries.

### 8.1 The Multiplicity Problem: Family-Wise Error Rate (FWER)
Under $m$ independent null hypotheses all being true:
$$\text{FWER} = \mathbb{P}(\text{at least one false rejection}) = 1 - (1 - \alpha)^m$$

For $m = 20$ tests at $\alpha = 0.05$:
$$\text{FWER} = 1 - (0.95)^{20} \approx 1 - 0.3585 = \mathbf{64.15\%}$$
There is a $64\%$ chance of claiming a "statistically significant breakthrough" that is purely fictional!

### 8.2 Bonferroni Correction (FWER Control)
The Bonferroni correction controls the FWER at level $\alpha$ using Boole's inequality:
$$\mathbb{P}\left(\bigcup_{i=1}^m \{p_i \le \alpha'\}\right) \le \sum_{i=1}^m \mathbb{P}(p_i \le \alpha') = m \alpha' = \alpha \implies \boxed{\alpha_{\text{Bonf}} = \frac{\alpha}{m}}$$
Reject $H_{0, i}$ if and only if $p_i \le \frac{\alpha}{m}$.
*Drawback*: Highly conservative; destroys statistical power when $m$ is large.

### 8.3 Benjamini-Hochberg (BH) Procedure (False Discovery Rate Control)
The False Discovery Rate ($\text{FDR}$) is the expected proportion of false positives among all rejected hypotheses:
$$\text{FDR} = \mathbb{E}\left[\frac{V}{R} \;\middle|\; R > 0\right] \cdot \mathbb{P}(R > 0)$$
where $V$ is the number of false discoveries (false positives) and $R$ is total discoveries (rejections).

**The Benjamini-Hochberg Algorithm (1995)**:
1. Sort all $m$ raw $p$-values in ascending order:
   $$p_{(1)} \le p_{(2)} \le \dots \le p_{(m)}$$
2. For a target FDR level $Q \in (0, 1)$ (e.g., $Q = 0.05$), compute critical thresholds:
   $$c_k = \frac{k}{m} \cdot Q, \quad k = 1, \dots, m$$
3. Find the largest index $k^*$ such that:
   $$k^* = \max \left\{ k \;\middle|\; p_{(k)} \le \frac{k}{m} Q \right\}$$
4. Reject all null hypotheses associated with $p_{(1)}, p_{(2)}, \dots, p_{(k^*)}$. If no such $k$ exists, reject nothing.

$$\boxed{p_{(k)} \le \frac{k}{m} Q}$$

The BH procedure is substantially more powerful than Bonferroni while guaranteeing $\text{FDR} \le \frac{m_0}{m} Q \le Q$ for independent or positively dependent test statistics.

---

## 9. Summary Architecture of Statistical Validation in Production

```
                               Raw Evaluation Data
                      (n test folds or N test samples)
                                       │
                      ┌────────────────┴────────────────┐
                      ▼                                 ▼
           Model A vs Model B Pairs           Sample Bootstrap
           d_i = Score_A,i - Score_B,i        Sample N with replacement B times
                      │                                 │
         ┌────────────┴────────────┐                    ▼
         ▼                         ▼           Bootstrap Distribution θ*
  Parametric Check          Non-Parametric      • 95% Percentile CI [θ*_2.5%, θ*_97.5%]
  • Normality test          • Wilcoxon Rank     • Basic Bias-Corrected CI
  • Paired t-test (t, df)   • Exact Permutation
         │                         │
         └────────────┬────────────┘
                      ▼
             Raw p-value p_obs
                      │
         ┌────────────┴────────────┐
         ▼                         ▼
   Single Hypothesis      Multiple Comparisons (m > 1)
   p_obs < α = 0.05?      • Bonferroni: p_i ≤ α / m
                          • Benjamini-Hochberg FDR: p_(k) ≤ (k / m) * Q
```
