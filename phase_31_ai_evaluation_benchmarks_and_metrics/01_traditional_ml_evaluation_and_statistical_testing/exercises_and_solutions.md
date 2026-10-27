# Topic 01: Exercises & Solutions — Traditional ML Evaluation and Statistical Testing

---

### Exercise 1: Paired $t$-Test and Wilcoxon Signed-Rank Test on 10-Fold CV

An applied AI research team evaluates two classifier architectures (Candidate Model $A$ and Baseline Model $B$) across 10 identical cross-validation folds. The test accuracy scores (in percentages) are recorded below:

| Fold $i$ | Model $A$ ($X_{A, i}$) | Model $B$ ($X_{B, i}$) |
| :--- | :--- | :--- |
| 1 | 85.4 | 83.2 |
| 2 | 87.1 | 86.5 |
| 3 | 82.0 | 82.0 |
| 4 | 88.6 | 85.1 |
| 5 | 84.5 | 83.0 |
| 6 | 89.2 | 88.0 |
| 7 | 81.3 | 82.5 |
| 8 | 86.7 | 84.2 |
| 9 | 85.0 | 83.1 |
| 10 | 87.8 | 85.4 |

**Questions:**
1. Compute the paired difference vector $d_i = X_{A, i} - X_{B, i}$, the sample mean difference $\bar{d}$, the sample standard deviation $s_d$, and the standard error of the mean $\text{SE}(\bar{d})$.
2. Calculate the Student's paired $t$-statistic and identify degrees of freedom. At $\alpha = 0.05$ (two-tailed critical value $t_{\text{crit}} = 2.262$ for $\nu = 9$), do you reject $H_0$?
3. Perform the Wilcoxon signed-rank test:
   - Filter out zero differences.
   - Rank the absolute differences $|d_i|$.
   - Calculate $W^+$ and $W^-$.
   - For $n_r = 9$ and $\alpha = 0.05$ (two-tailed Wilcoxon critical value $W_{\text{crit}} = 5$), state whether $H_0$ is rejected.

---

#### Solution to Exercise 1

**1. Paired Differences and Sample Statistics:**
Compute $d_i = X_{A, i} - X_{B, i}$:
- $d_1 = 85.4 - 83.2 = +2.2$
- $d_2 = 87.1 - 86.5 = +0.6$
- $d_3 = 82.0 - 82.0 = 0.0$
- $d_4 = 88.6 - 85.1 = +3.5$
- $d_5 = 84.5 - 83.0 = +1.5$
- $d_6 = 89.2 - 88.0 = +1.2$
- $d_7 = 81.3 - 82.5 = -1.2$
- $d_8 = 86.7 - 84.2 = +2.5$
- $d_9 = 85.0 - 83.1 = +1.9$
- $d_{10} = 87.8 - 85.4 = +2.4$

Mean difference:
$$\bar{d} = \frac{2.2 + 0.6 + 0.0 + 3.5 + 1.5 + 1.2 - 1.2 + 2.5 + 1.9 + 2.4}{10} = \frac{14.6}{10} = \mathbf{+1.46\%}$$

Sum of squared deviations $\sum (d_i - \bar{d})^2$:
- $(2.2 - 1.46)^2 = 0.74^2 = 0.5476$
- $(0.6 - 1.46)^2 = (-0.86)^2 = 0.7396$
- $(0.0 - 1.46)^2 = (-1.46)^2 = 2.1316$
- $(3.5 - 1.46)^2 = 2.04^2 = 4.1616$
- $(1.5 - 1.46)^2 = 0.04^2 = 0.0016$
- $(1.2 - 1.46)^2 = (-0.26)^2 = 0.0676$
- $(-1.2 - 1.46)^2 = (-2.66)^2 = 7.0756$
- $(2.5 - 1.46)^2 = 1.04^2 = 1.0816$
- $(1.9 - 1.46)^2 = 0.44^2 = 0.1936$
- $(2.4 - 1.46)^2 = 0.94^2 = 0.8836$
- $\sum (d_i - \bar{d})^2 = 16.884$

Sample variance and standard deviation:
$$s_d^2 = \frac{16.884}{9} \approx 1.876 \implies s_d \approx \sqrt{1.876} \approx \mathbf{1.3697\%}$$

Standard error:
$$\text{SE}(\bar{d}) = \frac{s_d}{\sqrt{10}} = \frac{1.3697}{3.1623} \approx \mathbf{0.4331\%}$$

**2. Student's Paired $t$-Test:**
$$t = \frac{\bar{d}}{\text{SE}(\bar{d})} = \frac{1.46}{0.4331} \approx \mathbf{3.371}$$
- Degrees of freedom: $\nu = 10 - 1 = \mathbf{9}$.
- Two-tailed critical value at $\alpha = 0.05$: $t_{\text{crit}} = 2.262$.
- Since $|t| = 3.371 > 2.262$, we **reject the null hypothesis $H_0$** ($p \approx 0.0082 < 0.05$). Candidate Model $A$ is statistically significantly superior to Baseline Model $B$.

**3. Wilcoxon Signed-Rank Test:**
- Drop fold 3 ($d_3 = 0$), so $n_r = 9$.
- Non-zero differences $|d_i|$:
  * Fold 2: $|+0.6| = 0.6 \implies$ Rank 1
  * Fold 6: $|+1.2| = 1.2$ (tied with Fold 7)
  * Fold 7: $|-1.2| = 1.2 \implies$ Tied ranks for positions 2 and 3: $\frac{2 + 3}{2} = 2.5$
  * Fold 5: $|+1.5| = 1.5 \implies$ Rank 4
  * Fold 9: $|+1.9| = 1.9 \implies$ Rank 5
  * Fold 1: $|+2.2| = 2.2 \implies$ Rank 6
  * Fold 10: $|+2.4| = 2.4 \implies$ Rank 7
  * Fold 8: $|+2.5| = 2.5 \implies$ Rank 8
  * Fold 4: $|+3.5| = 3.5 \implies$ Rank 9

Sum of ranks:
- Negative rank sum (only Fold 7 has $d_i < 0$):
  $$W^- = 2.5$$
- Positive rank sum:
  $$W^+ = 1 + 2.5 + 4 + 5 + 6 + 7 + 8 + 9 = 42.5$$
- Total sum check: $W^+ + W^- = 42.5 + 2.5 = 45 = \frac{9 \times 10}{2} = 45$ (verified).
- Test statistic:
  $$W = \min(W^+, W^-) = \mathbf{2.5}$$
- For $n_r = 9$ at $\alpha = 0.05$, the two-tailed critical threshold is $W_{\text{crit}} = 5$. Since $W = 2.5 \le 5$, we **reject $H_0$** ($p \approx 0.016 < 0.05$). Both parametric and non-parametric tests confirm Model $A$'s superiority.

---

### Exercise 2: Percentile vs. Empirical (Basic) Bootstrap Confidence Intervals

An engineer computes the test Macro-F1 score of a model on an evaluation set of 1,000 samples, obtaining a sample point estimate of:
$$\hat{\theta} = 0.820$$

The engineer runs $B = 2{,}000$ bootstrap iterations. The resulting distribution of bootstrap estimates $\hat{\theta}^{*(b)}$ yields:
- $2.5^{\text{th}}$ percentile ($\hat{\theta}^*_{(0.025)}$): $0.795$
- $50.0^{\text{th}}$ percentile (median $\hat{\theta}^*_{(0.500)}$): $0.824$
- $97.5^{\text{th}}$ percentile ($\hat{\theta}^*_{(0.975)}$): $0.840$

**Questions:**
1. Compute the $95\%$ Percentile Bootstrap Confidence Interval.
2. Compute the $95\%$ Empirical (Basic) Bootstrap Confidence Interval.
3. Compare the two intervals and explain why the Basic Bootstrap interval shifts relative to the Percentile interval when the bootstrap distribution is skewed or biased relative to $\hat{\theta}$.

---

#### Solution to Exercise 2

**1. Percentile Bootstrap Interval:**
$$\text{CI}_{\text{percentile}} = \left[\hat{\theta}^*_{(0.025)}, \; \hat{\theta}^*_{(0.975)}\right] = \mathbf{[0.795, \; 0.840]}$$

**2. Empirical (Basic) Bootstrap Interval:**
The basic bootstrap formula is:
$$\text{CI}_{\text{basic}} = \left[2\hat{\theta} - \hat{\theta}^*_{(0.975)}, \; 2\hat{\theta} - \hat{\theta}^*_{(0.025)}\right]$$

Substitute $\hat{\theta} = 0.820$, $\hat{\theta}^*_{(0.975)} = 0.840$, and $\hat{\theta}^*_{(0.025)} = 0.795$:
- Lower bound: $2(0.820) - 0.840 = 1.640 - 0.840 = \mathbf{0.800}$
- Upper bound: $2(0.820) - 0.795 = 1.640 - 0.795 = \mathbf{0.845}$

$$\text{CI}_{\text{basic}} = \mathbf{[0.800, \; 0.845]}$$

**3. Comparison and Intuition:**
- Notice that the bootstrap median ($0.824$) is higher than the original point estimate $\hat{\theta} = 0.820$, indicating an upward bootstrap bias ($\text{Bias} \approx 0.824 - 0.820 = +0.004$).
- The Percentile Bootstrap directly reports the distribution of $\hat{\theta}^*$, making it sensitive to bootstrap bias and skewness.
- The Basic Bootstrap models the error distribution $\delta^* = \hat{\theta}^* - \hat{\theta}$. If the bootstrap estimates tend to overshoot $\hat{\theta}$, the basic bootstrap subtracts this overshoot from the estimate, shifting the interval to the left:
  $$\hat{\theta} - \delta^*_{(0.975)} = \hat{\theta} - (\hat{\theta}^*_{(0.975)} - \hat{\theta}) = 2\hat{\theta} - \hat{\theta}^*_{(0.975)}$$
- As a result, the basic interval corrects for asymmetric distribution shifts.

---

### Exercise 3: Multiple Hypothesis Testing: Bonferroni vs. Benjamini-Hochberg (FDR)

A research lab tests $m = 10$ different architecture variants against the production baseline. The raw two-tailed $p$-values are sorted below:

| Variant $k$ | Architecture Description | Raw $p_{(k)}$ |
| :--- | :--- | :--- |
| 1 | Architecture $M_7$ | 0.0021 |
| 2 | Architecture $M_3$ | 0.0045 |
| 3 | Architecture $M_9$ | 0.0098 |
| 4 | Architecture $M_1$ | 0.0182 |
| 5 | Architecture $M_6$ | 0.0240 |
| 6 | Architecture $M_4$ | 0.0385 |
| 7 | Architecture $M_{10}$ | 0.0460 |
| 8 | Architecture $M_2$ | 0.0810 |
| 9 | Architecture $M_8$ | 0.1540 |
| 10 | Architecture $M_5$ | 0.6200 |

Target significance level: $\alpha = 0.05$. Target False Discovery Rate: $Q = 0.05$.

**Questions:**
1. Without multiple testing correction, which architectures are deemed statistically significant? How many false discoveries would you expect by chance?
2. Apply the **Bonferroni correction**. Determine the adjusted threshold $\alpha_{\text{Bonf}}$ and list the architectures rejected under this threshold.
3. Apply the **Benjamini-Hochberg procedure** at $Q = 0.05$. Compute the critical threshold $c_k = \frac{k}{m} Q$ for each $k \in \{1, \dots, 10\}$, determine the cutoff index $k^*$, and list the architectures discovered.
4. Contrast the conclusions: what is the engineering risk of using unadjusted vs. Bonferroni vs. Benjamini-Hochberg?

---

#### Solution to Exercise 3

**1. Unadjusted Comparison ($\alpha = 0.05$):**
Any architecture with raw $p_i \le 0.05$ is claimed significant:
- Variants 1 through 7 have $p_i \le 0.05$: $\{M_7, M_3, M_9, M_1, M_6, M_4, M_{10}\}$ (**7 models declared superior**).
- Under $m = 10$ tests where all null hypotheses are true, the probability of at least one false positive is:
  $$\text{FWER} = 1 - (1 - 0.05)^{10} = 1 - 0.5987 = \mathbf{40.13\%}$$
  Unadjusted testing creates a $40\%$ risk of shipping a false positive to production.

**2. Bonferroni Correction:**
$$\alpha_{\text{Bonf}} = \frac{\alpha}{m} = \frac{0.05}{10} = \mathbf{0.0050}$$
Evaluate $p_{(k)} \le 0.0050$:
- $k = 1$ ($M_7$): $p_{(1)} = 0.0021 \le 0.0050 \implies$ **Reject $H_0$**
- $k = 2$ ($M_3$): $p_{(2)} = 0.0045 \le 0.0050 \implies$ **Reject $H_0$**
- $k = 3$ ($M_9$): $p_{(3)} = 0.0098 > 0.0050 \implies$ Fail to reject
- All subsequent $p_{(k)} > 0.0050$.
- **Bonferroni discoveries**: $\{M_7, M_3\}$ (**2 models declared superior**).
- Bonferroni is extremely conservative; it likely discarded genuine improvements (high Type II error rate).

**3. Benjamini-Hochberg (BH) Procedure ($Q = 0.05$):**
Compute $c_k = \frac{k}{10} \times 0.05 = 0.005 \times k$:

| $k$ | Model | Raw $p_{(k)}$ | BH Threshold $c_k = \frac{k}{10} \times 0.05$ | Condition $p_{(k)} \le c_k$? |
| :--- | :--- | :--- | :--- | :--- |
| 1 | $M_7$ | 0.0021 | 0.0050 | **YES** |
| 2 | $M_3$ | 0.0045 | 0.0100 | **YES** |
| 3 | $M_9$ | 0.0098 | 0.0150 | **YES** |
| 4 | $M_1$ | 0.0182 | 0.0200 | **YES** |
| 5 | $M_6$ | 0.0240 | 0.0250 | **YES** |
| 6 | $M_4$ | 0.0385 | 0.0300 | NO ($0.0385 > 0.0300$) |
| 7 | $M_{10}$| 0.0460 | 0.0350 | NO ($0.0460 > 0.0350$) |
| 8 | $M_2$ | 0.0810 | 0.0400 | NO |
| 9 | $M_8$ | 0.1540 | 0.0450 | NO |
| 10| $M_5$ | 0.6200 | 0.0500 | NO |

- The largest index satisfying $p_{(k)} \le c_k$ is $k^* = 5$ (Architecture $M_6$, $0.0240 \le 0.0250$).
- Reject $H_0$ for all $k \le 5$: $\{M_7, M_3, M_9, M_1, M_6\}$.
- **BH Discoveries**: **5 models declared superior**.

**4. Engineering Trade-off Summary:**
- **Unadjusted**: Declared 7 models superior; high risk of false discoveries that degrade production performance.
- **Bonferroni**: Declared 2 models superior; controls FWER $\le 0.05$, but throws away 3 genuine, reproducible improvements.
- **Benjamini-Hochberg**: Declared 5 models superior; guarantees that among the 5 deployed candidates, the expected proportion of false positives is at most $5\%$, striking the optimal balance between safety and innovation velocity.
