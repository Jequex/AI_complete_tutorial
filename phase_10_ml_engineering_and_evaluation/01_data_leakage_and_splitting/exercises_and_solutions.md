# Topic 10.1: Exercises & Solutions — Data Leakage & Splitting

---

## Problem Set Overview

This exercise suite tests your theoretical comprehension, mathematical derivation skills, and real-world engineering intuition for preventing data leakage and designing robust validation schemes.

---

### Problem 1: The Subtle Leak in Customer Churn Prediction (Engineering Case Study)

A machine learning engineer at a telecommunications firm is tasked with building a model to predict customer churn ($y \in \{0, 1\}$) at the end of each calendar month. The engineer queries the data warehouse for all active users during the month of June and includes the following features:
1. `tenure_months`: Number of months the customer has held an active subscription.
2. `avg_monthly_bill`: Mean monthly spend over the last 6 months.
3. `support_tickets_count`: Total support calls logged during June.
4. `customer_retention_discount_applied`: A boolean indicating whether the customer received a special loyalty discount from customer support.
5. `data_usage_gb`: Gigabytes of mobile data used in June.

The engineer fits an XGBoost model, achieves **99.4% ROC-AUC** in 5-fold cross-validation, and deploys it. In July, the model predicts churn with only **53.1% accuracy** (scarcely better than random guessing).

**Questions:**
1. Identify the exact variable causing data leakage and explain why it is leaked.
2. Formulate the temporal condition that any valid training feature must satisfy.
3. How should the feature engineering pipeline be refactored to restore legitimate predictive utility?

---

### Solution 1

1. **Root Cause:**
   The culprit is `customer_retention_discount_applied`. 
   In telecom business operations, retention discounts are only offered to customers *after* they call customer support explicitly demanding to cancel their subscription. Therefore, `customer_retention_discount_applied = 1` occurs *as a consequence* of churn intent, rather than as a prior causal indicator. At deployment time (the beginning of July), this discount has not yet been triggered for users who will churn later in the month.

2. **Temporal Condition:**
   Let $t_{\text{pred}}$ be the exact timestamp at which the model generates inference, and let $t_{\text{event}}$ be the timestamp when the target outcome $y$ physically occurs. Any feature $x_j$ must satisfy:
   $$\text{timestamp}(x_j) \le t_{\text{pred}} < t_{\text{event}}$$
   Any feature logged where $\text{timestamp}(x_j) > t_{\text{pred}}$ constitutes target leakage.

3. **Pipeline Refactoring:**
   - Drop the concurrent retention discount flag.
   - If historical retention offers are relevant, engineer historical lag features: e.g., `retention_discounts_in_prior_year`, computed strictly over the window $[t_{\text{pred}} - 365\text{ days}, t_{\text{pred}}]$.

---

### Problem 2: Mathematical Proof of Preprocessing Mean Leakage

Suppose a dataset $\mathcal{D} = \{x_1, \dots, x_N\}$ is drawn from a 1D distribution with true mean $\mu^*$ and variance $\sigma^2$. We intend to standardize the data:
$$z_i = \frac{x_i - \mu}{\sigma}$$
Consider an engineer who computes the sample mean $\hat{\mu}_{\text{global}} = \frac{1}{N}\sum_{i=1}^N x_i$ across the entire dataset of size $N$, and then creates a test split of size $M$ ($M < N$) with remaining training set size $N - M$.

**Questions:**
1. Show that for any test sample $x_k \in \mathcal{D}_{\text{test}}$, the transformed training samples $z_i \in \mathcal{D}_{\text{train}}$ are mathematically dependent on $x_k$.
2. Compute the exact covariance $\text{Cov}(z_i, x_k)$ between a training feature and a test sample when $\hat{\mu}_{\text{global}}$ is used, assuming $x_i$ and $x_k$ are i.i.d. with variance $\sigma^2$.
3. What is the correct protocol?

---

### Solution 2

1. **Proof of Dependency:**
   The standardized training sample is:
   $$z_i = \frac{x_i - \hat{\mu}_{\text{global}}}{\hat{\sigma}_{\text{global}}}$$
   Writing out $\hat{\mu}_{\text{global}}$ explicitly:
   $$\hat{\mu}_{\text{global}} = \frac{1}{N}\sum_{j \in \mathcal{D}_{\text{train}}} x_j + \frac{1}{N}\sum_{m \in \mathcal{D}_{\text{test}}} x_m$$
   Therefore:
   $$z_i = \frac{x_i - \left(\frac{1}{N}\sum_{j \in \text{train}} x_j + \frac{1}{N}\sum_{m \in \text{test}} x_m\right)}{\hat{\sigma}_{\text{global}}}$$
   The partial derivative of the training feature $z_i$ with respect to the test sample $x_k$ ($k \in \text{test}$) is:
   $$\frac{\partial z_i}{\partial x_k} = -\frac{1}{N \hat{\sigma}_{\text{global}}} \ne 0$$
   Because the partial derivative is non-zero, every training feature value is a direct mathematical function of the test set, violating sample independence.

2. **Covariance Calculation:**
   Treating $\sigma$ as approximately fixed for first-order analysis:
   $$\text{Cov}(z_i, x_k) = \text{Cov}\left(\frac{x_i - \frac{1}{N}\sum_{j=1}^N x_j}{\sigma}, x_k\right)$$
   Expanding the covariance operator:
   $$\text{Cov}(z_i, x_k) = \frac{1}{\sigma}\left(\text{Cov}(x_i, x_k) - \frac{1}{N}\sum_{j=1}^N \text{Cov}(x_j, x_k)\right)$$
   Since samples are i.i.d., $\text{Cov}(x_i, x_k) = 0$ for $i \ne k$.
   For the summation, only the term where $j = k$ is non-zero, yielding $\text{Cov}(x_k, x_k) = \text{Var}(x_k) = \sigma^2$:
   $$\text{Cov}(z_i, x_k) = \frac{1}{\sigma}\left(0 - \frac{1}{N}\sigma^2\right) = -\frac{\sigma}{N}$$
   The covariance is strictly negative ($-\frac{\sigma}{N}$), introducing a negative correlation between training instances and test observations.

3. **Correct Protocol:**
   Compute $\hat{\mu}_{\text{train}}$ and $\hat{\sigma}_{\text{train}}$ strictly using indices $i \in \mathcal{D}_{\text{train}}$. Apply those fixed scalar values to transform $\mathcal{D}_{\text{test}}$ without re-estimating them.

---

### Problem 3: Stratified K-Fold vs. Standard K-Fold Probability Analysis

Consider a rare medical diagnostic dataset with $N = 100$ total patient samples. Exactly $M = 5$ patients test positive for a rare disease ($y = 1$), and $95$ test negative ($y = 0$). We perform standard 5-Fold Cross Validation where samples are assigned to 5 folds of size 20 uniformly at random without replacement.

**Questions:**
1. What is the probability that a randomly chosen fold contains **zero** positive disease cases?
2. What is the probability that at least one fold out of the five contains **zero** positive cases?
3. How does Stratified K-Fold resolve this?

---

### Solution 3

1. **Probability a Single Fold has 0 Positives:**
   The fold selects $k = 20$ samples from $N = 100$ without replacement. The number of positives follows a Hypergeometric distribution:
   $$P(X = 0) = \frac{\binom{5}{0}\binom{95}{20}}{\binom{100}{20}}$$
   Calculating:
   $$\binom{95}{20} = \frac{95!}{20! \cdot 75!}, \quad \binom{100}{20} = \frac{100!}{20! \cdot 80!}$$
   $$\frac{\binom{95}{20}}{\binom{100}{20}} = \frac{80 \times 79 \times 78 \times 77 \times 76}{100 \times 99 \times 98 \times 97 \times 96} \approx 0.3188 \quad (31.88\%)$$
   There is an approximately 31.9% chance that any specific validation fold contains zero disease cases!

2. **Probability that at least one fold has 0 positives:**
   Partitioning 5 items into 5 bins of size 20. The only way *no* bin has 0 positives is if each of the 5 bins receives exactly 1 positive!
   The probability that all 5 folds receive exactly 1 positive is:
   $$P(\text{all folds get 1}) = \frac{\binom{20}{1}\binom{20}{1}\binom{20}{1}\binom{20}{1}\binom{20}{1}}{\binom{100}{5}} = \frac{20^5}{\binom{100}{5}} = \frac{3,200,000}{75,287,520} \approx 0.0425 \quad (4.25\%)$$
   Therefore, the probability that *at least one fold receives zero positives* is:
   $$P(\ge 1 \text{ fold has 0 positives}) = 1 - 0.0425 = 0.9575 \quad (95.75\%)$$
   Under standard random K-Fold, **over 95% of the time**, at least one fold will have zero positive cases, making sensitivity / recall completely undefined ($0/0$) on that fold!

3. **Stratified Solution:**
   Stratified K-Fold partitions positive and negative classes separately:
   - Positives ($5$ items): assigns exactly $5/5 = 1$ positive item to each fold.
   - Negatives ($95$ items): assigns exactly $95/5 = 19$ negative items to each fold.
   Every fold is guaranteed to have exactly 1 positive and 19 negatives (5.0% prevalence), eliminating fold variance.
