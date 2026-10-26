# Topic 10.2: Exercises & Solutions — Feature Engineering & Scaling

---

## Problem Set Overview

These exercises explore the mathematical mechanics of feature scaling, the optimization theory behind gradient descent conditioning, and categorical encoding trade-offs.

---

### Problem 1: The Dummy Variable Trap & Matrix Singularity

Suppose a linear regression model has an intercept term $w_0$ (associated with a constant column of ones $x_{i,0} = 1$) and a categorical feature $C$ with $K = 3$ mutually exclusive categories: $\{\text{Red}, \text{Green}, \text{Blue}\}$.

An engineer encodes this categorical variable using full One-Hot Encoding into 3 binary columns:
$$x_{i,1} = \mathbb{I}(C_i = \text{Red}), \quad x_{i,2} = \mathbb{I}(C_i = \text{Green}), \quad x_{i,3} = \mathbb{I}(C_i = \text{Blue})$$

The design matrix $X \in \mathbb{R}^{N \times 4}$ has columns:
$$X = \begin{bmatrix} \mathbf{1} & \mathbf{x}_1 & \mathbf{x}_2 & \mathbf{x}_3 \end{bmatrix}$$

**Questions:**
1. Prove that the matrix $X^T X$ is singular (not invertible), meaning the Ordinary Least Squares normal equations $(X^T X)w = X^T y$ cannot be solved uniquely.
2. What are the two distinct ways to resolve this singularity?
3. How does $L_2$ regularization (Ridge regression) affect this singularity?

---

### Solution 1

1. **Proof of Singularity:**
   Because the three categories are mutually exclusive and exhaustive for every observation $i$:
   $$x_{i,1} + x_{i,2} + x_{i,3} = 1, \quad \forall i \in \{1, \dots, N\}$$
   In vector notation across all $N$ rows:
   $$\mathbf{x}_1 + \mathbf{x}_2 + \mathbf{x}_3 = \mathbf{1}$$
   Rearranging:
   $$1 \cdot \mathbf{1} - 1 \cdot \mathbf{x}_1 - 1 \cdot \mathbf{x}_2 - 1 \cdot \mathbf{x}_3 = \mathbf{0}$$
   Let $v = \begin{bmatrix} 1 & -1 & -1 & -1 \end{bmatrix}^T \ne \mathbf{0}$. We compute:
   $$X v = 1 \cdot \mathbf{1} - \mathbf{x}_1 - \mathbf{x}_2 - \mathbf{x}_3 = \mathbf{0}$$
   Because there exists a non-zero vector $v$ such that $X v = \mathbf{0}$, the columns of $X$ are linearly dependent.
   Consequently, $\text{rank}(X) \le 3 < 4$.
   For the Gram matrix $X^T X \in \mathbb{R}^{4 \times 4}$:
   $$(X^T X) v = X^T (X v) = X^T \mathbf{0} = \mathbf{0}$$
   Hence, $v$ is an eigenvector of $X^T X$ with eigenvalue $\lambda = 0$.
   Therefore, $\det(X^T X) = 0$, proving that $X^T X$ is non-invertible (singular). $\blacksquare$

2. **Resolutions:**
   - **Drop-one Dummy Coding:** Drop one category column (e.g. `Red`) so that the matrix contains only $[ \mathbf{1}, \mathbf{x}_{\text{Green}}, \mathbf{x}_{\text{Blue}} ]$. The intercept then represents the baseline mean for `Red`, and the remaining coefficients represent differential offsets.
   - **Fit without Intercept:** Omit the constant column of ones ($\mathbf{1}$), allowing each category to receive its own distinct baseline mean weight.

3. **Ridge Regularization Impact:**
   Ridge regression adds $\lambda I$ to the normal equations: $(X^T X + \lambda I) w = X^T y$.
   For any $\lambda > 0$, all eigenvalues shift by $\lambda$:
   $$\lambda_j(X^T X + \lambda I) = \lambda_j(X^T X) + \lambda \ge \lambda > 0$$
   This eliminates exact singularity, making the matrix strictly positive definite and invertible even with full one-hot encoding.

---

### Problem 2: Outlier Breakdown Analysis: StandardScaler vs. RobustScaler

Consider a 1-dimensional dataset of $N = 21$ salary values:
$$X = [40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, M]$$
where $20$ salaries lie symmetrically between $40$ and $60$ (thousand dollars), and the 21st salary is an extreme billionaire outlier $M = 10,000$.

**Questions:**
1. Compute the sample median $Q_2$, first quartile $Q_1$, third quartile $Q_3$, and $\text{IQR}$ of $X$.
2. Compute the transformed value of the median salary ($x = 50$) under `RobustScaler`. Does it change if $M \to \infty$?
3. Compute the sample mean $\bar{x}$ and standard deviation $s$ as functions of $M$. Show what happens to the standard score of $x = 50$ as $M \to \infty$.

---

### Solution 2

1. **Order Statistics:**
   The dataset has $N = 21$ sorted values.
   - Median index: $\frac{21 + 1}{2} = 11^{\text{th}}$ value: $Q_2 = 50$.
   - Lower half (first 10 values: 40 to 49): $Q_1 = \frac{44 + 45}{2} = 44.5$.
   - Upper half (values 12 to 21: 51 to $M$): $Q_3 = \frac{56 + 57}{2} = 56.5$.
   - $\text{IQR} = Q_3 - Q_1 = 56.5 - 44.5 = 12.0$.

2. **Robust Scaler for $x = 50$:**
   $$x_{\text{robust}} = \frac{x - Q_2}{\text{IQR}} = \frac{50 - 50}{12.0} = 0.0$$
   Because $M$ only occupies the top rank (the 21st position), whether $M = 70$ or $M = 10,000,000$, the 11th, 5th, and 16th rank values do not budge. The median point is perfectly centered at $0.0$ regardless of $M$.

3. **StandardScaler Breakdown:**
   The sum of the first 20 numbers is $\sum_{i=1}^{20} (40 + i - 1) = 20 \times 40 + \frac{19 \times 20}{2} = 800 + 190 = 990$.
   With outlier $M$:
   $$\bar{x} = \frac{990 + M}{21}$$
   As $M \to \infty$:
   $$\bar{x} \approx \frac{M}{21} \to \infty$$
   The variance is dominated by $(M - \bar{x})^2$:
   $$s \approx \sqrt{\frac{(M - M/21)^2}{21}} = \frac{20}{21\sqrt{21}} M \approx 0.2078 M$$
   The standardized Z-score of typical salary $x = 50$ is:
   $$z(50) = \frac{50 - \frac{990+M}{21}}{s} \approx \frac{-M/21}{0.2078 M} \approx -0.229$$
   Notice that as $M \to \infty$, the standard deviation grows at rate $\mathcal{O}(M)$, shrinking the difference between all 20 normal people towards zero! A salary of \$40,000 and a salary of \$60,000 become indistinguishable under Z-score scaling when an extreme outlier is present.

---

### Problem 3: Target Encoding with Empirical Bayes Smoothing

In a click-through rate (CTR) prediction system, the feature `publisher_domain` has thousands of unique domains.
The dataset has an overall global average CTR of $\mu_{\text{global}} = 0.05$ (5%).
We configure Micci-Barreca target encoding with smoothing parameter $m = 10$:
$$S(c) = \frac{n_c \bar{y}_c + m \mu_{\text{global}}}{n_c + m}$$

**Questions:**
1. Domain A is a tiny blog with $n_A = 2$ impressions, both of which resulted in clicks ($\bar{y}_A = 1.0$). Compute $S(A)$.
2. Domain B is a massive news portal with $n_B = 10,000$ impressions and $600$ clicks ($\bar{y}_B = 0.06$). Compute $S(B)$.
3. Explain how the hyperparameter $m$ functions as a Bayesian prior sample size.

---

### Solution 3

1. **Domain A (Small sample $n_A = 2$):**
   $$S(A) = \frac{2 \times 1.0 + 10 \times 0.05}{2 + 10} = \frac{2.0 + 0.5}{12} = \frac{2.5}{12} \approx 0.2083 \quad (20.83\%)$$
   Without smoothing, raw target encoding would assign $1.0$ (100% CTR), grossly overestimating click probability for this tiny site based on 2 clicks. The smoothing pulls it down towards the global prior of $0.05$.

2. **Domain B (Large sample $n_B = 10,000$):**
   $$S(B) = \frac{10,000 \times 0.06 + 10 \times 0.05}{10,000 + 10} = \frac{600 + 0.5}{10,010} = \frac{600.5}{10,010} \approx 0.05999 \quad (6.00\%)$$
   Because $n_B \gg m$, the empirical observation overwhelms the prior, preserving the true signal.

3. **Bayesian Interpretation:**
   The equation is identical to the posterior mean of a Beta-Binomial conjugate model where we begin with a prior equivalent to having already observed $m$ pseudo-samples with mean $\mu_{\text{global}}$. The hyperparameter $m$ defines the threshold of evidence required before we trust an observed category average over the population baseline.
