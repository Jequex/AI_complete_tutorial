# Topic 10.2: Feature Engineering, Scaling & Categorical Representations

---

## 1. Executive Summary & The 12-Year-Old Analogy

Machine learning algorithms do not understand physical units. They do not know what a square foot, a dollar, an age in years, or a zip code means; they see only floating-point numbers.

If one feature is measured in square feet (ranging from $600$ to $5,000$) and another feature is measured in bedrooms (ranging from $1$ to $5$), optimization algorithms such as Gradient Descent will treat a unit step in bedrooms identically to a unit step in square feet. This causes the loss surface to become an agonizingly narrow, eccentric canyon where gradients bounce violently off the walls instead of progressing toward the minimum.

```
      UNSCALED LOSS SURFACE                         STANDARDIZED LOSS SURFACE
   (Condition number κ >> 10,000)                     (Condition number κ ≈ 1)
 ┌──────────────────────────────────────┐       ┌──────────────────────────────────────┐
 │     \   /\   /\   /                  │       │              . - ~ ~ - .             │
 │      \ /  \ /  \ /    Oscillations   │       │          .-'             '-.         │
 │  ─────X────X────X───► along steep    │       │        .'         │         '.       │
 │      / \  / \  / \    canyon walls   │       │       /           │           \      │
 │     /   \/   \/   \                  │       │      │      ──────┼──────►     │     │
 │                                      │       │       \      Direct path      /      │
 │  Steep: Square Feet (Scale ~10^3)    │       │        '.   to minimum      .'       │
 │  Flat:  Bedrooms (Scale ~10^0)       │       │          '-.             .-'         │
 └──────────────────────────────────────┘       │              ' - . . - '             │
                                                └──────────────────────────────────────┘
```

---

### The 12-Year-Old Analogy: Converting Inches and Miles to the Same Scale

Imagine a team of two builders constructing a bridge across a river:
- Builder Bob measures all his distances in **Miles** (numbers like $0.001$, $0.004$).
- Builder Alice measures all her distances in **Inches** (numbers like $120,000$, $480,000$).

When the boss asks, *"Add these two blueprints together to calculate the total steel needed,"* adding Bob's $0.002$ to Alice's $240,000$ yields $240,000.002$. Bob's contribution is completely wiped out by Alice's massive numbers, even though Bob might have measured the most crucial section of the bridge!

**Feature Scaling** is simply giving Alice and Bob the exact same ruler (standardizing their units so both have a mean of 0 and a spread of 1) before allowing the computer to compute distances, angles, or gradients.

---

## 2. Mathematical Symbol Decoder

| Symbol | Mathematical Domain | Description & Engineering Meaning |
| :--- | :--- | :--- |
| $x \in \mathbb{R}^N$ | Vector | Column vector representing a continuous feature across $N$ instances |
| $\mu = \mathbb{E}[X]$ | Scalar | Arithmetic sample mean $\frac{1}{N}\sum_{i=1}^N x_i$ |
| $\sigma = \sqrt{\text{Var}(X)}$ | Scalar | Sample standard deviation $\sqrt{\frac{1}{N}\sum_{i=1}^N (x_i - \mu)^2}$ |
| $x_{\min}, x_{\max}$ | Scalars | Empirical minimum and maximum observed in the training split |
| $Q_1, Q_2, Q_3$ | Scalars | 25th percentile, Median (50th), and 75th percentile |
| $\text{IQR} = Q_3 - Q_1$ | Scalar | Interquartile Range (middle 50% data dispersion) |
| $H = \nabla^2 \mathcal{L}(\theta)$ | $\mathbb{R}^{d \times d}$ matrix | Hessian matrix of loss function second derivatives |
| $\kappa(H) = \frac{\lambda_{\max}}{\lambda_{\min}}$ | $\mathbb{R}_{\ge 1}$ | Condition number of the Hessian: ratio of max to min curvature |
| $S_c$ | Scalar | Smoothed target encoding for category $c$ |
| $m \in \mathbb{R}_{>0}$ | Hyperparameter | Smoothing weight / prior pseudo-count in Bayesian target encoding |

---

## 3. Mathematical Foundations of Feature Scaling

### 3.1 Z-Score Standardization
Standardization shifts the data distribution to zero mean and unit variance:
$$z_i = \frac{x_i - \mu_{\text{train}}}{\sigma_{\text{train}}}$$

- **Key Properties:**
  - Preserves zero entries if mean is zero, but destroys sparsity in sparse matrices unless handled carefully.
  - Does not bound values to a finite interval (values can extend to $\pm 3\sigma, \pm 5\sigma$).
  - Optimal default for algorithms assuming normally distributed inputs or gradient-based optimizers.

### 3.2 Min-Max Normalization
Min-Max scaling compresses or stretches features into a bounded range $[a, b]$, typically $[0, 1]$:
$$x'_i = a + \frac{x_i - x_{\min}}{x_{\max} - x_{\min}}(b - a)$$

- **Key Properties:**
  - Guarantees bounded interval $[0, 1]$.
  - Highly sensitive to outliers: a single massive outlier compresses all standard data into a microscopic band near $0.0$.
  - Essential for neural network layers with bounded activations (e.g., sigmoid, image pixel intensities $[0, 255] \to [0, 1]$).

### 3.3 Robust Scaling (Median & IQR)
When data contains heavy outliers, both $\mu$ and $\sigma$ become distorted. The **Robust Scaler** uses rank-based statistics:
$$x_{\text{robust}} = \frac{x_i - \text{Median}(X)}{\text{IQR}(X)} = \frac{x_i - Q_2}{Q_3 - Q_1}$$
Because percentiles depend on ranks rather than magnitude sums, extreme outliers leave the median and IQR unaffected.

---

## 4. Why Scaling Speeds Up Gradient Descent: The Hessian Proof

Consider Mean Squared Error for linear regression:
$$\mathcal{L}(w) = \frac{1}{2N}\|Xw - y\|_2^2$$
The Hessian matrix (second partial derivatives with respect to weights $w$) is:
$$H = \nabla^2 \mathcal{L}(w) = \frac{1}{N} X^T X$$

The eigenvalues $\lambda_1 \le \lambda_2 \le \dots \le \lambda_d$ of $H$ determine the principal curvatures of the quadratic loss bowl.

The maximum stable learning rate for gradient descent without divergence is:
$$\eta_{\max} < \frac{2}{\lambda_{\max}(H)}$$

Meanwhile, the rate of convergence along the slowest eigen-direction is governed by $\lambda_{\min}(H)$:
$$\|w_t - w^*\| \le \left(1 - \eta \lambda_{\min}\right)^t \|w_0 - w^*\|$$

Substituting $\eta \approx \frac{1}{\lambda_{\max}}$:
$$\text{Convergence Factor} = 1 - \frac{\lambda_{\min}}{\lambda_{\max}} = 1 - \frac{1}{\kappa(H)}$$
where $\kappa(H) = \frac{\lambda_{\max}}{\lambda_{\min}}$ is the **Condition Number**.

- **Unscaled features:** If feature 1 has variance $10^6$ and feature 2 has variance $1$, then $\kappa \approx 10^6$. The convergence factor is $1 - 10^{-6} = 0.999999$, requiring millions of iterations to converge!
- **Standardized features:** All features have variance $1.0$, rotating and conditioning $X^T X$ so that $\kappa \approx 1$. Gradient descent steps directly toward the global minimum in a handful of steps.

---

## 5. Categorical Feature Encoding

Categorical variables represent discrete symbols (e.g. `City`, `Device_Type`, `Blood_Type`).

```
                              CATEGORICAL ENCODING
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        ▼                              ▼                              ▼
  ONE-HOT ENCODING              ORDINAL ENCODING               TARGET ENCODING
  (Nominal categories,          (Ordered categories,          (High-cardinality,
   low cardinality < 20)         e.g., Low < Med < High)       e.g., Zip Codes, IDs)
```

### 5.1 One-Hot Encoding (OHE)
Maps category $c \in \{1, \dots, K\}$ to a sparse binary vector $e_c \in \{0, 1\}^K$.
- **The Dummy Variable Trap:** In linear models with an intercept term $b$, including all $K$ columns introduces exact multicollinearity ($\sum_{j=1}^K x_{i,j} = 1$). To avoid a singular $X^T X$ matrix, drop one reference category ($K - 1$ columns).
- **Limitation:** Cardinality explosion. If a feature has $50,000$ unique zip codes, OHE expands the matrix by $50,000$ sparse columns.

### 5.2 Ordinal Encoding
Maps categories to monotonically increasing integers:
$$\text{Size} \in \{\text{'Small'}, \text{'Medium'}, \text{'Large'}\} \implies [0, 1, 2]$$
- **Caution:** Only valid when an intrinsic, monotonic ordering exists. Using ordinal encoding for nominal categories (e.g., $\text{'Cat'}=0, \text{'Dog'}=1, \text{'Fish'}=2$) imposes a false metric distance: the model believes a Fish is twice as large as a Dog!

### 5.3 Target Encoding with Empirical Bayes Smoothing
For high-cardinality nominal features (e.g. zip codes, merchant IDs), **Target Encoding** replaces each category with the average target value observed for that category:
$$\bar{y}_c = \frac{1}{n_c}\sum_{i \in \text{Category } c} y_i$$
However, rare categories ($n_c = 1$ or $2$) overfit instantly to noise. We apply **Micci-Barreca smoothing**:
$$S(c) = \lambda(n_c) \bar{y}_c + (1 - \lambda(n_c)) \mu_{\text{global}}$$
where the blending weight is:
$$\lambda(n_c) = \frac{n_c}{n_c + m}$$
- When $n_c \gg m$: $\lambda \to 1$, trusting the category's empirical mean.
- When $n_c \ll m$: $\lambda \to 0$, shrinking the estimate to the global dataset prior $\mu_{\text{global}}$.

---

## 6. Non-Linear Transformations & Feature Synthesis

### 6.1 Log Transforms for Right-Skewed Distributions
Variables such as income, prices, web traffic, and file sizes follow heavy-tailed power-law distributions. Extreme values pull linear regression lines away from the median population.
$$\tilde{x} = \log(1 + x) \quad (\text{log1p, safe for } x \ge 0)$$
This contracts extreme values and expands compressed lower values, transforming skewed data into an approximately Gaussian bell curve.

### 6.2 Interaction Terms & Polynomial Expansion
Linear models assume features contribute additively: $\hat{y} = w_1 x_1 + w_2 x_2 + b$. 
If the effect of feature $x_1$ depends on the level of $x_2$, we synthesize an **interaction feature**:
$$x_3 = x_1 \times x_2$$
- Example: Predicting real estate value: `Area` $\times$ `CrimeRate`.
- Quadratic expansion adds $[x_1^2, x_2^2, x_1 x_2]$, enabling linear models to capture parabolic curvature and saddle surfaces.

---

## 7. Which Algorithms Require Scaling?

| Algorithm Class | Scaling Required? | Mathematical Reason |
| :--- | :---: | :--- |
| **Linear / Logistic Regression** | **YES** | Regularization ($L_1/L_2$) penalizes large coefficients unfairly; GD converges slowly |
| **Neural Networks (MLP, CNN, RNN)** | **YES** | Activations saturate; gradients vanish or explode if inputs have large magnitude |
| **K-Nearest Neighbors (KNN)** | **CRITICAL** | Euclidean distance $d(u,v) = \sqrt{\sum (u_j - v_j)^2}$ is completely dominated by large-scale features |
| **Principal Component Analysis (PCA)** | **CRITICAL** | Eigenvectors maximize variance; unscaled large-variance columns hijack all principal components |
| **Support Vector Machines (SVM)** | **YES** | Margin optimization and RBF kernels $e^{-\gamma \|u-v\|^2}$ rely on Euclidean distances |
| **Decision Trees / Random Forests / XGBoost** | **NO** | Splits evaluate ordering thresholds ($x_j \le \theta$), which are invariant to monotonic transformations |
