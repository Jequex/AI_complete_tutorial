# Topic 4.4: Exercises & Solutions — Expectation, Variance, and Covariance

Welcome to the problem set for **Topic 4.4: Expectation, Variance, Covariance & Correlation**.

Grab a pencil and notebook. Work through each problem before revealing the collapsible solutions below.

---

## 📝 Problem Set Overview

| Problem | Focus | Difficulty | Practical AI Connection |
| :---: | :--- | :---: | :--- |
| **1** | **Proof that Zero Covariance Does NOT Imply Independence** | Medium | Understanding why linear models fail on nonlinear relationships |
| **2** | **Derivation of He (Kaiming) Weight Initialization Variance** | Medium | Preventing vanishing and exploding gradients in deep ReLU networks |
| **3** | **Covariance Matrix & Pearson Correlation by Hand** | Challenging | Dimensionality reduction, PCA, and multi-feature standardization |

---

## Problem 1: Zero Covariance Does NOT Imply Independence

### 🧒 The Context: Maya's Non-Linear Parabola Trap
It is well known that if two variables are independent, their covariance is zero:
$$X \perp Y \implies \text{Cov}(X, Y) = 0$$
However, many beginners falsely assume the reverse is true!
Covariance measures only *linear* relationships. A non-linear relationship can have exact zero covariance while being $100\%$ dependent.

Let $X$ be uniformly distributed on $[-1, 1]$:
$$f_X(x) = \frac{1}{2} \quad \text{for } x \in [-1, 1]$$
Let $Y = X^2$.

### Tasks:
1. Compute the expected value $\mathbb{E}[X]$.
2. Compute the expected value $\mathbb{E}[Y]$.
3. Compute the expected value of the product $\mathbb{E}[XY]$.
4. Compute the covariance $\text{Cov}(X, Y) = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]$.
5. Explain why $X$ and $Y$ are completely dependent despite having zero covariance.

---

<details>
<summary>Click to view solution for Problem 1</summary>

### Detailed Step-by-Step Solution:

#### Step 1: Compute $\mathbb{E}[X]$
Because the uniform distribution is symmetric around $0$:
$$\mathbb{E}[X] = \int_{-1}^1 x \left(\frac{1}{2}\right) dx = \left[ \frac{x^2}{4} \right]_{-1}^1 = \frac{1}{4} - \frac{1}{4} = \mathbf{0.0}$$

#### Step 2: Compute $\mathbb{E}[Y]$
Since $Y = X^2$:
$$\mathbb{E}[Y] = \mathbb{E}[X^2] = \int_{-1}^1 x^2 \left(\frac{1}{2}\right) dx = \left[ \frac{x^3}{6} \right]_{-1}^1 = \frac{1}{6} - \left(-\frac{1}{6}\right) = \mathbf{\frac{2}{6} = \frac{1}{3}}$$

#### Step 3: Compute $\mathbb{E}[XY]$
$$XY = X(X^2) = X^3$$
$$\mathbb{E}[XY] = \mathbb{E}[X^3] = \int_{-1}^1 x^3 \left(\frac{1}{2}\right) dx = \left[ \frac{x^4}{8} \right]_{-1}^1 = \frac{1}{8} - \frac{1}{8} = \mathbf{0.0}$$
*(The integral of any odd function over a symmetric interval is always zero!)*

#### Step 4: Compute Covariance
$$\text{Cov}(X, Y) = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y] = 0 - (0 \times \frac{1}{3}) = \mathbf{0.0}$$

#### Step 5: Why They Are Dependent
$Y$ is a deterministic function of $X$ ($Y = X^2$). If Maya tells you $X = 0.5$, you know with $100\%$ certainty that $Y = 0.25$.
They are completely dependent, yet because their relationship is a symmetric parabola rather than a straight line, linear covariance sees zero correlation!
</details>

---

## Problem 2: Derivation of the He (Kaiming) Weight Initialization

### 🧒 The Context: Timmy's 100-Layer Neural Network
When training a deep neural network with ReLU activations, if initial weights are chosen slightly too large, activations explode to infinity; if slightly too small, activations shrink to zero.
Consider a single neuron:
$$y = \sum_{i=1}^{D_{\text{in}}} w_i x_i$$
Assume weights $w_i$ and inputs $x_i$ are independent with mean zero ($\mathbb{E}[w_i] = 0, \mathbb{E}[x_i] = 0$).

### Tasks:
1. Show that the variance of the linear sum before activation is:
   $$\text{Var}(y) = D_{\text{in}} \text{Var}(w) \text{Var}(x)$$
2. Because a ReLU activation $\text{ReLU}(z) = \max(0, z)$ zeroes out half of all values, it halves the signal variance:
   $$\text{Var}(\text{ReLU}(y)) = \frac{1}{2} \text{Var}(y)$$
3. Solve for the required weight variance $\text{Var}(w)$ such that the signal variance remains preserved from layer to layer: $\text{Var}(\text{ReLU}(y)) = \text{Var}(x)$.

---

<details>
<summary>Click to view solution for Problem 2</summary>

### Detailed Step-by-Step Solution:

#### Step 1: Variance of Independent Product Terms
For two independent zero-mean variables $w_i$ and $x_i$:
$$\mathbb{E}[w_i x_i] = \mathbb{E}[w_i] \mathbb{E}[x_i] = 0 \times 0 = 0$$
$$\text{Var}(w_i x_i) = \mathbb{E}[(w_i x_i)^2] - 0 = \mathbb{E}[w_i^2] \mathbb{E}[x_i^2] = \text{Var}(w_i) \text{Var}(x_i)$$

Summing across all $D_{\text{in}}$ independent input features:
$$\text{Var}(y) = \sum_{i=1}^{D_{\text{in}}} \text{Var}(w_i x_i) = \mathbf{D_{\text{in}} \text{Var}(w) \text{Var}(x)}$$

#### Step 2: Post-ReLU Variance Halving
Because the ReLU function sets all negative values to zero, for a symmetric distribution around zero, exactly half the variance is lost:
$$\text{Var}(\text{ReLU}(y)) = \frac{1}{2} \text{Var}(y) = \frac{1}{2} D_{\text{in}} \text{Var}(w) \text{Var}(x)$$

#### Step 3: Preserving Variance Across Layers
To prevent activations from exploding or dying as signals pass through 100 layers, we require:
$$\text{Var}(\text{ReLU}(y)) = \text{Var}(x)$$
$$\frac{1}{2} D_{\text{in}} \text{Var}(w) \text{Var}(x) = \text{Var}(x)$$
Divide both sides by $\text{Var}(x)$:
$$\frac{1}{2} D_{\text{in}} \text{Var}(w) = 1 \implies \mathbf{\text{Var}(w) = \frac{2}{D_{\text{in}}}}$$

$$\mathbf{\sigma_w = \sqrt{\frac{2}{D_{\text{in}}}}}$$

**Deep Learning Takeaway:** This is the exact formula for **He / Kaiming Initialization** (`torch.nn.init.kaiming_normal_`), published by Kaiming He in 2015, which enabled deep ResNets to train stably for the first time in history!
</details>

---

## Problem 3: Covariance Matrix & Pearson Correlation by Hand

### 🧒 The Context: Maya & Timmy's 3-Feature Dataset
We have $N = 3$ centered feature vectors for three students:
- Feature 1: Study hours $X = [2, 0, -2]^T$
- Feature 2: Practice quizzes $Y = [4, 0, -4]^T$
- Feature 3: Sleep hours $Z = [-2, 0, 2]^T$

### Tasks:
1. Form data matrix $A \in \mathbb{R}^{3 \times 3}$ and calculate the $3 \times 3$ sample covariance matrix $\Sigma = \frac{1}{N - 1} A^T A$.
2. Compute the Pearson correlation matrix $P$.
3. Interpret the correlation between Feature 1 and Feature 2, and between Feature 1 and Feature 3.

---

<details>
<summary>Click to view solution for Problem 3</summary>

### Detailed Step-by-Step Solution:

#### Step 1: Compute Sample Covariance Matrix $\Sigma = \frac{1}{3 - 1} A^T A = \frac{1}{2} A^T A$
Matrix $A$ with observations as rows:
$$A = \begin{bmatrix} 2 & 4 & -2 \\ 0 & 0 & 0 \\ -2 & -4 & 2 \end{bmatrix}$$

Multiplying $A^T A$:
- Entry $(1, 1)$: $2^2 + 0^2 + (-2)^2 = 4 + 0 + 4 = 8$
- Entry $(1, 2)$: $(2)(4) + 0 + (-2)(-4) = 8 + 8 = 16$
- Entry $(1, 3)$: $(2)(-2) + 0 + (-2)(2) = -4 - 4 = -8$
- Entry $(2, 2)$: $4^2 + 0^2 + (-4)^2 = 16 + 16 = 32$
- Entry $(2, 3)$: $(4)(-2) + 0 + (-4)(2) = -8 - 8 = -16$
- Entry $(3, 3)$: $(-2)^2 + 0^2 + 2^2 = 4 + 4 = 8$

Dividing by $N - 1 = 2$:
$$\Sigma = \frac{1}{2} \begin{bmatrix} 8 & 16 & -8 \\ 16 & 32 & -16 \\ -8 & -16 & 8 \end{bmatrix} = \mathbf{\begin{bmatrix} 4.0 & 8.0 & -4.0 \\ 8.0 & 16.0 & -8.0 \\ -4.0 & -8.0 & 4.0 \end{bmatrix}}$$

#### Step 2: Compute Pearson Correlation Matrix $P$
Standard deviations:
- $\sigma_1 = \sqrt{\Sigma_{11}} = \sqrt{4} = 2.0$
- $\sigma_2 = \sqrt{\Sigma_{22}} = \sqrt{16} = 4.0$
- $\sigma_3 = \sqrt{\Sigma_{33}} = \sqrt{4} = 2.0$

Correlation entries $\rho_{ij} = \frac{\Sigma_{ij}}{\sigma_i \sigma_j}$:
- $\rho_{12} = \frac{8.0}{2.0 \times 4.0} = \frac{8.0}{8.0} = \mathbf{+1.00}$
- $\rho_{13} = \frac{-4.0}{2.0 \times 2.0} = \frac{-4.0}{4.0} = \mathbf{-1.00}$
- $\rho_{23} = \frac{-16.0}{4.0 \times 2.0} = \frac{-16.0}{8.0} = \mathbf{-1.00}$

$$P = \begin{bmatrix} +1.00 & +1.00 & -1.00 \\ +1.00 & +1.00 & -1.00 \\ -1.00 & -1.00 & +1.00 \end{bmatrix}$$

#### Step 3: Interpretation
- Study hours and practice quizzes have **perfect positive correlation ($\rho = +1.00$)**: students who study more do proportionately more quizzes!
- Study hours and sleep hours have **perfect negative correlation ($\rho = -1.00$)**: students who study more sacrifice sleep in exact proportion!
</details>
