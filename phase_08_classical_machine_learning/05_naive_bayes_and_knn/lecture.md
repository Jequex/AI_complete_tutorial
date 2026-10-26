# Topic 8.5: Naive Bayes & K-Nearest Neighbors (KNN)

---

## 1. Executive Summary & The 12-Year-Old Analogy

So far, we have studied parametric models (Linear Regression, Logistic Regression, SVMs) that compress training data into a fixed set of weights ($w$ and $b$).

Topic 8.5 explores two classic, foundational, non-traditional paradigms:
1. **Naive Bayes:** A **generative probabilistic model** that applies Bayes' Theorem with an audacious simplifying assumption—that all input features are conditionally independent given the class label.
2. **K-Nearest Neighbors (KNN):** A **non-parametric instance-based lazy learner** that performs zero training computation, memorizing the dataset and making predictions on-the-fly by querying the most similar historical instances.

```
       NAIVE BAYES (Generative)                    KNN (Instance-Based)
   Model P(X, Y) via Bayes' Rule              Look at k closest neighbors
 
    +-------+      P(x_1 | y)                   * (Class A)
    | Class | ---> P(x_2 | y)               *       *
    |   y   | ---> P(x_3 | y)                   ? (Query point)
    +-------+                                   o     o (Class B)
                                                   o
```

---

### The 12-Year-Old Analogy

#### Part A: Naive Bayes (The Mailroom Spam Detective)
Imagine you work in the school mailroom sorting letters into **Spam** ($y=1$) or **Friendly** ($y=0$).
- You make a spreadsheet of words. You notice that 80% of Spam letters contain the word *"FREE"*, and 70% contain *"WINNER"*. In Friendly letters, *"FREE"* only appears 1% of the time.
- When a new letter arrives saying *"FREE WINNER TODAY"*, you calculate the likelihood by multiplying the word probabilities together: $P(\text{"FREE"} \mid \text{Spam}) \times P(\text{"WINNER"} \mid \text{Spam})$.
- **Why is it called "Naive"?** Because in human language, words don't appear independently! If a letter says *"FREE"*, it is much more likely to also say *"WINNER"*. Naive Bayes cheerfully ignores this grammar connection and pretends each word was rolled on an independent pair of dice. Surprisingly, despite this silly assumption, it sorts email with astonishing accuracy!
- **Laplace Smoothing (The "Never Say Never" Rule):** If a brand-new word *"KANGAROO"* appears in an email for the first time, its probability in Spam would be 0. If you multiply by zero, the entire score becomes zero, erasing all other evidence! To prevent this, you pretend you saw every word in the dictionary at least once before (Add-1 Smoothing).

#### Part B: K-Nearest Neighbors ("Tell Me Who Your Neighbors Are")
Imagine you transfer to a brand-new middle school and walk into the cafeteria at lunchtime. You don't know any of the clubs yet, but you want to identify which group is sitting at Table 4.
- You look at the **$k = 5$ closest kids** sitting at that table.
- If 4 of them are wearing soccer jerseys and cleats, and 1 is holding a trumpet, you conclude by a 4-to-1 majority vote: *"This is the Soccer Team table!"*
- There is **no test or studying required beforehand** (Lazy Learning). The algorithm just memorizes where everyone is sitting and measures distances with a ruler.
- **The Curse of Dimensionality:** If you only compare students by 2 traits (age and height), it is easy to find close neighbors. But if you compare them across 1,000 traits (every song they've ever listened to, every shoe they've ever owned), everyone is equally far apart from everyone else! Proximity loses its meaning.

---

## 2. Mathematical Symbol Decoder

| Symbol | Mathematical Domain | Description & Meaning |
| :--- | :--- | :--- |
| $y \in \{1, \dots, C\}$ | Discrete set | Target class label |
| $x = [x_1, \dots, x_d]^T$ | Feature vector | $d$-dimensional observation |
| $P(y = c)$ | Prior probability | Baseline frequency of class $c$ in training data: $\frac{N_c}{N}$ |
| $P(x_j \mid y = c)$ | Likelihood | Conditional probability of observing feature $x_j$ given class $c$ |
| $P(y = c \mid x)$ | Posterior | Probability of class $c$ after observing feature evidence $x$ |
| $\alpha$ | Smoothing parameter | Laplace additive smoothing parameter (typically $\alpha = 1$) |
| $k \in \mathbb{N}$ | Hyperparameter | Number of nearest neighbors polled in KNN voting |
| $d_p(u, v)$ | Metric function | Minkowski distance of order $p$: $\left(\sum_{j=1}^d |u_j - v_j|^p\right)^{1/p}$ |
| $\mathcal{N}_k(x)$ | Index set | Set of indices of the $k$ training samples closest to query $x$ |

---

## 3. Naive Bayes: Mathematics & Formulations

### 3.1 Bayes' Theorem with Conditional Independence

By Bayes' Rule:
$$P(y = c \mid x) = \frac{P(y = c) P(x \mid y = c)}{P(x)}$$

Since the denominator $P(x) = \sum_c P(y=c) P(x \mid y=c)$ is constant across all candidate classes $c$, we maximize the numerator:
$$\hat{y} = \arg\max_{c \in \{1, \dots, C\}} P(y = c) P(x_1, x_2, \dots, x_d \mid y = c)$$

#### The Naive Conditional Independence Assumption
We assume that all features $x_1, \dots, x_d$ are mutually independent given class $y$:
$$P(x_1, x_2, \dots, x_d \mid y = c) = \prod_{j=1}^d P(x_j \mid y = c)$$

Thus, the classification rule becomes:
$$\hat{y} = \arg\max_{c} P(y = c) \prod_{j=1}^d P(x_j \mid y = c)$$

---

### 3.2 Log-Space Computation (Preventing Underflow)

Multiplying dozens of probabilities $p_j < 0.05$ causes severe floating-point underflow ($p_1 \times p_2 \times \dots \to 0.0$). We take the natural logarithm, converting products into numerically stable additions:

$$\hat{y} = \arg\max_{c} \left[ \ln P(y = c) + \sum_{j=1}^d \ln P(x_j \mid y = c) \right]$$

---

### 3.3 Feature Likelihood Distributions

#### A. Multinomial Naive Bayes (Discrete Counts & Text)
For bag-of-words text classification, where $x_j$ is the count of word $j$:
$$P(x_j \mid y = c) = \frac{N_{jc} + \alpha}{N_c + \alpha |V|}$$
where:
- $N_{jc}$ is the number of times word $j$ appears in all training documents of class $c$.
- $N_c = \sum_j N_{jc}$ is the total word count in class $c$.
- $|V|$ is the total vocabulary size.
- $\alpha \ge 0$ is the **Laplace smoothing** parameter ($\alpha = 1$ is standard add-one smoothing).

#### B. Gaussian Naive Bayes (Continuous Real Features)
When features $x_j \in \mathbb{R}$ are continuous, we assume they follow a univariate normal distribution within each class:
$$P(x_j \mid y = c) = \frac{1}{\sqrt{2\pi \sigma_{jc}^2}} \exp\left( -\frac{(x_j - \mu_{jc})^2}{2\sigma_{jc}^2} \right)$$
where $\mu_{jc}$ and $\sigma_{jc}^2$ are the sample mean and variance of feature $j$ across class $c$.

---

## 4. K-Nearest Neighbors (KNN): Mathematics & Properties

### 4.1 Instance-Based Lazy Learning

KNN does not construct an explicit internal model or optimize a parameter vector during training.
- **Training Phase:** Simply stores the training set $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$. Time complexity: $\mathcal{O}(1)$.
- **Inference Phase:** To classify a query point $x_q$, KNN computes the distance to all $N$ stored instances, sorts them, and identifies the $k$ closest neighbors $\mathcal{N}_k(x_q)$. Time complexity: $\mathcal{O}(N d)$.

The predicted class is the majority vote among the $k$ neighbors:
$$\hat{y}(x_q) = \arg\max_{c} \sum_{i \in \mathcal{N}_k(x_q)} \mathbb{I}(y_i = c)$$

---

### 4.2 Distance Metrics

The geometry of the neighborhood is determined by the distance metric:
1. **Euclidean Distance ($L_2$ norm):**
   $$d_2(u, v) = \|u - v\|_2 = \sqrt{\sum_{j=1}^d (u_j - v_j)^2}$$
2. **Manhattan Distance ($L_1$ norm / Taxicab distance):**
   $$d_1(u, v) = \|u - v\|_1 = \sum_{j=1}^d |u_j - v_j|$$
3. **Minkowski Distance ($L_p$ norm):**
   $$d_p(u, v) = \|u - v\|_p = \left( \sum_{j=1}^d |u_j - v_j|^p \right)^{1/p}$$

---

### 4.3 Hyperparameter $k$: The Bias-Variance Trade-off

| Value of $k$ | Decision Boundary | Model Complexity | Bias & Variance | Risk |
| :--- | :--- | :--- | :--- | :--- |
| **$k = 1$** | Highly jagged, islands around points | Maximum complexity | **Low Bias, High Variance** | Overfitting; memorizes noise/outliers |
| **Moderate $k$ (e.g. 5–15)** | Smooth, robust curves | Balanced | **Balanced** | Optimal generalization |
| **Large $k \to N$** | Trivial, predicts global majority | Minimal complexity | **High Bias, Low Variance** | Underfitting; ignores local structure |

*(Rule of thumb: Choose an odd $k$ for binary classification to eliminate tie votes).*

---

### 4.4 The Curse of Dimensionality

In high-dimensional spaces ($d \gg 1$), Euclidean intuition completely collapses.
Consider a unit hypercube $[0, 1]^d$. The volume of the hypercube is $1^d = 1$.
To capture a fraction $r$ of the data points, a hypercube neighborhood must have side length $e_d = r^{1/d}$.
- In $d = 1$: to capture 10% of data ($r = 0.10$), edge length is $e_1 = 0.10$.
- In $d = 10$: $e_{10} = 0.10^{1/10} \approx 0.794$ (you must cover 80% of each dimension!).
- In $d = 100$: $e_{100} = 0.10^{1/100} \approx 0.977$ (you must cover 98% of each dimension!).

In high dimensions, **the concept of "nearest" neighbor becomes meaningless**, because the ratio of the distance to the nearest neighbor versus the farthest neighbor approaches 1:
$$\lim_{d \to \infty} \frac{\text{dist}_{\max} - \text{dist}_{\min}}{\text{dist}_{\min}} = 0$$

All points become roughly equidistant from one another!
