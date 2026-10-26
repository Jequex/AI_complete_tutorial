# Topic 10.3: Exercises & Solutions — Evaluation Metrics & Calibration

---

## Problem Set Overview

These problems explore the mathematical proofs underlying classification metrics, rank equivalence of ROC-AUC, and the formal decomposition of the Brier score.

---

### Problem 1: Mathematical Proof of the Wilcoxon-Mann-Whitney Equivalence to AUC-ROC

Let $\mathcal{D}_+ = \{x_1^+, \dots, x_{N_+}^+\}$ denote the set of all positive instances, and $\mathcal{D}_- = \{x_1^-, \dots, x_{N_-}^-\}$ denote the set of all negative instances.
Let $s(x) \in \mathbb{R}$ be the real-valued score assigned by a continuous classification model. Assume there are no ties ($s(u) \ne s(v)$ for all $u \ne v$).

The Wilcoxon-Mann-Whitney rank statistic is defined as:
$$\hat{A} = \frac{1}{N_+ N_-} \sum_{i=1}^{N_+} \sum_{j=1}^{N_-} \mathbb{I}\left(s(x_i^+) > s(x_j^-)\right)$$

**Questions:**
1. Prove that the area under the empirical ROC curve (AUC-ROC) calculated via the trapezoidal rule is mathematically identical to $\hat{A}$.
2. What is the value of $\hat{A}$ for a completely random ranking?
3. What is the computational complexity of evaluating $\hat{A}$ naively vs. using sorting?

---

### Solution 1

1. **Proof of Equivalence:**
   Consider all $N = N_+ + N_-$ instances sorted in descending order of their predicted scores $s(x_{(1)}) > s(x_{(2)}) > \dots > s(x_{(N)})$.
   Let the index $k$ sweep from $1$ to $N$. At each step $k$, we set the decision threshold just below $s(x_{(k)})$.
   
   The ROC curve starts at $(0, 0)$. As $k$ increases:
   - If the $k$-th sample is positive ($y_{(k)} = 1$), the True Positive count increases by $1$. The curve takes a vertical step upwards of size $\Delta y = \frac{1}{N_+}$.
   - If the $k$-th sample is negative ($y_{(k)} = 0$), the False Positive count increases by $1$. The curve takes a horizontal step to the right of size $\Delta x = \frac{1}{N_-}$.
   
   The total area under the curve is the sum of the areas of the vertical columns beneath each horizontal step.
   A horizontal step occurs every time a negative instance $x_j^-$ is encountered. The height of the curve at that moment is the fraction of all positive instances that were ranked *above* this negative instance:
   $$\text{Height}(x_j^-) = \frac{1}{N_+} \sum_{i=1}^{N_+} \mathbb{I}\left(s(x_i^+) > s(x_j^-)\right)$$
   Multiplying this height by the width of the horizontal step $\Delta x = \frac{1}{N_-}$ and summing across all $N_-$ negative instances:
   $$\text{AUC} = \sum_{j=1}^{N_-} \left[\frac{1}{N_+} \sum_{i=1}^{N_+} \mathbb{I}\left(s(x_i^+) > s(x_j^-)\right)\right] \frac{1}{N_-} = \frac{1}{N_+ N_-} \sum_{i=1}^{N_+} \sum_{j=1}^{N_-} \mathbb{I}\left(s(x_i^+) > s(x_j^-)\right) = \hat{A}$$
   $\blacksquare$

2. **Random Ranking:**
   For a randomly shuffled sequence of scores, by symmetry $P(s(X^+) > s(X^-)) = 0.5$.
   Therefore, $\mathbb{E}[\hat{A}] = 0.5$.

3. **Computational Complexity:**
   - Naive pairwise comparison evaluates $N_+ \times N_-$ pairs: $\mathcal{O}(N_+ N_-) = \mathcal{O}(N^2)$.
   - Sorting the $N$ instances takes $\mathcal{O}(N \log N)$. Once sorted, a single linear pass ($\mathcal{O}(N)$) computes the ranks and yields $\hat{A}$ via the sum of ranks formula:
     $$\hat{A} = \frac{\sum_{i=1}^{N_+} \text{Rank}(x_i^+) - \frac{N_+(N_+ + 1)}{2}}{N_+ N_-}$$

---

### Problem 2: Harmonic Mean vs. $F_\beta$ Score Tuning

The generalized $F_\beta$ score weights Recall $\beta$ times as heavily as Precision:
$$F_\beta = (1 + \beta^2) \frac{P \cdot R}{\beta^2 P + R}$$

**Questions:**
1. Show that when $\beta = 1$, $F_\beta$ simplifies to the standard harmonic mean $F_1$.
2. In a cancer screening application where missing a malignancy (False Negative) is 10 times more costly than ordering an unnecessary biopsy (False Positive), should you choose $\beta > 1$ or $\beta < 1$?
3. Compute $F_2$ and $F_{0.5}$ for a diagnostic system with Precision $P = 0.50$ and Recall $R = 0.90$.

---

### Solution 2

1. **Derivation for $\beta = 1$:**
   $$F_1 = (1 + 1^2) \frac{P \cdot R}{1^2 P + R} = 2 \frac{P \cdot R}{P + R} = \frac{2}{\frac{P+R}{P R}} = \frac{2}{\frac{1}{R} + \frac{1}{P}} = \left(\frac{P^{-1} + R^{-1}}{2}\right)^{-1}$$
   This is exactly the harmonic mean of Precision and Recall.

2. **Screening Application Choice:**
   In cancer screening, missing a tumor (FN) must be avoided at all costs, requiring high Recall.
   Since $\beta$ represents the relative importance of Recall over Precision, we choose $\beta > 1$ (e.g. $\beta = 2$ or $\beta = 3$) to emphasize Recall.

3. **Numerical Calculations ($P = 0.50, R = 0.90$):**
   - **$F_2$ ($\beta = 2$):**
     $$F_2 = (1 + 4) \frac{0.50 \times 0.90}{4 \times 0.50 + 0.90} = 5 \times \frac{0.45}{2.00 + 0.90} = \frac{2.25}{2.90} \approx 0.7759 \quad (77.6\%)$$
   - **$F_{0.5}$ ($\beta = 0.5$):**
     $$F_{0.5} = (1 + 0.25) \frac{0.50 \times 0.90}{0.25 \times 0.50 + 0.90} = 1.25 \times \frac{0.45}{0.125 + 0.90} = \frac{0.5625}{1.025} \approx 0.5488 \quad (54.9\%)$$
   Because Recall ($0.90$) is higher than Precision ($0.50$), $F_2$ rewards the model ($0.776$) while $F_{0.5}$ penalizes the model for lower precision ($0.549$).

---

### Problem 3: The Murphy Decomposition of the Brier Score

The Brier Score $\text{BS} = \frac{1}{N}\sum_{i=1}^N (p_i - y_i)^2$ evaluates probabilistic forecast accuracy.
In 1973, atmospheric scientist Allan Murphy proved that when predictions are grouped into $K$ unique forecast probability bins $\{f_1, \dots, f_K\}$ with bin sizes $N_k$, the Brier score decomposes into three distinct components:
$$\text{BS} = \text{Reliability} - \text{Resolution} + \text{Uncertainty}$$

where $\bar{y} = \frac{1}{N}\sum y_i$ is the base rate prevalence, and $\bar{y}_k$ is the observed empirical positive fraction in bin $k$.

$$\text{Reliability} = \frac{1}{N}\sum_{k=1}^K N_k (f_k - \bar{y}_k)^2$$
$$\text{Resolution} = \frac{1}{N}\sum_{k=1}^K N_k (\bar{y}_k - \bar{y})^2$$
$$\text{Uncertainty} = \bar{y}(1 - \bar{y})$$

**Questions:**
1. What does each of the three terms represent physically?
2. If a model outputs the constant base rate probability $f_i = \bar{y}$ for all instances (the uninformative dummy forecaster), compute its Reliability, Resolution, and Brier Score.
3. Why does Platt scaling improve the Brier score by reducing Reliability error without diminishing Resolution?

---

### Solution 3

1. **Physical Interpretations:**
   - **Reliability (Calibration Error):** Measures how close predicted probabilities $f_k$ are to the true observed frequencies $\bar{y}_k$ within each bin. Lower is better (0 is perfect calibration).
   - **Resolution (Discrimination Power):** Measures how much the conditional probabilities $\bar{y}_k$ deviate from the global prior $\bar{y}$. A model that sorts confident positives into one bin and negatives into another has high resolution. Higher is better (it subtracts from total error).
   - **Uncertainty (Inherent Entropy):** Measures the irreducible variance of the dataset $\text{Var}(Y) = \bar{y}(1 - \bar{y})$. It depends solely on class prevalence and cannot be altered by any model.

2. **Dummy Forecaster ($f_i = \bar{y}$):**
   There is only 1 bin ($K=1$) where $f_1 = \bar{y}$ and $\bar{y}_1 = \bar{y}$:
   - $\text{Reliability} = \frac{N}{N} (\bar{y} - \bar{y})^2 = 0$. (The dummy forecaster is *perfectly calibrated*!).
   - $\text{Resolution} = \frac{N}{N} (\bar{y} - \bar{y})^2 = 0$. (It has *zero discrimination* ability).
   - $\text{Brier Score} = 0 - 0 + \bar{y}(1 - \bar{y}) = \bar{y}(1 - \bar{y})$.
   For $\bar{y} = 0.05$, $\text{BS} = 0.05 \times 0.95 = 0.0475$.

3. **Platt Scaling Impact:**
   Platt scaling applies a monotonic logistic transformation $\sigma(A s + B)$ to the predicted margins. Because the transformation is strictly monotonic (preserving the ranking order of instances), it leaves the sorting between positive and negative instances intact (maintaining high Resolution). Simultaneously, it aligns the scores so $f_k \approx \bar{y}_k$, driving the Reliability error term towards zero.
