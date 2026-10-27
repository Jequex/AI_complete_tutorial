# Topic 02: Exercises & Solutions — Classification, Calibration & Thresholding

---

### Exercise 1: Exact ROC-AUC, PR-AUC, and the Mann-Whitney $U$ Equivalence

An evaluation benchmark contains $N = 10$ test instances evaluated by a classifier. The instances are sorted by descending predicted probability score $\hat{p}_i$:

| Index $i$ | True Label $y_i$ | Model Score $\hat{p}_i$ |
| :--- | :--- | :--- |
| 1 | 1 | 0.95 |
| 2 | 1 | 0.90 |
| 3 | 0 | 0.85 |
| 4 | 1 | 0.70 |
| 5 | 1 | 0.60 |
| 6 | 0 | 0.55 |
| 7 | 0 | 0.45 |
| 8 | 1 | 0.35 |
| 9 | 0 | 0.25 |
| 10 | 0 | 0.10 |

Total positives: $N_+ = 5$. Total negatives: $N_- = 5$.

**Questions:**
1. Generate the table of unique operating points $(\text{FPR}, \text{TPR})$ by sweeping the threshold $\theta$ through all unique scores. Compute $\text{AUC}_{\text{ROC}}$ using the trapezoidal integration formula.
2. Calculate the Mann-Whitney $U$-statistic directly by counting all pairs $(i, j)$ with $y_i = 1$ and $y_j = 0$ where $\hat{p}_i > \hat{p}_j$. Verify that $\frac{U}{N_+ N_-} = \text{AUC}_{\text{ROC}}$.
3. Compute the Precision-Recall curve operating points $(\text{Recall}, \text{Precision})$ and compute the Area Under the PR Curve ($\text{PR-AUC}$ / Average Precision).

---

#### Solution to Exercise 1

**1. ROC Operating Points and Trapezoidal Integration:**
We sweep threshold $\theta$ downwards from $> 0.95$ to $\le 0.10$:

| Step $k$ | Threshold $\theta$ | $\text{TP}$ | $\text{FP}$ | $\text{TPR} = \frac{\text{TP}}{5}$ | $\text{FPR} = \frac{\text{FP}}{5}$ | $\Delta \text{FPR}$ | Trapezoid Area $\frac{\text{TPR}_k + \text{TPR}_{k-1}}{2} \Delta \text{FPR}$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | $> 0.95$ | 0 | 0 | 0.0 | 0.0 | - | - |
| 1 | 0.95 | 1 | 0 | 0.2 | 0.0 | 0.0 | 0.0 |
| 2 | 0.90 | 2 | 0 | 0.4 | 0.0 | 0.0 | 0.0 |
| 3 | 0.85 | 2 | 1 | 0.4 | 0.2 | 0.2 | $\frac{0.4 + 0.4}{2} \times 0.2 = 0.08$ |
| 4 | 0.70 | 3 | 1 | 0.6 | 0.2 | 0.0 | 0.0 |
| 5 | 0.60 | 4 | 1 | 0.8 | 0.2 | 0.0 | 0.0 |
| 6 | 0.55 | 4 | 2 | 0.8 | 0.4 | 0.2 | $\frac{0.8 + 0.8}{2} \times 0.2 = 0.16$ |
| 7 | 0.45 | 4 | 3 | 0.8 | 0.6 | 0.2 | $\frac{0.8 + 0.8}{2} \times 0.2 = 0.16$ |
| 8 | 0.35 | 5 | 3 | 1.0 | 0.6 | 0.0 | 0.0 |
| 9 | 0.25 | 5 | 4 | 1.0 | 0.8 | 0.2 | $\frac{1.0 + 1.0}{2} \times 0.2 = 0.20$ |
| 10 | 0.10 | 5 | 5 | 1.0 | 1.0 | 0.2 | $\frac{1.0 + 1.0}{2} \times 0.2 = 0.20$ |

Summing the trapezoid areas:
$$\text{AUC}_{\text{ROC}} = 0.08 + 0.16 + 0.16 + 0.20 + 0.20 = \mathbf{0.80}$$

**2. Mann-Whitney $U$-Statistic Calculation:**
Positive instances have scores: $S_+ = \{0.95, 0.90, 0.70, 0.60, 0.35\}$.
Negative instances have scores: $S_- = \{0.85, 0.55, 0.45, 0.25, 0.10\}$.

For each positive instance, count how many negative instances it strictly exceeds:
- $\hat{p} = 0.95$ exceeds all 5 negatives: count = 5
- $\hat{p} = 0.90$ exceeds all 5 negatives: count = 5
- $\hat{p} = 0.70$ exceeds 4 negatives ($0.55, 0.45, 0.25, 0.10$): count = 4
- $\hat{p} = 0.60$ exceeds 4 negatives ($0.55, 0.45, 0.25, 0.10$): count = 4
- $\hat{p} = 0.35$ exceeds 2 negatives ($0.25, 0.10$): count = 2

Sum of pairwise comparisons:
$$U = 5 + 5 + 4 + 4 + 2 = \mathbf{20}$$
Normalized by total pairs:
$$\frac{U}{N_+ \cdot N_-} = \frac{20}{5 \times 5} = \frac{20}{25} = \mathbf{0.80}$$
The identity $\text{AUC}_{\text{ROC}} \equiv \frac{U}{N_+ N_-} = 0.80$ is verified exactly.

**3. PR Curve and Average Precision ($\text{PR-AUC}$):**
At each threshold where a positive instance is retrieved, compute Recall and Precision:

| Positive Rank $k$ | Score | $\text{TP}$ | $\text{FP}$ | Recall $R_k = \frac{\text{TP}}{5}$ | Precision $P_k = \frac{\text{TP}}{\text{TP}+\text{FP}}$ | $\Delta R_k = R_k - R_{k-1}$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 0.95 | 1 | 0 | 0.2 | $1 / 1 = 1.000$ | 0.2 |
| 2 | 0.90 | 2 | 0 | 0.4 | $2 / 2 = 1.000$ | 0.2 |
| 3 | 0.70 | 3 | 1 | 0.6 | $3 / 4 = 0.750$ | 0.2 |
| 4 | 0.60 | 4 | 1 | 0.8 | $4 / 5 = 0.800$ | 0.2 |
| 5 | 0.35 | 5 | 3 | 1.0 | $5 / 8 = 0.625$ | 0.2 |

Average Precision ($\text{PR-AUC}$):
$$\text{PR-AUC} = \sum_{k=1}^5 \Delta R_k \cdot P_k = 0.2(1.000 + 1.000 + 0.750 + 0.800 + 0.625) = 0.2(4.175) = \mathbf{0.835}$$

---

### Exercise 2: Murphy's Brier Score Decomposition

A medical diagnostic model outputs predicted disease probabilities for $N = 20$ patients. The predictions are grouped into $M = 4$ calibration bins:

| Bin $m$ | Confidence Range | Bin Count $n_m$ | Sum of Probabilities $\sum \hat{p}_i$ | Sum of Actual Positives $\sum y_i$ |
| :--- | :--- | :--- | :--- | :--- |
| 1 | $[0.0, 0.25)$ | 6 | 0.60 | 1 |
| 2 | $[0.25, 0.50)$ | 4 | 1.40 | 1 |
| 3 | $[0.50, 0.75)$ | 5 | 3.25 | 3 |
| 4 | $[0.75, 1.00]$ | 5 | 4.45 | 4 |

Total dataset positives: $\sum_{i=1}^{20} y_i = 1 + 1 + 3 + 4 = 9 \implies \bar{y} = \frac{9}{20} = 0.45$.

**Questions:**
1. For each bin $m$, compute average confidence $\bar{p}_m$ and empirical positive rate $\bar{y}_m$.
2. Compute the **Reliability ($\text{REL}$)** component.
3. Compute the **Resolution ($\text{RES}$)** component.
4. Compute the **Uncertainty ($\text{UNC}$)** component.
5. Compute the total Brier Score using Murphy's decomposition $\text{BS} = \text{REL} - \text{RES} + \text{UNC}$.

---

#### Solution to Exercise 2

**1. Bin Statistics:**
- Bin 1: $\bar{p}_1 = \frac{0.60}{6} = 0.10$, $\bar{y}_1 = \frac{1}{6} \approx 0.1667$
- Bin 2: $\bar{p}_2 = \frac{1.40}{4} = 0.35$, $\bar{y}_2 = \frac{1}{4} = 0.2500$
- Bin 3: $\bar{p}_3 = \frac{3.25}{5} = 0.65$, $\bar{y}_3 = \frac{3}{5} = 0.6000$
- Bin 4: $\bar{p}_4 = \frac{4.45}{5} = 0.89$, $\bar{y}_4 = \frac{4}{5} = 0.8000$

**2. Reliability ($\text{REL}$):**
$$\text{REL} = \sum_{m=1}^4 \frac{n_m}{N} (\bar{p}_m - \bar{y}_m)^2$$
- Bin 1: $\frac{6}{20}(0.10 - 0.1667)^2 = 0.30 \times (-0.0667)^2 = 0.30 \times 0.004449 \approx 0.001335$
- Bin 2: $\frac{4}{20}(0.35 - 0.25)^2 = 0.20 \times (0.10)^2 = 0.20 \times 0.0100 = 0.002000$
- Bin 3: $\frac{5}{20}(0.65 - 0.60)^2 = 0.25 \times (0.05)^2 = 0.25 \times 0.0025 = 0.000625$
- Bin 4: $\frac{5}{20}(0.89 - 0.80)^2 = 0.25 \times (0.09)^2 = 0.25 \times 0.0081 = 0.002025$

$$\text{REL} \approx 0.001335 + 0.002000 + 0.000625 + 0.002025 = \mathbf{0.005985}$$

**3. Resolution ($\text{RES}$):**
Base rate $\bar{y} = 0.45$.
$$\text{RES} = \sum_{m=1}^4 \frac{n_m}{N} (\bar{y}_m - \bar{y})^2$$
- Bin 1: $0.30 \times (0.1667 - 0.45)^2 = 0.30 \times (-0.2833)^2 = 0.30 \times 0.08028 \approx 0.024084$
- Bin 2: $0.20 \times (0.25 - 0.45)^2 = 0.20 \times (-0.20)^2 = 0.20 \times 0.0400 = 0.008000$
- Bin 3: $0.25 \times (0.60 - 0.45)^2 = 0.25 \times (0.15)^2 = 0.25 \times 0.0225 = 0.005625$
- Bin 4: $0.25 \times (0.80 - 0.45)^2 = 0.25 \times (0.35)^2 = 0.25 \times 0.1225 = 0.030625$

$$\text{RES} \approx 0.024084 + 0.008000 + 0.005625 + 0.030625 = \mathbf{0.068334}$$

**4. Uncertainty ($\text{UNC}$):**
$$\text{UNC} = \bar{y}(1 - \bar{y}) = 0.45 \times (1 - 0.45) = 0.45 \times 0.55 = \mathbf{0.247500}$$

**5. Total Brier Score:**
$$\text{BS} = \text{REL} - \text{RES} + \text{UNC} = 0.005985 - 0.068334 + 0.247500 = \mathbf{0.185151}$$

---

### Exercise 3: Expected Calibration Error & Temperature Scaling

A modern vision transformer outputs logits $z_i$ that produce overconfident probabilities $\hat{p}_i = \sigma(z_i)$. On a validation cohort of 6 samples:

| Sample $i$ | Logit $z_i$ | Raw Prob $\hat{p}_i = \sigma(z_i)$ | Label $y_i$ |
| :--- | :--- | :--- | :--- |
| 1 | +2.20 | 0.900 | 1 |
| 2 | +1.73 | 0.850 | 1 |
| 3 | +1.39 | 0.800 | 0 |
| 4 | -1.39 | 0.200 | 1 |
| 5 | -1.73 | 0.150 | 0 |
| 6 | -2.20 | 0.100 | 0 |

Consider 2 equal-width bins: Bin 1 ($[0.0, 0.5]$) and Bin 2 ($(0.5, 1.0]$).

**Questions:**
1. Compute the raw Expected Calibration Error ($\text{ECE}$) and Maximum Calibration Error ($\text{MCE}$) for these 2 bins.
2. Suppose temperature scaling is applied with learned parameter $T = 2.0$. Compute the calibrated logits $z_i' = z_i / 2.0$ and new probabilities $\hat{p}_i(T=2.0)$.
3. Recompute $\text{ECE}$ after temperature scaling.
4. Show why the classification accuracy at default threshold $0.5$ and the ranking order of the samples remain completely unchanged.

---

#### Solution to Exercise 3

**1. Raw ECE and MCE:**
- Bin 1 ($[0.0, 0.5]$): Samples 4, 5, 6.
  * Count: $n_1 = 3$.
  * Average confidence: $\text{conf}(B_1) = \frac{0.20 + 0.15 + 0.10}{3} = 0.150$.
  * Empirical positive accuracy: $\text{acc}(B_1) = \frac{1 + 0 + 0}{3} = \frac{1}{3} \approx 0.3333$.
  * Absolute gap: $|0.3333 - 0.150| = 0.1833$.
- Bin 2 ($(0.5, 1.0]$): Samples 1, 2, 3.
  * Count: $n_2 = 3$.
  * Average confidence: $\text{conf}(B_2) = \frac{0.90 + 0.85 + 0.80}{3} = 0.850$.
  * Empirical positive accuracy: $\text{acc}(B_2) = \frac{1 + 1 + 0}{3} = \frac{2}{3} \approx 0.6667$.
  * Absolute gap: $|0.6667 - 0.850| = 0.1833$.

$$\text{ECE}_{\text{raw}} = \frac{3}{6}(0.1833) + \frac{3}{6}(0.1833) = \mathbf{0.1833 \text{ (or } 18.33\%)}$$
$$\text{MCE}_{\text{raw}} = \max(0.1833, 0.1833) = \mathbf{0.1833}$$

**2. Temperature Scaled Probabilities ($T = 2.0$):**
- Sample 1: $z_1' = 2.20 / 2 = 1.10 \implies \hat{p}_1' = \sigma(1.10) \approx \mathbf{0.7503}$
- Sample 2: $z_2' = 1.73 / 2 = 0.865 \implies \hat{p}_2' = \sigma(0.865) \approx \mathbf{0.7037}$
- Sample 3: $z_3' = 1.39 / 2 = 0.695 \implies \hat{p}_3' = \sigma(0.695) \approx \mathbf{0.6671}$
- Sample 4: $z_4' = -1.39 / 2 = -0.695 \implies \hat{p}_4' = \sigma(-0.695) \approx \mathbf{0.3329}$
- Sample 5: $z_5' = -1.73 / 2 = -0.865 \implies \hat{p}_5' = \sigma(-0.865) \approx \mathbf{0.2963}$
- Sample 6: $z_6' = -2.20 / 2 = -1.10 \implies \hat{p}_6' = \sigma(-1.10) \approx \mathbf{0.2497}$

**3. Calibrated ECE:**
- Bin 1: Samples 4, 5, 6 ($n_1 = 3$).
  * Average confidence: $\frac{0.3329 + 0.2963 + 0.2497}{3} = \frac{0.8789}{3} \approx 0.2930$.
  * Empirical positive rate: $\text{acc}(B_1) = \frac{1}{3} \approx 0.3333$.
  * Absolute gap: $|0.3333 - 0.2930| = 0.0403$.
- Bin 2: Samples 1, 2, 3 ($n_2 = 3$).
  * Average confidence: $\frac{0.7503 + 0.7037 + 0.6671}{3} = \frac{2.1211}{3} \approx 0.7070$.
  * Empirical positive rate: $\text{acc}(B_2) = \frac{2}{3} \approx 0.6667$.
  * Absolute gap: $|0.6667 - 0.7070| = 0.0403$.

$$\text{ECE}_{\text{calibrated}} = \frac{3}{6}(0.0403) + \frac{3}{6}(0.0403) = \mathbf{0.0403 \text{ (or } 4.03\%)}$$
Calibration error was slashed from **$18.33\%$ down to $4.03\%$**, a **$4.5\times$ reduction** in miscalibration!

**4. Invariance Proof:**
- For any $T > 0$, $\sigma(z / T) \ge 0.5 \iff z / T \ge 0 \iff z \ge 0$. Since the sign of $z$ is preserved, predictions at threshold $0.5$ do not change. Accuracy remains identical ($4 / 6 \approx 66.7\%$).
- Furthermore, since $f(z) = z / T$ is strictly monotonic, $z_a > z_b \iff z_a / T > z_b / T$. The pairwise ranking of every sample is preserved, meaning ROC-AUC is mathematically identical before and after temperature scaling.
