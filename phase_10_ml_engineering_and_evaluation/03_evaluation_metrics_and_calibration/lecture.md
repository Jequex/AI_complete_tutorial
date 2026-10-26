# Topic 10.3: Evaluation Metrics, Trade-offs & Probability Calibration

---

## 1. Executive Summary & The 12-Year-Old Analogy

Accuracy is the single most dangerous metric in applied machine learning.

If an engineer trains an AI system to detect fraudulent credit card transactions—where $99.9\%$ of transactions are legitimate and $0.1\%$ are fraudulent—a trivial "dummy model" that classifies every single transaction as non-fraudulent achieves **$99.9\%$ accuracy**. Yet, this model is completely useless; every criminal escapes undetected.

In real-world engineering, evaluation is not about maximizing a single headline number. It requires navigating fundamental trade-offs: **Precision** (avoiding false alarms) versus **Recall** (catching every positive event), analyzing global threshold rankings (**ROC-AUC** vs. **PR-AUC**), and verifying whether predicted probabilities reflect real-world frequencies (**Probability Calibration**).

```
                      THE CONFUSION MATRIX (BINARY OUTCOMES)
                                     Actual Positive (Y=1)       Actual Negative (Y=0)
                            ┌───────────────────────────┬───────────────────────────┐
 Predicted Positive (Ŷ=1)   │   TRUE POSITIVE (TP)      │   FALSE POSITIVE (FP)     │
                            │   "Caught the Fraud!"     │   "False Alarm!"          │
                            ├───────────────────────────┼───────────────────────────┤
 Predicted Negative (Ŷ=0)   │   FALSE NEGATIVE (FN)     │   TRUE NEGATIVE (TN)      │
                            │   "Missed the Fraud!"     │   "Correctly Ignored"     │
                            └───────────────────────────┴───────────────────────────┘
```

---

### The 12-Year-Old Analogy: The Medical Test for a Rare Disease

Imagine a disease that affects only 1 out of every 1,000 children in a school district.

- **The Lazy Doctor (Accuracy = 99.9%):**
  A lazy doctor walks into the clinic and tells every single child: *"You are 100% healthy!"* without examining them. 
  Out of 1,000 children, the doctor is right 999 times! His accuracy is $99.9\%$. But the one sick child never gets medicine and ends up in the emergency room.
- **Precision (Trusting Positive Tests):**
  The lab creates a blood test. If the test says a child is sick, what are the odds they *actually* have the disease? If the test sounds false alarms constantly, healthy children undergo scary, painful treatments. High Precision means *"When the alarm rings, you can trust it."*
- **Recall (Catching Every Sick Child):**
  Out of all children who actually have the disease, how many did the test identify? High Recall means *"No sick child slips through undetected."*
- **The Trade-off (The Smoke Alarm Knob):**
  If you turn your kitchen smoke alarm sensitivity up to maximum, it catches every fire (100% Recall), but it also screams every time you toast a piece of bread (terrible Precision). If you turn sensitivity down so it never beeps at toast, it might sleep through a real kitchen fire (terrible Recall).
- **Probability Calibration (Honest Weather Forecasts):**
  If a meteorologist says there is a *"70% chance of rain"* on 100 different days, it had better rain on roughly 70 of those days. If it only rains on 10 of those days, the forecaster is overconfident and uncalibrated.

---

## 2. Mathematical Symbol Decoder

| Symbol | Mathematical Domain | Description & Engineering Interpretation |
| :--- | :--- | :--- |
| $\text{TP}, \text{FP}, \text{FN}, \text{TN}$ | Non-negative integers | Counts of True Positives, False Positives, False Negatives, True Negatives |
| $P = \frac{\text{TP}}{\text{TP} + \text{FP}}$ | Scalar $\in [0, 1]$ | Precision (Positive Predictive Value): fraction of predicted positives that are real |
| $R = \frac{\text{TP}}{\text{TP} + \text{FN}}$ | Scalar $\in [0, 1]$ | Recall / Sensitivity / True Positive Rate (TPR): fraction of actual positives captured |
| $\text{TNR} = \frac{\text{TN}}{\text{TN} + \text{FP}}$ | Scalar $\in [0, 1]$ | Specificity / True Negative Rate: fraction of actual negatives correctly identified |
| $\text{FPR} = \frac{\text{FP}}{\text{TN} + \text{FP}}$ | Scalar $\in [0, 1]$ | False Positive Rate ($1 - \text{Specificity}$): rate of false alarms among negatives |
| $F_1$ | Scalar $\in [0, 1]$ | Harmonic mean of Precision and Recall |
| $\text{MCC}$ | Scalar $\in [-1, +1]$ | Matthews Correlation Coefficient (discrete Pearson correlation between $Y$ and $\hat{Y}$) |
| $\tau \in [0, 1]$ | Threshold scalar | Decision boundary: predict $\hat{y}_i = 1$ if $\hat{p}_i \ge \tau$, else $0$ |
| $\text{AUC-ROC}$ | Scalar $\in [0.5, 1.0]$ | Area under the Receiver Operating Characteristic curve |
| $\text{AUC-PR}$ | Scalar $\in [0, 1]$ | Area under the Precision-Recall curve |
| $\text{BS}$ | Scalar $\ge 0$ | Brier Score: Mean Squared Error of predicted probabilities against binary truth |

---

## 3. Mathematical Analysis of Classification Metrics

### 3.1 Why the Harmonic Mean for $F_1$?
The $F_1$-score is defined as:
$$F_1 = 2 \cdot \frac{P \cdot R}{P + R} = \left(\frac{P^{-1} + R^{-1}}{2}\right)^{-1}$$

**Mathematical Proof of Asymmetric Penalty:**
Consider an arithmetic mean $A(P, R) = \frac{P + R}{2}$ versus harmonic mean $H(P, R) = \frac{2 P R}{P + R}$.
Suppose a model catches all positives ($R = 1.0$) by predicting everything as positive, resulting in catastrophic precision $P = 0.01$.
- Arithmetic Mean: $A(0.01, 1.0) = \frac{1.01}{2} = 0.505$ (A respectable score!).
- Harmonic Mean: $H(0.01, 1.0) = \frac{2 \times 0.01 \times 1.0}{1.01} = \frac{0.02}{1.01} \approx 0.0198$ (Near zero!).

By the AM-HM inequality:
$$\mathcal{H}(P, R) \le \mathcal{G}(P, R) \le \mathcal{A}(P, R)$$
with equality if and only if $P = R$. The harmonic mean approaches 0 if *either* precision or recall collapses to 0, preventing an engineer from cheating by gaming one metric at the total expense of the other.

### 3.2 Matthews Correlation Coefficient (MCC)
While $F_1$ ignores True Negatives ($\text{TN}$), the Matthews Correlation Coefficient includes all four quadrants of the confusion matrix:
$$\text{MCC} = \frac{\text{TP} \times \text{TN} - \text{FP} \times \text{FN}}{\sqrt{(\text{TP} + \text{FP})(\text{TP} + \text{FN})(\text{TN} + \text{FP})(\text{TN} + \text{FN})}}$$
- $\text{MCC} = +1$: Perfect prediction.
- $\text{MCC} = 0$: Completely uninformative coin-flip prediction.
- $\text{MCC} = -1$: Total inverse prediction.

MCC is invariant to which class is labeled "positive" vs "negative" and provides a reliable scalar summary even under 1000:1 class imbalance.

---

## 4. Threshold-Free Curves: ROC-AUC vs. PR-AUC

Most classifiers output a continuous score $\hat{s}_i \in \mathbb{R}$ or probability $\hat{p}_i \in [0, 1]$. Applying a threshold $\tau$ converts continuous scores into binary labels:
$$\hat{y}_i = \mathbb{I}(\hat{p}_i \ge \tau)$$

Sweeping $\tau$ from $1.0$ down to $0.0$ generates parametric performance curves.

```
          RECEIVER OPERATING CHARACTERISTIC (ROC)               PRECISION-RECALL (PR) CURVE
      TPR ▲                                              Precision ▲
          │         .---~'                                         │ 1.0├───────╮
          │       .'                                               │    │       '--.
          │      /                                                 │    │           '--.
          │     /       AUC-ROC ≈ 0.95                             │    │    AUC-PR     \
          │    /                                                   │    │                \
          │   /                                                    │    │                 \
          │  /  Diagonal = Random Guess (0.50)                     │0.05├─── ─── ─── ─── ─── (Random baseline = prevalence)
          └──┴────────────────────────► FPR                        └──┴────────────────────────► Recall
          0.0                         1.0                          0.0                         1.0
```

### 4.1 ROC Curve (Receiver Operating Characteristic)
Plots **True Positive Rate** (Recall) on the y-axis against **False Positive Rate** on the x-axis:
$$\text{TPR}(\tau) = \frac{\text{TP}(\tau)}{\text{TP}(\tau) + \text{FN}(\tau)}, \quad \text{FPR}(\tau) = \frac{\text{FP}(\tau)}{\text{TN}(\tau) + \text{FP}(\tau)}$$

**Probabilistic Interpretation of AUC-ROC:**
The Area Under the ROC Curve is mathematically equivalent to the probability that the model ranks a randomly chosen positive sample higher than a randomly chosen negative sample:
$$\text{AUC-ROC} = P\left(\hat{s}(X^+) > \hat{s}(X^-)\right)$$
where $X^+ \sim \mathcal{D}_{\text{pos}}$ and $X^- \sim \mathcal{D}_{\text{neg}}$.

### 4.2 Precision-Recall Curve: Why PR Dominates in High Imbalance
In fraud, cancer detection, and cyber-security, actual positives comprise $< 0.1\%$ of the dataset ($N_{\text{neg}} \gg N_{\text{pos}}$).
Notice the denominator of FPR: $\text{FPR} = \frac{\text{FP}}{\text{TN} + \text{FP}} \approx \frac{\text{FP}}{1,000,000}$.
If a model generates $1,000$ false alarms on a million negative samples, $\text{FPR} = \frac{1,000}{1,000,000} = 0.001$ ($0.1\%$). 
On the ROC curve, this looks like a triumph: the curve hugs the upper-left axis, yielding an $\text{AUC-ROC} > 0.98$.

However, if there were only $100$ actual fraud cases in the entire dataset:
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} \le \frac{100}{100 + 1000} \approx 9.0\%$$
For every 1 genuine alert, investigators investigate 10 false alarms! 
The **Precision-Recall curve** directly exposes this disaster because Precision divides by $\text{FP}$, without $\text{TN}$ diluting the denominator.

---

## 5. Probability Calibration & Reliability

A model that outputs $\hat{p} = 0.85$ should not merely rank instances; it should mean that out of 100 samples with this score, approximately 85 belong to the positive class.

Many popular models produce uncalibrated probabilities:
- **Naive Bayes:** Independence assumptions produce extreme, pushed-to-the-edges probabilities ($0.00001$ or $0.99999$).
- **Support Vector Machines:** Maximize geometric margin without natural probabilistic meaning.
- **Boosted Trees (XGBoost/LightGBM):** Logit predictions are often conservative or distorted near boundaries.

### 5.1 The Brier Score
The Brier Score measures the mean squared error between forecast probabilities and true binary indicators ($y \in \{0, 1\}$):
$$\text{BS} = \frac{1}{N}\sum_{i=1}^N (\hat{p}_i - y_i)^2$$
A strictly proper scoring rule: $\text{BS} = 0$ is perfect probability estimation; $\text{BS} = 0.25$ is the score of a perpetual $0.5$ guesser.

### 5.2 Reliability Diagrams
Data is binned into $M$ intervals (e.g. $[0, 0.1], [0.1, 0.2], \dots, [0.9, 1.0]$). For each bin $B_m$:
$$\bar{p}_m = \frac{1}{|B_m|}\sum_{i \in B_m} \hat{p}_i, \quad \bar{y}_m = \frac{1}{|B_m|}\sum_{i \in B_m} y_i$$
A well-calibrated classifier produces a reliability curve hugging the identity diagonal $\bar{y}_m = \bar{p}_m$.

### 5.3 Platt Scaling
Platt scaling calibrates raw model margins or logits $s_i$ by fitting a 1D logistic regression on a separate validation set:
$$P(Y=1 \mid s_i) = \frac{1}{1 + \exp(A \cdot s_i + B)}$$
where scalar parameters $A$ and $B$ are estimated via Maximum Likelihood.
