# Topic 02: Classification, Calibration & Thresholding

## 1. Introduction & 12-Year-Old Intuition

Imagine you have a friend named Charlie who loves predicting the weather:
- Every morning, Charlie looks at the sky and says: *"There is a 90% chance of rain today!"*
- You carry your heavy umbrella to school every single day because Charlie sounded so confident.
- At the end of the year, you look back at the 100 days when Charlie predicted a 90% chance of rain. 
- It actually rained on only **55 of those days**!

Charlie is **uncalibrated**. When someone tells you there is a 90% chance of rain, it should actually rain 90 out of 100 times. If it only rains 55 times, Charlie is wildly **overconfident**.

Now consider another friend, Dana:
- Dana always says: *"If it's sunny at dawn, it won't rain; if it's cloudy, it will."*
- Dana's predictions put all rainy days in one bucket and sunny days in another bucket. Dana has great **discrimination / ranking** (high ROC-AUC), but if you ask Dana for exact probabilities, Dana might assign arbitrary numbers like 0.99 and 0.01.

In mission-critical AI systems (autonomous vehicles, cancer diagnosis, credit underwriting, content moderation), a model does not just output binary classes; it outputs **confidence scores** $p \in [0, 1]$. 
- If an autonomous vehicle model outputs $p = 0.999$ confidence that the road ahead is clear, the control software will not slow down.
- If deep neural networks are systematically overconfident (as proven by Guo et al., 2017), using raw softmax probabilities directly as risk estimates can cause catastrophic failures.

This topic teaches the dual pillars of evaluation:
1. **Discrimination**: How well does the model separate positives from negatives across varying thresholds? ($\text{ROC-AUC}, \text{PR-AUC}$)
2. **Calibration**: Does a predicted probability of $80\%$ truly mean the event occurs with an empirical frequency of $80\%$? ($\text{ECE}, \text{MCE}, \text{Brier Score}$, Temperature Scaling)

---

## 2. Mathematical Symbol Decoder

| Symbol | Name | Domain | Description & Engineering Significance |
| :--- | :--- | :--- | :--- |
| $y_i$ | Ground Truth Label | $\{0, 1\}$ | Binary class indicator ($1$ = positive class, $0$ = negative class). |
| $\hat{p}_i$ | Predicted Probability | $[0, 1]$ | Model confidence output $\mathbb{P}(Y=1 \mid X=x_i)$. |
| $z_i$ | Logit Score | $\mathbb{R}$ | Unnormalized log-odds output before sigmoid/softmax: $\hat{p}_i = \sigma(z_i)$. |
| $\theta$ | Decision Threshold | $[0, 1]$ | Cutoff above which positive class is predicted: $\hat{y}_i = \mathbb{I}(\hat{p}_i \ge \theta)$. |
| $\text{TPR}(\theta)$ | True Positive Rate / Recall | $[0, 1]$ | $\frac{\text{TP}}{\text{TP} + \text{FN}} = \mathbb{P}(\hat{p} \ge \theta \mid Y = 1)$. |
| $\text{FPR}(\theta)$ | False Positive Rate | $[0, 1]$ | $\frac{\text{FP}}{\text{FP} + \text{TN}} = \mathbb{P}(\hat{p} \ge \theta \mid Y = 0)$. |
| $\text{PPV}(\theta)$ | Precision / Positive Predictive Value | $[0, 1]$ | $\frac{\text{TP}}{\text{TP} + \text{FP}} = \mathbb{P}(Y = 1 \mid \hat{p} \ge \theta)$. |
| $\text{AUC}_{\text{ROC}}$ | Area Under ROC Curve | $[0, 1]$ | Probability that a randomly chosen positive sample scores higher than a negative sample. |
| $\text{AUC}_{\text{PR}}$ | Area Under PR Curve | $[0, 1]$ | Average Precision over all recall levels; crucial metric under severe class imbalance. |
| $\text{BS}$ | Brier Score | $[0, 1]$ | Mean squared error between predicted probabilities and binary outcomes $\frac{1}{N}\sum (\hat{p}_i - y_i)^2$. |
| $B_m$ | Calibration Bin $m$ | Subset of $\{1..N\}$ | Samples partitioned by predicted confidence into bin $m \in \{1, \dots, M\}$. |
| $\text{acc}(B_m)$ | Bin Empirical Accuracy | $[0, 1]$ | Fraction of true positives in bin $B_m$: $\frac{1}{|B_m|}\sum_{i \in B_m} y_i$. |
| $\text{conf}(B_m)$ | Bin Average Confidence | $[0, 1]$ | Mean predicted probability in bin $B_m$: $\frac{1}{|B_m|}\sum_{i \in B_m} \hat{p}_i$. |
| $\text{ECE}$ | Expected Calibration Error | $[0, 1]$ | Weighted average gap between accuracy and confidence across all bins. |
| $\text{MCE}$ | Maximum Calibration Error | $[0, 1]$ | Worst-case calibration gap across all bins: $\max_m |\text{acc}(B_m) - \text{conf}(B_m)|$. |
| $T$ | Calibration Temperature | $\mathbb{R}^+$ | Positive scalar learned on validation set to rescale logits: $\hat{p}_i(T) = \sigma(z_i / T)$. |

---

## 3. Confusion Matrix & Threshold-Varying Metrics

For a chosen threshold $\theta \in [0, 1]$, instances with $\hat{p}_i \ge \theta$ are predicted positive ($\hat{y}_i = 1$), else negative ($\hat{y}_i = 0$).

```
                      Actual Positive (y = 1)          Actual Negative (y = 0)
Predicted Positive  ┌────────────────────────────────┬────────────────────────────────┐
(p_hat ≥ θ)         │       True Positive (TP)       │      False Positive (FP)       │
                    ├────────────────────────────────┼────────────────────────────────┤
Predicted Negative  │       False Negative (FN)      │       True Negative (TN)       │
(p_hat < θ)         └────────────────────────────────┴────────────────────────────────┘
```

Core derived metrics:
- **True Positive Rate (TPR / Recall / Sensitivity)**: $\text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{\text{TP}}{N_+}$
- **False Positive Rate (FPR / Fall-out)**: $\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}} = \frac{\text{FP}}{N_-}$
- **Precision (Positive Predictive Value)**: $\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$
- **Specificity (True Negative Rate)**: $\text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}} = 1 - \text{FPR}$
- **$F_1$ Score**: $F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2\text{TP}}{2\text{TP} + \text{FP} + \text{FN}}$

---

## 4. The ROC Curve & ROC-AUC

The **Receiver Operating Characteristic (ROC)** curve plots $\text{TPR}(\theta)$ on the y-axis versus $\text{FPR}(\theta)$ on the x-axis for all $\theta \in [0, 1]$.

```
TPR (Recall)
  1.0 ┌───────────────────────────────────────┐ Perfect Model (AUC = 1.0)
      │                                       │
      │                  . · * * *            │
      │             . · '                     │
      │         · '                           │ Realistic Model (AUC = 0.88)
      │     . '                               │
      │ . '                                   │ Random Classifier (AUC = 0.50)
  0.0 └───────────────────────────────────────┴─▶ FPR
     0.0                                     1.0
```

### 4.1 Trapezoidal Integration
Given sorted unique thresholds creating points $(x_k, y_k) = (\text{FPR}_k, \text{TPR}_k)$ for $k = 0, \dots, K$ where $(x_0, y_0) = (0, 0)$ and $(x_K, y_K) = (1, 1)$:
$$\boxed{\text{AUC}_{\text{ROC}} = \sum_{k=1}^K \frac{y_k + y_{k-1}}{2} (x_k - x_{k-1})}$$

### 4.2 The Wilcoxon-Mann-Whitney Identity
A profound mathematical property of $\text{AUC}_{\text{ROC}}$ is that it is exactly equal to the probability that a randomly drawn positive instance receives a higher score than a randomly drawn negative instance:
$$\boxed{\text{AUC}_{\text{ROC}} = \mathbb{P}\left(\hat{p}_+ > \hat{p}_-\right) + \frac{1}{2}\mathbb{P}\left(\hat{p}_+ = \hat{p}_-\right)}$$

**Proof**:
Let $S_+ = \{i \mid y_i = 1\}$ with $|S_+| = N_+$ and $S_- = \{j \mid y_j = 0\}$ with $|S_-| = N_-$.
Define the Mann-Whitney kernel:
$$K(\hat{p}_i, \hat{p}_j) = \begin{cases} 1 & \text{if } \hat{p}_i > \hat{p}_j \\ 0.5 & \text{if } \hat{p}_i = \hat{p}_j \\ 0 & \text{if } \hat{p}_i < \hat{p}_j \end{cases}$$
The Mann-Whitney $U$-statistic is $U = \sum_{i \in S_+} \sum_{j \in S_-} K(\hat{p}_i, \hat{p}_j)$.
Integrating the step functions across all possible cutoffs shows:
$$\text{AUC}_{\text{ROC}} = \frac{U}{N_+ \cdot N_-} = \frac{1}{N_+ N_-} \sum_{i \in S_+} \sum_{j \in S_-} K(\hat{p}_i, \hat{p}_j) \quad \blacksquare$$

---

## 5. Precision-Recall Curves & Why ROC-AUC Misleads on Imbalanced Data

Consider a fraud detection problem with $N = 1{,}000{,}000$ transactions where only $N_+ = 1{,}000$ are fraudulent ($0.1\%$ prevalence, $N_- = 999{,}000$).

Suppose a model produces:
- $\text{TP} = 900$, $\text{FN} = 100 \implies \text{TPR} = \frac{900}{1000} = 90\%$
- $\text{FP} = 9{,}990$, $\text{TN} = 989{,}010 \implies \text{FPR} = \frac{9{,}990}{999{,}000} \approx 1.0\%$

On the ROC curve:
- $\text{TPR} = 0.90$ and $\text{FPR} = 0.01$. The ROC curve looks virtually flawless!
- But look at the **Precision**:
  $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{900}{900 + 9{,}990} = \frac{900}{10{,}890} \approx \mathbf{8.26\%}$$
- More than $91\%$ of all flagged fraud alerts are false alarms!

> [!IMPORTANT]
> The denominator of $\text{FPR}$ is $N_-$. When $N_- \gg N_+$, a massive number of false positives generates a tiny change in $\text{FPR}$. Precision, however, pits $\text{TP}$ directly against $\text{FP}$. **Under class imbalance, always report Precision-Recall AUC (PR-AUC / Average Precision)**.

$$\boxed{\text{PR-AUC} = \sum_{k=1}^K (R_k - R_{k-1}) \cdot P_k}$$
where $R_k$ and $P_k$ are Recall and Precision at the $k$-th score threshold.

---

## 6. The Brier Score & Its Canonical Decomposition

The **Brier Score** (Brier, 1950) is the strictly proper scoring rule measuring mean squared probability error:
$$\boxed{\text{BS} = \frac{1}{N}\sum_{i=1}^N (\hat{p}_i - y_i)^2}$$
where $\text{BS} \in [0, 1]$, and lower is better.

### 6.1 Murphy's Canonical Decomposition (1973)
Partition the predictions into $M$ discrete confidence bins $B_1, \dots, B_M$.
For bin $m$:
- Bin size: $n_m = |B_m|$
- Average confidence: $\bar{p}_m = \frac{1}{n_m}\sum_{i \in B_m} \hat{p}_i$
- Empirical positive rate: $\bar{y}_m = \frac{1}{n_m}\sum_{i \in B_m} y_i$
- Base rate (overall positive prevalence): $\bar{y} = \frac{1}{N}\sum_{i=1}^N y_i$

The Brier score decomposes into three orthogonal components:
$$\boxed{\text{BS} = \text{Reliability} - \text{Resolution} + \text{Uncertainty}}$$

```
                       BRIER SCORE DECOMPOSITION
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
   RELIABILITY (REL)         RESOLUTION (RES)         UNCERTAINTY (UNC)
   (Calibration error)       (Discrimination power)   (Inherent randomness)
   Lower is better           Higher is better         Constant for dataset
   0 for perfect model       Maximized when p_m = y   y_bar * (1 - y_bar)
   Σ (n_m/N)*(p_bar_m - y_m)^2 Σ (n_m/N)*(y_m - y_bar)^2
```

1. **Reliability ($\text{REL}$)**: Measures calibration. How closely average confidence matches empirical accuracy within each bin:
   $$\text{REL} = \sum_{m=1}^M \frac{n_m}{N} (\bar{p}_m - \bar{y}_m)^2$$
2. **Resolution ($\text{RES}$)**: Measures discrimination. How far bin probabilities deviate from the prior base rate:
   $$\text{RES} = \sum_{m=1}^M \frac{n_m}{N} (\bar{y}_m - \bar{y})^2$$
3. **Uncertainty ($\text{UNC}$)**: Dataset entropy / intrinsic irreducible noise:
   $$\text{UNC} = \bar{y}(1 - \bar{y})$$

---

## 7. Expected Calibration Error (ECE) & Reliability Diagrams

A model is **perfectly calibrated** if:
$$\mathbb{P}(Y = 1 \mid \hat{p} = p) = p, \quad \forall p \in [0, 1]$$

### 7.1 Equal-Width Binning Formulation
Partition the unit interval $[0, 1]$ into $M$ bins $I_m = (\frac{m-1}{M}, \frac{m}{M}]$.
Let $B_m = \{i \mid \hat{p}_i \in I_m\}$.
- **Bin Accuracy**: $\text{acc}(B_m) = \frac{1}{|B_m|}\sum_{i \in B_m} y_i$
- **Bin Confidence**: $\text{conf}(B_m) = \frac{1}{|B_m|}\sum_{i \in B_m} \hat{p}_i$

The **Expected Calibration Error (ECE)** is the sample-weighted absolute difference:
$$\boxed{\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left|\text{acc}(B_m) - \text{conf}(B_m)\right|}$$

The **Maximum Calibration Error (MCE)** bounds the worst-case safety hazard:
$$\boxed{\text{MCE} = \max_{m \in \{1, \dots, M\}} \left|\text{acc}(B_m) - \text{conf}(B_m)\right|}$$

```
Confidence vs Accuracy (Reliability Diagram)
  Accuracy
    1.0 ┌───────────────────────────────────────/ Perfect Calibration (y = x)
        │                                     / 
        │                                   /   ■ Overconfident Gap
        │                                 /     ■ (conf > acc)
        │                               / ■ 
        │                             / ■ ■
        │                           / ■ ■ ■
        │                         / ■ ■ ■ ■
        │                       / ■ ■ ■ ■ ■
    0.0 └──────────────────────┴────────┴───────┴─▶ Confidence
       0.0                                     1.0
```

---

## 8. Post-Hoc Calibration: Platt Scaling & Temperature Scaling

Guo et al. (2017, *On Calibration of Modern Neural Networks*) proved that while 1990s shallow neural networks were well-calibrated, modern deep networks with batch normalization, high depth, and weight decay are systematically **overconfident**.

### 8.1 Temperature Scaling
Let $z_i \in \mathbb{R}$ be the uncalibrated logit for instance $i$ before the sigmoid function: $\hat{p}_i = \sigma(z_i) = \frac{1}{1 + e^{-z_i}}$.
Temperature scaling introduces a single learned scalar parameter $T > 0$:
$$\boxed{\hat{p}_i(T) = \sigma\left(\frac{z_i}{T}\right) = \frac{1}{1 + e^{-z_i / T}}}$$

- If $T > 1$: Logits are softened toward $0$, reducing overconfident probabilities toward $0.5$.
- If $T < 1$: Logits are sharpened, heightening confidence.
- If $T = 1$: Original uncalibrated probabilities are preserved.

### 8.2 Optimization Objective
$T$ is optimized on a held-out validation set to minimize negative log-likelihood (binary cross-entropy):
$$\mathcal{L}(T) = -\frac{1}{N_{\text{val}}}\sum_{i=1}^{N_{\text{val}}} \left[ y_i \log \sigma\left(\frac{z_i}{T}\right) + (1 - y_i) \log \left(1 - \sigma\left(\frac{z_i}{T}\right)\right) \right]$$

### 8.3 Invariance Theorems of Temperature Scaling
Because $f(z) = z / T$ is a strictly monotonically increasing function for any $T > 0$:
1. **Rank Order Invariance**: For any two instances $i$ and $j$, $z_i > z_j \iff z_i / T > z_j / T$.
2. **ROC-AUC Invariance**: $\text{AUC}_{\text{ROC}}(T) = \text{AUC}_{\text{ROC}}(\text{uncalibrated})$.
3. **Threshold-0.5 Accuracy Invariance**: If $z_i > 0$, then $z_i / T > 0$; predictions at default threshold $0.5$ do not change!
4. **Calibration Rectification**: $\text{ECE}$ and Brier Score improve dramatically without compromising classification accuracy or ranking ability.
