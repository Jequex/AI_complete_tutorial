# Topic 5.2: Confidence Intervals, Hypothesis Testing & AI A/B Testing

---

## 1. Executive Summary & Intuitive Foundations

### Why Single-Number Guesses Lie
In Topic 5.1, we computed point estimates like the sample mean $\bar{X}$.
Suppose an AI engineer fine-tunes a new model (Model B) and evaluates it against the baseline (Model A) on 200 customer support chats:
- **Model A Accuracy:** $82.0\%$
- **Model B Accuracy:** $85.0\%$

Did the fine-tuning genuinely make the model better, or was Model B just lucky with an easier set of 200 questions?
If you ship Model B to production based purely on this $3\%$ bump, you might crash user retention if the difference was pure random noise!

To make scientific decisions, AI engineers need two tools:
1. **Confidence Intervals:** Put error bars around our metrics (*"We are 95% confident the true accuracy is between $81.2\%$ and $88.8\%$"*).
2. **Hypothesis Testing:** Calculate the exact probability that an observed improvement was merely a fluke of random luck!

---

### The 12-Year-Old Analogy: The Courtroom Trial & The Weighted Coin
Imagine your friend brings a coin to a game and flips 8 Heads out of 10 flips. You suspect the coin is rigged (biased toward heads).

In statistics, we set up a **Courtroom Trial**:
1. **The Null Hypothesis ($H_0$, "Innocent until proven guilty"):**
   We assume the coin is completely fair ($P(\text{Heads}) = 0.50$).
2. **The Alternative Hypothesis ($H_1$, "Guilty as charged"):**
   The coin is unfair ($P(\text{Heads}) > 0.50$).
3. **The Evidence:**
   8 Heads in 10 flips.
4. **The $p$-value:**
   We ask: *"If the coin was 100% fair, how often would someone get 8 or more Heads just by pure luck?"*
   By binomial probability, that chance is about **$5.5\%$**.
5. **The Verdict:**
   A standard jury threshold is $5\%$ ($\alpha = 0.05$). Since $5.5\% > 5\%$, there is not enough evidence "beyond a reasonable doubt" to convict the coin of cheating. You cannot reject the null hypothesis!

---

## 2. Confidence Intervals: Bounding Our Ignorance

A **Confidence Interval (CI)** gives a range of plausible values for an unknown population parameter $\theta$ at a specified confidence level (e.g. $95\%$ or $99\%$).

### Derivation from the Central Limit Theorem
By the Central Limit Theorem, the distribution of the sample mean $\bar{X}$ across repeated samples of size $n$ is approximately Normal:
$$\bar{X} \sim \mathcal{N}\left(\mu, \, \frac{\sigma^2}{n}\right)$$

Standardizing into a $z$-score:
$$Z = \frac{\bar{X} - \mu}{\sigma / \sqrt{n}} \sim \mathcal{N}(0, 1)$$

For a $95\%$ confidence level ($\alpha = 0.05$), the standard normal distribution contains $95\%$ of its probability between $-1.96$ and $+1.96$:
$$P\left( -1.96 \le \frac{\bar{X} - \mu}{\sigma / \sqrt{n}} \le +1.96 \right) = 0.95$$

Multiply all terms by the standard error $\text{SE} = \frac{\sigma}{\sqrt{n}}$:
$$P\left( -1.96 \frac{\sigma}{\sqrt{n}} \le \bar{X} - \mu \le +1.96 \frac{\sigma}{\sqrt{n}} \right) = 0.95$$

Rearranging the inequalities to isolate the true population mean $\mu$ in the center:
$$\mathbf{P\left( \bar{X} - 1.96 \frac{\sigma}{\sqrt{n}} \le \mu \le \bar{X} + 1.96 \frac{\sigma}{\sqrt{n}} \right) = 0.95}$$

Thus, the **95% Confidence Interval** is:
$$\mathbf{\text{CI}_{95\%} = \left[ \bar{X} - 1.96 \frac{\sigma}{\sqrt{n}}, \quad \bar{X} + 1.96 \frac{\sigma}{\sqrt{n}} \right]}$$

> [!NOTE]
> **Correct Interpretation of a Confidence Interval:**
> A $95\%$ confidence interval does **NOT** mean there is a $95\%$ probability that the true parameter lies in this specific numerical interval (the parameter is a fixed constant, not random!).
> It means: *"If we repeated this experiment 100 times and calculated 100 intervals, approximately 95 of those intervals would successfully trap the true population parameter $\mu$."*

---

### Unknown Population Variance & Student's $t$-Distribution
In real-world AI evaluation, we almost never know the true population standard deviation $\sigma$. We must estimate it using the sample standard deviation $S$.

When substituting $S$ for $\sigma$, the distribution is no longer Normal—it has heavier tails because $S$ adds extra uncertainty. The ratio follows **Student's $t$-distribution** with $\nu = n - 1$ degrees of freedom:
$$T = \frac{\bar{X} - \mu}{S / \sqrt{n}} \sim t_{n-1}$$

The confidence interval becomes:
$$\mathbf{\text{CI}_{1-\alpha} = \left[ \bar{X} - t_{\alpha/2, \, n-1} \frac{S}{\sqrt{n}}, \quad \bar{X} + t_{\alpha/2, \, n-1} \frac{S}{\sqrt{n}} \right]}$$
As $n \to \infty$, $t_{n-1}$ converges smoothly to the standard normal $z$-distribution.

---

## 3. Mathematical Decoder Table

| Symbol | Mathematical Name | Plain-English Meaning | Typical Value |
| :--- | :--- | :--- | :--- |
| $H_0$ | Null Hypothesis | The default assumption that there is no real effect or difference | e.g. $\mu_A = \mu_B$ |
| $H_1$ | Alternative Hypothesis | The claim you want to prove (the discovery) | e.g. $\mu_B > \mu_A$ |
| $\alpha$ | Significance Level | Maximum allowed risk of a False Alarm (Type I error) | $0.05$ (5%) or $0.01$ (1%) |
| $\beta$ | Type II Error Rate | Probability of missing a real improvement (False Negative) | $0.10$ or $0.20$ |
| $1 - \beta$ | Statistical Power | Probability of successfully detecting a real effect | $80\%$ to $95\%$ |
| $p$-value | Probability Value | Chance of seeing results this extreme assuming $H_0$ is true | Calculated from test statistic |
| $z_{\text{crit}} / t_{\text{crit}}$ | Critical Value | The threshold line on the distribution curve beyond which $H_0$ is rejected | e.g. $1.96$ for two-tailed 95% |

---

## 4. The Decision Matrix: Type I vs Type II Errors

Every statistical hypothesis test results in one of four possible outcomes:

```
                                      THE GROUND TRUTH
                          H_0 is True                 H_0 is False
                       (No Real Difference)       (Model B is Truly Better)
                    +--------------------------+---------------------------+
  Decision:         |       TYPE I ERROR       |      CORRECT DECISION     |
  Reject H_0        |      (False Alarm)       |       (True Positive)     |
  (Ship Model B)    |     Probability = α      |    Statistical Power = 1-β|
                    +--------------------------+---------------------------+
  Decision:         |     CORRECT DECISION     |       TYPE II ERROR       |
  Fail to Reject H_0|      (True Negative)     |       (Missed Discovery)  |
  (Keep Model A)    |     Probability = 1-α    |     Probability = β       |
                    +--------------------------+---------------------------+
```

1. **Type I Error ($\alpha$):** You ship a model that is actually no better (or worse) than baseline. You fooled yourself by a lucky evaluation run.
2. **Type II Error ($\beta$):** You discard an awesome model that was genuinely superior because your benchmark had too few samples to confirm it.
3. **Statistical Power ($1 - \beta$):** The sensitivity of your test. Higher sample sizes dramatically increase power!

---

## 5. What a $p$-value Actually Is (and What It Is NOT)

### Formal Definition:
$$\mathbf{p\text{-value} = P(\text{Observing data as or more extreme than } D \mid H_0 \text{ is true})}$$

If the $p$-value is smaller than your chosen threshold $\alpha$ (typically $p < 0.05$):
- **Reject $H_0$**: The observed data is too improbable under the assumption of mere chance. We conclude there is a statistically significant effect.

### Critical Misconceptions in AI:
- ❌ **FALSE:** "A $p$-value of $0.03$ means there is a $97\%$ chance Model B is better." (That is $P(H_1 | D)$, which requires Bayesian inference!).
- ❌ **FALSE:** "A tiny $p$-value means the model improvement is huge." (A tiny $p$-value only means the difference is unlikely to be zero; with $1,000,000$ test tokens, a tiny, useless improvement of $0.0001\%$ will have $p < 0.0001$!).

---

## 6. Real-World AI Application: A/B Testing Model Checkpoints

Suppose you are evaluating whether a quantized 4-bit model (Model B) degrades user acceptance compared to an unquantized 16-bit model (Model A) on a production live-traffic split.

- **Model A:** $n_A = 1{,}000$ requests, $x_A = 780$ thumbs-up ($\hat{p}_A = 0.780$)
- **Model B:** $n_B = 1{,}000$ requests, $x_B = 730$ thumbs-up ($\hat{p}_B = 0.730$)

Does the 4-bit quantization genuinely hurt user satisfaction?

### Step 1: Set Hypotheses
$$H_0: p_A = p_B \quad (\text{No difference in satisfaction})$$
$$H_1: p_A > p_B \quad (\text{Quantized model performs worse})$$

### Step 2: Pooled Proportion under $H_0$
$$\hat{p} = \frac{x_A + x_B}{n_A + n_B} = \frac{780 + 730}{1000 + 1000} = \frac{1510}{2000} = \mathbf{0.755}$$

### Step 3: Compute Standard Error of Difference
$$\text{SE}_{\text{diff}} = \sqrt{\hat{p}(1 - \hat{p})\left(\frac{1}{n_A} + \frac{1}{n_B}\right)} = \sqrt{(0.755)(0.245)\left(\frac{1}{1000} + \frac{1}{1000}\right)}$$
$$\text{SE}_{\text{diff}} = \sqrt{0.184975 \times 0.002} = \sqrt{0.00036995} \approx \mathbf{0.01923}$$

### Step 4: Calculate the $z$-Test Statistic
$$z = \frac{\hat{p}_A - \hat{p}_B}{\text{SE}_{\text{diff}}} = \frac{0.780 - 0.730}{0.01923} = \frac{0.050}{0.01923} \approx \mathbf{+2.60}$$

### Step 5: Evaluate $p$-value & Decision
For a standard normal distribution, the probability of exceeding $z = +2.60$ is:
$$p = 1 - \Phi(2.60) \approx 1 - 0.9953 = \mathbf{0.0047} \quad (0.47\%)$$

Since $p = 0.0047 < 0.01$:
**Reject $H_0$!** The performance degradation of the 4-bit model is statistically significant at the $99\%$ confidence level. Do NOT deploy Model B without further optimization!

---

## 7. Summary of Essential Takeaways

1. **Point estimates need intervals:** Never report benchmark numbers without confidence intervals or sample sizes.
2. **95% Confidence Interval:** $\bar{X} \pm 1.96 \frac{\sigma}{\sqrt{n}}$ captures the true mean in 95 out of 100 repeated trials.
3. **Hypothesis testing structure:** Assume null hypothesis $H_0$ is true until evidence proves it improbable beyond threshold $\alpha$.
4. **$p$-value:** The probability of seeing results this extreme under $H_0$.
5. **Two-sample proportions:** Essential for A/B testing LLM prompt variants, quantization schemes, and production feature flags.

