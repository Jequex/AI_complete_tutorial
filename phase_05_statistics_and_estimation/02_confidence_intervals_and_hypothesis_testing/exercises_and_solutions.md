# Topic 5.2: Exercises & Solutions — Confidence Intervals & Hypothesis Testing

---

## Problem Set Overview

| Problem | Focus | Difficulty |
| :--- | :--- | :--- |
| **1** | **95% Confidence Interval for LLM Token Latency** | Easy |
| **2** | **Two-Sample Proportion Hypothesis Test for Prompt Optimization** | Medium |
| **3** | **Required Sample Size Derivation for Statistical Power in A/B Testing** | Challenging |

---

## Problem 1: 95% Confidence Interval for LLM Token Latency

### The Context
An AI infrastructure engineer profiles a language model's time-per-output-token across $n = 64$ benchmark runs.
The sample statistics are:
- Sample Mean: $\bar{X} = 42.5 \text{ ms}$
- Sample Standard Deviation: $S = 8.0 \text{ ms}$

The engineering team has established a strict Service Level Agreement (SLA):
> *"The true average latency must not exceed $45.0 \text{ ms}$."*

### Task
1. Compute the **Standard Error** $\text{SE}(\bar{X})$.
2. Using the standard normal approximation ($z_{0.025} = 1.96$), compute the **95% Confidence Interval** for the true average latency $\mu$.
3. Based on this confidence interval, can the engineer guarantee with 95% confidence that the model satisfies the SLA?

---

### Solution 1

#### 1. Standard Error of the Mean
$$\text{SE}(\bar{X}) = \frac{S}{\sqrt{n}} = \frac{8.0}{\sqrt{64}} = \frac{8.0}{8} = \mathbf{1.0 \text{ ms}}$$

#### 2. 95% Confidence Interval
$$\text{Margin of Error} = z_{0.025} \times \text{SE} = 1.96 \times 1.0 = \mathbf{1.96 \text{ ms}}$$
$$\text{CI}_{95\%} = [\bar{X} - 1.96, \quad \bar{X} + 1.96] = [42.5 - 1.96, \quad 42.5 + 1.96] = \mathbf{[40.54 \text{ ms}, \quad 44.46 \text{ ms}]}$$

#### 3. SLA Compliance Verdict
The entire 95% confidence interval $[40.54, 44.46]$ lies strictly below the $45.0 \text{ ms}$ ceiling.
Therefore, the engineer can confidently conclude that the model meets the SLA requirements at the 95% confidence level.

---

## Problem 2: Two-Sample Proportion Hypothesis Test for Prompt Optimization

### The Context
An NLP engineer designs a new system prompt (Prompt B) with Chain-of-Thought reasoning to improve task completion over the standard prompt (Prompt A).
They run an A/B evaluation on random customer queries:
- **Prompt A:** $n_A = 400$ requests, $x_A = 240$ successful completions ($\hat{p}_A = 0.60$)
- **Prompt B:** $n_B = 400$ requests, $x_B = 272$ successful completions ($\hat{p}_B = 0.68$)

Significance level is set to $\alpha = 0.05$ (critical value $z_{0.025} = 1.96$ for a two-tailed test).

### Task
1. Formulate the Null Hypothesis $H_0$ and Alternative Hypothesis $H_1$.
2. Compute the **pooled proportion** $\hat{p}$ under $H_0$.
3. Compute the standard error of the difference $\text{SE}_{\text{diff}}$.
4. Compute the test statistic $z$.
5. Determine whether to reject $H_0$ at $\alpha = 0.05$. Should the team deploy Prompt B?

---

### Solution 2

#### 1. Hypotheses
$$H_0: p_B = p_A \quad (\text{Prompt B has no real effect})$$
$$H_1: p_B \neq p_A \quad (\text{Prompt B has a different success rate})$$

#### 2. Pooled Proportion
$$\hat{p} = \frac{x_A + x_B}{n_A + n_B} = \frac{240 + 272}{400 + 400} = \frac{512}{800} = \mathbf{0.640}$$

#### 3. Standard Error of the Difference
$$\text{SE}_{\text{diff}} = \sqrt{\hat{p}(1 - \hat{p}) \left( \frac{1}{n_A} + \frac{1}{n_B} \right)} = \sqrt{(0.64)(0.36) \left( \frac{1}{400} + \frac{1}{400} \right)}$$
$$\text{SE}_{\text{diff}} = \sqrt{0.2304 \times \frac{2}{400}} = \sqrt{0.2304 \times 0.005} = \sqrt{0.001152} \approx \mathbf{0.03394}$$

#### 4. The $z$-Test Statistic
$$z = \frac{\hat{p}_B - \hat{p}_A}{\text{SE}_{\text{diff}}} = \frac{0.680 - 0.600}{0.03394} = \frac{0.080}{0.03394} \approx \mathbf{+2.357}$$

#### 5. Statistical Decision
Since $|z| = 2.357 > 1.96$:
- Two-tailed $p$-value: $p = 2 \times (1 - \Phi(2.357)) \approx 2 \times (1 - 0.9908) = \mathbf{0.0184} \quad (1.84\%)$.
- Because $p = 0.0184 < 0.05$, **we reject the Null Hypothesis $H_0$**.
- **Conclusion:** Prompt B produces a statistically significant $8\%$ improvement in task success. Deploy Prompt B to production!

---

## Problem 3: Required Sample Size Derivation for Statistical Power in A/B Testing

### The Context
Before launching an expensive evaluation on human raters, an AI team needs to know how many queries per model ($n$) they must collect to detect a meaningful improvement without wasting budget.
- Baseline model accuracy: $p_1 = 0.80$
- Minimum detectable effect: $p_2 = 0.85$ ($\Delta = 0.05$)
- Significance level: $\alpha = 0.05$ (two-tailed, $z_{\alpha/2} = 1.96$)
- Desired Statistical Power: $1 - \beta = 0.80$ ($\beta = 0.20$, $z_\beta = 0.8416$)

### Task
1. State the standard sample size formula for comparing two independent proportions with equal group sizes:
   $$n = \frac{(z_{\alpha/2} + z_\beta)^2 \cdot [p_1(1 - p_1) + p_2(1 - p_2)]}{(p_2 - p_1)^2}$$
2. Calculate the variance term $[p_1(1 - p_1) + p_2(1 - p_2)]$.
3. Compute the required sample size $n$ per model.
4. If the team only has budget for $n = 300$ samples per model, what happens to their statistical power?

---

### Solution 3

#### 1. Formula Inputs
- $z_{\alpha/2} = 1.96$
- $z_\beta = 0.8416$
- Sum of critical values: $(1.96 + 0.8416) = 2.8016 \implies (2.8016)^2 \approx \mathbf{7.849}$

#### 2. Variance Sum
$$p_1(1 - p_1) = 0.80 \times 0.20 = 0.160$$
$$p_2(1 - p_2) = 0.85 \times 0.15 = 0.1275$$
$$\text{Variance Sum} = 0.160 + 0.1275 = \mathbf{0.2875}$$

#### 3. Sample Size Calculation
$$\Delta^2 = (0.85 - 0.80)^2 = (0.05)^2 = \mathbf{0.0025}$$
$$n = \frac{7.849 \times 0.2875}{0.0025} = \frac{2.2565875}{0.0025} \approx \mathbf{902.6} \implies \mathbf{903 \text{ samples per model}}$$
Total evaluations needed: $2 \times 903 = \mathbf{1{,}806 \text{ ratings}}$.

#### 4. Underpowered Sample Size Analysis ($n = 300$)
With only $n = 300$, the test is severely **underpowered**:
$$\frac{\sqrt{n} \cdot \Delta}{\sqrt{p_1(1-p_1) + p_2(1-p_2)}} - z_{\alpha/2} = \frac{\sqrt{300} \cdot 0.05}{\sqrt{0.2875}} - 1.96 = \frac{17.32 \times 0.05}{0.5362} - 1.96 \approx 1.615 - 1.96 = -0.345$$
Power drops to $\Phi(-0.345) \approx \mathbf{36.5\%}$!
The team has a nearly **64% chance of missing a real 5% breakthrough** due to an inadequate sample size!

