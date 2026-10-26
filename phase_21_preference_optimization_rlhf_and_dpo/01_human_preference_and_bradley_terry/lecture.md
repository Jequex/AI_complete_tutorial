# Topic 21.1: Human Preference Modeling & The Bradley-Terry Framework

---

## 1. Executive Summary & The 12-Year-Old Analogy

After Supervised Fine-Tuning (SFT), a language model understands conversational conventions, follows prompt formats, and generates coherent structured text. However, SFT suffers from a fundamental mathematical bottleneck: **it treats all demonstrations in the training set as equally optimal ground truth**. If a dataset contains human answers that are polite but factually hallucinated, or verbose but intellectually shallow, standard maximum likelihood training simply forces the model to memorize those flaws:

$$\mathcal{L}_{\text{SFT}}(\theta) = -\sum_{t=1}^T \log P_\theta(y_t \mid x, y_{<t})$$

To create frontier AI that exhibits nuanced judgment, deep helpfulness, and reliable safety, we must optimize models according to **human preference**. But how do we mathematically measure human preference?

```
+----------------------------------------------------------------------------------------------------+
|                               THE HUMAN PREFERENCE MEASUREMENT PROBLEM                             |
|                                                                                                    |
|  SCALAR RATING (UNRELIABLE):                                                                       |
|  "Rate this response from 1 to 10."                                                                |
|  Annotator Alice: "8/10 (I'm an easy grader)."                                                     |
|  Annotator Bob:   "4/10 (I only give 8+ to Pulitzer prize winners)."                               |
|  Annotator Carol (Morning): "7/10."  Annotator Carol (After Lunch): "5/10 (I have a headache)."    |
|  --> HIGH VARIANCE, SCALE DRIFT, ZERO UNIVERSAL CALIBRATION.                                       |
|                                                                                                    |
|  PAIRWISE COMPARISON (ROBUST & CONSISTENT):                                                        |
|  "Between Response A and Response B, which one is more helpful and truthful?"                      |
|  Annotator Alice: "A > B"                                                                          |
|  Annotator Bob:   "A > B"                                                                          |
|  Annotator Carol: "A > B"                                                                          |
|  --> HIGH INTER-ANNOTATOR AGREEMENT, SCALE-INVARIANT, COGNITIVELY SIMPLE.                          |
+----------------------------------------------------------------------------------------------------+
```

### The 12-Year-Old Analogy: The Soda Taste Test Challenge

Imagine you are hired by a soda company to find the most delicious soda recipe in the world:

1. **The Flawed Approach (The 1-to-10 Scorecard):**
   - You hand a glass of Soda A to your friend Charlie and say, *"Give this a score from 1 to 10."* Charlie says, *"7.4."*
   - Three hours later, you hand a glass of Soda B to your sister Maya and ask for a score. Maya says, *"6.8."*
   - Does Charlie's 7.4 mean Soda A is better than Soda B? No! Charlie might give almost everything a 7 or 8 because he hates sounding mean. Maya might be a harsh food critic who gives almost everything a 5. Even worse, if Charlie tests Soda A on an empty stomach after soccer practice, he might rate it an 8.5; if he tests it right after eating a chocolate donut, he might rate the exact same soda a 4.0!
   - Absolute numerical numbers from humans are wildly noisy and poorly calibrated.

2. **The Robust Approach (The Blind Sip Challenge):**
   - Instead, you pour Soda A into Cup 1 and Soda B into Cup 2.
   - You blindfold Charlie, Maya, and twenty other friends. You say: *"Take a sip of both. Which cup tastes better to you: Cup 1 or Cup 2?"*
   - Even though Charlie and Maya have totally different grading personalities, they can both easily point to Cup 1 and say: *"Cup 1 is clearly crisper and less sugary."*
   - Pairwise comparison removes the confusion of inventing an arbitrary numerical scale.

In 1952, statisticians Ralph A. Bradley and Milton E. Terry established the mathematical bridge that converts these intuitive pairwise comparisons $(y_w \succ y_l)$ into smooth, continuous, differentiable reward scores $r(x, y)$. This formulation is the mathematical bedrock of Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO).

---

## 2. Mathematical Symbol Decoder

| Symbol | Formal Domain | Description & Meaning |
| :--- | :--- | :--- |
| $x \in \mathcal{X}$ | Discrete Sequence | Context prompt or user instruction ($x = (x_1, \dots, x_M)$). |
| $y_w, y_l \in \mathcal{Y}$ | Discrete Sequences | Winning (preferred) completion $y_w$ and losing (dispreferred) completion $y_l$. |
| $y_1 \succ y_2 \mid x$ | Binary Relation | Binary preference event: "$y_1$ is preferred over $y_2$ given context $x$." |
| $r(x, y) \in \mathbb{R}$ | Real Scalar | Latent (unobserved) scalar reward assigned to completion $y$ for prompt $x$. |
| $\sigma(z) = \frac{1}{1 + e^{-z}}$ | Function $\mathbb{R} \to (0, 1)$ | Standard logistic sigmoid function. |
| $\mathcal{D}_{\text{pref}}$ | Preference Dataset | Curated collection of comparison tuples $(x, y_w, y_l)$. |
| $\Phi(z)$ | CDF | Standard normal cumulative distribution function (Thurstonian model). |
| $\mathcal{L}_{\text{MLE}}(\mathbf{r})$ | Objective Function | Log-likelihood of Bradley-Terry preference observations given latent rewards. |
| $K$ | Integer | Number of ranked alternative completions in Plackett-Luce generalization. |
| $\kappa$ (Kappa) | Metric $\in [-1, 1]$ | Cohen's or Fleiss' inter-annotator agreement coefficient. |

---

## 3. Why Scalar Scoring Fails in Human Annotation

Before examining Bradley-Terry, we must formalize why assigning absolute scalar scores $s \in [0, 10]$ fails both mathematically and psychologically:

### 3.1 Inter-Rater Inconsistency & Calibration Drift
Let $u^*(x, y) \in \mathbb{R}$ denote the true latent utility of completion $y$. An annotator $k$ evaluates the response and maps it to an observed scalar score $s^{(k)} \in [0, 10]$ through an unknown noisy personal transfer function $g_k$:

$$s^{(k)}(x, y) = g_k(u^*(x, y)) + \epsilon^{(k)}, \quad \epsilon^{(k)} \sim \mathcal{N}(0, \sigma_k^2)$$

Different human annotators exhibit drastically divergent characteristics:
1. **Bias Shift:** Generous annotators have $\mathbb{E}[g_{\text{easy}}(u)] = u + 2.0$, while strict annotators have $\mathbb{E}[g_{\text{strict}}(u)] = u - 1.5$.
2. **Scale Compression & Expansion:** Some annotators only utilize the range $[6.0, 8.0]$, while others utilize $[1.0, 10.0]$.
3. **Temporal Drift:** A human rater's scoring threshold drifts over the course of a 4-hour labeling shift due to cognitive fatigue.

As a result, the Pearson correlation across different human annotators on continuous 1-10 scales is notoriously low (often $\rho < 0.40$), and inter-annotator agreement $\kappa < 0.35$.

### 3.2 Ordinal Superiority
When the question is simplified to a binary comparison:
$$\text{Is } y_1 \text{ better than } y_2 \text{ given prompt } x?$$
The decision rule reduces to whether the perceived utility difference exceeds zero:

$$\mathbb{I}[y_1 \succ y_2] = \mathbb{I}[u^*(x, y_1) - u^*(x, y_2) + (\epsilon_1 - \epsilon_2) > 0]$$

Crucially, **any monotonic transformation applied to utility preserves the ordering**:
$$u(y_1) > u(y_2) \iff f(u(y_1)) > f(u(y_2)) \quad \forall f' > 0$$
Pairwise comparisons eliminate the need for annotators to establish an arbitrary origin or unit scale. In empirical alignment studies (e.g., Anthropic, OpenAI, DeepMind), pairwise human agreement jumps to $\kappa \approx 0.65 - 0.78$.

---

## 4. The Bradley-Terry Preference Model (1952)

The Bradley-Terry model posits that every completion $y$ has a latent, unobserved real-valued reward $r(x, y) \in \mathbb{R}$. When presented with two candidate completions $y_1$ and $y_2$, the probability that an annotator prefers $y_1$ over $y_2$ is governed by the relative exponential strength of their respective rewards:

$$P(y_1 \succ y_2 \mid x) = \frac{\exp(r(x, y_1))}{\exp(r(x, y_1)) + \exp(r(x, y_2))}$$

### 4.1 Reduction to the Logistic Sigmoid
Dividing both the numerator and the denominator by $\exp(r(x, y_1))$:

$$P(y_1 \succ y_2 \mid x) = \frac{1}{1 + \frac{\exp(r(x, y_2))}{\exp(r(x, y_1))}} = \frac{1}{1 + \exp\left(-(r(x, y_1) - r(x, y_2))\right)}$$

Recognizing the definition of the standard logistic sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$:

$$\boxed{P(y_1 \succ y_2 \mid x) = \sigma\left(r(x, y_1) - r(x, y_2)\right)}$$

```
          P(y1 > y2)
             ^
         1.0 |                                     ***************
             |                                ****
             |                             ***
         0.5 |-----------------------*****------------------------
             |                    ***
             |                ****
         0.0 | ***************
             +-----------------------+----------------------->
                    r(y1) < r(y2)    0     r(y1) > r(y2)
                                 r(y1) - r(y2)
```

### 4.2 Key Mathematical Properties of Bradley-Terry
1. **Symmetry & Anti-Symmetry:**
   $$P(y_2 \succ y_1 \mid x) = 1 - P(y_1 \succ y_2 \mid x) = 1 - \sigma(r_1 - r_2) = \sigma(r_2 - r_1)$$
2. **Indifference at Equality:**
   If $r(x, y_1) = r(x, y_2)$, then $r_1 - r_2 = 0$, and:
   $$P(y_1 \succ y_2 \mid x) = \sigma(0) = \frac{1}{1 + e^0} = 0.5$$
3. **Translation Invariance (Gauge Freedom):**
   Adding an arbitrary constant $c \in \mathbb{R}$ to all reward values leaves preference probabilities completely unchanged:
   $$\sigma\left((r_1 + c) - (r_2 + c)\right) = \sigma(r_1 - r_2)$$
   To make latent rewards uniquely identifiable, we must enforce an anchoring condition, such as centering $\sum_{i} r_i = 0$ or fixing $r_0 = 0$.

---

## 5. Maximum Likelihood Estimation (MLE) of Latent Rewards

Given a preference dataset $\mathcal{D} = \{(x^{(k)}, y_w^{(k)}, y_l^{(k)})\}_{k=1}^N$ where $y_w^{(k)}$ was preferred over $y_l^{(k)}$, the likelihood of the dataset under parameters $\mathbf{r}$ is:

$$\mathcal{P}(\mathcal{D} \mid \mathbf{r}) = \prod_{k=1}^N P(y_w^{(k)} \succ y_l^{(k)} \mid x^{(k)}) = \prod_{k=1}^N \sigma\left(r(x^{(k)}, y_w^{(k)}) - r(x^{(k)}, y_l^{(k)})\right)$$

Taking the natural logarithm yields the log-likelihood objective:

$$\mathcal{L}_{\text{MLE}}(\mathbf{r}) = \sum_{k=1}^N \log \sigma\left(r(x^{(k)}, y_w^{(k)}) - r(x^{(k)}, y_l^{(k)})\right)$$

### 5.1 First-Order Derivative (Gradient)
Recall the derivative of the log-sigmoid function:
$$\frac{d}{dz} \log \sigma(z) = \frac{1}{\sigma(z)} \cdot \sigma(z)(1 - \sigma(z)) = 1 - \sigma(z) = \sigma(-z)$$

Let $\Delta_k = r(x^{(k)}, y_w^{(k)}) - r(x^{(k)}, y_l^{(k)})$. Then the gradient with respect to the winning reward $r(y_w)$ and losing reward $r(y_l)$ is:

$$\frac{\partial \mathcal{L}_{\text{MLE}}}{\partial r(x^{(k)}, y_w^{(k)})} = 1 - \sigma(\Delta_k) = \sigma(-\Delta_k)$$
$$\frac{\partial \mathcal{L}_{\text{MLE}}}{\partial r(x^{(k)}, y_l^{(k)})} = -(1 - \sigma(\Delta_k)) = -\sigma(-\Delta_k)$$

**Intuition:** If the model already assigns a much higher reward to $y_w$ ($\Delta_k \gg 0$), $\sigma(-\Delta_k) \approx 0$, and the gradient is near zero. If the model incorrectly predicts that $y_l$ is superior ($\Delta_k < 0$), $\sigma(-\Delta_k) \approx 1$, producing a strong gradient pushing $r(y_w)$ upward and $r(y_l)$ downward.

### 5.2 Strict Concavity & Global Optimality
The second derivative with respect to $\Delta_k$ is:
$$\frac{\partial^2}{\partial \Delta_k^2} \log \sigma(\Delta_k) = -\sigma(\Delta_k)(1 - \sigma(\Delta_k)) < 0 \quad \forall \Delta_k \in \mathbb{R}$$
Because the Hessian is negative semi-definite everywhere, the log-likelihood is **strictly concave** (modulo the translational gauge freedom). Any gradient ascent or Newton-Raphson optimizer is guaranteed to converge to the global optimum.

---

## 6. Thurstonian Model vs. Bradley-Terry

The Bradley-Terry model is one specific formulation of random utility models. In 1927, Louis Leon Thurstone proposed the **Thurstonian Model (Law of Comparative Judgment, Case V)**:

```
Thurstone:       R_i = r_i + ε_i,   where ε_i ~ Normal(0, σ^2)
Bradley-Terry:   R_i = r_i + ε_i,   where ε_i ~ Gumbel(0, 1)
```

Under Thurstone's model, the perceived quality $R_i$ is normally distributed around true quality $r_i$. The difference between two independent normal variables is also normal:
$$R_1 - R_2 \sim \mathcal{N}\left(r_1 - r_2, 2\sigma^2\right)$$
Hence, the preference probability is given by the Gaussian Cumulative Distribution Function (the **Probit** model):

$$P_{\text{Thurstone}}(y_1 \succ y_2) = \Phi\left(\frac{r_1 - r_2}{\sqrt{2}\sigma}\right)$$

### Comparison Table

| Attribute | Bradley-Terry Model | Thurstonian (Case V) Model |
| :--- | :--- | :--- |
| **Noise Distribution** | Gumbel / Type I Extreme Value | Gaussian / Normal |
| **Link Function** | Logistic Sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$ | Probit / Normal CDF $\Phi(z)$ |
| **Tails** | Heavier tails (logistical) | Lighter exponential tails (Gaussian) |
| **Analytical Form** | Simple closed-form derivative $\sigma(z)(1 - \sigma(z))$ | Requires numerical error function $\text{erf}(z)$ |
| **Computational Efficiency** | Extremely fast, analytically clean | Slower evaluation of Gaussian integrals |
| **LLM Alignment Adoption** | **Industry Standard** (RLHF, DPO, Elo) | Rare in deep learning, common in psychometrics |

---

## 7. The Plackett-Luce Model ($K > 2$ Ranked Outputs)

In many annotation environments, annotators are presented with $K \ge 3$ candidate completions $\{y_1, y_2, \dots, y_K\}$ produced by different models or temperatures, and requested to rank them in order of quality: $y_{(1)} \succ y_{(2)} \succ \dots \succ y_{(K)}$.

The **Plackett-Luce model** (Plackett 1975, Luce 1959) generalizes Bradley-Terry to full permutations:

$$P(y_{(1)} \succ y_{(2)} \succ \dots \succ y_{(K)} \mid x) = \prod_{k=1}^{K-1} \frac{\exp(r(x, y_{(k)}))}{\sum_{j=k}^K \exp(r(x, y_{(j)}))}$$

```
Stage 1: Choose 1st place from all K items.             exp(r_1) / [exp(r_1) + ... + exp(r_K)]
Stage 2: Choose 2nd place from remaining K-1 items.     exp(r_2) / [exp(r_2) + ... + exp(r_K)]
...
Stage K-1: Choose (K-1)th place from remaining 2 items. exp(r_{K-1}) / [exp(r_{K-1}) + exp(r_K)]
```

When $K = 2$, Plackett-Luce simplifies directly to Bradley-Terry:
$$\frac{\exp(r_1)}{\exp(r_1) + \exp(r_2)} = \sigma(r_1 - r_2)$$

---

## 8. Transitivity, Stochastic Transitivity & The Condorcet Paradox

A critical assumption in scalar reward modeling is **transitivity**:
$$r(A) > r(B) \text{ and } r(B) > r(C) \implies r(A) > r(C)$$

In deterministic mathematics, real numbers are strictly transitive. However, human preferences are stochastic. In probabilistic modeling, transitivity is categorized into three levels of strength:

1. **Weak Stochastic Transitivity (WST):**
   $$P(A \succ B) \ge 0.5 \text{ and } P(B \succ C) \ge 0.5 \implies P(A \succ C) \ge 0.5$$
2. **Moderate Stochastic Transitivity (MST):**
   $$P(A \succ C) \ge \min(P(A \succ B), P(B \succ C))$$
3. **Strong Stochastic Transitivity (SST):**
   $$P(A \succ C) \ge \max(P(A \succ B), P(B \succ C))$$

The Bradley-Terry model mathematically enforces **Strong Stochastic Transitivity (SST)**. 

### Proof of SST in Bradley-Terry:
Let $\Delta_{AB} = r(A) - r(B) \ge 0$ and $\Delta_{BC} = r(B) - r(C) \ge 0$.
Then $\Delta_{AC} = r(A) - r(C) = \Delta_{AB} + \Delta_{BC} \ge \max(\Delta_{AB}, \Delta_{BC})$.
Since $\sigma(z)$ is strictly monotonically increasing:
$$\sigma(\Delta_{AC}) \ge \max(\sigma(\Delta_{AB}), \sigma(\Delta_{BC})) \implies P(A \succ C) \ge \max(P(A \succ B), P(B \succ C)) \quad \blacksquare$$

### The Condorcet Paradox in Human Preference
Real-world human judges can exhibit **intransitive preference cycles**:
$$A \succ B \succ C \succ A$$
For example, response $A$ is praised for conciseness over verbose $B$; response $B$ is praised for comprehensive depth over bulleted $C$; but response $C$ is praised for clean formatting over concise $A$.

When cyclical data is fed into a Bradley-Terry model, the model cannot assign scalar rewards that satisfy all three inequalities simultaneously. Instead, the maximum likelihood reward parameters will settle at a compromised stationary point where log-loss is minimized across the cycles. Recognizing and filtering noisy, intransitive annotator data is an essential preprocessing step in RLHF dataset curation.

---

## 9. Summary & Architectural Next Steps

- Scalar human ratings are plagued by inter-rater variance, calibration drift, and cognitive fatigue.
- Pairwise comparisons $(y_w \succ y_l \mid x)$ are scale-free, cognitively natural, and achieve high annotator agreement.
- The **Bradley-Terry model** maps latent continuous rewards $r(x, y)$ to pairwise probabilities via the logistic sigmoid $\sigma(r_w - r_l)$.
- In **Topic 21.2**, we will utilize this Bradley-Terry formulation to train a neural **Reward Model** $[d_{\text{model}} \to 1]$ that replaces human annotators during policy optimization.
