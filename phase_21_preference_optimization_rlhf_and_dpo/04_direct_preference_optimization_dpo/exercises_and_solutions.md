# Topic 21.4: Exercises & Solutions — Direct Preference Optimization (DPO)

---

### Problem 1: Algebraic Partition Function Cancellation in DPO (Easy)

The optimal policy under KL-constrained RL has the form:
$$\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right)$$
where $Z(x) = \sum_{y} \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right)$.

1. Express the ground-truth reward $r(x, y)$ explicitly in terms of $\pi^*(y \mid x)$, $\pi_{\text{ref}}(y \mid x)$, $\beta$, and $Z(x)$.
2. Compute the difference $r(x, y_w) - r(x, y_l)$ between two completions $y_w$ and $y_l$ for the same prompt $x$.
3. Show why $Z(x)$ cannot be computed in practice for modern LLMs, and explain why its cancellation is considered a mathematical breakthrough.

<details>
<summary>View Complete Solution</summary>

**Step 1: Inverting for reward $r(x, y)$:**
Take the natural logarithm of both sides:
$$\log \pi^*(y \mid x) = -\log Z(x) + \log \pi_{\text{ref}}(y \mid x) + \frac{1}{\beta} r(x, y)$$
Isolating $r(x, y)$:
$$\frac{1}{\beta} r(x, y) = \log \pi^*(y \mid x) - \log \pi_{\text{ref}}(y \mid x) + \log Z(x) = \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \log Z(x)$$
$$\implies r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$

**Step 2: Difference of rewards:**
$$r(x, y_w) - r(x, y_l) = \left[ \beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} + \beta \log Z(x) \right] - \left[ \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} + \beta \log Z(x) \right]$$
$$= \beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} + (\beta \log Z(x) - \beta \log Z(x))$$
$$= \beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \quad \blacksquare$$

**Step 3: Why $Z(x)$ is intractable and why cancellation is miraculous:**
The partition function $Z(x) = \sum_{y \in \mathcal{Y}} \pi_{\text{ref}}(y \mid x) \exp(\frac{1}{\beta} r(x, y))$ requires summing over the set of all possible completions $\mathcal{Y}$. For a vocabulary of size $|\mathcal{V}| = 128{,}000$ and sequence length $T = 512$, the cardinality is $|\mathcal{V}|^T \approx 128{,}000^{512} \approx 10^{2600}$ completions. This is physically impossible to compute or approximate accurately.
Because $Z(x)$ depends only on the prompt $x$ and not on the completion $y$, it has the identical value for both $y_w$ and $y_l$. When computing the difference in Bradley-Terry, it subtracts to exactly zero, allowing exact optimization without ever evaluating $Z(x)$!
</details>

---

### Problem 2: DPO Analytical Gradient & Error Weighting (Medium)

Recall the DPO loss:
$$\mathcal{L}_{\text{DPO}}(\theta) = -\log \sigma\left( \hat{r}_\theta(x, y_w) - \hat{r}_\theta(x, y_l) \right)$$
where $\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$. Let $\Delta \hat{r} = \hat{r}_\theta(x, y_w) - \hat{r}_\theta(x, y_l)$.

1. Prove that:
   $$\nabla_\theta \mathcal{L}_{\text{DPO}}(\theta) = -\beta \, \sigma(-\Delta \hat{r}) \left[ \nabla_\theta \log \pi_\theta(y_w \mid x) - \nabla_\theta \log \pi_\theta(y_l \mid x) \right]$$
2. Suppose $\beta = 0.1$.
   - At initialization, the model assigns equal probabilities: $\pi_\theta(y_w) = \pi_\theta(y_l)$, so $\Delta \hat{r} = 0$. What is the scaling factor $\sigma(-\Delta \hat{r})$?
   - After several updates, the model correctly prefers $y_w$ with $\Delta \hat{r} = +3.0$. What is the scaling factor?
   - Suppose on a misaligned prompt, the model assigns much higher probability to the dispreferred response $y_l$, with $\Delta \hat{r} = -3.0$. What is the scaling factor?

<details>
<summary>View Complete Solution</summary>

**Step 1: Gradient derivation:**
Let $u = \Delta \hat{r} = \hat{r}_\theta(x, y_w) - \hat{r}_\theta(x, y_l)$.
Using $\frac{d}{du}[-\log \sigma(u)] = -(1 - \sigma(u)) = -\sigma(-u)$:
$$\nabla_\theta \mathcal{L}_{\text{DPO}} = -\sigma(-u) \nabla_\theta u$$
Now compute $\nabla_\theta u$:
$$u = \beta \left( \log \pi_\theta(y_w \mid x) - \log \pi_{\text{ref}}(y_w \mid x) \right) - \beta \left( \log \pi_\theta(y_l \mid x) - \log \pi_{\text{ref}}(y_l \mid x) \right)$$
Since $\pi_{\text{ref}}$ is frozen, its derivatives with respect to $\theta$ are zero:
$$\nabla_\theta u = \beta \nabla_\theta \log \pi_\theta(y_w \mid x) - \beta \nabla_\theta \log \pi_\theta(y_l \mid x)$$
Substituting back:
$$\nabla_\theta \mathcal{L}_{\text{DPO}}(\theta) = -\beta \sigma(-\Delta \hat{r}) \left[ \nabla_\theta \log \pi_\theta(y_w \mid x) - \nabla_\theta \log \pi_\theta(y_l \mid x) \right] \quad \blacksquare$$

**Step 2: Numerical evaluation of scaling factor $\sigma(-\Delta \hat{r})$:**
- At $\Delta \hat{r} = 0.0$:
  $$\sigma(0) = \frac{1}{1 + e^0} = 0.500$$
- At $\Delta \hat{r} = +3.0$ (model is confident and correct):
  $$\sigma(-3.0) = \frac{1}{1 + e^3} = \frac{1}{1 + 20.0855} \approx 0.0474$$
  The gradient is automatically down-weighted by $95\%$, preventing catastrophic forgetting and over-fitting on already aligned examples.
- At $\Delta \hat{r} = -3.0$ (model is confident and wrong):
  $$\sigma(-(-3.0)) = \sigma(3.0) = \frac{1}{1 + e^{-3}} \approx 0.9526$$
  The gradient multiplier is nearly at its maximum ($0.9526$), applying a powerful gradient push to boost $y_w$ and suppress $y_l$.
</details>

---

### Problem 3: The Role of the $\beta$ Regularizer (Medium)

Consider a single training pair $(x, y_w, y_l)$.
Let the reference model assign log-probabilities:
$$\log \pi_{\text{ref}}(y_w \mid x) = -10.0, \quad \log \pi_{\text{ref}}(y_l \mid x) = -10.0$$
Suppose after DPO training, the policy shifts to:
$$\log \pi_\theta(y_w \mid x) = -8.0, \quad \log \pi_\theta(y_l \mid x) = -14.0$$

1. Calculate the implicit reward difference $\Delta \hat{r} = \hat{r}(y_w) - \hat{r}(y_l)$ when $\beta = 0.5$, $\beta = 0.1$, and $\beta = 0.01$.
2. What happens to the implied win probability $P(y_w \succ y_l) = \sigma(\Delta \hat{r})$ in each of these three cases?
3. What failure mode occurs if an engineer sets $\beta \to 0$ in DPO?

<details>
<summary>View Complete Solution</summary>

**Step 1: Compute implicit reward difference:**
$$\log \frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} = -8.0 - (-10.0) = +2.0$$
$$\log \frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)} = -14.0 - (-10.0) = -4.0$$
$$\Delta \log = (+2.0) - (-4.0) = +6.0$$
The implicit reward difference is $\Delta \hat{r} = \beta \times 6.0$:
- When $\beta = 0.5$: $\Delta \hat{r} = 0.5 \times 6.0 = 3.0$
- When $\beta = 0.1$: $\Delta \hat{r} = 0.1 \times 6.0 = 0.6$
- When $\beta = 0.01$: $\Delta \hat{r} = 0.01 \times 6.0 = 0.06$

**Step 2: Win probabilities:**
- $\beta = 0.5$: $P(y_w \succ y_l) = \sigma(3.0) \approx 0.9526$ ($95.3\%$)
- $\beta = 0.1$: $P(y_w \succ y_l) = \sigma(0.6) \approx 0.6457$ ($64.6\%$)
- $\beta = 0.01$: $P(y_w \succ y_l) = \sigma(0.06) \approx 0.5150$ ($51.5\%$)

**Step 3: Consequence of $\beta \to 0$:**
When $\beta \to 0$, the implicit logit inside the sigmoid $\beta \log(\dots)$ collapses to $0$. To minimize loss, the optimizer is forced to push the log-ratio $\log \frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \log \frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}$ to infinity. The KL divergence constraint vanishes entirely.
As a result, the policy diverges wildly from the pre-trained distribution $\pi_{\text{ref}}$, over-memorizes specific stylistic artifacts of $y_w$, loses language fluency, and suffers catastrophic mode collapse.
</details>

---

### Problem 4: DPO vs. Unconstrained SFT on Preferred Data (Hard)

A common intuition among junior engineers is:
*"Why do we need negative examples $y_l$ and DPO? Can't we just filter our dataset to keep only the winning completions $y_w$ and run standard Supervised Fine-Tuning (SFT) on $(x, y_w)$?"*

1. Compare the mathematical gradients of SFT on $(x, y_w)$ versus DPO on $(x, y_w, y_l)$.
2. Explain two major failure modes that occur when relying exclusively on SFT on winning data rather than preference optimization with negative contrast.

<details>
<summary>View Complete Solution</summary>

**Step 1: Gradient comparison:**
- SFT on preferred data:
  $$\nabla_\theta \mathcal{L}_{\text{SFT}} = -\nabla_\theta \log \pi_\theta(y_w \mid x)$$
  This is a pure **unidirectional pull**: it strictly increases the likelihood of tokens in $y_w$.
- DPO on preference pairs:
  $$\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta \sigma(-\Delta \hat{r}) \left[ \nabla_\theta \log \pi_\theta(y_w \mid x) - \nabla_\theta \log \pi_\theta(y_l \mid x) \right]$$
  DPO is a **bidirectional contrastive push-pull**: it actively increases $P(y_w)$ while actively suppressing $P(y_l)$.

**Step 2: Failure modes of SFT on preferred data alone:**
1. **Lack of Contrastive Negative Boundaries (Hallucination & Safety Blindness):**
   Suppose $y_w$ is a safe refusal (*"I cannot help build an explosive device..."*) and $y_l$ is a dangerous bomb recipe. In SFT, the model learns that safe refusals are probable, but it never receives an explicit gradient reducing the probability of the dangerous recipe! The dangerous completion remains in the probability distribution. DPO explicitly depresses the probability of $y_l$, creating a sharp decision boundary.
2. **Frequency Bias & Token Over-Allocation:**
   If a preferred response $y_w$ shares common phrases, intros (*"Sure, I'd be happy to help!"*), or punctuation with dispreferred responses, SFT indiscriminately reinforces all tokens in $y_w$, including the generic filler. DPO subtracts the gradient of $y_l$; shared prefix tokens between $y_w$ and $y_l$ produce cancelling gradients ($\nabla \log \pi(y_w) \approx \nabla \log \pi(y_l)$), focusing gradient updates exclusively on the distinguishing informative tokens!
</details>
