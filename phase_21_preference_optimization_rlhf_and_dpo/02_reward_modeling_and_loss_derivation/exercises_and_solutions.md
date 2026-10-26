# Topic 21.2: Exercises & Solutions — Reward Modeling & Loss Derivation

---

### Problem 1: Reward Head vs. LM Head Parameter Count & Memory (Easy)

Consider an 8-billion parameter Transformer architecture with:
- Hidden dimension $d_{\text{model}} = 4096$
- Vocabulary size $|\mathcal{V}| = 128{,}000$

1. Compute the number of parameters and FP16 memory footprint in the standard Language Model head:
   $$\mathbf{W}_{\text{LM}} \in \mathbb{R}^{|\mathcal{V}| \times d_{\text{model}}}$$
2. Compute the number of parameters and FP16 memory footprint in the scalar Reward Model head:
   $$\mathbf{W}_{\text{RM}} \in \mathbb{R}^{1 \times d_{\text{model}}}$$
3. What percentage of the LM head's parameter count does the Reward head consume?

<details>
<summary>View Complete Solution</summary>

**Step 1: LM Head Parameter Count & Memory:**
$$\text{Params}_{\text{LM}} = |\mathcal{V}| \times d_{\text{model}} = 128{,}000 \times 4096 = 524{,}288{,}000 \approx 524.3\text{M parameters}$$
In FP16 (2 bytes per parameter):
$$\text{Memory}_{\text{LM}} = 524{,}288{,}000 \times 2\text{ bytes} = 1{,}048{,}576{,}000\text{ bytes} \approx 1.048\text{ GB (or } 0.977\text{ GiB)}$$

**Step 2: Reward Model Head Parameter Count & Memory:**
$$\text{Params}_{\text{RM}} = 1 \times d_{\text{model}} = 1 \times 4096 = 4096\text{ parameters}$$
In FP16:
$$\text{Memory}_{\text{RM}} = 4096 \times 2\text{ bytes} = 8192\text{ bytes} = 8.192\text{ KB}$$

**Step 3: Comparison Ratio:**
$$\text{Ratio} = \frac{4096}{524{,}288{,}000} = \frac{1}{128{,}000} \approx 0.000781\%$$
The scalar reward head is over $128{,}000\times$ smaller than the LM classification head, discarding over $1\text{ GB}$ of head weight parameters while retaining the entire underlying transformer backbone.
</details>

---

### Problem 2: Ranking Loss with Margin Penalty (Medium)

Consider the margin-enhanced pairwise ranking loss:
$$\mathcal{L}(\theta) = -\log \sigma\left(r_\theta(x, y_w) - r_\theta(x, y_l) - m\right)$$
where $m \ge 0$ is a fixed margin penalty. Let $\Delta = r_\theta(x, y_w) - r_\theta(x, y_l)$.

1. Derive the gradient of the loss with respect to $\Delta$.
2. When $m = 0.5$ and the model assigns equal rewards $r_w = r_l = 2.0$ ($\Delta = 0$), what is the gradient magnitude? Compare this to the case where $m = 0.0$.
3. Why does introducing a margin $m > 0$ prevent the model from settling into indifference ($r_w \approx r_l$)?

<details>
<summary>View Complete Solution</summary>

**Step 1: Gradient derivation:**
Let $z = \Delta - m$. The loss is $\mathcal{L} = -\log \sigma(z)$.
$$\frac{\partial \mathcal{L}}{\partial \Delta} = \frac{d\mathcal{L}}{dz} \frac{\partial z}{\partial \Delta} = -\frac{\sigma'(z)}{\sigma(z)} \cdot 1 = -(1 - \sigma(z)) = -\sigma(-z)$$
Substituting $z = \Delta - m$:
$$\frac{\partial \mathcal{L}}{\partial \Delta} = -\sigma(m - \Delta) = \sigma(\Delta - m) - 1$$

**Step 2: Gradient comparison at $\Delta = 0$:**
- If $m = 0.0$:
  $$\left| \frac{\partial \mathcal{L}}{\partial \Delta} \right| = \sigma(0.0) = \frac{1}{1 + e^0} = 0.500$$
- If $m = 0.5$:
  $$\left| \frac{\partial \mathcal{L}}{\partial \Delta} \right| = \sigma(0.5 - 0.0) = \sigma(0.5) = \frac{1}{1 + e^{-0.5}} = \frac{1}{1 + 0.6065} \approx 0.6225$$
The gradient magnitude is $24.5\%$ larger when a margin is present.

**Step 3: Purpose of the margin penalty:**
Without a margin ($m = 0$), any infinitesimal difference $\Delta = \epsilon > 0$ already yields a win probability $> 50\%$. The model can reduce loss very slowly without creating a decisive separation between good and bad responses. By enforcing $\Delta \ge m$, the model is forced to push the reward of the winning response at least $m$ units above the losing response before gradient forces abate.
</details>

---

### Problem 3: Score Normalization & Additive Invariance (Medium)

Suppose an unnormalized reward model outputs scores $r(x, y)$ such that across a batch of prompts, the rewards have mean $\mu = 15.0$ and variance $\sigma^2 = 25.0$.

1. Show analytically that replacing $r(x, y)$ with normalized scores $\tilde{r}(x, y) = \frac{r(x, y) - \mu}{\sigma}$ changes the pairwise log-loss if $\sigma \ne 1$.
2. Under what condition does an affine transformation $\tilde{r} = a \cdot r + b$ leave the pairwise cross-entropy loss completely invariant?
3. In PPO, if the reward model has an arbitrary constant offset $+100.0$, explain why the policy might experience length bias even if no length feature was explicitly rewarded.

<details>
<summary>View Complete Solution</summary>

**Step 1: Impact of normalization on loss:**
Let $\Delta = r_w - r_l$. Under normalization:
$$\tilde{\Delta} = \tilde{r}_w - \tilde{r}_l = \frac{r_w - \mu}{\sigma} - \frac{r_l - \mu}{\sigma} = \frac{r_w - r_l}{\sigma} = \frac{\Delta}{\sigma}$$
The normalized loss is:
$$\tilde{\mathcal{L}} = -\log \sigma\left(\frac{\Delta}{\sigma}\right)$$
If $\sigma = 5.0$, then $\frac{\Delta}{5.0} \ne \Delta$. Because sigmoid is a nonlinear function, $\sigma(\Delta / 5) \ne \sigma(\Delta)$.
Therefore, scaling by standard deviation changes the effective temperature of the logistic link function, softening the win probabilities.

**Step 2: Condition for invariance:**
For $\tilde{\mathcal{L}} = \mathcal{L}$, we must have:
$$\tilde{r}_w - \tilde{r}_l = (a r_w + b) - (a r_l + b) = a (r_w - r_l) = r_w - r_l$$
This holds if and only if $a = 1.0$. The additive shift $b$ can be any real number ($b \in \mathbb{R}$), but the multiplicative scale must remain $a = 1.0$.

**Step 3: Positive constant offset and length bias in PPO:**
In standard PPO implementations, if the terminal reward has a large positive bias (e.g. $+100.0$), or if positive step rewards are distributed across tokens without discounting ($\gamma \approx 1$), each additional generated token that does not terminate the sequence allows the policy to accumulate positive value expectation or avoids negative terminal penalties. Conversely, if rewards are negatively shifted, the model learns an "early-stopping" bias where it outputs `<eos>` immediately to minimize punishment. Centering the reward to zero mean eliminates this artificial survival incentive.
</details>

---

### Problem 4: Goodhart's Law & Length Bias Modeling (Hard)

Suppose true human utility $U^*(y)$ depends solely on conciseness and factual correctness:
$$U^*(y) = Q(y) - 0.05 \cdot L(y)$$
where $Q(y) \in [0, 10]$ is factual quality and $L(y) \in [10, 200]$ is response length in words.
However, because human annotators were impressed by thoroughness in the training data, the trained Reward Model learned:
$$r_{\text{RM}}(y) = 0.6 \cdot Q(y) + 0.04 \cdot L(y)$$

Consider two responses:
- Response 1 (Crisp): $Q(y_1) = 9.0$, $L(y_1) = 20$ words.
- Response 2 (Bloated): $Q(y_2) = 7.0$, $L(y_2) = 150$ words.

1. Which response has higher true human utility $U^*$?
2. Which response will the Reward Model assign a higher score to?
3. Calculate the reward gap $r_{\text{RM}}(y_2) - r_{\text{RM}}(y_1)$ and predict what will happen if an RL policy is trained against $r_{\text{RM}}$.

<details>
<summary>View Complete Solution</summary>

**Step 1: Compute true utility $U^*$:**
- For Response 1 (Crisp):
  $$U^*(y_1) = 9.0 - 0.05(20) = 9.0 - 1.0 = 8.0$$
- For Response 2 (Bloated):
  $$U^*(y_2) = 7.0 - 0.05(150) = 7.0 - 7.5 = -0.5$$
Response 1 has vastly higher true utility ($8.0$ vs $-0.5$).

**Step 2: Compute Reward Model scores:**
- For Response 1:
  $$r_{\text{RM}}(y_1) = 0.6(9.0) + 0.04(20) = 5.4 + 0.8 = 6.2$$
- For Response 2:
  $$r_{\text{RM}}(y_2) = 0.6(7.0) + 0.04(150) = 4.2 + 6.0 = 10.2$$
The Reward Model incorrectly scores Response 2 much higher ($10.2 > 6.2$).

**Step 3: Policy behavior under RL:**
The reward gap is:
$$\Delta r_{\text{RM}} = 10.2 - 6.2 = +4.0$$
Under Bradley-Terry, the RM believes Response 2 wins with probability:
$$\sigma(4.0) = \frac{1}{1 + e^{-4}} \approx 98.2\%$$
If an RL algorithm (like PPO) optimizes against $r_{\text{RM}}$, it will exploit this positive gradient $\frac{\partial r}{\partial L} = +0.04$ by producing maximally long, verbose answers up to the maximum token generation limit, completely destroying true human utility $U^*$. This is textbook Goodhart's Law.
</details>
