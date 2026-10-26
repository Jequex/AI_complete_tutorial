# Topic 20.2: Exercises & Solutions — Supervised Fine-Tuning & Loss Masking

---

### Problem 1: Numerical Calculation of Masked vs. Unmasked Cross-Entropy (Easy)

Consider a toy vocabulary of size $V = 4$.
We have a concatenated sequence of 4 tokens:
- Prompt tokens: $\mathbf{x} = [x_1, x_2]$ (indices $t=0, 1$)
- Response tokens: $\mathbf{y} = [y_1, y_2]$ (indices $t=2, 3$)

Suppose the model outputs the following predicted probability vectors $P(w_{t+1} \mid w_{\le t})$:
- At $t=0$ (predicting $x_2$, correct class index 1): $P_0 = [0.1, 0.4, 0.3, 0.2]$
- At $t=1$ (predicting $y_1$, correct class index 3): $P_1 = [0.2, 0.1, 0.1, 0.6]$
- At $t=2$ (predicting $y_2$, correct class index 0): $P_2 = [0.8, 0.1, 0.05, 0.05]$

1. Identify the target label sequence $\mathbf{t} = [t_0, t_1, t_2]$ under:
   - (a) Unmasked training
   - (b) Masked SFT training (using $-100$ for masked positions).
2. Compute the exact loss value for both unmasked training $\mathcal{L}_{\text{unmasked}}$ and masked training $\mathcal{L}_{\text{SFT}}$ using natural logarithms.

<details>
<summary>View Complete Solution</summary>

**Step 1: Target Label Sequences:**
- (a) **Unmasked training:** Every next token is evaluated.
  - At $t=0$, target is $x_2$ (class 1).
  - At $t=1$, target is $y_1$ (class 3).
  - At $t=2$, target is $y_2$ (class 0).
  $$\mathbf{t}_{\text{unmasked}} = [1, 3, 0]$$

- (b) **Masked SFT training:**
  - Position $t=0$ is within the prompt predicting a prompt token $x_2$. It must be masked out: $t_0 = -100$.
  - Position $t=1$ is the prompt transition token predicting the first response token $y_1$. It is evaluated: $t_1 = 3$.
  - Position $t=2$ is within the response predicting $y_2$. It is evaluated: $t_2 = 0$.
  $$\mathbf{t}_{\text{masked}} = [-100, 3, 0]$$

**Step 2: Loss Calculations:**
Negative log-likelihoods for each target:
- At $t=0$, $p(1) = 0.4 \implies -\ln(0.4) \approx 0.9163$
- At $t=1$, $p(3) = 0.6 \implies -\ln(0.6) \approx 0.5108$
- At $t=2$, $p(0) = 0.8 \implies -\ln(0.8) \approx 0.2231$

**Unmasked Loss:**
Averages all 3 transitions:
$$\mathcal{L}_{\text{unmasked}} = \frac{-\ln(0.4) - \ln(0.6) - \ln(0.8)}{3} = \frac{0.9163 + 0.5108 + 0.2231}{3} = \frac{1.6502}{3} \approx 0.5501$$

**Masked SFT Loss:**
Averages only the 2 response transitions:
$$\mathcal{L}_{\text{SFT}} = \frac{-\ln(0.6) - \ln(0.8)}{2} = \frac{0.5108 + 0.2231}{2} = \frac{0.7339}{2} \approx 0.3670$$
Notice that unmasked training inflated the loss by $50\%$ solely due to the prompt prediction error.
</details>

---

### Problem 2: Gradient Flow Analysis on Masked Positions (Medium)

Let an autoregressive layer have output projection weights $W \in \mathbb{R}^{d \times V}$, producing logits $\mathbf{z}_t = \mathbf{h}_t W$.
The loss over a sequence of length $T$ is:
$$\mathcal{L} = \frac{1}{N_{\text{active}}} \sum_{t=1}^T \ell_t, \qquad \ell_t = \begin{cases} -\log\left(\frac{e^{z_{t, y_t}}}{\sum_{j=1}^V e^{z_{t, j}}}\right) & \text{if } y_t \ne -100 \\ 0 & \text{if } y_t = -100 \end{cases}$$
where $N_{\text{active}} = \sum_{t=1}^T \mathbb{I}[y_t \ne -100]$.

1. Derive the partial derivative $\frac{\partial \mathcal{L}}{\partial W}$ in terms of hidden states $\mathbf{h}_t$, predicted probabilities $\mathbf{p}_t$, and one-hot ground-truth vectors $\mathbf{e}_{y_t}$.
2. Prove that if positions $t \in \{1, \dots, M-1\}$ are masked with $y_t = -100$, these positions contribute strictly zero to the parameter update $\Delta W$.

<details>
<summary>View Complete Solution</summary>

**Step 1: Derivative Derivation:**
For any active position $t$ where $y_t \ne -100$:
$$\frac{\partial \ell_t}{\partial z_{t, k}} = p_{t, k} - \mathbb{I}[k = y_t] = (\mathbf{p}_t - \mathbf{e}_{y_t})_k$$
For any masked position $t$ where $y_t = -100$:
$$\ell_t \equiv 0 \implies \frac{\partial \ell_t}{\partial \mathbf{z}_t} = \mathbf{0}$$

Applying the chain rule for matrix multiplication $\mathbf{z}_t = \mathbf{h}_t W$:
$$\frac{\partial \ell_t}{\partial W} = \mathbf{h}_t^T \frac{\partial \ell_t}{\partial \mathbf{z}_t}$$
Summing over the full sequence:
$$\frac{\partial \mathcal{L}}{\partial W} = \frac{1}{N_{\text{active}}} \sum_{t=1}^T \frac{\partial \ell_t}{\partial W} = \frac{1}{N_{\text{active}}} \sum_{t=1}^T \mathbb{I}[y_t \ne -100] \cdot \mathbf{h}_t^T (\mathbf{p}_t - \mathbf{e}_{y_t})$$

**Step 2: Proof of Zero Gradient Contribution:**
For all prompt positions $t \in \{1, \dots, M-1\}$, $y_t = -100$.
Therefore, $\mathbb{I}[y_t \ne -100] = 0$.
The summation simplifies to:
$$\frac{\partial \mathcal{L}}{\partial W} = \frac{1}{N_{\text{active}}} \sum_{t=M}^T \mathbf{h}_t^T (\mathbf{p}_t - \mathbf{e}_{y_t})$$
The gradient contribution from prompt tokens is:
$$\sum_{t=1}^{M-1} \frac{\partial \ell_t}{\partial W} = \sum_{t=1}^{M-1} \mathbf{h}_t^T \mathbf{0} = \mathbf{0}$$
Hence, $\Delta W = -\eta \frac{\partial \mathcal{L}}{\partial W}$ is completely independent of the prompt token targets.
</details>

---

### Problem 3: Gradient Dilution & Effective Sample Efficiency (Hard)

A team fine-tunes an LLM on a multi-document question answering dataset:
- Average prompt length: $\bar{M} = 3{,}600$ tokens (long retrieved context).
- Average answer length: $\bar{K} = 400$ tokens (concise synthesis).
- Batch size: $B = 32$ sequences.

1. What fraction $\rho$ of total sequence tokens are actively updating the model's answering capability under:
   - (a) Unmasked training
   - (b) Masked SFT training
2. Suppose training without masking causes the model to update weights with gradient vector $\mathbf{g}_{\text{unmasked}} = \alpha \mathbf{g}_{\text{prompt}} + (1 - \alpha) \mathbf{g}_{\text{resp}}$. Compute the scalar coefficient $\alpha$.
3. If an engineer claims: *"Training without masking is strictly better because the model sees 10x more tokens of supervision"*, provide a rigorous refutation based on optimization dynamics and multi-task gradient interference.

<details>
<summary>View Complete Solution</summary>

**Step 1: Fraction of Active Answer Supervision Tokens:**
Total sequence length per sample: $T = \bar{M} + \bar{K} = 3600 + 400 = 4000$ tokens.
- (a) **Unmasked training:** All 4,000 tokens generate loss. However, only 400 tokens are answers.
  The fraction of tokens representing the target task (answering questions) is:
  $$\rho_{\text{unmasked}} = \frac{400}{4000} = 10\%$$
- (b) **Masked SFT training:**
  The prompt tokens have target $-100$ and zero loss. The loss denominator is exactly $N_{\text{active}} = 400$.
  Every single gradient-bearing token ($100\%$) directly corresponds to the assistant's answer:
  $$\rho_{\text{masked}} = \frac{400}{400} = 100\%$$

**Step 2: Coefficient $\alpha$:**
In unmasked training, the gradient is the unweighted mean over all $T$ positions:
$$\mathbf{g}_{\text{unmasked}} = \frac{1}{4000} \left( \sum_{t=1}^{3600} \mathbf{g}_t + \sum_{t=3601}^{4000} \mathbf{g}_t \right) = \frac{3600}{4000} \mathbf{g}_{\text{prompt}} + \frac{400}{4000} \mathbf{g}_{\text{resp}}$$
Thus:
$$\alpha = \frac{3600}{4000} = 0.90 \quad (90\%)$$
$90\%$ of the gradient magnitude is driven by the prompt text.

**Step 3: Refutation of the Engineer's Claim:**
1. **Capacity Stealing:** Neural networks have finite representation bandwidth. Minimizing $\mathcal{L}_{\text{prompt}}$ forces hidden layers to allocate attention heads and MLP neurons to predicting background text (e.g. Wikipedia formatting, legal disclaimers, boilerplate) rather than conditional synthesis.
2. **Negative Gradient Transfer (Orthogonality & Conflict):**
   The gradient vectors $\mathbf{g}_{\text{prompt}}$ and $\mathbf{g}_{\text{resp}}$ point in conflicting directions:
   $$\langle \mathbf{g}_{\text{prompt}}, \mathbf{g}_{\text{resp}} \rangle < 0$$
   Optimizing the prompt distribution updates the weights to model raw document continuation, which actively undoes instruction-following alignments learned from $\mathbf{g}_{\text{resp}}$.
3. **Inference Asymmetry:** At test time, the model is **never** asked to generate the prompt—the prompt is provided by the user. Training on prompt prediction optimizes performance on an objective that is never evaluated in production.
</details>
