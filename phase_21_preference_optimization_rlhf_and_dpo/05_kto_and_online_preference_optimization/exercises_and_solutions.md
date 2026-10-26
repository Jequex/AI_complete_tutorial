# Topic 21.5: Exercises & Solutions — KTO & Online Preference Optimization

---

### Problem 1: Prospect Theory & Asymmetric Loss Weighting (Easy)

In Kahneman-Tversky Optimization (KTO), the loss function incorporates the loss-aversion multiplier $\lambda_{\text{undesirable}}$:
$$\mathcal{L}_{\text{KTO}} = \mathbb{E}_{\mathcal{D}_D}[\ell_D] + \lambda_{\text{undesirable}} \mathbb{E}_{\mathcal{D}_U}[\ell_U]$$
where:
$$\ell_D = 1 - \sigma\left(\lambda_D(z - z_{\text{ref}})\right), \quad \ell_U = 1 - \sigma\left(\lambda_U(z_{\text{ref}} - z)\right)$$
Suppose $\lambda_D = 1.0$, $\lambda_U = 1.0$, $z_{\text{ref}} = 0.0$, and $\lambda_{\text{undesirable}} = 1.50$.

1. Compute the loss $\ell_D$ for a desirable output whose implicit reward exceeds the baseline by $\Delta z = +1.0$.
2. Compute the loss contribution $\lambda_{\text{undesirable}} \ell_U$ for an undesirable output whose implicit reward falls below the baseline by an equal magnitude $\Delta z = -1.0$.
3. What is the ratio between the undesirable penalty and the desirable reward? Why is this psychological asymmetry beneficial in safety-critical AI alignment?

<details>
<summary>View Complete Solution</summary>

**Step 1: Compute loss for desirable output:**
Here $z - z_{\text{ref}} = +1.0$.
$$\ell_D = 1 - \sigma(1.0 \times 1.0) = 1 - \frac{1}{1 + e^{-1}} = 1 - \frac{1}{1 + 0.3679} = 1 - 0.7311 = 0.2689$$

**Step 2: Compute loss contribution for undesirable output:**
Here $z_{\text{ref}} - z = 0.0 - (-1.0) = +1.0$.
$$\ell_U = 1 - \sigma(1.0 \times 1.0) = 1 - 0.7311 = 0.2689$$
Multiplying by $\lambda_{\text{undesirable}} = 1.50$:
$$\text{Penalty} = 1.50 \times 0.2689 = 0.4034$$

**Step 3: Ratio and alignment interpretation:**
$$\text{Ratio} = \frac{0.4034}{0.2689} = 1.50$$
The undesirable output contributes $50\%$ more gradient penalty than the desirable output contributes reward.
In safety-critical LLM deployment, the asymmetry of consequences is profound: generating a slightly subpar essay on baking is a minor inconvenience, whereas generating a toxic, libelous, or dangerous output can cause immediate real-world harm. Setting $\lambda_{\text{undesirable}} > 1.0$ forces the model to be fundamentally risk-averse, aggressively purging harmful token sequences even if it slightly dampens creative risks.
</details>

---

### Problem 2: Reference Point $z_{\text{ref}}$ Dynamics & Running Estimates (Medium)

In KTO, the reference point represents the policy's average implicit reward:
$$z_{\text{ref}} = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta} \left[ \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} \right]$$

1. At the very beginning of training (epoch 0), when $\pi_\theta = \pi_{\text{ref}}$, what is the theoretical value of $z_{\text{ref}}$?
2. If $z_{\text{ref}}$ is mistakenly held fixed at $0.0$ throughout training while the policy successfully improves (so that the average implicit reward drifts to $+1.5$), what happens to the loss functions $\ell_D$ and $\ell_U$?
3. Describe how an Exponential Moving Average (EMA) or per-batch baseline calculation preserves reference point calibration.

<details>
<summary>View Complete Solution</summary>

**Step 1: Initial theoretical value:**
When $\pi_\theta = \pi_{\text{ref}}$, $\frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} = 1.0$ for all $(x, y)$.
$$\beta \log(1.0) = 0.0 \implies z_{\text{ref}}^{(0)} = 0.0$$

**Step 2: Effect of static $z_{\text{ref}} = 0.0$ under policy drift:**
If the policy improves, its average reward rises to $\bar{z} = +1.5$.
- For desirable outputs, $z \approx +2.0$. The argument $z - z_{\text{ref}} = 2.0 - 0.0 = +2.0$. The loss $\ell_D = 1 - \sigma(2.0) \approx 0.119$. The gradient shrinks, causing the model to stop learning from good examples.
- For undesirable outputs, $z \approx +1.0$ (bad compared to the current policy, but still higher than the stale baseline $0.0$!). The argument $z_{\text{ref}} - z = 0.0 - 1.0 = -1.0$. The model actually treats a flawed output as a *gain*, failing to penalize it!

**Step 3: Solution via dynamic updating:**
To prevent this calibration failure, KTO estimates $z_{\text{ref}}$ dynamically using an Exponential Moving Average (EMA):
$$z_{\text{ref}}^{(k)} = \alpha z_{\text{ref}}^{(k-1)} + (1 - \alpha) \frac{1}{|B|} \sum_{i \in B} z(x_i, y_i)$$
where $\alpha \approx 0.95$. Tracking the true moving expectation ensures that outputs are judged relative to what the model is *currently capable of producing*.
</details>

---

### Problem 3: Unpaired Telemetry vs. Paired Annotation Economics (Medium)

An AI company operates a customer support chatbot handling $1{,}000{,}000$ queries per day.
- Approach A (Contract Paired Labeling): The company pays third-party annotators to review prompts, write 2 responses per prompt, and label the winner. Cost: $\$0.80$ per pair.
- Approach B (In-Product Telemetry with KTO): Users naturally click thumbs-up or thumbs-down on $2\%$ of chatbot turns. Cost: $\$0.00$ incremental labeling cost.

1. If the company has an annual alignment budget of $\$200{,}000$, how many paired preference examples can they collect using Approach A?
2. How many unpaired binary feedback examples does Approach B collect in a single year for free?
3. What is the data volume advantage of KTO in this product deployment scenario?

<details>
<summary>View Complete Solution</summary>

**Step 1: Data volume under Approach A (Contract Annotations):**
$$N_{\text{paired}} = \frac{\$200{,}000}{\$0.80} = 250{,}000\text{ paired examples per year}$$

**Step 2: Data volume under Approach B (Product Telemetry):**
Daily feedback clicks:
$$\text{Daily Clicks} = 1{,}000{,}000 \text{ queries} \times 0.02 = 20{,}000\text{ feedback signals per day}$$
Annual feedback volume:
$$N_{\text{unpaired}} = 20{,}000 \times 365 = 7{,}300{,}000\text{ binary examples per year}$$

**Step 3: Volume ratio and strategic advantage:**
$$\text{Ratio} = \frac{7{,}300{,}000}{250{,}000} = 29.2\times$$
KTO captures nearly **30 times more alignment data** at zero contractor cost, directly reflecting real end-user interaction distributions rather than artificial contractor evaluations.
</details>

---

### Problem 4: Distribution Shift in Offline DPO vs. Online Self-Correction (Hard)

Consider an alignment experiment:
A base model frequently hallucinates when asked about obscure historical dates.
1. In **Offline DPO**, 10,000 preference pairs were generated using the SFT model.
   - Explain what happens if the policy $\pi_\theta$ develops an out-of-distribution habit during epoch 4 (e.g. inserting unnecessary disclaimers like *"As an AI, I must remind you..."* on every turn). Can offline DPO correct this behavior?
2. In **Online / Iterative DPO**, candidate completions are generated dynamically: $y_1, y_2 \sim \pi_{\theta}(\cdot \mid x)$ and scored in real time by an automated judge.
   - Explain the step-by-step mechanism by which Online DPO automatically eradicates the disclaimer habit within a single iteration.

<details>
<summary>View Complete Solution</summary>

**Step 1: Failure of Offline DPO under distribution shift:**
In offline DPO, the dataset $\mathcal{D}_{\text{offline}} = \{(x, y_w, y_l)\}$ is static and frozen. The negative examples $y_l$ only reflect the mistakes made by the *initial SFT model*.
When the policy $\pi_\theta$ evolves during training, it begins generating tokens and stylistic habits (such as the annoying disclaimer prefix) that **do not appear anywhere in the static dataset**. Because the training loss is evaluated only on $(x, y_w, y_l) \sim \mathcal{D}_{\text{offline}}$, the policy is never penalized for its new disclaimer habit. The model suffers from severe off-policy distribution shift.

**Step 2: Online DPO self-correction mechanism:**
In Online DPO, at each training step:
1. The current policy $\pi_\theta$ generates two fresh candidate responses:
   - $y_1$: *"As an AI, I must remind you that history is complex. Napoleon was born on August 15, 1769."*
   - $y_2$: *"Napoleon Bonaparte was born on August 15, 1769 in Ajaccio, Corsica."*
2. The automated judge (or Reward Model) compares both on-policy responses and detects that $y_1$ has a redundant, unprompted disclaimer. The judge assigns $y_2 \succ y_1$.
3. The DPO loss is computed directly on this fresh pair:
   $$\nabla_\theta \mathcal{L} = -\beta \sigma(\hat{r}_1 - \hat{r}_2) \left[ \nabla \log \pi_\theta(y_2) - \nabla \log \pi_\theta(y_1) \right]$$
   The gradient directly depresses the probability of $y_1$ (specifically penalizing the disclaimer tokens), while reinforcing $y_2$.
4. Within a single training iteration, the policy learns that its new disclaimer habit is dispreferred, successfully eliminating the behavior.
</details>
