# Topic 02: Exercises & Solutions — Activation Patching & Causal Tracing

## Exercise 1: Logit Difference vs. Output Probability Restoration

In activation patching experiments, researchers almost universally measure **Logit Difference** $\Delta L = \text{logit}(y_*) - \text{logit}(y_{\text{alt}})$ rather than direct output probabilities $P(y_*)$.

1. Explain mathematically why the softmax probability $P(y_*)$ is prone to extreme saturation and uncalibrated compression at the extremes ($P \approx 0$ or $P \approx 1$).
2. Show that under a binary vocabulary or when other logits remain approximately constant, the logit difference $\Delta L$ has a direct linear relationship with the log-odds ratio:
   $$\Delta L = \ln \left(\frac{P(y_*)}{P(y_{\text{alt}})}\right)$$
3. Suppose a clean run has $\text{logit}(y_*) = 28.0$ and $\text{logit}(y_{\text{alt}}) = 18.0$. A corrupted run has $\text{logit}(y_*) = 12.0$ and $\text{logit}(y_{\text{alt}}) = 22.0$. An intermediate activation patch yields $\text{logit}(y_*) = 24.0$ and $\text{logit}(y_{\text{alt}}) = 19.0$. Calculate the normalized restoration metric $\mathcal{R}$.

### Solution

#### 1. Softmax Saturation
The softmax function $P(y_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$ exhibits severe vanishing gradients when $|z_i - z_j|$ is large. If clean logits differ by 20 ($z_* - z_{\text{alt}} = 20$), both a difference of 20 and a difference of 30 produce $P(y_*) \approx 0.99999999$. Conversely, small perturbations near zero logit difference cause wild swings in probability. Logits remain uncompressed and unbounded in $\mathbb{R}$, providing a faithful linear measure of model evidence.

#### 2. Log-Odds Relationship
Using the definition of softmax:
$$\frac{P(y_*)}{P(y_{\text{alt}})} = \frac{e^{\text{logit}(y_*)} / \sum_k e^{\text{logit}(k)}}{e^{\text{logit}(y_{\text{alt}})} / \sum_k e^{\text{logit}(k)}} = \frac{e^{\text{logit}(y_*)}}{e^{\text{logit}(y_{\text{alt}})}} = \exp\left(\text{logit}(y_*) - \text{logit}(y_{\text{alt}})\right)$$
Taking the natural logarithm of both sides:
$$\ln\left(\frac{P(y_*)}{P(y_{\text{alt}})}\right) = \text{logit}(y_*) - \text{logit}(y_{\text{alt}}) = \Delta L$$
Thus, logit difference is precisely the **log-odds ratio** between the clean and counterfactual predictions.

#### 3. Numerical Calculation
- Clean difference: $\Delta L_{\text{clean}} = 28.0 - 18.0 = +10.0$
- Corrupted difference: $\Delta L_{\text{corrupt}} = 12.0 - 22.0 = -10.0$
- Patched difference: $\Delta L_{\text{patched}} = 24.0 - 19.0 = +5.0$

Normalized restoration:
$$\mathcal{R} = \frac{\Delta L_{\text{patched}} - \Delta L_{\text{corrupt}}}{\Delta L_{\text{clean}} - \Delta L_{\text{corrupt}}} = \frac{5.0 - (-10.0)}{10.0 - (-10.0)} = \frac{15.0}{20.0} = \mathbf{0.75 \quad (75.0\%)}$$
Patching this activation recovers $75\%$ of the total causal preference for the correct entity.

---

## Exercise 2: Natural Indirect Effect (NIE) and Pearl's Mediation Analysis

Consider a causal DAG representing a 2-layer Transformer:
- Treatment: $X \in \{x_{\text{clean}}, x_{\text{corrupt}}\}$ (input prompt)
- Mediator: $M = \mathbf{a}_1$ (Layer 1 MLP activation at subject token)
- Direct Path: $X \to Y$ (bypassing $M$, e.g., through shallow attention or residual stream)
- Output: $Y = \Delta L$

1. Write the formal Pearl structural causal formula for the **Total Effect (TE)**.
2. Define the **Natural Indirect Effect (NIE)** corresponding to activation patching of mediator $M$.
3. If an experiment shows $\text{TE} = 40.0$ and the patched run restoring $M = \mathbf{a}_{\text{clean}}$ yields an effect of $34.0$, what fraction of the total causal effect is mediated through Layer 1 MLP?

### Solution

#### 1. Total Effect (TE)
The Total Effect measures the full change in outcome when changing treatment from corrupted to clean:
$$\text{TE} = \mathbb{E}[Y(X = x_{\text{clean}}, M(x_{\text{clean}}))] - \mathbb{E}[Y(X = x_{\text{corrupt}}, M(x_{\text{corrupt}}))]$$

#### 2. Natural Indirect Effect (NIE)
The Natural Indirect Effect measures how much the outcome changes when the mediator $M$ is set to its clean value while the input is held at the corrupted condition:
$$\text{NIE} = \mathbb{E}[Y(X = x_{\text{corrupt}}, M(x_{\text{clean}}))] - \mathbb{E}[Y(X = x_{\text{corrupt}}, M(x_{\text{corrupt}}))]$$
This is mathematically identical to the numerator of the normalized restoration metric $\Delta L_{\text{patched}} - \Delta L_{\text{corrupt}}$ in activation patching.

#### 3. Fraction Mediated
$$\text{Fraction Mediated} = \frac{\text{NIE}}{\text{TE}} = \frac{34.0}{40.0} = \mathbf{0.85 \quad (85.0\%)}$$
$85\%$ of the factual recall signal travels indirectly through Layer 1 MLP, proving it is the dominant causal channel.

---

## Exercise 3: Subject-to-Last-Token Circuit Decomposition

In causal tracing experiments, why does patching the MLP at the *subject* token restore performance, whereas patching the MLP at the *last* token has zero effect?

1. Analyze the mathematical operation of the Transformer MLP block:
   $$\text{MLP}(\mathbf{h}_t) = W_2 \text{GeLU}(W_1 \mathbf{h}_t + \mathbf{b}_1) + \mathbf{b}_2$$
   Is there any cross-token communication inside an MLP block?
2. How does the information computed by the subject token's MLP reach the last token position so that the unembedding matrix $W_U$ can read it?
3. Design a two-stage patching intervention to prove that attention head $H$ at layer 3 is the specific mover transporting the subject's MLP output to the last token.

### Solution

#### 1. Token-Wise Independence of MLPs
The MLP operates strictly as a **position-wise** feed-forward network. For each token position $t \in \{0, 1, \dots, T-1\}$, $\text{MLP}(\mathbf{h}_t)$ depends *only* on $\mathbf{h}_t$ at that specific index. There are no operations across different tokens $t \neq t'$. Therefore, an MLP at the last token cannot access or retrieve facts about the subject unless that information was already transported into the last token's residual stream.

#### 2. Information Transport via Attention
The only mechanism for cross-token communication in a Transformer is the **Multi-Head Attention** mechanism:
$$\mathbf{h}_t^{\text{attn}} = \sum_{j \le t} A_{t, j} W_O W_V \mathbf{h}_j$$
At the last token $t = T-1$, an attention head with high attention weight $A_{T-1, \text{subject}} \approx 1$ reads the subject's residual stream (where the subject MLP wrote the fact) and copies it into the last token's residual stream.

#### 3. Two-Stage Proof Intervention
To prove head $H$ at layer 3 is the mover:
- **Test A (Sufficient Mover)**: Keep the entire corrupted run unchanged, but patch *only* the output of attention head $H$ at layer 3 at the last token position with its clean activation. If $\mathcal{R} \approx 100\%$, head $H$ is sufficient to restore the fact.
- **Test B (Necessary Mover / Knockout)**: Run the clean prompt, but overwrite (ablate/zero) head $H$'s attention pattern from the last token to the subject token. If the model's factual accuracy collapses to random chance while all other heads remain intact, head $H$ is causally necessary.
