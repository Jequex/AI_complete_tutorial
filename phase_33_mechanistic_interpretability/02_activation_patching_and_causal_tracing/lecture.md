# Topic 02: Activation Patching & Causal Tracing

## 1. Introduction & 12-Year-Old Intuition

Imagine you are watching a high-stakes detective movie. A bank vault was unlocked, but there are 40 bank employees working across 12 floors of the skyscraper. Who entered the passcode into the computer system?

If you just look at photos of who was in the building, everyone looks suspicious—they are all busy typing at their desks. Correlation does not tell you who actually opened the vault door.

To find the true culprit with 100% certainty, you run a **controlled intervention experiment**:
1. **The Clean Run**: On a normal day, Agent Alice is working on Floor 4, and the vault unlocks with code "PARIS".
2. **The Corrupted Run**: On an alternate day, Alice is replaced by Agent Bob, and the system fails to unlock (it says "ROME").
3. **The Activation Patch (The Swap)**: You rerun the alternate day with Bob, but at the exact moment Floor 4 computes its work, you secretly sneak Alice's notebook onto the desk of Floor 4, while keeping everything else in the building identical!
   - If the vault suddenly unlocks with "PARIS", you have proven beyond doubt that **Floor 4's notebook was the causal driver** of the passcode.
   - If the vault still fails, Floor 4 had nothing to do with it.

In Mechanistic Interpretability, this technique is called **Activation Patching** (or causal mediation analysis). Instead of guessing what a neuron does by looking at correlations, we surgically swap activations between a "clean" run and a "corrupted" run. By measuring whether the model's output flips back to the original answer, we can draw a precise causal circuit map through millions of neural connections.

---

## 2. Mathematical Symbol Decoder

| Symbol | Mathematical Entity | Units / Domain | Operational Meaning & Interpretability Role |
| :--- | :--- | :--- | :--- |
| $x_{\text{clean}}$ | Clean Input Prompt | Token sequence | The prompt containing the factual subject (e.g., *"The Eiffel Tower is in"*). |
| $x_{\text{corrupt}}$ | Corrupted Input Prompt | Token sequence | Counterfactual prompt with an altered subject or Gaussian noise (e.g., *"The Colosseum is in"*). |
| $y_*$ | Target Clean Output | Token ID | The factual ground-truth token (e.g., *"Paris"*). |
| $y_{\text{alt}}$ | Alternative Output | Token ID | The counterfactual token associated with corruption (e.g., *"Rome"*). |
| $\mathbf{a}_l^{(t)}$ | Internal Activation | $\mathbb{R}^d$ | Hidden state vector at layer $l$, token index $t$, and specific module (MLP, Attention, or Residual). |
| $\mathbf{a}_{\text{clean}}, \mathbf{a}_{\text{corrupt}}$ | Baseline Activations | $\mathbb{R}^d$ | Activations captured during the uninterrupted clean and corrupted forward passes. |
| $\text{logit}(y)$ | Pre-softmax Logit | Real scalar | The unnormalized score assigned by the model's output unembedding layer to token $y$. |
| $\Delta L_{\text{clean}}$ | Clean Logit Difference | Real scalar | $\text{logit}_{\text{clean}}(y_*) - \text{logit}_{\text{clean}}(y_{\text{alt}})$. |
| $\Delta L_{\text{corrupt}}$ | Corrupted Logit Diff | Real scalar | $\text{logit}_{\text{corrupt}}(y_*) - \text{logit}_{\text{corrupt}}(y_{\text{alt}})$. |
| $\text{ATE}$ | Average Treatment Effect | Real scalar | Causal impact of the intervention: $\mathbb{E}[\Delta L_{\text{patched}} - \Delta L_{\text{corrupt}}]$. |
| $\mathcal{R}(l, t)$ | Normalized Restoration | Real ratio (often $[0, 1]$) | Percentage of clean model capability recovered by patching state $(l, t)$. |

---

## 3. Causal Intervention Methodology: Clean, Corrupted, Patched

Activation patching isolates causal dependencies using a 3-step protocol:

```
1. CLEAN RUN:
   "The Eiffel Tower is located in" ──▶ [ Forward Pass ] ──▶ Logit("Paris") = +37.4
                                            │ Cache Activations a_clean(l, t)

2. CORRUPTED RUN:
   "The Colosseum is located in"   ──▶ [ Forward Pass ] ──▶ Logit("Paris") = -5.5
                                            │ Baseline drop in target logit

3. PATCHED RUN:
   "The Colosseum is located in"   ──▶ [ Forward Pass ]
                                            │ At (Layer l, Pos t):
                                            │ Overwrite a_corrupt ◄── a_clean(l, t)
                                            ▼
                                       [ Forward Pass ] ──▶ Logit("Paris") = +31.2
```

### 3.1 The Normalized Restoration Metric
To measure whether an individual activation $\mathbf{a}(l, t)$ causally mediates the factual output, we define the **Normalized Logit Difference Restoration**:

$$\boxed{\mathcal{R}(l, t) = \frac{\Delta L_{\text{patched}}(l, t) - \Delta L_{\text{corrupt}}}{\Delta L_{\text{clean}} - \Delta L_{\text{corrupt}}}}$$

where:
$$\Delta L = \text{logit}(y_*) - \text{logit}(y_{\text{alt}})$$

- **$\mathcal{R}(l, t) \approx 1.0$ (or $100\%$)**: Overwriting this single activation alone completely restores the model's factual memory, proving it is a critical information bottleneck.
- **$\mathcal{R}(l, t) \approx 0.0$**: The component has no causal contribution to the fact at this stage of processing.

---

## 4. Causal Tracing (Meng et al. 2022, ROME)

In **Locating and Editing Factual Associations in GPT (ROME)**, Meng et al. (2022) applied causal tracing to autoregressive language models (GPT-2, GPT-J, GPT-NeoX).

### The Landmark Discovery: Two Distinct Functional Phases
When patching individual components across all layers $l \in [1, L]$ and token positions $t \in [1, T]$, a striking two-phase architecture emerged:

```
Layer L
  ▲
  │                                     [Late Attention Heads]
  │                                     Route fact from subject to final token
  │                                     Position: Last Token
Mid │               [Early/Mid MLPs]
  │               Recall factual association
  │               Position: Subject Token
0 └─────────────────────────────────────────────────────────────▶
    Token 0 (Subject)       Token 1 ("is")       Token 2 (Last Token)
```

1. **Early/Mid-Layer MLPs at the Subject Token**:
   - When patching the MLP activations at the subject token (e.g., *"Eiffel Tower"*), restoration reaches up to $80\text{--}90\%$.
   - **Conclusion**: Factual associations are stored as associative memory keys in intermediate MLP layers!
2. **Late-Layer Attention Heads at the Final Token**:
   - When patching attention heads at the prompt's final token (e.g., *"in"*), restoration spikes again.
   - **Conclusion**: Late attention heads act as *information movers*, reading the retrieved factual concept from the subject's residual stream and copying it to the last position for unembedding.

---

## 5. Causal Mediation Analysis & The Average Treatment Effect (ATE)

Under Pearl's structural causal model (SCM) framework, the neural network computation is a directed acyclic graph (DAG) where nodes represent intermediate activations $\mathbf{A}$, the input context is variable $X$, and the output logit difference is $Y$.

### The Do-Calculus Formulation
The uncorrupted prediction corresponds to $P(Y \mid X = x_{\text{clean}})$.
The intervention corresponds to setting node $\mathbf{A}(l, t)$ to its clean value $\mathbf{a}_{\text{clean}}$ while holding the input at $X = x_{\text{corrupt}}$:

$$P\left(Y \mid \text{do}(X = x_{\text{corrupt}}, \; \mathbf{A}(l, t) = \mathbf{a}_{\text{clean}})\right)$$

The **Average Treatment Effect (ATE)** or Total Indirect Effect (TIE) of node $\mathbf{A}(l, t)$ is:

$$\text{ATE}(l, t) = \mathbb{E}\left[ Y \mid \text{do}(\mathbf{A}(l, t) = \mathbf{a}_{\text{clean}}, X = x_{\text{corrupt}}) \right] - \mathbb{E}\left[ Y \mid X = x_{\text{corrupt}} \right]$$

When applied across a distribution of factual triplets $(s, r, o)$ (subject, relation, object), the expectation $\mathbb{E}[\text{ATE}]$ quantifies the exact causal power of the layer/head across the entire domain.

---

## 6. Interchange Intervention Training (IIT)

Beyond passive interpretability, causal interventions can be used for model training and hypothesis testing via **Interchange Intervention Training (IIT)** (Geiger et al., 2021).

### The Problem:
How do we know if a neural network actually computes a specific high-level algorithmic program (e.g., hierarchical syntax parsing or modular arithmetic)?

### The IIT Algorithm:
1. Define a high-level symbolic causal model $\mathcal{M}_{\text{symbolic}}$ (the causal hypothesis).
2. Sample two inputs $x_1, x_2$.
3. Compute an interchange intervention in $\mathcal{M}_{\text{symbolic}}$: swap intermediate variable $V$ from $x_1$ into $x_2$, obtaining counterfactual output $y^*$.
4. Align $V$ with a specific neural subspace or head in the deep model $\mathcal{M}_{\text{neural}}$.
5. Perform activation patching on $\mathcal{M}_{\text{neural}}$ and backpropagate the cross-entropy loss between the patched output and $y^*$:

$$\mathcal{L}_{\text{IIT}} = \mathcal{L}_{\text{CE}}\left(\mathcal{M}_{\text{neural}}(x_2 \mid \text{patch}(V \leftarrow x_1)), \; y^*\right)$$

By training with $\mathcal{L}_{\text{IIT}}$, we can explicitly guide the network to realize human-interpretable causal circuits.

---

## 7. Practical Engineering Rules for Causal Tracing
1. **Choose Corrupted Prompts Carefully**: The corrupted prompt must match the syntax, syllable count, and format of the clean prompt so that differences in intermediate activations isolate *only* the semantic fact.
2. **Noise vs Token Replacement**: In ROME, Meng et al. corrupted subject embeddings by adding Gaussian noise $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, 3\sigma^2)$. Counterfactual token replacement (e.g., *"Rome"* vs *"Paris"*) provides a complementary semantic contrast.
3. **Isolate Components**: Always patch Residual Stream, Attention Heads, and MLPs independently to avoid confounding the information storage mechanism with the routing mechanism.
