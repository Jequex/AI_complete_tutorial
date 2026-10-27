# Phase 33: Mechanistic Interpretability

Welcome to **Phase 33: Mechanistic Interpretability** of the Complete Artificial Intelligence & Machine Learning Engineering Curriculum.

In classical machine learning, neural networks are treated as opaque, black-box statistical predictors evaluated purely on macroscopic validation metrics:
> *"The model predicted token $X$ with probability $P(X)$."*

**Mechanistic Interpretability (MechInterp)** reverses this paradigm. Drawing on reverse-engineering methodologies from computer systems, neuroscience, and algebraic geometry, our goal is to treat neural weights and activations as readable, reverse-engineerable computational circuits:
> *"Here is the internal causal computation, linear subspace, and attention head composition that produced token $X$."*

In Phase 33, we deconstruct the full modern mechanistic interpretability toolkit—from residual stream geometry and linear probing to causal activation patching, the Transformer Circuits framework ($QK$ and $OV$ decompositions), two-layer induction head circuits, the Superposition Hypothesis and Sparse Autoencoders (SAEs), and active model steering with surgical rank-one weight editing (ROME).

---

## Master Architectural Roadmap

```
                    PHASE 33: MECHANISTIC INTERPRETABILITY
                                       │
       ┌───────────────────────────────┴───────────────────────────────┐
       ▼                                                               ▼
  [01] Linear Probing & Geometry                                  [02] Causal Activation Patching
• Residual Stream as Communication Bus                          • Clean, Corrupted, & Patched Runs
• Linear Representation Hypothesis: v_c                         • Causal Tracing (Meng et al. 2022)
• Logistic Regression & Ridge Probes                            • Average Treatment Effect (ATE) on Logits
• Probe Accuracy vs Cosine Alignment                            • Normalized Logit Restoration R(l, t)
• Hewitt & Liang Selectivity Test                               • Early Subject MLP vs Late Head Routing
• Ruling Out Probe Memorization Capacity                        • Interchange Intervention Training (IIT)
       │                                                               │
       └───────────────────────────────┬───────────────────────────────┘
                                       │
                                       ▼
                     [03] Attention Head Circuits & Induction
                  • Transformer Circuits Framework (Elhage et al.)
                  • QK-Circuit: Where to look, A = softmax(x^T W_QK x)
                  • OV-Circuit: What to copy, Delta x = A (W_OV x)
                  • Eigenvalue Analysis of Copy Matrices W_U W_OV W_E
                  • Two-Layer Induction Circuit: [A][B] ... [A] -> [B]
                  • Layer 0 Previous-Token Head * Layer 1 Induction Head
                  • The Pretraining "Induction Bump" Phase Change
                                       │
       ┌───────────────────────────────┴───────────────────────────────┐
       ▼                                                               ▼
  [04] Polysemanticity & SAEs                                     [05] Steering & Weight Editing
• The Superposition Hypothesis: D >> d                          • Representation Engineering (Zou et al.)
• High-D Almost-Orthogonality (Johnson-Lindenstrauss)           • Difference-in-Means Steering Vectors
• Polysemanticity: Single Neurons Firing on Clashing Concepts   • Runtime Residual Intervention: h' = h + alpha * v
• Overcomplete Dictionaries: z = ReLU(W_enc * x + b_enc)        • Rank-One Model Editing (ROME)
• L1 Sparsity Loss: ||x - x_hat||^2 + lambda * ||z||_1          • Key-Value Associative Memories W * k = v
• Normalizing Decoder Columns ||w_dec|| = 1                     • Closed-Form Update: W' = W + Lambda * (C^-1 * k)^T
• Resolving Superposition into Monosemantic Latents             • Efficacy 100% & Zero Catastrophic Interference
```

---

## Mathematical Pillars & Core Derivations

### 1. The Residual Stream & The Linear Representation Hypothesis
In modern autoregressive Transformers, representations propagate through a shared linear highway:
$$\mathbf{h}_L = \mathbf{h}_0 + \sum_{l=0}^{L-1} \Delta \mathbf{h}_l^{\text{attn}} + \sum_{l=0}^{L-1} \Delta \mathbf{h}_l^{\text{mlp}}$$

The **Linear Representation Hypothesis** asserts that semantic features and factual knowledge correspond to 1D directions $\mathbf{v}_c \in \mathbb{R}^d$ ($\|\mathbf{v}_c\| = 1$). A linear probe optimizes:
$$\mathcal{L}(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^N \left[ y^{(i)} \ln \sigma(\mathbf{w}^\top \mathbf{h}^{(i)} + b) + (1 - y^{(i)}) \ln(1 - \sigma(\mathbf{w}^\top \mathbf{h}^{(i)} + b)) \right] + \frac{\lambda}{2} \|\mathbf{w}\|_2^2$$
To rule out probe memorization, we evaluate **Probe Selectivity**:
$$\boxed{\text{Selectivity} = \text{Accuracy}_{\text{Real Task}} - \text{Accuracy}_{\text{Random Control}}}$$
A selectivity exceeding $35\%$ confirms genuine native linear representation, rejecting probe parameter overfitting.

---

### 2. Causal Mediation Analysis & The Normalized Restoration Metric
To move beyond correlational observations, **Activation Patching** intervenes on internal network nodes between a clean run $x_{\text{clean}}$ and a corrupted/counterfactual run $x_{\text{corrupt}}$. The causal importance of layer $l$ at token position $t$ is quantified via the **Normalized Logit Difference Restoration**:
$$\boxed{\mathcal{R}(l, t) = \frac{\Delta L_{\text{patched}}(l, t) - \Delta L_{\text{corrupt}}}{\Delta L_{\text{clean}} - \Delta L_{\text{corrupt}}}}$$
where $\Delta L = \text{logit}(y_*) - \text{logit}(y_{\text{alt}})$. In factual recall (Meng et al., 2022), causal tracing reveals two localized circuits:
1. **Early/Mid-Layer MLPs at the Subject Token**: Storing and retrieving the associative factual memory ($\mathcal{R} \approx 85\%$).
2. **Late-Layer Attention Heads at the Last Token**: Routing the retrieved concept from the subject to the output logits ($\mathcal{R} \approx 100\%$).

---

### 3. The Transformer Circuits Framework & Induction Heads
Self-attention decomposes into two independent bilinear circuits (Elhage et al., 2021):
- **$QK$-Circuit**: $\alpha_{i, j} = \frac{\mathbf{x}_i^\top (W_Q^\top W_K) \mathbf{x}_j}{\sqrt{d_{\text{head}}}}$, controlling *where information is moved*.
- **$OV$-Circuit**: $\Delta \mathbf{x}_i = \sum_j A_{i, j} (W_O W_V) \mathbf{x}_j$, controlling *what information is copied*.

A single layer cannot execute general in-context completion $[A][B] \dots [A] \to [B]$. It requires composing a **two-layer Induction Circuit**:
- **Layer 0 (Previous-Token Head)**: Writes the identity of token $j-1$ ($[A]$) into the residual stream of token $j$ ($[B]$).
- **Layer 1 (Induction Head)**: Compares current token $i$ ($[A]$) with the previous-token signal stored at position $j$, causing attention to spike to position $j$, where its $OV$-circuit copies $[B]$ directly to output logits!

---

### 4. The Superposition Hypothesis & Sparse Autoencoders (SAEs)
Because natural features are sparse ($P(f_k \neq 0) \ll 1$), neural networks represent $D \gg d$ features by packing them into almost-orthogonal directions in high-dimensional space (Johnson-Lindenstrauss lemma):
$$\mathbf{x} = \sum_{k=1}^D f_k \mathbf{v}_k, \quad \text{where } |\langle \mathbf{v}_i, \mathbf{v}_j \rangle| \le \epsilon$$
This causes individual coordinate neurons to be **polysemantic** (activating for unrelated concepts). **Sparse Autoencoders (SAEs)** untangle superposition by learning an overcomplete dictionary:
$$\mathbf{z} = \text{ReLU}(W_{\text{enc}} \mathbf{x} + \mathbf{b}_{\text{enc}}), \quad \hat{\mathbf{x}} = W_{\text{dec}} \mathbf{z} + \mathbf{b}_{\text{dec}}$$
$$\boxed{\mathcal{L}_{\text{SAE}} = \frac{1}{B} \sum_{i=1}^B \left( \|\mathbf{x}^{(i)} - \hat{\mathbf{x}}^{(i)}\|_2^2 + \lambda \sum_{j=1}^D |z_j^{(i)}| \right) \quad \text{s.t.} \quad \|\mathbf{w}_j^{\text{dec}}\|_2 = 1}$$
The unit-norm constraint on decoder columns prevents scaling degeneracies, allowing the SAE to recover clean, monosemantic latent directions.

---

### 5. Representation Engineering & Rank-One Model Editing (ROME)
At inference time, model behavior can be steered along conceptual axes using difference-of-means vectors:
$$\mathbf{v}_{\text{steer}} = \frac{\frac{1}{N}\sum \mathbf{h}_+ - \frac{1}{M}\sum \mathbf{h}_-}{\|\frac{1}{N}\sum \mathbf{h}_+ - \frac{1}{M}\sum \mathbf{h}_-\|_2}, \quad \mathbf{h}' = \mathbf{h} + \alpha \mathbf{v}_{\text{steer}}$$

To permanently edit or insert factual memories without catastrophic forgetting, **ROME** models MLP projections as linear associative memories $W \mathbf{k} \approx \mathbf{v}$. Solving the constrained optimization problem under background knowledge covariance $\mathbf{C} = \mathbb{E}[\mathbf{k} \mathbf{k}^\top]$ yields the exact closed-form rank-1 update:
$$\boxed{W' = W + \boldsymbol{\Lambda} \left( \mathbf{C}^{-1} \mathbf{k}_* \right)^\top}$$
$$\boxed{\boldsymbol{\Lambda} = \frac{\mathbf{v}_* - W \mathbf{k}_*}{\mathbf{k}_*^\top \mathbf{C}^{-1} \mathbf{k}_*}}$$
Pre-multiplying by $\mathbf{C}^{-1}$ projects the weight update into the null space of pre-existing knowledge, achieving $100\%$ target update efficacy with near-zero drift on prior memories.

---

## Phase Navigation & Directory Index

| Topic | Directory | Core Mathematical & Engineering Concepts | Resources |
| :--- | :--- | :--- | :--- |
| **01** | `01_linear_probing_and_representation_geometry/` | Residual stream communication bus, Linear Representation Hypothesis, Logistic classification probes, Cosine alignment, Hewitt & Liang Selectivity test. | [Lecture (HTML)](01_linear_probing_and_representation_geometry/lecture.html) &bull; [Exercises](01_linear_probing_and_representation_geometry/exercises_and_solutions.html) &bull; [Experiment (Python)](01_linear_probing_and_representation_geometry/experiment.py) |
| **02** | `02_activation_patching_and_causal_tracing/` | Clean vs Corrupted runs, Causal Tracing (ROME), Average Treatment Effect (ATE), Normalized logit difference restoration $\mathcal{R}(l, t)$, Subject MLP vs Late Attention. | [Lecture (HTML)](02_activation_patching_and_causal_tracing/lecture.html) &bull; [Exercises](02_activation_patching_and_causal_tracing/exercises_and_solutions.html) &bull; [Experiment (Python)](02_activation_patching_and_causal_tracing/experiment.py) |
| **03** | `03_attention_head_mechanisms_and_induction_heads/` | Transformer Circuits, $QK$-circuits, $OV$-circuits, Copy matrix spectrum $W_U W_{OV} W_E$, 2-layer Induction Head circuit composition, In-context pattern completion. | [Lecture (HTML)](03_attention_head_mechanisms_and_induction_heads/lecture.html) &bull; [Exercises](03_attention_head_mechanisms_and_induction_heads/exercises_and_solutions.html) &bull; [Experiment (Python)](03_attention_head_mechanisms_and_induction_heads/experiment.py) |
| **04** | `04_polysemanticity_and_sparse_autoencoders/` | The Superposition Hypothesis ($D \gg d$), Johnson-Lindenstrauss almost-orthogonality, Polysemantic neurons, Sparse Autoencoders (SAEs), $L_1$ regularized dictionary learning. | [Lecture (HTML)](04_polysemanticity_and_sparse_autoencoders/lecture.html) &bull; [Exercises](04_polysemanticity_and_sparse_autoencoders/exercises_and_solutions.html) &bull; [Experiment (Python)](04_polysemanticity_and_sparse_autoencoders/experiment.py) |
| **05** | `05_model_steering_and_weight_editing/` | Representation Engineering (RepE), Contrastive steering vectors, Runtime residual intervention $\mathbf{h}' = \mathbf{h} + \alpha \mathbf{v}$, Closed-form ROME update $W' = W + \boldsymbol{\Lambda} (\mathbf{C}^{-1}\mathbf{k})^\top$. | [Lecture (HTML)](05_model_steering_and_weight_editing/lecture.html) &bull; [Exercises](05_model_steering_and_weight_editing/exercises_and_solutions.html) &bull; [Experiment (Python)](05_model_steering_and_weight_editing/experiment.py) |

---

## 🚀 Running All Phase 33 Experiments

Every single experiment in Phase 33 is written from first principles in **pure Python 3 standard library** with zero external dependencies (no PyTorch, no NumPy, no SciPy, no GPU required):

```bash
# Topic 01: Linear Probing & Representation Geometry
python3 phase_33_mechanistic_interpretability/01_linear_probing_and_representation_geometry/experiment.py

# Topic 02: Activation Patching & Causal Tracing
python3 phase_33_mechanistic_interpretability/02_activation_patching_and_causal_tracing/experiment.py

# Topic 03: Attention Head Mechanisms & Induction Heads
python3 phase_33_mechanistic_interpretability/03_attention_head_mechanisms_and_induction_heads/experiment.py

# Topic 04: Polysemanticity & Sparse Autoencoders (SAEs)
python3 phase_33_mechanistic_interpretability/04_polysemanticity_and_sparse_autoencoders/experiment.py

# Topic 05: Model Steering & Weight Editing (ROME)
python3 phase_33_mechanistic_interpretability/05_model_steering_and_weight_editing/experiment.py
```
