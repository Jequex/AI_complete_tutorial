# Topic 05: Model Steering & Weight Editing

## 1. Introduction & 12-Year-Old Intuition

Imagine you have a loyal robot assistant named Arthur. Arthur is intelligent and knows millions of facts, but he has two small flaws:
1. **Mood Control (Steering)**: Sometimes Arthur sounds grumpy or pessimistic, and you wish you had a simple dial on his chest labeled *"Politeness & Optimism"*. Turning the dial to $+3$ should make Arthur cheerful and helpful, while turning it to $-3$ makes him grumpy, without having to send him back to the factory to be rebuilt.
2. **Memory Correction (Weight Editing)**: Arthur firmly believes that *"The capital of the Moon is Seattle"*. If you try to teach him the truth by yelling at him (standard prompt engineering) or retraining him from scratch (pretraining), it takes months and millions of dollars. Worse, if you fine-tune him hastily, he might forget how to do math or forget that Paris is in France!

What if you could open a tiny hatch in Arthur's head, find the single memory chip storing the Moon's capital, and surgically alter one single circuit connection so that *"Moon $\to$ Luna City"*, while leaving all of Arthur's other memories $100\%$ untouched?

In Mechanistic Interpretability, these two superpowers are:
- **Activation Steering (Representation Engineering)**: Adding a concept direction vector $\mathbf{v}_{\text{steer}}$ directly to the model's intermediate thoughts during generation.
- **Rank-One Model Editing (ROME)**: Calculating a mathematically exact, closed-form rank-1 update matrix $\Delta W = \boldsymbol{\Lambda} \mathbf{k}_*^\top$ that inserts or edits a specific factual memory while mathematically preserving the covariance of all pre-existing knowledge.

---

## 2. Mathematical Symbol Decoder

| Symbol | Mathematical Entity | Domain | Interpretability Role |
| :--- | :--- | :--- | :--- |
| $\mathcal{D}_+, \mathcal{D}_-$ | Contrastive Datasets | Text prompts | Pairs of positive vs negative prompts isolating a single concept. |
| $\mathbf{h}_l^{(i)}$ | Residual Stream Activation | $\mathbb{R}^d$ | Intermediate representation at layer $l$ for sample $i$. |
| $\mathbf{v}_{\text{steer}}$ | Steering Vector | $\mathbb{R}^d, \|\mathbf{v}\|=1$ | Normalized difference-in-means vector representing the concept. |
| $\alpha \in \mathbb{R}$ | Steering Strength / Multiplier | Real scalar | Controls intervention magnitude; positive amplifies, negative suppresses. |
| $\mathbf{h}'_l$ | Steered Activation | $\mathbb{R}^d$ | Intervened residual state: $\mathbf{h}'_l = \mathbf{h}_l + \alpha \mathbf{v}_{\text{steer}}$. |
| $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ | Target Weight Matrix | Linear layer (MLP $W_2$) | Feed-forward projection matrix storing key-value associative memories. |
| $\mathbf{k}_* \in \mathbb{R}^{d_{\text{in}}}$ | Subject Key Vector | $d_{\text{in}}$-dim vector | Internal representation of the subject entity to be edited (e.g., *"Space Needle"*). |
| $\mathbf{v}_* \in \mathbb{R}^{d_{\text{out}}}$ | Target Value Vector | $d_{\text{out}}$-dim vector | Desired output representation (e.g., *"Seattle"*). |
| $\mathbf{C} \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}$ | Knowledge Covariance Matrix | Positive definite matrix | $\mathbf{C} = \mathbb{E}[\mathbf{k} \mathbf{k}^\top]$, capturing the distribution of all unedited keys. |
| $\boldsymbol{\Lambda} \in \mathbb{R}^{d_{\text{out}}}$ | ROME Update Vector | $d_{\text{out}}$-dim vector | Direction of rank-1 adjustment: $W' = W + \boldsymbol{\Lambda} (\mathbf{C}^{-1} \mathbf{k}_*)^\top$. |

---

## 3. Activation Steering: Representation Engineering (RepE)

Introduced by **Zou et al. (2023)**, **Representation Engineering (RepE)** treats intermediate activations as a readable and writable control interface.

### 3.1 Computing the Difference-in-Means Steering Vector
Given a contrastive pair dataset (e.g., honest vs deceptive, optimistic vs pessimistic, or sycophantic vs objective):
- Positive set: $\{\mathbf{h}_+^{(1)}, \dots, \mathbf{h}_+^{(N)}\}$
- Negative set: $\{\mathbf{h}_-^{(1)}, \dots, \mathbf{h}_-^{(M)}\}$

The concept direction is computed via difference-of-means:

$$\mathbf{v}_{\text{diff}} = \frac{1}{N} \sum_{i=1}^N \mathbf{h}_+^{(i)} - \frac{1}{M} \sum_{j=1}^M \mathbf{h}_-^{(j)}$$
$$\mathbf{v}_{\text{steer}} = \frac{\mathbf{v}_{\text{diff}}}{\|\mathbf{v}_{\text{diff}}\|_2}$$

*(Alternatively, $\mathbf{v}_{\text{steer}}$ is the first principal component from PCA on the contrastive difference vectors).*

### 3.2 Runtime Steering Intervention
During autoregressive text generation, at layer $l$, the residual stream activation $\mathbf{h}_l$ is intervened:

$$\boxed{\mathbf{h}'_l = \mathbf{h}_l + \alpha \cdot \mathbf{v}_{\text{steer}}}$$

- **$\alpha > 0$**: Injects the target behavior (e.g., making the model hyper-truthful or euphoric).
- **$\alpha = 0$**: Standard unsteered baseline generation.
- **$\alpha < 0$**: Inverts the concept (e.g., inducing refusal or cynicism).

Because steering operates entirely at inference time in the residual stream, it requires **zero gradient updates, zero fine-tuning, and zero weight changes**.

---

## 4. Direct Weight Editing: Rank-One Model Editing (ROME)

While steering changes behavior globally, **Model Editing** aims to correct isolated factual errors with surgical precision.

In **Locating and Editing Factual Associations in GPT (ROME)** (Meng et al., 2022), the second linear layer of a Transformer MLP block ($W_{\text{proj}} \in \mathbb{R}^{d \times d_{\text{mlp}}}$) is modeled as a linear associative memory:

$$W \mathbf{k}_i \approx \mathbf{v}_i$$

where $\mathbf{k}_i$ is a key vector representing a subject (e.g., *"The Eiffel Tower"*) and $\mathbf{v}_i$ is the value vector representing the factual attribute (e.g., *"Paris"*).

### 4.1 The Constrained Optimization Problem
We wish to find a new weight matrix $W'$ such that:
1. **Efficacy**: The target key $\mathbf{k}_*$ outputs the new desired value $\mathbf{v}_*$:
   $$W' \mathbf{k}_* = \mathbf{v}_*$$
2. **Specificity (Minimal Interference)**: The output on all other pre-existing factual keys $\mathbf{k}$ is distorted as little as possible:
   $$\min_{W'} \sum_{\mathbf{k} \in \mathcal{K}} \|W' \mathbf{k} - W \mathbf{k}\|_2^2 \quad \text{subject to} \quad W' \mathbf{k}_* = \mathbf{v}_*$$

Writing this in continuous expectation form with unedited key covariance $\mathbf{C} = \mathbb{E}[\mathbf{k} \mathbf{k}^\top]$:

$$\min_{W'} \text{Tr}\left( (W' - W) \mathbf{C} (W' - W)^\top \right) \quad \text{s.t.} \quad W' \mathbf{k}_* = \mathbf{v}_*$$

---

## 5. Mathematical Proof of the Closed-Form ROME Update

We formulate the Lagrangian for the optimization problem. Let $\Delta W = W' - W$.
The objective is:

$$\mathcal{L}(\Delta W, \boldsymbol{\nu}) = \frac{1}{2} \text{Tr}\left(\Delta W \mathbf{C} \Delta W^\top\right) + \boldsymbol{\nu}^\top \left( \mathbf{v}_* - (W + \Delta W) \mathbf{k}_* \right)$$

where $\boldsymbol{\nu} \in \mathbb{R}^{d_{\text{out}}}$ is the vector of Lagrange multipliers.

### Step 1: Derivative with Respect to $\Delta W$
$$\frac{\partial \mathcal{L}}{\partial \Delta W} = \Delta W \mathbf{C} - \boldsymbol{\nu} \mathbf{k}_*^\top = 0$$
$$\Delta W \mathbf{C} = \boldsymbol{\nu} \mathbf{k}_*^\top$$
Multiplying on the right by $\mathbf{C}^{-1}$ (assuming $\mathbf{C}$ is invertible via regularization $\mathbf{C} + \lambda \mathbf{I}$):

$$\Delta W = \boldsymbol{\nu} (\mathbf{C}^{-1} \mathbf{k}_*)^\top$$

This proves that the optimal perturbation $\Delta W$ is **strictly a rank-one matrix**!

### Step 2: Solving for the Multiplier $\boldsymbol{\nu}$
Substitute $\Delta W$ into the hard constraint $W' \mathbf{k}_* = (W + \Delta W)\mathbf{k}_* = \mathbf{v}_*$:

$$W \mathbf{k}_* + \Delta W \mathbf{k}_* = \mathbf{v}_*$$
$$W \mathbf{k}_* + \boldsymbol{\nu} (\mathbf{C}^{-1} \mathbf{k}_*)^\top \mathbf{k}_* = \mathbf{v}_*$$

Notice that $(\mathbf{C}^{-1} \mathbf{k}_*)^\top \mathbf{k}_* = \mathbf{k}_*^\top \mathbf{C}^{-1} \mathbf{k}_*$ is a **scalar**:

$$\boldsymbol{\nu} \left( \mathbf{k}_*^\top \mathbf{C}^{-1} \mathbf{k}_* \right) = \mathbf{v}_* - W \mathbf{k}_*$$
$$\boldsymbol{\nu} = \frac{\mathbf{v}_* - W \mathbf{k}_*}{\mathbf{k}_*^\top \mathbf{C}^{-1} \mathbf{k}_*}$$

Defining $\boldsymbol{\Lambda} \triangleq \boldsymbol{\nu}$:

$$\boxed{\boldsymbol{\Lambda} = \frac{\mathbf{v}_* - W \mathbf{k}_*}{\mathbf{k}_*^\top \mathbf{C}^{-1} \mathbf{k}_*}}$$
$$\boxed{W' = W + \boldsymbol{\Lambda} \left( \mathbf{C}^{-1} \mathbf{k}_* \right)^\top}$$

### Key Properties of the Solution:
1. **Exact Recall**:
   $$W' \mathbf{k}_* = W \mathbf{k}_* + \frac{\mathbf{v}_* - W \mathbf{k}_*}{\mathbf{k}_*^\top \mathbf{C}^{-1} \mathbf{k}_*} \left( \mathbf{k}_*^\top \mathbf{C}^{-1} \mathbf{k}_* \right) = W \mathbf{k}_* + (\mathbf{v}_* - W \mathbf{k}_*) = \mathbf{v}_*$$
   The target fact is recalled with $100\%$ mathematical precision.
2. **Minimal Covariance Drift**:
   For any key $\mathbf{k}_j$ orthogonal to $\mathbf{k}_*$ under the metric tensor $\mathbf{C}^{-1}$ ($\mathbf{k}_j^\top \mathbf{C}^{-1} \mathbf{k}_* = 0$), $W' \mathbf{k}_j = W \mathbf{k}_j$ with **zero drift**.

---

## 6. Summary Comparison: Steering vs Weight Editing

| Dimension | Activation Steering (RepE) | Weight Editing (ROME) |
| :--- | :--- | :--- |
| **Intervention Target** | Residual stream activations $\mathbf{h}_l$ | MLP weight matrix $W_{\text{proj}}$ |
| **Persistence** | Ephemeral (active only during steered inference) | Permanent (model checkpoint weights are modified) |
| **Scope** | Global behavioral shift (sentiment, honesty, style) | Surgical, localized factual assertion ($s, r, o$) |
| **Compute Cost** | Single vector addition at runtime ($\mathcal{O}(d)$) | Matrix inversion $\mathbf{C}^{-1}$ once ($\mathcal{O}(d^3)$) |
| **Side Effects** | Broad behavioral modulation across all queries | Preserves existing key-value covariance |
