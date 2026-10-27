# Topic 05: Exercises & Solutions — Model Steering & Weight Editing

## Exercise 1: Step-by-Step Derivation of the Closed-Form ROME Update

Consider the linear associative memory problem in ROME (Meng et al., 2022).
Let $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ be the pre-edit MLP projection matrix.
Let $\mathbf{C} = \sum_{i=1}^M \mathbf{k}_i \mathbf{k}_i^\top \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}$ be the positive definite covariance matrix of background keys.
We seek an updated matrix $W'$ that minimizes total squared distortion on the background keys while satisfying the exact equality constraint for the target factual pair $(\mathbf{k}_*, \mathbf{v}_*)$:

$$\min_{W'} \sum_{i=1}^M \|(W' - W) \mathbf{k}_i\|_2^2 \quad \text{subject to} \quad W' \mathbf{k}_* = \mathbf{v}_*$$

1. Express the objective function in matrix trace form using $\Delta W = W' - W$ and covariance matrix $\mathbf{C}$.
2. Form the Lagrangian $\mathcal{L}(\Delta W, \boldsymbol{\nu})$ with Lagrange multiplier vector $\boldsymbol{\nu} \in \mathbb{R}^{d_{\text{out}}}$.
3. Differentiate $\mathcal{L}$ with respect to $\Delta W$ to prove that $\Delta W$ must have algebraic rank 1.
4. Solve for $\boldsymbol{\nu}$ and write down the final closed-form update rule for $W'$.

### Solution

#### 1. Trace Formulation
For any vector $\mathbf{u}$, $\|\mathbf{u}\|_2^2 = \text{Tr}(\mathbf{u} \mathbf{u}^\top) = \text{Tr}(\mathbf{u}^\top \mathbf{u})$.
$$\sum_{i=1}^M \|\Delta W \mathbf{k}_i\|_2^2 = \sum_{i=1}^M \text{Tr}\left( \Delta W \mathbf{k}_i \mathbf{k}_i^\top \Delta W^\top \right) = \text{Tr}\left( \Delta W \left( \sum_{i=1}^M \mathbf{k}_i \mathbf{k}_i^\top \right) \Delta W^\top \right) = \text{Tr}\left( \Delta W \mathbf{C} \Delta W^\top \right)$$
Thus, the optimization problem is:
$$\min_{\Delta W} \frac{1}{2} \text{Tr}\left( \Delta W \mathbf{C} \Delta W^\top \right) \quad \text{subject to} \quad (W + \Delta W) \mathbf{k}_* = \mathbf{v}_*$$

#### 2. Lagrangian Formulation
$$\mathcal{L}(\Delta W, \boldsymbol{\nu}) = \frac{1}{2} \text{Tr}\left( \Delta W \mathbf{C} \Delta W^\top \right) + \boldsymbol{\nu}^\top \left( \mathbf{v}_* - W \mathbf{k}_* - \Delta W \mathbf{k}_* \right)$$

#### 3. Rank-1 Structure of $\Delta W$
Using the standard matrix derivative identity $\frac{\partial}{\partial X} \text{Tr}(X A X^\top) = 2 X A$ (for symmetric $A = \mathbf{C}$):
$$\frac{\partial \mathcal{L}}{\partial \Delta W} = \Delta W \mathbf{C} - \boldsymbol{\nu} \mathbf{k}_*^\top = 0$$
$$\Delta W \mathbf{C} = \boldsymbol{\nu} \mathbf{k}_*^\top$$
Post-multiplying both sides by $\mathbf{C}^{-1}$:
$$\Delta W = \boldsymbol{\nu} (\mathbf{C}^{-1} \mathbf{k}_*)^\top$$
Because $\Delta W$ is the outer product of column vector $\boldsymbol{\nu}$ and row vector $(\mathbf{C}^{-1} \mathbf{k}_*)^\top$, its column rank is at most 1. Hence, the optimal update is **strictly a rank-one matrix**.

#### 4. Solving for $\boldsymbol{\nu}$ and Final Update
Substitute $\Delta W = \boldsymbol{\nu} (\mathbf{C}^{-1} \mathbf{k}_*)^\top$ into the constraint $\Delta W \mathbf{k}_* = \mathbf{v}_* - W \mathbf{k}_*$:
$$\left( \boldsymbol{\nu} (\mathbf{C}^{-1} \mathbf{k}_*)^\top \right) \mathbf{k}_* = \mathbf{v}_* - W \mathbf{k}_*$$
$$\boldsymbol{\nu} \left( \mathbf{k}_*^\top \mathbf{C}^{-1} \mathbf{k}_* \right) = \mathbf{v}_* - W \mathbf{k}_*$$
Since $\mathbf{k}_*^\top \mathbf{C}^{-1} \mathbf{k}_*$ is a non-zero scalar (by positive definiteness of $\mathbf{C}$):
$$\boldsymbol{\nu} = \frac{\mathbf{v}_* - W \mathbf{k}_*}{\mathbf{k}_*^\top \mathbf{C}^{-1} \mathbf{k}_*}$$
Defining $\boldsymbol{\Lambda} = \boldsymbol{\nu}$, the final closed-form update rule is:
$$\boxed{W' = W + \frac{\mathbf{v}_* - W \mathbf{k}_*}{\mathbf{k}_*^\top \mathbf{C}^{-1} \mathbf{k}_*} \left( \mathbf{C}^{-1} \mathbf{k}_* \right)^\top}$$

---

## Exercise 2: Why Covariance Inversion Prevents Catastrophic Forgetting

Suppose a naive engineer implements a rank-one weight update *without* the covariance matrix $\mathbf{C}^{-1}$:
$$W_{\text{naive}}' = W + \frac{\mathbf{v}_* - W \mathbf{k}_*}{\|\mathbf{k}_*\|_2^2} \mathbf{k}_*^\top$$

1. Verify whether $W_{\text{naive}}' \mathbf{k}_* = \mathbf{v}_*$.
2. Let $\mathbf{k}_j$ be a pre-existing factual key that is correlated with $\mathbf{k}_*$ such that $\mathbf{k}_*^\top \mathbf{k}_j = \rho > 0$. Compute the output drift $\|W_{\text{naive}}' \mathbf{k}_j - W \mathbf{k}_j\|_2$.
3. Contrast this with the ROME update $W_{\text{ROME}}'$: how does pre-multiplying by $\mathbf{C}^{-1}$ project the update into a subspace that is orthogonal to the major principal components of background knowledge?

### Solution

#### 1. Naive Efficacy
$$W_{\text{naive}}' \mathbf{k}_* = W \mathbf{k}_* + \frac{\mathbf{v}_* - W \mathbf{k}_*}{\|\mathbf{k}_*\|_2^2} (\mathbf{k}_*^\top \mathbf{k}_*) = W \mathbf{k}_* + (\mathbf{v}_* - W \mathbf{k}_*) = \mathbf{v}_*$$
The naive update satisfies the target constraint $W' \mathbf{k}_* = \mathbf{v}_*$.

#### 2. Naive Drift on Correlated Keys
For any key $\mathbf{k}_j$:
$$W_{\text{naive}}' \mathbf{k}_j = W \mathbf{k}_j + \frac{\mathbf{v}_* - W \mathbf{k}_*}{\|\mathbf{k}_*\|_2^2} (\mathbf{k}_*^\top \mathbf{k}_j) = W \mathbf{k}_j + \frac{\rho}{\|\mathbf{k}_*\|_2^2} (\mathbf{v}_* - W \mathbf{k}_*)$$
The output drift is:
$$\|W_{\text{naive}}' \mathbf{k}_j - W \mathbf{k}_j\|_2 = \frac{|\rho|}{\|\mathbf{k}_*\|_2^2} \|\mathbf{v}_* - W \mathbf{k}_*\|_2$$
If $\rho > 0$, the memory for subject $j$ suffers severe distortion proportional to the correlation $\rho$, causing **catastrophic forgetting**.

#### 3. ROME's Covariance Shield
In ROME, the update direction is $\mathbf{u} = \mathbf{C}^{-1} \mathbf{k}_*$.
In the basis of principal eigenvectors of $\mathbf{C} = \sum \sigma_i^2 \mathbf{e}_i \mathbf{e}_i^\top$, $\mathbf{C}^{-1}$ scales components along major data axes by $\frac{1}{\sigma_i^2} \ll 1$.
Consequently:
$$\mathbf{k}_j^\top (\mathbf{C}^{-1} \mathbf{k}_*) \approx 0$$
for any key $\mathbf{k}_j$ lying in the high-variance data manifold! The update $\Delta W$ acts strictly along the null or low-variance directions of existing knowledge, leaving all existing factual outputs unchanged.

---

## Exercise 3: Steering Vector Gain $\alpha$ and Phase Transitions

During inference, an activation steering vector $\mathbf{v}_{\text{steer}}$ is injected at layer $l$: $\mathbf{h}_l' = \mathbf{h}_l + \alpha \mathbf{v}_{\text{steer}}$.

1. As $\alpha \to \infty$, what happens to the downstream softmax logits and text generation quality?
2. If layer $l+1$ is followed by a LayerNorm operation:
   $$\text{LN}(\mathbf{h}) = \frac{\mathbf{h} - \mu}{\sigma} \odot \boldsymbol{\gamma} + \boldsymbol{\beta}$$
   Show how large $\alpha$ alters the normalization denominator $\sigma$ and explain the "saturation penalty".
3. What is the recommended empirical range for $\alpha$ to preserve fluent grammar while steering semantic sentiment?

### Solution

#### 1. Degenerate High-Gain Behavior
As $\alpha \to \infty$, the steering vector completely drowns out the prompt tokens. The model's entropy collapses, causing it to emit repetitive token loops, syntax gibberish, or single-token stuttering.

#### 2. LayerNorm Saturation Penalty
The variance of $\mathbf{h}' = \mathbf{h} + \alpha \mathbf{v}$ scales quadratically:
$$\sigma^2(\mathbf{h}') \approx \sigma^2(\mathbf{h}) + \alpha^2 \sigma^2(\mathbf{v})$$
For large $\alpha$, $\sigma(\mathbf{h}') \approx \alpha \sigma(\mathbf{v})$.
The normalized activation becomes:
$$\text{LN}(\mathbf{h}') \approx \frac{\mathbf{h} + \alpha \mathbf{v}}{\alpha \sigma(\mathbf{v})} = \frac{1}{\alpha} \frac{\mathbf{h}}{\sigma(\mathbf{v})} + \frac{\mathbf{v}}{\sigma(\mathbf{v})}$$
Notice that the original prompt information $\mathbf{h}$ is scaled down by $\frac{1}{\alpha} \to 0$! All contextual reasoning, grammar tracking, and instructions are effectively extinguished.

#### 3. Recommended Empirical Range
In practice (e.g., Zou et al., 2023; Turner et al., 2023), $\alpha$ should be set such that:
$$\|\alpha \mathbf{v}_{\text{steer}}\|_2 \in [0.1, 0.5] \cdot \|\mathbf{h}_l\|_2$$
Typically $\alpha \in [1.0, 3.5]$ when $\|\mathbf{v}_{\text{steer}}\|_2 = 1$, producing clean behavioral redirection while maintaining grammatical coherence.
