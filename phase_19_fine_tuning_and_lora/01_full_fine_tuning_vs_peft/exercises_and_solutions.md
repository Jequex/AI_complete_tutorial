# Topic 19.1: Exercises & Solutions — Full Fine-Tuning vs. PEFT & Catastrophic Forgetting

---

### Problem 1: Full Fine-Tuning vs. PEFT VRAM Footprint Calculation (Easy)

An AI engineer plans to adapt a 14-billion parameter model ($N = 14 \times 10^9$) on a medical dataset.
Consider two training configurations using standard mixed precision (FP16 model forward/backward pass, FP32 master weights, and AdamW optimizer):

1. **Configuration A: Full Fine-Tuning (FFT)**, where all $N$ parameters are updated.
2. **Configuration B: Parameter-Efficient Fine-Tuning (PEFT)**, where the base model is frozen in FP16, and an adapter containing $0.2\%$ of the total parameters ($M = 0.002 N$) is trained.

**Tasks:**
1. Calculate the exact static GPU memory footprint (in Gigabytes, $1\text{ GB} = 1024^3\text{ bytes}$) required by Configuration A (weights, gradients, and AdamW optimizer states).
2. Calculate the exact static memory footprint required by Configuration B.
3. Compute the percentage reduction in static memory achieved by PEFT.

<details>
<summary>View Complete Solution</summary>

**Step 1: Configuration A (Full Fine-Tuning)**
The canonical static memory formula for mixed-precision AdamW is:
$$M_{\text{static}} = 16 \text{ bytes per parameter}$$

Breakdown for $N = 14 \times 10^9$:
- Model Weights (FP16): $14 \times 10^9 \times 2 = 28 \times 10^9\text{ bytes}$
- Gradients (FP16): $14 \times 10^9 \times 2 = 28 \times 10^9\text{ bytes}$
- Master Weights (FP32): $14 \times 10^9 \times 4 = 56 \times 10^9\text{ bytes}$
- AdamW First Moment $m_t$ (FP32): $14 \times 10^9 \times 4 = 56 \times 10^9\text{ bytes}$
- AdamW Second Moment $v_t$ (FP32): $14 \times 10^9 \times 4 = 56 \times 10^9\text{ bytes}$

Total bytes:
$$\text{Total Bytes} = 14 \times 10^9 \times 16 = 224 \times 10^9\text{ bytes}$$

Converting to gigabytes ($1\text{ GB} = 1,073,741,824\text{ bytes}$):
$$M_{\text{FFT}} = \frac{224 \times 10^9}{1024^3} \approx \frac{224 \times 10^9}{1,073,741,824} \approx \mathbf{208.62\text{ GB}}$$
*(Requires at least three 80GB H100 GPUs solely for static storage).*

**Step 2: Configuration B (PEFT with 0.2% Parameters)**
- Frozen Base Model: Only the model weights are stored in FP16. Since base weights are frozen, they require **0 bytes** of gradients and **0 bytes** of optimizer states:
  $$M_{\text{base}} = 14 \times 10^9 \times 2\text{ bytes} = 28 \times 10^9\text{ bytes}$$
- Trainable Adapter Parameters: $M = 0.002 \times 14 \times 10^9 = 28 \times 10^6\text{ parameters}$ ($28\text{ million}$).
  The adapter parameters require the full $16\text{ bytes/param}$:
  $$M_{\text{adapter}} = 28 \times 10^6 \times 16\text{ bytes} = 0.448 \times 10^9\text{ bytes}$$

Total static memory:
$$\text{Total Bytes} = 28 \times 10^9 + 0.448 \times 10^9 = 28.448 \times 10^9\text{ bytes}$$

Converting to gigabytes:
$$M_{\text{PEFT}} = \frac{28.448 \times 10^9}{1024^3} \approx \mathbf{26.49\text{ GB}}$$

**Step 3: Memory Reduction Percentage**
$$\text{Reduction} = \frac{208.62 - 26.49}{208.62} \times 100\% = \frac{182.13}{208.62} \times 100\% \approx \mathbf{87.30\%}$$

*(The PEFT model fits comfortably on a single 32GB or 40GB GPU, whereas FFT requires a multi-GPU cluster).*

</details>

---

### Problem 2: First- and Second-Order Gradient Interference Calculation (Medium)

Consider a toy 2-parameter neural network $\theta = [\theta_1, \theta_2]^T$. 
The model has been pre-trained on Task A to near optimality, but retains slight gradient $\mathbf{g}_A = \nabla_\theta \mathcal{L}_A(\theta)$ and local Hessian $\mathbf{H}_A = \nabla^2_\theta \mathcal{L}_A(\theta)$:
$$\mathbf{g}_A = \begin{bmatrix} 1.0 \\ -2.0 \end{bmatrix}, \quad \mathbf{H}_A = \begin{bmatrix} 4.0 & 0.0 \\ 0.0 & 2.0 \end{bmatrix}$$

We now perform one step of unconstrained gradient descent on Task B with learning rate $\eta = 0.1$:
$$\Delta \theta = -\eta \mathbf{g}_B$$

Consider two scenarios for Task B's gradient $\mathbf{g}_B$:
- **Scenario I (Orthogonal Task):** $\mathbf{g}_B^{(I)} = \begin{bmatrix} 2.0 \\ 1.0 \end{bmatrix}$
- **Scenario II (Conflicting / Adversarial Task):** $\mathbf{g}_B^{(II)} = \begin{bmatrix} -3.0 \\ 2.0 \end{bmatrix}$

Using the second-order Taylor approximation:
$$\Delta \mathcal{L}_A \approx -\eta \langle \mathbf{g}_A, \mathbf{g}_B \rangle + \frac{\eta^2}{2} \mathbf{g}_B^T \mathbf{H}_A \mathbf{g}_B$$

Calculate the exact change in Task A loss $\Delta \mathcal{L}_A$ for both scenarios, and explain why Scenario II results in catastrophic forgetting.

<details>
<summary>View Complete Solution</summary>

**Scenario I: Orthogonal Gradient $\mathbf{g}_B^{(I)} = [2.0, 1.0]^T$**
1. First-Order Inner Product:
   $$\langle \mathbf{g}_A, \mathbf{g}_B^{(I)} \rangle = (1.0)(2.0) + (-2.0)(1.0) = 2.0 - 2.0 = 0.0$$
   The first-order linear interference is exactly zero:
   $$-\eta \langle \mathbf{g}_A, \mathbf{g}_B^{(I)} \rangle = -0.1 \times 0.0 = 0.0$$

2. Second-Order Curvature Penalty:
   $$\mathbf{H}_A \mathbf{g}_B^{(I)} = \begin{bmatrix} 4.0 & 0.0 \\ 0.0 & 2.0 \end{bmatrix} \begin{bmatrix} 2.0 \\ 1.0 \end{bmatrix} = \begin{bmatrix} 8.0 \\ 2.0 \end{bmatrix}$$
   $$\mathbf{g}_B^{(I)T} \mathbf{H}_A \mathbf{g}_B^{(I)} = (2.0)(8.0) + (1.0)(2.0) = 16.0 + 2.0 = 18.0$$
   $$\frac{\eta^2}{2} \mathbf{g}_B^{(I)T} \mathbf{H}_A \mathbf{g}_B^{(I)} = \frac{0.01}{2} \times 18.0 = 0.005 \times 18.0 = +0.09$$

3. Net Change in Task A Loss:
   $$\Delta \mathcal{L}_A^{(I)} \approx 0.0 + 0.09 = \mathbf{+0.09}$$
   Task A loss experiences mild drift caused strictly by second-order manifold curvature.

---

**Scenario II: Conflicting Gradient $\mathbf{g}_B^{(II)} = [-3.0, 2.0]^T$**
1. First-Order Inner Product:
   $$\langle \mathbf{g}_A, \mathbf{g}_B^{(II)} \rangle = (1.0)(-3.0) + (-2.0)(2.0) = -3.0 - 4.0 = -7.0$$
   Notice that $\langle \mathbf{g}_A, \mathbf{g}_B \rangle < 0$ (obtuse angle / destructive interference).
   $$-\eta \langle \mathbf{g}_A, \mathbf{g}_B^{(II)} \rangle = -0.1 \times (-7.0) = \mathbf{+0.70}$$
   The first-order term is strictly positive and large!

2. Second-Order Curvature Penalty:
   $$\mathbf{H}_A \mathbf{g}_B^{(II)} = \begin{bmatrix} 4.0 & 0.0 \\ 0.0 & 2.0 \end{bmatrix} \begin{bmatrix} -3.0 \\ 2.0 \end{bmatrix} = \begin{bmatrix} -12.0 \\ 4.0 \end{bmatrix}$$
   $$\mathbf{g}_B^{(II)T} \mathbf{H}_A \mathbf{g}_B^{(II)} = (-3.0)(-12.0) + (2.0)(4.0) = 36.0 + 8.0 = 44.0$$
   $$\frac{\eta^2}{2} \mathbf{g}_B^{(II)T} \mathbf{H}_A \mathbf{g}_B^{(II)} = \frac{0.01}{2} \times 44.0 = 0.005 \times 44.0 = \mathbf{+0.22}$$

3. Net Change in Task A Loss:
   $$\Delta \mathcal{L}_A^{(II)} \approx +0.70 + 0.22 = \mathbf{+0.92}$$

**Key Insight:**
In Scenario II, the negative gradient inner product produces a **$10\times$ larger loss increase** ($+0.92$ vs $+0.09$). Over repeated iterations, this positive feedback loop exponentially degrades Task A performance, which is the foundational mathematical cause of **Catastrophic Forgetting**.

</details>

---

### Problem 3: Bayesian Derivation of Elastic Weight Consolidation (Hard)

To combat catastrophic forgetting in full networks without architectural adapters, Kirkpatrick et al. (2017) proposed **Elastic Weight Consolidation (EWC)**.

Given dataset $\mathcal{D}_A$ for Task A and dataset $\mathcal{D}_B$ for Task B, the log-posterior over model parameters $\theta$ conditioned on both tasks is:
$$\log P(\theta \mid \mathcal{D}_A, \mathcal{D}_B) = \log P(\mathcal{D}_B \mid \theta) + \log P(\theta \mid \mathcal{D}_A) - \log P(\mathcal{D}_B)$$

**Tasks:**
1. Perform a second-order Taylor expansion of the log-posterior $\log P(\theta \mid \mathcal{D}_A)$ around the Task A optimal parameter vector $\theta_A^*$.
2. Prove that under the Laplace approximation, maximizing the joint posterior is equivalent to minimizing the Task B loss with a Fisher-weighted quadratic regularization penalty:
   $$\mathcal{L}_{\text{EWC}}(\theta) = \mathcal{L}_B(\theta) + \sum_{i} \frac{\lambda}{2} F_i (\theta_i - \theta_{A, i}^*)^2$$
3. Explain why calculating the full Fisher Information Matrix is intractable for modern LLMs ($N \ge 7\text{B}$), thereby explaining why PEFT methods like LoRA superseded EWC in production LLM workflows.

<details>
<summary>View Complete Solution</summary>

**Step 1: Taylor Expansion of Task A Log-Posterior**
Let $f(\theta) = \log P(\theta \mid \mathcal{D}_A)$. Expanding $f(\theta)$ in a Taylor series around the mode $\theta_A^*$:
$$f(\theta) \approx f(\theta_A^*) + \nabla_\theta f(\theta_A^*)^T (\theta - \theta_A^*) + \frac{1}{2} (\theta - \theta_A^*)^T \nabla^2_\theta f(\theta_A^*) (\theta - \theta_A^*)$$

At the maximum a posteriori (MAP) point $\theta_A^*$, the gradient vanishes:
$$\nabla_\theta f(\theta_A^*) = \mathbf{0}$$

Therefore:
$$\log P(\theta \mid \mathcal{D}_A) \approx \log P(\theta_A^* \mid \mathcal{D}_A) - \frac{1}{2} (\theta - \theta_A^*)^T \left(-\nabla^2_\theta \log P(\theta \mid \mathcal{D}_A)\Big|_{\theta_A^*}\right) (\theta - \theta_A^*)$$

**Step 2: Connecting the Hessian to the Fisher Information Matrix**
The negative expected Hessian of the log-likelihood is asymptotically equal to the **Fisher Information Matrix** $F$:
$$\mathbb{E}\left[ -\nabla^2_\theta \log P(\mathcal{D}_A \mid \theta) \right] = F = \mathbb{E}_{x \sim \mathcal{D}_A}\left[ \nabla_\theta \log P(x \mid \theta) \nabla_\theta \log P(x \mid \theta)^T \right]$$

Assuming a flat prior on $\theta$, the negative Hessian of the posterior equals the Fisher Matrix $F$. Approximating $F$ by its diagonal elements $F_{ii}$ (diagonal Fisher approximation):
$$\log P(\theta \mid \mathcal{D}_A) \approx \text{const} - \frac{1}{2} \sum_i F_{ii} (\theta_i - \theta_{A, i}^*)^2$$

Substituting this into the joint posterior:
$$\log P(\theta \mid \mathcal{D}_A, \mathcal{D}_B) \propto -\mathcal{L}_B(\theta) - \frac{\lambda}{2} \sum_i F_i (\theta_i - \theta_{A, i}^*)^2$$

Negating to convert log-likelihood maximization into loss minimization gives the canonical EWC objective:
$$\boxed{\mathcal{L}_{\text{EWC}}(\theta) = \mathcal{L}_B(\theta) + \sum_{i} \frac{\lambda}{2} F_i (\theta_i - \theta_{A, i}^*)^2}$$

**Step 3: Why PEFT/LoRA Superseded EWC in Modern LLMs**
1. **Computational & Memory Storage Limits:** Even storing just the *diagonal* of the Fisher matrix requires an extra $4N$ bytes in FP32. For an 8B model, that is an additional $32\text{ GB}$ of static memory, worsening the already severe Memory Wall. Storing the full non-diagonal Fisher matrix ($N \times N = 64 \times 10^{18}$ entries) is physically impossible.
2. **Optimizer Overhead:** Computing the EWC penalty requires calculating gradients with respect to every single parameter during Task B training, meaning all $N$ parameters still require FP16 gradients and FP32 AdamW optimizer moments ($16N$ bytes total).
3. **PEFT Elegance:** Instead of soft quadratic springs that fight against optimization, PEFT mathematically freezes $\theta_0$ ($0\text{ memory for gradients/optimizer}$) and restricts adaptation strictly to a low-dimensional subspace ($r \ll d$).

</details>
