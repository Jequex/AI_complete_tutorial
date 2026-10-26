# Topic 7.3: Adaptive Learning Rates: AdaGrad, RMSProp, Adam & AdamW

---

## 1. Executive Summary & The 12-Year-Old Analogy

In deep neural networks with millions or billions of parameters, not all weights are created equal. Word embedding tokens that appear rarely (e.g., "platypus") receive gradients once every 10,000 steps, while common tokens ("the", "is") receive gradients at every single step.

If we apply a single global learning rate $\eta$:
- Rare features never update enough to learn anything meaningful.
- Common features receive giant gradient bursts that cause them to oscillate or diverge.

**Adaptive learning rate optimizers** solve this by giving **each individual parameter its own customized, dynamically adjusted step size**.

```
Global Learning Rate (SGD):
  All weights get same step size η regardless of gradient scale!
  w_rare  ──> [tiny step]
  w_dense ──> [HUGE STEP (Overshoots!)]

Adaptive Learning Rate (Adam / AdamW):
  Step size scaled inversely by historical gradient magnitude!
  w_rare  ──> [scaled UP to learn effectively]
  w_dense ──> [scaled DOWN to remain stable]
```

---

### The 12-Year-Old Analogy: The Smart All-Terrain Hiking Shoes

Imagine hiking across an island where the terrain alternates between sharp jagged rocks, sticky mud, and smooth sand:

1. **AdaGrad (The Hardening Shoes):**
   Your shoes harden every time they hit the ground. On rocky ground they get stiff to protect your feet. But there is a catch: **they never soften up again**. After 10 minutes, the shoes turn into 50-pound solid cement blocks, and you can no longer take a single step (the step size shrinks to zero and optimization freezes!).
2. **RMSProp (The Smart Memory-Foam Shoes):**
   Geoffrey Hinton invented shoes with dynamic memory foam. They stiffen up on rocks and soften up on sand, but they only remember the last 5 minutes of your walk. They never permanently turn into cement!
3. **Adam (The Rocket-Powered Smart Shoes):**
   Diederik Kingma and Jimmy Ba combined the memory foam shoes (RMSProp) with rocket boosters (Polyak momentum). Plus, they added an auto-pilot "soft starter" so the rockets don't jerk you off your feet on the very first step (bias correction).
4. **AdamW (The Clean Backpack):**
   Loshchilov and Hutter noticed a major bug in how people cleaned their backpacks in Adam: if your shoes were moving slowly, your backpack cleaner threw out way too much stuff, and if your shoes were moving fast, it threw out almost nothing! AdamW completely disconnects weight shrinkage from the shoes, ensuring your backpack stays light and balanced everywhere.

---

## 2. Mathematical Symbol Decoder

| Symbol | Mathematical Domain | Description & Meaning |
| :--- | :--- | :--- |
| $\theta_t \in \mathbb{R}^d$ | Vector space | Model parameter weights at step $t$ |
| $g_t = \nabla \mathcal{L}(\theta_t)$ | Vector space | Current mini-batch gradient vector |
| $G_t \in \mathbb{R}^d$ | Vector space | Cumulative sum of squared gradients (AdaGrad) |
| $m_t \in \mathbb{R}^d$ | Vector space | First raw moment vector (exponentially decaying average of $g_t$) |
| $v_t \in \mathbb{R}^d$ | Vector space | Second raw moment vector (exponentially decaying average of $g_t^2$) |
| $\beta_1 \in [0, 1)$ | Scalar | Exponential decay rate for first moment (standard: $0.9$) |
| $\beta_2 \in [0, 1)$ | Scalar | Exponential decay rate for second moment (standard: $0.999$) |
| $\hat{m}_t, \hat{v}_t$ | Vector space | Bias-corrected first and second moment estimators |
| $\epsilon > 0$ | Small scalar | Numerical stabilization factor (standard: $10^{-8}$) |
| $\lambda \ge 0$ | Scalar | Decoupled weight decay coefficient in AdamW |

---

## 3. The Evolution: AdaGrad to RMSProp

### 3.1 AdaGrad (Duchi et al., 2011)
AdaGrad scales coordinate updates inversely by the square root of all historical squared gradients:
$$G_t = G_{t-1} + g_t \odot g_t$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t} + \epsilon} \odot g_t$$
where $\odot$ represents element-wise (Hadamard) multiplication.

- **Advantage:** Coordinates with infrequent or small gradients receive large updates; coordinates with frequent, high-magnitude gradients receive small, tempered updates.
- **The Fatal Flaw:** Because $g_t^2 \ge 0$, the accumulator $G_t$ is **strictly monotonically increasing**:
  $$G_t = \sum_{k=1}^t g_k^2 \implies \lim_{t \to \infty} G_t = \infty$$
  Therefore, the effective learning rate $\frac{\eta}{\sqrt{G_t} + \epsilon} \to 0$. In deep learning, AdaGrad's learning rate decays to near-zero long before the model has traversed complex non-convex terrain.

### 3.2 RMSProp (Hinton, 2012)
Geoffrey Hinton replaced AdaGrad's infinite historical sum with an **Exponential Moving Average (EMA)** of squared gradients:
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t \odot g_t$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t} + \epsilon} \odot g_t$$

- Typically, $\beta_2 = 0.99$.
- The effective memory horizon is $\frac{1}{1 - \beta_2} \approx 100$ steps.
- Because older gradients decay exponentially, $v_t$ reflects only the **recent local curvature**, completely eliminating AdaGrad's premature freezing problem!

---

## 4. The Adam Optimizer (Kingma & Ba, 2014)

Adam (**Ada**ptive **M**oment Estimation) combines the velocity accumulation of Momentum (first moment $m_t$) with the adaptive per-coordinate scaling of RMSProp (second raw moment $v_t$).

### 4.1 The Raw Updates
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

### 4.2 Mathematical Derivation of Bias Corrections

When initialized at $m_0 = 0$ and $v_0 = 0$, both $m_t$ and $v_t$ are heavily biased toward zero in the early iterations (especially since $\beta_2 = 0.999$, meaning $(1-\beta_2) = 0.001$).

Let us derive the exact bias correction factor for $m_t$.
Unrolling the recurrence:
$$m_t = (1 - \beta_1) g_t + \beta_1 m_{t-1}$$
$$m_t = (1 - \beta_1) g_t + \beta_1 (1 - \beta_1) g_{t-1} + \beta_1^2 m_{t-2}$$
$$m_t = (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i} g_i$$

Taking mathematical expectations on both sides, assuming the true underlying gradient expectation $\mathbb{E}[g_i] \approx \mathbb{E}[g_t]$ has stabilized locally:
$$\mathbb{E}[m_t] = \mathbb{E}\left[ (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i} g_i \right]$$
$$\mathbb{E}[m_t] = \mathbb{E}[g_t] (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i}$$

Let $k = t - i$. As $i$ ranges from $1$ to $t$, $k$ ranges from $0$ to $t-1$:
$$\sum_{i=1}^t \beta_1^{t-i} = \sum_{k=0}^{t-1} \beta_1^k = \frac{1 - \beta_1^t}{1 - \beta_1}$$

Substituting this finite geometric sum back:
$$\mathbb{E}[m_t] = \mathbb{E}[g_t] (1 - \beta_1) \left( \frac{1 - \beta_1^t}{1 - \beta_1} \right) = \mathbb{E}[g_t] (1 - \beta_1^t)$$

Therefore, $m_t$ systematically underestimates $\mathbb{E}[g_t]$ by a factor of $(1 - \beta_1^t)$. To obtain an **unbiased estimator**, we divide by $(1 - \beta_1^t)$:
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t} \implies \mathbb{E}[\hat{m}_t] = \mathbb{E}[g_t]$$

By an identical algebraic derivation for the second raw moment $v_t$:
$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t} \implies \mathbb{E}[\hat{v}_t] = \mathbb{E}[g_t^2]$$

$\blacksquare$

### 4.3 Full Adam Algorithm
1. Compute gradient: $g_t = \nabla \mathcal{L}(\theta_t)$
2. Update biased first moment: $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$
3. Update biased second raw moment: $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$
4. Compute bias-corrected moments:
   $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
5. Apply update:
   $$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

---

## 5. AdamW: Decoupled Weight Decay (Loshchilov & Hutter, 2017)

In classical SGD, adding an $L_2$ penalty to the loss function is mathematically identical to applying weight decay:
$$\mathcal{L}_{\text{reg}}(\theta) = \mathcal{L}(\theta) + \frac{1}{2} \lambda \|\theta\|^2 \implies \nabla \mathcal{L}_{\text{reg}}(\theta) = \nabla \mathcal{L}(\theta) + \lambda \theta$$
$$\theta_{t+1} = \theta_t - \eta (\nabla \mathcal{L}(\theta_t) + \lambda \theta_t) = (1 - \eta \lambda)\theta_t - \eta \nabla \mathcal{L}(\theta_t)$$

However, in Adam, **$L_2$ regularization is NOT equivalent to weight decay**, leading to a severe flaw.

### 5.1 The Flaw of $L_2$ Regularization in Adam
When we add $L_2$ regularization to the loss in standard Adam, the gradient passed to Adam is:
$$g_t^{\text{reg}} = g_t + \lambda \theta_t$$
This modified gradient enters both the first moment $m_t$ and the second moment $v_t$.
The resulting update step becomes:
$$\theta_{t+1} \approx \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \left( \hat{m}_t + \lambda \theta_t \right)$$

Notice what happens to the weight shrinkage term:
$$\text{Effective Weight Decay} = \frac{\eta \lambda}{\sqrt{\hat{v}_t} + \epsilon} \theta_t$$

- **For weights with large gradients ($v_t \gg 1$):** The weight decay rate is divided by a huge number, meaning their weights are barely penalized at all!
- **For weights with small or sparse gradients ($v_t \ll 1$):** The weight decay rate is divided by a tiny number, meaning their weights are heavily penalized and shrunk toward zero!
- This is the exact **opposite** of what regularization should do.

### 5.2 Decoupled Weight Decay (AdamW)
Loshchilov & Hutter proposed **decoupling** the weight decay completely from the gradient moment updates:
$$\theta_{t+1} = \theta_t - \eta \lambda \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

```
Standard Adam with L2:
  g_t = ∇L(θ) + λ θ   <-- Injected into gradient moments (Coupled!)
  θ_{t+1} = θ_t - η * (m_t / √v_t)

AdamW (Decoupled Weight Decay):
  g_t = ∇L(θ)         <-- Pure data gradient in moments!
  θ_{t+1} = (1 - η λ) θ_t - η * (m_t / √v_t)  <-- Uniform shrinkage!
```

- Every single weight decays at the uniform rate $(1 - \eta \lambda)$, regardless of historical gradient scale.
- AdamW drastically improves generalization and is the **universal default optimizer for all modern Large Language Models (LLMs)** including GPT-4, LLaMA, Mistral, and Claude.

---

## 6. Summary Comparison: Adaptive Optimizers

| Property | AdaGrad | RMSProp | Adam | AdamW |
| :--- | :--- | :--- | :--- | :--- |
| **First Moment (Momentum)** | No | No | Yes ($\beta_1=0.9$) | Yes ($\beta_1=0.9$) |
| **Second Moment (Adaptive)** | Cumulative $\sum g^2$ | EMA ($(1-\beta_2)g^2$) | EMA ($(1-\beta_2)g^2$) | EMA ($(1-\beta_2)g^2$) |
| **Bias Correction** | None | None | Yes ($1-\beta_1^t, 1-\beta_2^t$) | Yes ($1-\beta_1^t, 1-\beta_2^t$) |
| **Weight Decay Treatment** | Coupled ($L_2$) | Coupled ($L_2$) | Coupled ($L_2$) | **Decoupled ($(1-\eta\lambda)\theta$)** |
| **Premature Freezing** | High risk | Solved | Solved | Solved |
| **Modern AI Usage** | Rare (Sparse NLP only) | Reinforcement Learning | Common | **Universal Standard for Transformers** |
