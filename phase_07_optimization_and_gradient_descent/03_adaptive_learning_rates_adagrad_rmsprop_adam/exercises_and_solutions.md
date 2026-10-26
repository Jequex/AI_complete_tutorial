# Topic 7.3: Exercises & Solutions — Adaptive Learning Rates

---

### Problem 1: Manual Step-by-Step Adam Hand Calculation (Medium)

Consider optimizing a 1D parameter $\theta$ initialized at $\theta_0 = 1.0$.
Hyperparameters:
- Learning rate $\eta = 0.01$
- Decay rates $\beta_1 = 0.9, \beta_2 = 0.99$
- Epsilon $\epsilon = 10^{-6}$
- Initial moments $m_0 = 0, v_0 = 0$

Suppose the optimizer receives gradients:
- Step 1 ($t=1$): $g_1 = 0.40$
- Step 2 ($t=2$): $g_2 = 0.20$

Calculate exact numerical values for:
1. $m_1, v_1, \hat{m}_1, \hat{v}_1$, and the updated parameter $\theta_1$.
2. $m_2, v_2, \hat{m}_2, \hat{v}_2$, and the updated parameter $\theta_2$.
3. What would the update $\Delta \theta_1$ have been **without** bias correction? What percentage error would that introduce?

#### Solution
1. **Step 1 ($t = 1$):**
   - Biased moments:
     $$m_1 = \beta_1 m_0 + (1 - \beta_1) g_1 = 0.9(0) + 0.1(0.40) = 0.040$$
     $$v_1 = \beta_2 v_0 + (1 - \beta_2) g_1^2 = 0.99(0) + 0.01(0.40^2) = 0.01(0.16) = 0.00160$$
   - Bias correction factors ($1 - \beta_1^1 = 0.10$, $1 - \beta_2^1 = 0.01$):
     $$\hat{m}_1 = \frac{m_1}{1 - \beta_1^1} = \frac{0.040}{0.10} = 0.400$$
     $$\hat{v}_1 = \frac{v_1}{1 - \beta_2^1} = \frac{0.00160}{0.01} = 0.160$$
   - Parameter update:
     $$\Delta \theta_1 = -\frac{\eta}{\sqrt{\hat{v}_1} + \epsilon} \hat{m}_1 = -\frac{0.01}{\sqrt{0.160} + 10^{-6}} (0.400) = -\frac{0.01}{0.400 + 10^{-6}} (0.400) \approx -0.010$$
     $$\theta_1 = \theta_0 + \Delta \theta_1 = 1.0 - 0.010 = 0.990$$

2. **Step 2 ($t = 2$):**
   - Biased moments:
     $$m_2 = \beta_1 m_1 + (1 - \beta_1) g_2 = 0.9(0.040) + 0.1(0.20) = 0.036 + 0.020 = 0.056$$
     $$v_2 = \beta_2 v_1 + (1 - \beta_2) g_2^2 = 0.99(0.0016) + 0.01(0.04) = 0.001584 + 0.00040 = 0.001984$$
   - Bias correction factors ($1 - \beta_1^2 = 1 - 0.81 = 0.19$, $1 - \beta_2^2 = 1 - 0.9801 = 0.0199$):
     $$\hat{m}_2 = \frac{0.056}{0.19} \approx 0.29474$$
     $$\hat{v}_2 = \frac{0.001984}{0.0199} \approx 0.099698 \implies \sqrt{\hat{v}_2} \approx 0.31575$$
   - Parameter update:
     $$\Delta \theta_2 = -\frac{0.01}{0.31575} (0.29474) \approx -0.009335$$
     $$\theta_2 = \theta_1 + \Delta \theta_2 = 0.990 - 0.009335 = 0.980665$$

3. **Impact of Omitting Bias Correction at Step 1:**
   Without bias correction:
   $$\Delta \theta_1^{\text{biased}} = -\frac{\eta}{\sqrt{v_1} + \epsilon} m_1 = -\frac{0.01}{\sqrt{0.00160}} (0.040) = -\frac{0.01}{0.040} (0.040) = -0.010$$
   While in this 1D case $\frac{m_1}{\sqrt{v_1}} = \frac{(1-\beta_1)g}{\sqrt{(1-\beta_2)g^2}} = \frac{0.1}{0.1} = 1$, whenever $\beta_1 \ne \sqrt{\beta_2}$ (such as standard $\beta_1=0.9, \beta_2=0.999$ where $\sqrt{1-\beta_2} = \sqrt{0.001} \approx 0.0316 \ne 0.1$), the uncorrected step would be distorted by $\frac{0.1}{0.0316} \approx 3.16\times$ (a **316% overshoot**!), demonstrating why bias correction is mandatory.

---

### Problem 2: AdaGrad Monotonic Freezing Proof (Easy)

In AdaGrad, the update for coordinate $j$ is:
$$\theta_{t+1, j} = \theta_{t, j} - \frac{\eta}{\sqrt{G_{t, j}} + \epsilon} g_{t, j}, \quad \text{where } G_{t, j} = \sum_{k=1}^t g_{k, j}^2$$

1. Suppose a weight receives a steady, non-zero gradient $g_{k, j} = c > 0$ at every step. Express $G_{t, j}$ in terms of $t$ and $c$.
2. Compute the step size $\Delta \theta_{t, j} = |\theta_{t+1, j} - \theta_{t, j}|$ as a function of $t$.
3. Show that the total distance traversed by this parameter from step 1 to step $T$ scales as $\mathcal{O}(\sqrt{T})$, whereas standard SGD traverses $\mathcal{O}(T)$.
4. Explain why this causes AdaGrad to freeze prematurely in deep networks.

#### Solution
1. **Accumulator $G_{t, j}$:**
   $$G_{t, j} = \sum_{k=1}^t c^2 = t \cdot c^2$$

2. **Step Size Decay:**
   Ignoring $\epsilon \ll \sqrt{G_t}$:
   $$\Delta \theta_{t, j} = \frac{\eta}{\sqrt{t \cdot c^2}} c = \frac{\eta}{\sqrt{t}}$$
   The step size decays at the aggressive rate of $\mathcal{O}(1/\sqrt{t})$.

3. **Cumulative Distance Traversed:**
   The total distance traveled over $T$ steps is:
   $$D_{\text{AdaGrad}}(T) = \sum_{t=1}^T \frac{\eta}{\sqrt{t}} \approx \eta \int_1^T t^{-1/2} dt = 2\eta (\sqrt{T} - 1) = \mathcal{O}(\sqrt{T})$$
   In contrast, for standard SGD with constant learning rate $\eta$:
   $$D_{\text{SGD}}(T) = \sum_{t=1}^T \eta c = \eta c T = \mathcal{O}(T)$$

4. **Premature Freezing:**
   Because $\frac{1}{\sqrt{t}} \to 0$, by epoch 100,000 the effective step size is hundreds of times smaller than at initialization. If the model enters a new, complex loss landscape later in training, AdaGrad's accumulator cannot forget its past history, making it completely unable to adapt or cross the remaining barrier.

---

### Problem 3: AdamW vs $L_2$ Regularization Mathematical Proof (Hard)

Let $\mathcal{L}(\theta)$ be the unregularized empirical loss.

1. Show that minimizing $\mathcal{L}_{\text{reg}}(\theta) = \mathcal{L}(\theta) + \frac{1}{2} \lambda \|\theta\|_2^2$ with Adam produces an effective weight update:
   $$\theta_{t+1} \approx \theta_t - \frac{\eta \lambda}{\sqrt{\hat{v}_t} + \epsilon} \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$
2. Contrast this with AdamW's decoupled update:
   $$\theta_{t+1} = (1 - \eta \lambda) \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$
3. Consider two weights in a Transformer:
   - Weight $A$ (frequently used token): average gradient magnitude is large ($\hat{v}_A = 100$).
   - Weight $B$ (rare token): average gradient magnitude is small ($\hat{v}_B = 0.01$).
   Compute the effective shrinkage coefficient on Weight $A$ vs Weight $B$ under standard Adam with $L_2$ decay vs AdamW (assume $\eta = 0.001, \lambda = 0.1, \epsilon=0$).

#### Solution
1. **$L_2$ Coupled Gradient in Adam:**
   With $L_2$ regularization, the objective is $\mathcal{L}_{\text{reg}}(\theta) = \mathcal{L}(\theta) + \frac{\lambda}{2} \|\theta\|^2$.
   The gradient is $\nabla \mathcal{L}_{\text{reg}}(\theta) = \nabla \mathcal{L}(\theta) + \lambda \theta$.
   Because Adam feeds this total gradient into its moment buffers:
   $$\hat{m}_t \approx \hat{m}_t^{\text{loss}} + \lambda \theta_t$$
   The parameter update is:
   $$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \left( \hat{m}_t^{\text{loss}} + \lambda \theta_t \right) = \theta_t - \underbrace{\frac{\eta \lambda}{\sqrt{\hat{v}_t} + \epsilon} \theta_t}_{\text{Effective Regularization}} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t^{\text{loss}}$$

2. **Decoupled Update in AdamW:**
   AdamW computes moment statistics $\hat{m}_t, \hat{v}_t$ exclusively on the data loss gradient $\nabla \mathcal{L}(\theta)$. The weight decay is applied directly as a linear shrinkage:
   $$\theta_{t+1} = \theta_t - \eta \lambda \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t = (1 - \eta \lambda) \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

3. **Numerical Comparison on Weights A and B:**
   We are given $\eta = 0.001, \lambda = 0.1 \implies \eta \lambda = 10^{-4}$.
   - **Under AdamW (Decoupled):**
     $$\text{Shrinkage on } A = \eta \lambda = 10^{-4}$$
     $$\text{Shrinkage on } B = \eta \lambda = 10^{-4}$$
     Both weights experience exact, uniform regularization of $0.01\%$ per step.
   - **Under Standard Adam with $L_2$ Penalty:**
     $$\text{Shrinkage on } A = \frac{\eta \lambda}{\sqrt{\hat{v}_A}} = \frac{10^{-4}}{\sqrt{100}} = \frac{10^{-4}}{10} = 10^{-5}$$
     $$\text{Shrinkage on } B = \frac{\eta \lambda}{\sqrt{\hat{v}_B}} = \frac{10^{-4}}{\sqrt{0.01}} = \frac{10^{-4}}{0.1} = 10^{-3}$$
     **Catastrophic Distortion:**
     $$\frac{\text{Shrinkage on } B}{\text{Shrinkage on } A} = \frac{10^{-3}}{10^{-5}} = 100\times$$
     Weight $B$ (the rare token, which already has low data evidence) is shrunk **100 times harder** than Weight $A$! This causes rare tokens to be obliterated from the model representation, which is why AdamW is strictly required for modern NLP models.

---

### Problem 4: Scale Invariance of Adam Step Sizes (Medium)

Suppose the loss function is scaled by a positive constant $c > 0$: $\tilde{\mathcal{L}}(\theta) = c \cdot \mathcal{L}(\theta)$.
Show that the Adam parameter update $\Delta \theta$ is invariant to $c$ (assuming $\epsilon \to 0$ and moments are steady).

#### Solution
If $\tilde{\mathcal{L}}(\theta) = c \mathcal{L}(\theta)$, then the gradient is scaled by $c$:
$$\tilde{g}_t = c \cdot g_t$$
Since first moment $m_t$ is a linear combination of gradients:
$$\tilde{m}_t = c \cdot m_t \implies \hat{\tilde{m}}_t = c \cdot \hat{m}_t$$
Since second moment $v_t$ is a linear combination of squared gradients:
$$\tilde{v}_t = c^2 \cdot v_t \implies \hat{\tilde{v}}_t = c^2 \cdot \hat{v}_t \implies \sqrt{\hat{\tilde{v}}_t} = c \sqrt{\hat{v}_t}$$
Substituting into the update step (with $\epsilon \approx 0$):
$$\Delta \tilde{\theta}_t = -\frac{\eta}{\sqrt{\hat{\tilde{v}}_t}} \hat{\tilde{m}}_t = -\frac{\eta}{c \sqrt{\hat{v}_t}} (c \hat{m}_t) = -\frac{\eta}{\sqrt{\hat{v}_t}} \hat{m}_t = \Delta \theta_t$$
The scaling constant $c$ cancels out completely in the numerator and denominator!
This proves that Adam's step size is automatically invariant to arbitrary constant scalings of the loss function, explaining why Adam is so robust across wildly diverse loss functions.
