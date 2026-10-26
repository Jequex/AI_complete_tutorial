# Topic 13.3: Weight Initialization Schemes & Deep Variance Dynamics

---

## 1. Executive Summary & The 12-Year-Old Analogy

Before a neural network ever sees a single training sample or computes its first gradient, its parameters must be initialized. In shallow networks (1 to 3 layers), arbitrary random weights often suffice. However, in deep architectures (20, 50, or 100+ layers), **improper initialization guarantees catastrophic failure before training even starts**.

If initial weights are chosen slightly too large, activation variance multiplies across successive layers, triggering exponential **numerical explosion** ($\text{NaN} / \infty$). If weights are chosen slightly too small, activations shrink exponentially with each matrix multiplication, decaying to $0.0000$ and inducing **vanishing gradients** where all learning freezes.

Modern deep learning solved this through rigorous statistical analysis of variance propagation:
- **Xavier / Glorot Initialization (2010):** Engineered for symmetric, zero-centered activations (Linear, Tanh).
- **He / Kaiming Initialization (2015):** Engineered for Rectified Linear Units (ReLU/GELU), accounting for the fact that ReLU eliminates half of the probability distribution mass.

```
                    ┌────────────────────────────────────────────────────────┐
                    │      THE 50-LAYER ACTIVATION VARIANCE GAUNTLET         │
                    └────────────────────────────────────────────────────────┘
                    
       Layer 0 (Input):  Var(x) = 1.0000
       
       [ Naive Small (std=0.05) ] ──► Layer 10: 10^-8  ──► Layer 50: 10^-41 (Dead Signal)
       [ Naive Large (std=1.00) ] ──► Layer 10: 10^12  ──► Layer 50: Overflow / NaN
       [ Xavier (Glorot) + ReLU ] ──► Layer 10: 10^-3  ──► Layer 50: 10^-16 (0.5^L Decay)
       [ He / Kaiming (2/n_in)  ] ──► Layer 10: 0.67   ──► Layer 50: 1.10   (STABLE!)
```

---

### The 12-Year-Old Analogy: Tuning the Microphone Volume Before the Concert

Imagine a giant rock concert with 50 amplifiers connected in a continuous chain:
- **The Whisperer (Weights Too Small):**
  If each amplifier turns down the sound by just 10% ($0.9\times$), then by the 50th amplifier, $(0.9)^{50} \approx 0.005$. The singer's voice is completely silenced. Nobody in the stadium hears anything, and the sound crew has zero signal to tune.
- **The Ear-Splitting Screech (Weights Too Large):**
  If each amplifier turns up the volume by 10% ($1.1\times$), then by the 50th amplifier, $(1.1)^{50} \approx 117.4\times$. The sound explodes into ear-splitting feedback, blows out the speakers, and trips the stadium circuit breakers ($\text{NaN}$ overflow).
- **The Goldilocks Sound Engineer (Xavier & He Initialization):**
  A master sound engineer knows that each amplifier has a specific electronic resistance. To keep the singer's volume at exactly $1.0\times$ from microphone to the 50th speaker, every amplifier must have its volume dial calibrated mathematically to $1 / \sqrt{\text{wires}}$.
  Furthermore, if an effect pedal chops off the bottom half of the sound wave (like a ReLU pedal!), the volume drops by half. To fix that, the engineer dials the volume up by $\sqrt{2}$ to compensate!

---

## 2. Mathematical Symbol Decoder

| Symbol | Mathematical Domain | Description & Engineering Interpretation |
| :--- | :--- | :--- |
| $n_{\text{in}}$ | $\mathbb{N}$ | Fan-in: number of incoming input features/connections to a neuron. |
| $n_{\text{out}}$ | $\mathbb{N}$ | Fan-out: number of outgoing output features/connections from a neuron. |
| $\text{Var}(x)$ | $\mathbb{R}_{\ge 0}$ | Variance of the activation distribution across neurons in a layer. |
| $\text{Var}(w) = \sigma_w^2$ | $\mathbb{R}_{\ge 0}$ | Variance of the initialized parameter distribution. |
| $y = \sum_{i=1}^{n_{\text{in}}} w_i x_i$ | $\mathbb{R}$ | Linear pre-activation before non-linear thresholding. |
| $\sigma_{\text{Glorot}}^2$ | $\mathbb{R}_{> 0}$ | Xavier variance: $\frac{2}{n_{\text{in}} + n_{\text{out}}}$ harmonizing forward and backward passes. |
| $\sigma_{\text{He}}^2$ | $\mathbb{R}_{> 0}$ | He / Kaiming variance: $\frac{2}{n_{\text{in}}}$ compensating for half-rectified distributions. |

---

## 3. Mathematical Proof: Variance Propagation in Linear Layers

Consider a single linear neuron computing:
$$y = \sum_{i=1}^{n_{\text{in}}} w_i x_i$$
Assume:
1. $w_i$ and $x_i$ are independent, identically distributed (i.i.d.) random variables.
2. Weights are zero-mean: $\mathbb{E}[w_i] = 0$.
3. Inputs are zero-mean: $\mathbb{E}[x_i] = 0$.

For independent zero-mean variables:
$$\text{Var}(w_i x_i) = \mathbb{E}[w_i^2 x_i^2] - (\mathbb{E}[w_i x_i])^2 = \mathbb{E}[w_i^2] \mathbb{E}[x_i^2] - 0 = \text{Var}(w_i) \text{Var}(x_i)$$

Summing across all $n_{\text{in}}$ independent input features:
$$\text{Var}(y) = \sum_{i=1}^{n_{\text{in}}} \text{Var}(w_i x_i) = n_{\text{in}} \text{Var}(w) \text{Var}(x)$$

$$\text{Var}(y) = n_{\text{in}} \sigma_w^2 \sigma_x^2$$

### The Condition for Stability
To prevent the variance of activations from expanding or collapsing from layer to layer, we demand:
$$\text{Var}(y) = \text{Var}(x)$$
$$n_{\text{in}} \sigma_w^2 = 1 \implies \sigma_w^2 = \frac{1}{n_{\text{in}}}$$

---

## 4. Xavier / Glorot Initialization (2010)

Xavier Glorot and Yoshua Bengio analyzed both the forward pass and the backward gradient pass.

### 4.1 Forward vs Backward Harmony
- **Forward Pass:** Preserving activation variance requires:
  $$\sigma_w^2 = \frac{1}{n_{\text{in}}}$$
- **Backward Pass:** In reverse-mode autograd, the gradient $\Delta x = \Delta y W^T$ flows backward across $n_{\text{out}}$ connections. Preserving gradient variance requires:
  $$\sigma_w^2 = \frac{1}{n_{\text{out}}}$$

When $n_{\text{in}} \ne n_{\text{out}}$, both conditions cannot be satisfied simultaneously. Glorot and Bengio took the **harmonic mean** of the two conditions:
$$\sigma_w^2 = \frac{2}{n_{\text{in}} + n_{\text{out}}}$$

### 4.2 Standard Distributions
- **Normal Xavier:**
  $$W \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}} + n_{\text{out}}}\right)$$
- **Uniform Xavier:** Since a uniform distribution $\mathcal{U}(-a, a)$ has variance $\text{Var} = \frac{a^2}{3}$:
  $$a = \sqrt{3 \sigma^2} = \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}} \implies W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}, \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}\right)$$

---

## 5. He / Kaiming Initialization (2015)

While Xavier initialization works well for symmetric activations like Tanh ($\tanh(0) = 0$), it **fails catastrophically for ReLU**:
$$\text{ReLU}(z) = \max(0, z)$$

### 5.1 The Factor-of-Two Penalty
If pre-activation $z \sim \mathcal{N}(0, \sigma_z^2)$ is symmetric around zero:
- $\text{ReLU}(z) = z$ for $z > 0$ (probability $50\%$)
- $\text{ReLU}(z) = 0$ for $z \le 0$ (probability $50\%$)

Evaluating the second moment:
$$\mathbb{E}[\text{ReLU}(z)^2] = \int_{-\infty}^\infty \max(0, z)^2 p(z) \, dz = \int_0^\infty z^2 p(z) \, dz = \frac{1}{2} \mathbb{E}[z^2] = \frac{1}{2} \text{Var}(z)$$

Therefore, every ReLU layer **halves the variance** of the forward signal!
$$\text{Var}(x_l) = \frac{1}{2} n_{\text{in}} \sigma_w^2 \text{Var}(x_{l-1})$$

If Xavier initialization is used ($\sigma_w^2 = \frac{1}{n_{\text{in}}}$):
$$\text{Var}(x_l) = \frac{1}{2} \text{Var}(x_{l-1})$$
Across $L=50$ layers, the signal shrinks by:
$$\left(\frac{1}{2}\right)^{50} \approx 8.88 \times 10^{-16}$$
The network's activations vanish completely into zero!

### 5.2 The Kaiming Formula
To cancel the $\frac{1}{2}$ factor, Kaiming He et al. set:
$$\frac{1}{2} n_{\text{in}} \sigma_w^2 = 1 \implies \sigma_w^2 = \frac{2}{n_{\text{in}}}$$

- **Kaiming Normal (`fan_in`):**
  $$W \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}}}\right)$$
- **Kaiming Uniform (`fan_in`):**
  $$W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{\text{in}}}}, \sqrt{\frac{6}{n_{\text{in}}}}\right)$$

---

## 6. Initialization Cheat Sheet for Practitioners

| Activation Function | Recommended Initialization Scheme | PyTorch Function |
| :--- | :--- | :--- |
| **Linear / Identity** | Xavier Normal / Uniform | `torch.nn.init.xavier_normal_` |
| **Tanh / Sigmoid** | Xavier Normal / Uniform | `torch.nn.init.xavier_normal_` |
| **ReLU / LeakyReLU** | He / Kaiming Normal (`mode='fan_in'`) | `torch.nn.init.kaiming_normal_` |
| **GELU / SwiGLU (LLMs)** | Truncated Normal ($\sigma \approx 0.02$) | `torch.nn.init.trunc_normal_` |
| **Biases (All Layers)** | Constant Zero ($\mathbf{b} = \mathbf{0}$) | `torch.nn.init.zeros_` |

---

## 7. Summary & Key Takeaways

1. In deep networks, activation variance compounds multiplicatively across layers: $\text{Var}(x_L) \propto \left(n_{\text{in}} \sigma_w^2\right)^L$.
2. Xavier initialization stabilizes linear and Tanh networks by setting $\sigma^2 = \frac{2}{n_{\text{in}} + n_{\text{out}}}$.
3. ReLU non-linearities zero out negative inputs, discarding $50\%$ of distribution variance.
4. He / Kaiming initialization compensates by setting $\sigma^2 = \frac{2}{n_{\text{in}}}$, preserving activation scale across 50+ deep layers.
5. Always initialize bias vectors to zero to prevent unbalancing initial activation symmetry.
