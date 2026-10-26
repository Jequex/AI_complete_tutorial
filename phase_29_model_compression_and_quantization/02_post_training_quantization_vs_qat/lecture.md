# Topic 02: Post-Training Quantization (PTQ) vs. Quantization-Aware Training (QAT)

## 1. Introduction & 12-Year-Old Intuition

Imagine you are a professional gymnast performing a routine on a wooden balance beam that is **4 inches wide** (floating-point precision). You have practiced your entire life on this beam, landing flips and somersaults with absolute grace.

Suddenly, an Olympic judge announces two different challenges:
- **Challenge A (Post-Training Quantization / PTQ)**: The competition starts in 30 seconds. You must perform your exact same routine, but the beam has just been shaved down to a **wire-thin tightrope that is only 1 inch wide** (INT4). You are given zero practice time. What happens? You try your old muscle memory, your foot slips by a millimeter, and you crash onto the mat.
- **Challenge B (Quantization-Aware Training / QAT)**: The judge gives you two weeks to train on the 1-inch tightrope before the competition. During practice, whenever your foot slips, your coach gently guides your balance back to center. By the time the real competition starts, your brain and muscles have adapted completely: you score a perfect 10 even on the narrow tightrope!

**PTQ** takes a finished, pre-trained model and snaps all its numbers into integer slots without retraining. It is lightning-fast (taking seconds or minutes) but can cause sudden accuracy collapse when precision drops too low (e.g., INT4 or INT2).

**QAT** simulates the integer roundoff errors during training. The neural network learns to adjust its weights so that after rounding, the outputs remain stable and accurate.

---

## 2. Mathematical Symbol Decoder

| Symbol | Name | Formal Type | Intuition / Role |
| :--- | :--- | :--- | :--- |
| $\mathcal{D}_{\text{calib}}$ | Calibration Dataset | Unlabeled dataset $\{x_1, \dots, x_M\}$ | Small subset (e.g. 128–512 samples) used in PTQ to collect activation statistics. |
| $c \in \mathbb{R}^+$ | Clipping Threshold | Positive real scalar | Upper boundary $[-c, c]$ or $[0, c]$ mapped to integer limits. |
| $D_{\text{KL}}(P \parallel Q)$ | KL Divergence | Information metric ($\text{nats} / \text{bits}$) | Relative entropy between FP32 histogram $P$ and simulated quantized distribution $Q$. |
| $\text{clip}(u, a, b)$ | Saturation Clamp | Operator: $\min(\max(u, a), b)$ | Constrains unbounded values into hardware integer ranges. |
| $\hat{x}_{\text{fake}}$ | Fake Quantized Value | Real scalar $\in \mathbb{R}$ | Floating-point number that only takes discrete values $s \cdot q$. |
| $\text{STE}$ | Straight-Through Estimator | Gradient heuristic | Surrogate gradient replacing zero-derivative rounding with identity. |
| $s \in \mathbb{R}^+$ | Step Size / Scale | Learnable parameter in LSQ | Step size optimized directly via SGD/Adam alongside model weights. |
| $g_s$ | LSQ Gradient Scale | Scalar $\frac{1}{\sqrt{N q_{\max}}}$ | Normalization factor preventing exploding step-size gradients across large tensors. |

---

## 3. Post-Training Quantization (PTQ) & Calibration Strategies

In PTQ, weights $\mathbf{W}$ are static and immediately accessible. Activations $\mathbf{X}$, however, depend dynamically on the input data distribution. PTQ runs a small **calibration set** (e.g. 100–500 representative input samples) through the FP32 network to collect activation histograms and compute optimal scales $s$.

```
[ FP32 Pretrained Model ]
           │
           ▼
[ Run Calibration Data ] ──> Collect Activation Histograms
           │
           ▼
[ Select Calibration Strategy ]
   ├── A. MinMax:           c = max(|X|)
   ├── B. Percentile:       c = Quantile(|X|, 99.99%)
   └── C. KL-Divergence:    c = argmin D_KL(P || Q)  [NVIDIA TensorRT]
           │
           ▼
[ Export INT8 Quantized Model ] (Ready for deployment in minutes!)
```

### 3.1 MinMax Calibration
Sets the clipping boundary to the absolute maximum value encountered:
$$c = \max_{x \in \mathcal{D}_{\text{calib}}} |x|, \quad s = \frac{c}{q_{\max}}$$
- **Advantage**: Zero clipping error.
- **Fatal Flaw**: Extreme sensitivity to activation outliers. If a single token in a Transformer produces an outlier value of $x = 120.0$ while $99.99\%$ of activations sit in $[-2.0, 2.0]$, MinMax stretches $s = \frac{120}{127} \approx 0.945$. The normal values $[-2.0, 2.0]$ are squashed into just 4 integer bins ($\{-2, -1, 0, 1, 2\}$), obliterating the layer's representational capacity.

### 3.2 Percentile / Histogram Calibration
Sorts all observed activations and clips at a fixed quantile (e.g. $p = 99.99\%$ or $99.9\%$):
$$c = \text{Quantile}(|x|, p)$$
- Sacrifices the top $0.01\%$ of extreme values to saturation ($q_{\max}$).
- In return, compresses step size $s$ by $5\times$ to $20\times$, providing dramatically finer resolution for the remaining $99.99\%$ of features.

### 3.3 KL-Divergence Calibration (The NVIDIA TensorRT Method)
TensorRT treats quantization not as geometric distance minimization, but as **minimizing information entropy loss**.
1. Collect a high-resolution histogram of activations (e.g., 2048 bins) representing continuous distribution $P$.
2. For each candidate threshold $T \in [128, 2048]$:
   - Clip histogram $P$ at bin $T$, bundling all probabilities from bins $> T$ into bin $T$.
   - Quantize the clipped histogram into 128 integer bins (for signed INT8).
   - Expand the 128 bins back into $T$ bins to create reconstructed distribution $Q$.
   - Compute the Kullback-Leibler divergence:
     $$D_{\text{KL}}(P \parallel Q) = \sum_{i=1}^T P(i) \ln\left( \frac{P(i)}{Q(i)} \right)$$
3. Pick $T^* = \arg\min_T D_{\text{KL}}(P \parallel Q)$ and set $s = \frac{T^* \cdot \text{bin\_width}}{127}$.

---

## 4. The Non-Differentiable Rounding Barrier & Fake Quantization

When compressing beyond INT8 down to INT4 or INT2, PTQ frequently breaks down because the quantization noise is too large for the frozen weights to absorb. We must fine-tune the weights using gradient descent.

### 4.1 The Rounding Barrier
Consider the loss $\mathcal{L}(w)$ with respect to quantized weight $\hat{w} = s \left\lfloor \frac{w}{s} \right\rceil$.
By the chain rule:
$$\frac{\partial \mathcal{L}}{\partial w} = \frac{\partial \mathcal{L}}{\partial \hat{w}} \cdot \frac{\partial \hat{w}}{\partial w}$$
Now consider the derivative of the round function $f(u) = \lfloor u \rceil$:
$$\frac{d \lfloor u \rceil}{du} = \begin{cases} 0, & u \notin \mathbb{Z} + 0.5 \\ \text{undefined}, & u \in \mathbb{Z} + 0.5 \end{cases}$$

Almost everywhere ($\text{a.e.}$), the derivative is **identically zero**!
If plugged into an autograd engine:
$$\frac{\partial \mathcal{L}}{\partial w} = \frac{\partial \mathcal{L}}{\partial \hat{w}} \times 0 = 0$$

All gradients evaluate to zero. Floating-point master weights receive zero update ($w \leftarrow w - \eta \cdot 0$), and training halts completely.

### 4.2 Fake Quantization Operator
To resolve this, QAT performs training in a **floating-point simulation framework**:
$$\hat{w}_{\text{fake}} = s \cdot \text{clip}\left( \left\lfloor \frac{w}{s} \right\rceil, \; q_{\min}, \; q_{\max} \right)$$
The numbers are stored as standard FP32 or FP16 tensors, but their numerical values are strictly clamped to integer multiples of $s$.

---

## 5. The Straight-Through Estimator (STE)

Introduced by Yoshua Bengio et al. (2013), the **Straight-Through Estimator (STE)** replaces the zero derivative of the rounding function with the identity pass-through during backpropagation:
$$\frac{\partial \lfloor u \rceil}{\partial u} \approx 1$$

### 5.1 Hard Saturation STE Formula
Taking the clipping bounds into account:
$$\hat{x}_{\text{fake}} = s \cdot \min\left( \max\left( \left\lfloor \frac{x}{s} \right\rceil, q_{\min} \right), q_{\max} \right)$$

The forward pass is discrete:
$$\text{Forward}: \quad \hat{x}_{\text{fake}} = Q_{\text{fake}}(x)$$

The backward pass uses the surrogate derivative:
$$\boxed{\frac{\partial \hat{x}_{\text{fake}}}{\partial x} = \begin{cases} 1, & q_{\min} s \le x \le q_{\max} s \\ 0, & \text{otherwise} \end{cases}}$$

```
Forward:   x  ───[ Fake Quantizer ]───> x_hat ───> Loss
                    │
Backward:  dL/dx <───[ STE (Pass-through if in bounds) ] <─── dL/dx_hat
```

### 5.2 Why STE Works
During forward propagation, the network experiences the exact errors, disturbances, and zero-crossings of the true integer hardware.
During backward propagation, the incoming gradient $\frac{\partial \mathcal{L}}{\partial \hat{w}_{\text{fake}}}$ flows straight into the master floating-point weights $w$.
The master weights take gradient descent steps:
$$w_{t+1} = w_t - \eta \frac{\partial \mathcal{L}}{\partial \hat{w}_{\text{fake}}}$$
Eventually, master weights migrate across the rounding thresholds into integer bins that minimize the total loss.

---

## 6. Learned Step Size Quantization (LSQ)

Rather than setting scale $s$ from static heuristics or calibration sets, **Learned Step Size Quantization (LSQ, Esser et al. 2020)** treats the step size $s$ as a continuous learnable parameter optimized by standard gradient descent simultaneously with the weights.

### 6.1 LSQ Forward Pass
For symmetric quantization with $q_{\max} = 2^{b-1} - 1$ and $q_{\min} = -2^{b-1}$:
$$\hat{x} = s \cdot \text{clip}\left( \left\lfloor \frac{x}{s} \right\rceil, \; q_{\min}, \; q_{\max} \right)$$

### 6.2 LSQ Analytical Gradient with Respect to Step Size $s$
Differentiating $\hat{x}$ with respect to $s$:
$$\frac{\partial \hat{x}}{\partial s} = \begin{cases} 
q_{\min}, & \frac{x}{s} < q_{\min} \\
q_{\max}, & \frac{x}{s} > q_{\max} \\
\left\lfloor \frac{x}{s} \right\rceil - \frac{x}{s}, & q_{\min} \le \frac{x}{s} \le q_{\max}
\end{cases}$$

Notice:
- If $x$ is clamped at the positive limit, $\frac{\partial \hat{x}}{\partial s} = q_{\max} > 0$. If loss wants $\hat{x}$ larger, gradient pushes $s$ up, expanding the dynamic range to reduce clamping!
- If $x$ is inside the grid, $\left\lfloor \frac{x}{s} \right\rceil - \frac{x}{s} = -\frac{e}{s}$. The gradient optimizes $s$ to minimize rounding residuals.

### 6.3 LSQ Gradient Scaling
Because a single scale factor $s$ affects all $N$ elements of a tensor, its accumulated gradient $\sum_{i=1}^N \frac{\partial \mathcal{L}}{\partial \hat{x}_i} \frac{\partial \hat{x}_i}{\partial s}$ can be orders of magnitude larger than weight gradients. LSQ stabilizes optimization using gradient scale factor $g$:
$$g = \frac{1}{\sqrt{N \cdot q_{\max}}}$$
$$\nabla_s^{\text{scaled}} = g \cdot \nabla_s$$

---

## 7. PTQ vs. QAT Comprehensive Engineering Comparison

| Criterion | Post-Training Quantization (PTQ) | Quantization-Aware Training (QAT) |
| :--- | :--- | :--- |
| **Compute Cost** | Seconds to minutes on a single CPU/GPU | Hours to days (requires training loop & backward pass) |
| **Data Requirement** | 100–500 unlabeled calibration samples | Full labeled training dataset (or domain pretraining corpus) |
| **Complexity** | Zero pipeline changes; pure post-processing | Custom training graph, fake quant nodes, STE implementation |
| **INT8 Accuracy** | Negligible degradation (<0.5% drop on most models) | Perfectly matches FP32 baseline |
| **INT4 Accuracy** | Severe accuracy drop or collapse on standard models | Retains >95–99% of original baseline performance |
| **Weight Migration** | Weights never move; only scales are calibrated | Master weights migrate to optimal integer basins |
