# Topic 02: Exercises & Solutions — PTQ vs. Quantization-Aware Training (QAT)

---

### Exercise 1: Analytical Derivation of the LSQ Step Size Gradient
**Problem:**
In Learned Step Size Quantization (LSQ), the simulated quantization operator for signed $b$-bit integers is given by:
$$\hat{x}(x, s) = s \cdot \text{clip}\left( \left\lfloor \frac{x}{s} \right\rceil, \; q_{\min}, \; q_{\max} \right)$$
where $s > 0$.
Assuming the Straight-Through Estimator $\frac{\partial \lfloor u \rceil}{\partial u} \approx 1$ holds for in-bounds values, rigorously derive the partial derivative $\frac{\partial \hat{x}}{\partial s}$ across all three piecewise regions:
1. Underflow: $\frac{x}{s} < q_{\min}$
2. In-bounds: $q_{\min} \le \frac{x}{s} \le q_{\max}$
3. Overflow: $\frac{x}{s} > q_{\max}$

<details>
<summary><strong>Solution</strong></summary>

Let $u = \frac{x}{s}$. Then $\frac{\partial u}{\partial s} = -\frac{x}{s^2}$.
We analyze $\hat{x}(x, s)$ piecewise:

**Region 1: Underflow ($u < q_{\min}$)**
In this region, the clamping function saturates at the lower boundary:
$$\hat{x} = s \cdot q_{\min}$$
Differentiating directly with respect to $s$:
$$\frac{\partial \hat{x}}{\partial s} = \frac{\partial}{\partial s} (s \cdot q_{\min}) = q_{\min}$$

**Region 3: Overflow ($u > q_{\max}$)**
In this region, the clamping function saturates at the upper boundary:
$$\hat{x} = s \cdot q_{\max}$$
Differentiating directly with respect to $s$:
$$\frac{\partial \hat{x}}{\partial s} = \frac{\partial}{\partial s} (s \cdot q_{\max}) = q_{\max}$$

**Region 2: In-bounds ($q_{\min} \le u \le q_{\max}$)**
Here, clipping is inactive:
$$\hat{x} = s \cdot \lfloor u \rceil = s \cdot \left\lfloor \frac{x}{s} \right\rceil$$
By the product rule:
$$\frac{\partial \hat{x}}{\partial s} = \left\lfloor \frac{x}{s} \right\rceil \cdot \frac{ds}{ds} + s \cdot \frac{\partial}{\partial s} \left( \left\lfloor \frac{x}{s} \right\rceil \right)$$
$$= \left\lfloor \frac{x}{s} \right\rceil + s \cdot \left( \left. \frac{\partial \lfloor v \rceil}{\partial v} \right|_{v = x/s} \cdot \frac{\partial}{\partial s}\left( \frac{x}{s} \right) \right)$$
Using the Straight-Through Estimator $\frac{\partial \lfloor v \rceil}{\partial v} \approx 1$:
$$\frac{\partial \hat{x}}{\partial s} \approx \left\lfloor \frac{x}{s} \right\rceil + s \cdot \left( 1 \cdot \left( -\frac{x}{s^2} \right) \right) = \left\lfloor \frac{x}{s} \right\rceil - \frac{x}{s}$$

Combining all three cases yields the exact LSQ derivative:
$$\boxed{\frac{\partial \hat{x}}{\partial s} = \begin{cases} 
q_{\min}, & \frac{x}{s} < q_{\min} \\
q_{\max}, & \frac{x}{s} > q_{\max} \\
\left\lfloor \frac{x}{s} \right\rceil - \frac{x}{s}, & q_{\min} \le \frac{x}{s} \le q_{\max}
\end{cases}}$$
</details>

---

### Exercise 2: Mathematical Analysis of STE Gradient Bias
**Problem:**
Let $f(x) = \lfloor x \rceil$ be the rounding function and $g(x) = x$ be the identity function used by the Straight-Through Estimator (STE).
Consider the scalar optimization problem:
$$\min_x \mathcal{L}(f(x))$$
where $\mathcal{L}(y) = \frac{1}{2}(y - y^*)^2$ for a constant target $y^* \in \mathbb{R}$.
1. Compute the true derivative $\frac{d\mathcal{L}}{dx}$ for almost all $x$.
2. Compute the STE surrogate gradient $g_{\text{STE}}(x) = \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{dg}{dx}$.
3. Show that for $x = 1.2$ and $y^* = 2.0$, the true gradient cannot provide a descent direction, but the STE surrogate gradient points strictly towards the global optimum.

<details>
<summary><strong>Solution</strong></summary>

1. **True Derivative:**
   $$\frac{d\mathcal{L}}{dx} = \frac{d\mathcal{L}}{dy} \cdot \frac{df}{dx} = (\lfloor x \rceil - y^*) \cdot 0 = 0 \quad (\forall x \notin \mathbb{Z} + 0.5)$$
   The true gradient is identically zero almost everywhere. Under standard gradient descent:
   $$x_{t+1} = x_t - \eta \cdot 0 = x_t$$
   The parameter never moves.

2. **STE Surrogate Gradient:**
   Under STE, $\frac{df}{dx}$ is replaced with $\frac{dg}{dx} = 1$:
   $$g_{\text{STE}}(x) = \left. \frac{d\mathcal{L}}{dy} \right|_{y = \lfloor x \rceil} \cdot 1 = \lfloor x \rceil - y^*$$

3. **Evaluation at $x = 1.2, y^* = 2.0$:**
   - Rounding: $\lfloor 1.2 \rceil = 1.0$.
   - Loss value: $\mathcal{L}(1.0) = \frac{1}{2}(1.0 - 2.0)^2 = 0.5$.
   - True gradient: $\frac{d\mathcal{L}}{dx} = 0$. Optimization is stuck.
   - STE surrogate gradient:
     $$g_{\text{STE}}(1.2) = 1.0 - 2.0 = -1.0$$
   - Applying a gradient descent update with learning rate $\eta = 0.5$:
     $$x_{\text{new}} = x - \eta \cdot g_{\text{STE}}(x) = 1.2 - 0.5 \times (-1.0) = 1.2 + 0.5 = 1.7$$
   - Evaluating after the update: $\lfloor 1.7 \rceil = 2.0$.
   - New loss: $\mathcal{L}(\lfloor 1.7 \rceil) = \frac{1}{2}(2.0 - 2.0)^2 = 0.0$!
   
   The STE step traversed the non-differentiable step boundary and reached the global minimum.
</details>

---

### Exercise 3: KL-Divergence Calibration on a Discrete Histogram
**Problem:**
Suppose calibration yields an activation histogram with 4 positive bins:
$$P = [0.40, \; 0.30, \; 0.20, \; 0.10]$$
We test two candidate clipping thresholds:
- **Candidate 1 ($T = 4$)**: Keep all 4 bins. Quantize to 2 bins by merging bins $(1, 2)$ and $(3, 4)$, then reconstruct $Q_1$ by distributing counts equally within each merged bin.
- **Candidate 2 ($T = 3$)**: Clip at bin 3. Bin 4 is folded into bin 3, giving $P_{\text{clip}} = [0.40, \; 0.30, \; 0.30]$. Quantize the first 2 bins and bin 3 into 2 bins.

Compute $D_{\text{KL}}(P \parallel Q_1)$ for Candidate 1 where $Q_1$ is obtained by averaging bin pairs:
$$Q_1(1) = Q_1(2) = \frac{0.40 + 0.30}{2} = 0.35$$
$$Q_1(3) = Q_1(4) = \frac{0.20 + 0.10}{2} = 0.15$$
Show step-by-step numerical calculation using natural logarithms.

<details>
<summary><strong>Solution</strong></summary>

Recall the definition of Kullback-Leibler divergence:
$$D_{\text{KL}}(P \parallel Q_1) = \sum_{i=1}^4 P(i) \ln\left( \frac{P(i)}{Q_1(i)} \right)$$

Evaluating each term:
- **Bin 1 ($i = 1$):** $P(1) = 0.40, \; Q_1(1) = 0.35$
  $$\frac{P(1)}{Q_1(1)} = \frac{0.40}{0.35} \approx 1.142857, \quad \ln(1.142857) \approx 0.133531$$
  $$P(1) \ln\left( \frac{P(1)}{Q_1(1)} \right) = 0.40 \times 0.133531 \approx 0.053413$$

- **Bin 2 ($i = 2$):** $P(2) = 0.30, \; Q_1(2) = 0.35$
  $$\frac{P(2)}{Q_1(2)} = \frac{0.30}{0.35} \approx 0.857143, \quad \ln(0.857143) \approx -0.154151$$
  $$P(2) \ln\left( \frac{P(2)}{Q_1(2)} \right) = 0.30 \times (-0.154151) \approx -0.046245$$

- **Bin 3 ($i = 3$):** $P(3) = 0.20, \; Q_1(3) = 0.15$
  $$\frac{P(3)}{Q_1(3)} = \frac{0.20}{0.15} \approx 1.333333, \quad \ln(1.333333) \approx 0.287682$$
  $$P(3) \ln\left( \frac{P(3)}{Q_1(3)} \right) = 0.20 \times 0.287682 \approx 0.057536$$

- **Bin 4 ($i = 4$):** $P(4) = 0.10, \; Q_1(4) = 0.15$
  $$\frac{P(4)}{Q_1(4)} = \frac{0.10}{0.15} \approx 0.666667, \quad \ln(0.666667) \approx -0.405465$$
  $$P(4) \ln\left( \frac{P(4)}{Q_1(4)} \right) = 0.10 \times (-0.405465) \approx -0.040547$$

Summing all four terms:
$$D_{\text{KL}}(P \parallel Q_1) = 0.053413 - 0.046245 + 0.057536 - 0.040547 = \mathbf{0.024157 \text{ nats}}$$

In TensorRT, this calculation is repeated for hundreds of candidate thresholds $T$, and the threshold achieving minimal KL-divergence is selected as the optimal calibration point.
</details>
