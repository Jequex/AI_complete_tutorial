# Topic 01: Uniform Quantization Mechanics

## 1. Introduction & 12-Year-Old Intuition

Imagine you have a giant set of high-end artist watercolor paints with **16.7 million distinct micro-shades** of color (this is 32-bit floating-point numbers, FP32). When you paint a massive mural, packing all 16.7 million paint bottles requires a gigantic, heavy shipping container that takes up enormous space and costs a fortune to transport.

Now imagine someone tells you:
*"You can only take a tiny wooden pencil box with **256 crayons** (8-bit integer, INT8), or even a pocket box with just **16 crayons** (4-bit integer, INT4). But your painting still needs to look almost identical to the human eye!"*

How do you do it?
1. **Find the palette range**: You look at your painting and discover the darkest shade used is deep navy blue ($x_{\min} = -5.0$) and the brightest is bright sun yellow ($x_{\max} = +5.0$).
2. **Lay down a ruler (Scale $s$)**: You divide that entire color span evenly into 255 equal steps. The size of each step is your **scale factor** $s$.
3. **Set the reference peg (Zero-Point $z$)**: You decide which crayon index represents pure black or zero.
4. **Snap every color (Quantization)**: Whenever you see an original paint shade, you snap it to the closest crayon in your box (`round(x / s) + z`).
5. **Reconstruct when painting (Dequantization)**: When looking at crayon number $q$, you estimate the original shade by multiplying back: $\hat{x} = s \times (q - z)$.

Because you rounded to the nearest crayon, you made a tiny mistake on each color—called **Quantization Noise**. 

**Uniform Quantization** is the mathematical science of choosing the best ruler ($s$) and reference peg ($z$) so that the memory footprint drops by $4\times$ (FP32 to INT8) or $8\times$ (FP32 to INT4) while keeping the quantization noise so whisper-quiet that the neural network performs its job without noticing the difference.

---

## 2. Mathematical Symbol Decoder

| Symbol | Name | Formal Domain | Intuition / Operational Meaning |
| :--- | :--- | :--- | :--- |
| $x \in \mathbb{R}$ | Continuous Value | Real number ($\text{FP32} / \text{FP16}$) | Original continuous weight or activation scalar. |
| $q \in \mathbb{Z}$ | Quantized Value | Discrete integer ($\text{INT8} / \text{INT4}$) | Discrete quantized integer code stored in memory or hardware registers. |
| $\hat{x} \in \mathbb{R}$ | Dequantized Value | Real number ($\mathbb{R}$) | Reconstructed real-valued approximation of $x$. |
| $b$ | Bit-width | Integer ($b \in \{2, 4, 8, 16\}$) | Number of bits used to represent each numerical entry. |
| $q_{\min}, q_{\max}$ | Quantization Bounds | Integer range | Supported integer limits; e.g. $[0, 255]$ for UINT8, $[-128, 127]$ for INT8. |
| $s \in \mathbb{R}^+$ | Scale Factor (Step Size) | Positive real scalar ($\Delta$) | Real-world distance represented by one integer increment ($1 \text{ LSB}$). |
| $z \in \mathbb{Z}$ | Zero-Point | Integer | Quantized integer index corresponding exactly to real zero ($x = 0.0$). |
| $e = x - \hat{x}$ | Quantization Error | Continuous interval $[-\frac{s}{2}, \frac{s}{2}]$ | Residual noise introduced by the deterministic rounding operation. |
| $\sigma_e^2$ | Noise Variance | $\mathbb{R}^+$ | Power of the quantization noise, theoretically $\frac{s^2}{12}$ under uniform error. |
| $\text{SQNR}$ | Signal-to-Noise Ratio | Decibels ($\text{dB}$) | Logarithmic ratio of signal variance to quantization error variance. |

---

## 3. Uniform Affine (Asymmetric) Quantization

In real-world deep neural networks, activations following ReLU or GeLU are strictly non-negative or skewed, e.g., $x \in [0.0, 18.5]$. Centering around zero would waste half of the available integer states. **Uniform Affine (Asymmetric) Quantization** provides an arbitrary zero-point shift to utilize the full integer grid.

### 3.1 Mapping Real Range to Integer Range
Given continuous tensor values bounded by $[x_{\min}, x_{\max}]$ and a target $b$-bit integer container with range $[q_{\min}, q_{\max}]$:
- For unsigned $b$-bit integers: $q_{\min} = 0$, $q_{\max} = 2^b - 1$.
- For signed $b$-bit integers: $q_{\min} = -2^{b-1}$, $q_{\max} = 2^{b-1} - 1$.

The scale factor $s$ defines the step size:
$$s = \frac{x_{\max} - x_{\min}}{q_{\max} - q_{\min}}$$

To ensure that real zero ($x = 0.0$) maps exactly to an integer in the discrete domain (vital for zero-padding, sparse activations, and ReLU zero-preservation), we compute the **zero-point** $z$:
$$0 = s \cdot (z - q_{\min}) + x_{\min} \implies z = \text{round}\left( -\frac{x_{\min}}{s} \right) + q_{\min}$$

Clamping $z$ guarantees it remains within the allowable integer range:
$$z = \text{clip}\left( \left\lfloor -\frac{x_{\min}}{s} \right\rceil + q_{\min}, \; q_{\min}, \; q_{\max} \right)$$

### 3.2 Quantization & Dequantization Operators
The continuous-to-integer quantization function $Q(x; s, z)$ is:
$$q = Q(x; s, z) = \text{clip}\left( \left\lfloor \frac{x}{s} \right\rceil + z, \; q_{\min}, \; q_{\max} \right)$$
where $\lfloor \cdot \rceil$ denotes rounding to the nearest integer.

The dequantization operator $\hat{Q}(q; s, z)$ reconstructs the real approximation:
$$\hat{x} = \hat{Q}(q; s, z) = s \cdot (q - z)$$

---

## 4. Symmetric Quantization

When tensor distributions are inherently symmetric around zero (such as weights initialized with zero-mean Gaussian or Xavier distributions), setting the zero-point strictly to zero ($z = 0$) eliminates zero-point subtraction overhead in hardware multiply-accumulate (MAC) units.

### 4.1 Symmetric Mapping
For signed $b$-bit representation with limits $q_{\max} = 2^{b-1} - 1$ and $q_{\min} = -2^{b-1}$:
$$s = \frac{\max(|x_{\min}|, |x_{\max}|)}{q_{\max}} = \frac{\|x\|_\infty}{2^{b-1} - 1}$$

With $z = 0$, the quantization formula simplifies to:
$$q = \text{clip}\left( \left\lfloor \frac{x}{s} \right\rceil, \; -2^{b-1}, \; 2^{b-1} - 1 \right)$$

And dequantization is a single scalar multiplication:
$$\hat{x} = s \cdot q$$

### 4.2 Asymmetric vs. Symmetric Comparison

| Dimension | Asymmetric (Uniform Affine) | Symmetric Uniform |
| :--- | :--- | :--- |
| **Zero-Point ($z$)** | Arbitrary integer $z \in [q_{\min}, q_{\max}]$ | Constrained to $z = 0$ |
| **Degrees of Freedom** | 2 parameters per tensor/channel ($s, z$) | 1 parameter per tensor/channel ($s$) |
| **Optimal For** | Skewed activations (ReLU, Sigmoid, GeLU) | Zero-centered weights ($\mathbf{W}$) |
| **Integer Arithmetic** | Incurs cross-terms: $\mathbf{X}\mathbf{W} = s_x s_w (q_x - z_x)(q_w - z_w)$ | Clean dot products: $\mathbf{X}\mathbf{W} = s_x s_w (q_x q_w)$ |
| **Zero Representation** | Exact: $x = 0 \iff q = z$ | Exact: $x = 0 \iff q = 0$ |

---

## 5. Quantization Noise & Theoretical Derivation of the 6.02 dB Rule

What is the fundamental theoretical limit of information loss when quantizing a signal?

### 5.1 The Uniform Quantization Noise Model (Bennett's Postulates)
Consider a continuous signal $x$ quantized with uniform step size $\Delta = s$.
The quantization error is:
$$e = x - \hat{x}$$

When the step size $\Delta$ is sufficiently small relative to the dynamic range of $x$, and the signal crosses many quantization thresholds, Bennett (1948) established that:
1. The error $e$ is uniformly distributed over the interval $\left[ -\frac{\Delta}{2}, \frac{\Delta}{2} \right]$:
   $$p(e) = \begin{cases} \frac{1}{\Delta}, & -\frac{\Delta}{2} \le e \le \frac{\Delta}{2} \\ 0, & \text{otherwise} \end{cases}$$
2. The error $e$ is zero-mean:
   $$\mathbb{E}[e] = \int_{-\Delta/2}^{\Delta/2} e \cdot \frac{1}{\Delta} \, de = 0$$
3. The quantization noise variance (noise power $P_{\text{noise}} = \sigma_e^2$) is:
   $$\sigma_e^2 = \mathbb{E}[e^2] = \int_{-\Delta/2}^{\Delta/2} e^2 \cdot \frac{1}{\Delta} \, de = \frac{1}{\Delta} \left[ \frac{e^3}{3} \right]_{-\Delta/2}^{\Delta/2} = \frac{1}{\Delta} \left( \frac{\Delta^3}{24} - \left( -\frac{\Delta^3}{24} \right) \right) = \frac{\Delta^2}{12}$$

$$\boxed{\sigma_e^2 = \frac{\Delta^2}{12} = \frac{s^2}{12}}$$

### 5.2 Signal-to-Quantization-Noise Ratio (SQNR)
The Signal-to-Quantization-Noise Ratio measures fidelity in logarithmic decibels:
$$\text{SQNR} \triangleq 10 \log_{10}\left( \frac{P_{\text{signal}}}{P_{\text{noise}}} \right) = 10 \log_{10}\left( \frac{\sigma_x^2}{\sigma_e^2} \right)$$

#### Case A: Full-Scale Sinusoidal Signal
Let $x(t) = A \sin(\omega t)$ spanning the full dynamic range $[-A, A]$ of a $b$-bit quantizer with $N = 2^b$ levels.
The step size is $\Delta = \frac{2A}{2^b} = A \cdot 2^{1-b}$.
1. **Signal Power**:
   $$P_{\text{signal}} = \frac{1}{T} \int_0^T A^2 \sin^2(\omega t) dt = \frac{A^2}{2}$$
2. **Noise Power**:
   $$P_{\text{noise}} = \frac{\Delta^2}{12} = \frac{(2A \cdot 2^{-b})^2}{12} = \frac{4A^2 \cdot 2^{-2b}}{12} = \frac{A^2 \cdot 2^{-2b}}{3}$$
3. **SQNR Calculation**:
   $$\text{SQNR} = 10 \log_{10} \left( \frac{A^2 / 2}{A^2 2^{-2b} / 3} \right) = 10 \log_{10} \left( \frac{3}{2} \cdot 2^{2b} \right)$$
   $$= 10 \log_{10}(1.5) + 10 \log_{10}(2^{2b})$$
   $$= 10 \log_{10}(1.5) + 20 b \log_{10}(2)$$
   
Evaluating the numerical constants:
$$20 \log_{10}(2) = 20 \times 0.30102999566 \approx 6.0206$$
$$10 \log_{10}(1.5) = 10 \times 0.176091259 \approx 1.7609$$

Therefore:
$$\boxed{\text{SQNR}_{\text{sinusoid}} \approx 6.02 \cdot b + 1.76 \text{ dB}}$$

#### Case B: Uniformly Distributed Signal
Let $x \sim U[-A, A]$.
1. **Signal Power**:
   $$P_{\text{signal}} = \frac{(2A)^2}{12} = \frac{A^2}{3}$$
2. **Noise Power**:
   $$P_{\text{noise}} = \frac{\Delta^2}{12} = \frac{(2A / 2^b)^2}{12} = \frac{A^2 \cdot 2^{-2b}}{3}$$
3. **SQNR Calculation**:
   $$\text{SQNR} = 10 \log_{10}\left( \frac{A^2 / 3}{A^2 2^{-2b} / 3} \right) = 10 \log_{10}(2^{2b}) = 20 b \log_{10}(2) \approx \mathbf{6.02 \cdot b \text{ dB}}$$

> **Fundamental Insight**: Every single bit ($+1$ bit) added to a quantizer cuts the step size $\Delta$ in half, which cuts the noise variance by $\frac{1}{4}$ ($4\times$), boosting the SQNR by exactly $10 \log_{10}(4) \approx 6.02 \text{ dB}$.

---

## 6. Dynamic Range vs. Precision: INT8 vs. INT4 Tradeoffs

| Bit-width | Distinct Levels | Memory Compression (vs FP32) | Theoretical SQNR (Uniform) | Primary Application in LLMs |
| :--- | :--- | :--- | :--- | :--- |
| **FP32** | $\approx 4.29 \times 10^9$ | $1\times$ (Baseline: 4 bytes/param) | $> 150 \text{ dB}$ | Master training weights |
| **FP16 / BF16** | 65,536 | $2\times$ (2 bytes/param) | $\approx 96 \text{ dB}$ | High-throughput GPU inference/training |
| **INT8** | 256 | $4\times$ (1 byte/param) | $\approx 48.16 \text{ dB}$ | Zero-shot PTQ, W8A8 matrix multiplication |
| **INT4** | 16 | $8\times$ (0.5 bytes/param) | $\approx 24.08 \text{ dB}$ | Extreme LLM compression (GPTQ, AWQ, GGUF) |

### 6.1 The Quantization Error Duality: Clamping vs. Rounding
When designing a quantizer for a bell-shaped distribution (e.g. Gaussian weights $\mathcal{N}(0, \sigma^2)$):
1. **Clipping (Clamping) Error**: If we set scale $s$ small to give fine resolution around zero, extreme values $|x| > s \cdot q_{\max}$ get clamped hard. Clamping error grows rapidly when clipping thresholds are too tight.
2. **Rounding (Granularity) Error**: If we widen the clipping threshold to capture distant outliers, step size $s$ grows larger, increasing the baseline noise variance $\sigma_e^2 = \frac{s^2}{12}$ across all non-outlier values.

The optimal quantization grid balances clipping error against rounding error to minimize Mean Squared Error (MSE).

---

## 7. Granularity Taxonomy: Per-Tensor vs. Per-Channel vs. Group-Wise

```
Per-Tensor Quantization:
[  Entire Matrix W (N x M)  ] ---> 1 Scale Factor s, 1 Zero-point z

Per-Channel / Per-Row Quantization:
[ Row 1 (1 x M)             ] ---> s_1, z_1
[ Row 2 (1 x M)             ] ---> s_2, z_2
[ Row N (1 x M)             ] ---> s_N, z_N

Group-Wise Quantization (e.g., Group Size G = 128):
[ G elements ][ G elements ][ G elements ] ...
    s_11          s_12          s_13
```

1. **Per-Tensor**: Simplest, lowest metadata overhead, but highly vulnerable if even one weight or activation channel has a large outlier.
2. **Per-Channel (Per-Output-Channel for Weights)**: Standard in modern INT8 inference. Different output features have independent dynamic ranges without interfering with one another.
3. **Group-Wise (e.g., Group Size 128 or 64 in INT4)**: Divides each row into blocks of $G$ consecutive weights. Each block gets its own independent scale factor $s$. This prevents isolated local spikes from degrading the precision of other weights in the same row.
