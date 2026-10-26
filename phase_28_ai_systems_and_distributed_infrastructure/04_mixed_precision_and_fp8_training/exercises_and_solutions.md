# Topic 04: Exercises & Solutions — Mixed Precision & FP8 Training

## Exercise 1: Bit-Level Floating-Point Decoding
**Problem:**
Decode the following 16-bit binary pattern:
$$\mathbf{b} = \texttt{0\_10001\_0100000000}_2$$
under two distinct interpretations:
1. IEEE 754 Half-Precision (**FP16**: 1 sign, 5 exponent with bias 15, 10 mantissa).
2. Brain Floating Point (**BF16**: 1 sign, 8 exponent with bias 127, 7 mantissa; for this part, use the first 16 bits of $\mathbf{b}$ formatted as $\texttt{0\_10001010\_0000000}_2$).

Also decode the 8-bit pattern $\mathbf{c} = \texttt{0\_1110\_101}_2$ under **FP8 E4M3** (1 sign, 4 exponent with bias 7, 3 mantissa).

<details>
<summary>Click to view Solution</summary>

### Solution:

#### 1. Under FP16 ($\texttt{0\_10001\_0100000000}_2$):
- **Sign Bit $s$**: $0 \implies$ Positive ($+1$).
- **Raw Exponent $E$**: $\texttt{10001}_2 = 16 + 1 = 17_{10}$.
- **Unbiased Exponent $e$**: $E - \text{bias} = 17 - 15 = \mathbf{2}$.
- **Mantissa Bits**: $\texttt{0100000000}_2 = 0 \times 2^{-1} + 1 \times 2^{-2} = \frac{1}{4} = 0.25$.
- **Normalized Value**:
  $$x = (+1) \times 2^2 \times (1 + 0.25) = 4 \times 1.25 = \mathbf{5.0}$$

#### 2. Under BF16 ($\texttt{0\_10001010\_0000000}_2$):
- **Sign Bit $s$**: $0 \implies$ Positive ($+1$).
- **Raw Exponent $E$**: $\texttt{10001010}_2 = 128 + 8 + 2 = 138_{10}$.
- **Unbiased Exponent $e$**: $E - \text{bias} = 138 - 127 = \mathbf{11}$.
- **Mantissa Bits**: $\texttt{0000000}_2 = 0.0$.
- **Normalized Value**:
  $$x = (+1) \times 2^{11} \times (1 + 0.0) = 2^{11} = \mathbf{2{,}048.0}$$

#### 3. Under FP8 E4M3 ($\texttt{0\_1110\_101}_2$):
- **Sign Bit $s$**: $0 \implies$ Positive ($+1$).
- **Raw Exponent $E$**: $\texttt{1110}_2 = 8 + 4 + 2 = 14_{10}$.
- **Unbiased Exponent $e$**: $E - \text{bias} = 14 - 7 = \mathbf{7}$.
- **Mantissa Bits**: $\texttt{101}_2 = 1 \times 2^{-1} + 0 \times 2^{-2} + 1 \times 2^{-3} = 0.5 + 0.125 = 0.625$.
- **Value**:
  $$x = (+1) \times 2^7 \times (1 + 0.625) = 128 \times 1.625 = \mathbf{208.0}$$
</details>

---

## Exercise 2: Loss Scaling Bounds Derivation
**Problem:**
During a Transformer training step in FP16, backward gradients are distributed such that:
- The smallest non-trivial gradient magnitude is $g_{\min} = 1.5 \times 10^{-7}$.
- The largest gradient magnitude (activation gradient outlier) is $g_{\max} = 1.8$.

FP16 parameters:
- Minimum normal value: $N_{\min} = 6.10 \times 10^{-5}$.
- Maximum finite value: $M_{\max} = 65{,}504.0$.

1. Derive the minimum loss scale factor $S_{\min}$ required to ensure that $g_{\min}$ does not fall into the subnormal or flush-to-zero regime.
2. Derive the maximum allowable loss scale factor $S_{\max}$ to ensure that $g_{\max}$ does not trigger an arithmetic overflow to $+\infty$.
3. What is the allowable range for $S$? Does a valid scaling factor exist? What power of 2 would you choose?

<details>
<summary>Click to view Solution</summary>

### Solution:

#### 1. Minimum Scale Factor $S_{\min}$:
To prevent underflow, the scaled gradient $g_{\text{scaled}} = S \cdot g_{\min}$ must be at least the smallest normal representable number in FP16:
$$S \cdot g_{\min} \ge N_{\min} \implies S \ge \frac{N_{\min}}{g_{\min}}$$
Substitute values:
$$S_{\min} = \frac{6.10 \times 10^{-5}}{1.5 \times 10^{-7}} = \frac{610}{1.5} \approx \mathbf{406.67}$$

#### 2. Maximum Scale Factor $S_{\max}$:
To prevent overflow, the largest scaled gradient must not exceed the maximum finite FP16 value:
$$S \cdot g_{\max} \le M_{\max} \implies S \le \frac{M_{\max}}{g_{\max}}$$
Substitute values:
$$S_{\max} = \frac{65{,}504.0}{1.8} \approx \mathbf{36{,}391.11}$$

#### 3. Allowable Range & Selection:
The allowable range for $S$ is:
$$406.67 \le S \le 36{,}391.11$$
A wide valid window exists!
Standard AMP implementations select powers of 2. We can choose:
$$S = 2^{10} = 1{,}024, \quad S = 2^{12} = 4{,}096, \quad \text{or } S = 2^{14} = 16{,}384$$
Choosing $S = 2^{14} = 16{,}384$ centers the scaled gradient distribution safely between $g_{\min} \cdot S \approx 2.46 \times 10^{-3}$ and $g_{\max} \cdot S \approx 29{,}491 < 65{,}504$.
</details>

---

## Exercise 3: Precision Regimes & Hardware Throughput Comparison
**Problem:**
You are training a 20-billion parameter model ($\Phi = 20 \times 10^9$) on a cluster of 32 NVIDIA H100 GPUs ($P_{\text{FP16}} = 989\text{ TFLOP/s}$, $P_{\text{FP8}} = 1{,}978\text{ TFLOP/s}$).
Batch size per step requires $1.2 \times 10^{14}\text{ FLOPs}$ of computation per GPU.

1. Calculate the weight storage memory (in GB) when weights are stored in FP32, FP16/BF16, and FP8.
2. Assuming compute-bound execution at $60\%$ Model FLOPs Utilization (MFU), calculate the step time for FP16 training vs FP8 training.
3. What is the theoretical speedup of FP8 over FP16?

<details>
<summary>Click to view Solution</summary>

### Solution:

#### 1. Weight Storage Memory:
- **FP32** ($4\text{ bytes/param}$): $20 \times 10^9 \times 4\text{ B} = \mathbf{80.0\text{ GB}}$
- **FP16 / BF16** ($2\text{ bytes/param}$): $20 \times 10^9 \times 2\text{ B} = \mathbf{40.0\text{ GB}}$ ($50\%$ savings)
- **FP8** ($1\text{ byte/param}$): $20 \times 10^9 \times 1\text{ B} = \mathbf{20.0\text{ GB}}$ ($75\%$ savings over FP32)

#### 2. Step Time Calculation at 60% MFU:
- **FP16 Training**:
  - Effective compute rate $= 0.60 \times 989\text{ TFLOP/s} = 593.4\text{ TFLOP/s} = 5.934 \times 10^{14}\text{ FLOP/s}$.
  - Step Time:
    $$T_{\text{FP16}} = \frac{1.2 \times 10^{14}\text{ FLOPs}}{5.934 \times 10^{14}\text{ FLOP/s}} \approx \mathbf{0.2022\text{ seconds}} = 202.2\text{ ms}$$
- **FP8 Training**:
  - Effective compute rate $= 0.60 \times 1{,}978\text{ TFLOP/s} = 1{,}186.8\text{ TFLOP/s} = 1.1868 \times 10^{15}\text{ FLOP/s}$.
  - Step Time:
    $$T_{\text{FP8}} = \frac{1.2 \times 10^{14}\text{ FLOPs}}{1.1868 \times 10^{15}\text{ FLOP/s}} \approx \mathbf{0.1011\text{ seconds}} = 101.1\text{ ms}$$

#### 3. Theoretical Speedup:
$$\text{Speedup} = \frac{T_{\text{FP16}}}{T_{\text{FP8}}} = \frac{202.2\text{ ms}}{101.1\text{ ms}} = \mathbf{2.0\times}$$
FP8 cuts step time in half while cutting weight and activation memory bandwidth in half.
</details>
