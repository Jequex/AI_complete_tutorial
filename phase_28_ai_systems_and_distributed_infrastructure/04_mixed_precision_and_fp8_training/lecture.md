# Topic 04: Mixed Precision & FP8 Training

## 1. Introduction & 12-Year-Old Intuition

Imagine you are an explorer mapping both giant mountain ranges and tiny grains of sand on a distant planet.

You have three measuring tapes:
1. **The Giant Engineering Tape (FP32)**: A 32-foot tape measure that has millimeters marked on it and extends all the way to outer space. It is incredibly accurate and never runs out of length, but it is heavy, clunky, and takes huge amounts of energy to carry around.
2. **The Pocket Ruler (FP16)**: An 8-inch pocket ruler. It is lightweight, fast to pull out, and lets you work $4\times$ faster. But it has two serious flaws:
   - *It is too short*: Any number larger than $65{,}504$ falls off the end and snaps the ruler (**Overflow / Infinity**).
   - *It cannot measure tiny specks*: Any speck smaller than a thousandth of an inch is rounded down to absolute zero (**Underflow**). When training a deep neural network, millions of tiny gradient nudges fall through the cracks and vanish into thin air!
3. **The Telescope Ruler (BF16)**: Someone took the 32-foot engineering tape, kept its full outer-space reach, but erased some of the finest millimeter markings. You can measure both galaxies and microscopic bugs without breaking the ruler or vanishing into zero.

**Mixed Precision Training** gives us the best of both worlds: we do the heavy arithmetic using lightweight pocket rulers to run at lightning speed on Tensor Cores, while keeping a master copy on the 32-foot engineering tape so no precision is permanently lost.

---

## 2. Mathematical Symbol Decoder

| Symbol | Concept / Metric | Formal Definition | Physical / Numerical Meaning |
| :--- | :--- | :--- | :--- |
| $\text{sign}$ | Sign Bit | $s \in \{0, 1\}$ | Indicates positive ($0$) or negative ($1$) numerical value ($(-1)^s$). |
| $e$ | Exponent Bits | Width in bits | Determines the dynamic range (scale order of magnitude). |
| $\text{bias}$ | Exponent Bias | $2^{e-1} - 1$ | Constant offset centered around zero to represent negative powers of 2. |
| $m$ | Mantissa (Fraction) Bits | Width in bits | Determines the numerical precision (significant digits). |
| $\epsilon_{\text{mach}}$ | Machine Epsilon | $2^{-m}$ | Smallest difference such that $1.0 + \epsilon_{\text{mach}} \neq 1.0$ in the format. |
| $S$ | Loss Scale Factor | Scalar (e.g. $2^{16} = 65{,}536$) | Constant multiplier applied to the loss to shift tiny gradients away from zero underflow. |
| $\mathbf{g}_{\text{scaled}}$ | Scaled Gradients | $S \cdot \nabla_\theta \mathcal{L}$ | Gradients scaled into the representable normal range of FP16. |
| $\text{FTZ}$ | Flush-To-Zero | Hardware Flag | Hardware optimization that forces subnormal numbers to exact $0.0$ for speed. |

---

## 3. Anatomy of Floating-Point Representations

Every standard IEEE binary floating-point number is expressed mathematically as:
$$x = (-1)^s \times 2^{E - \text{bias}} \times \left(1 + \sum_{i=1}^m b_i 2^{-i}\right)$$
where $s$ is the sign bit, $E$ is the raw unsigned exponent integer, and $b_i \in \{0, 1\}$ are the mantissa bits.

```
FP32 (Single Precision, 4 bytes):
+---+----------------+-----------------------------------------------+
| s |   8-bit Exp    |              23-bit Mantissa                  |
+---+----------------+-----------------------------------------------+

FP16 (Half Precision, 2 bytes):
+---+--------+--------------------+
| s | 5-bit  |   10-bit Mantissa  |
+---+--------+--------------------+

BF16 (Bfloat16, 2 bytes):
+---+----------------+--------------+
| s |   8-bit Exp    | 7-bit Mant   |
+---+----------------+--------------+

FP8 E4M3 (1 byte):       FP8 E5M2 (1 byte):
+---+------+-----+       +---+--------+----+
| s | 4-b  | 3-b |       | s | 5-bit  | 2-b|
+---+------+-----+       +---+--------+----+
```

### Comprehensive Format Comparison Table:
| Format | Total Bits | Bits $(s / e / m)$ | Exponent Bias | Smallest Subnormal | Smallest Normal | Maximum Finite Value | Precision (Decimals) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FP32** | 32 | $1 \,/\, 8 \,/\, 23$ | 127 | $1.40 \times 10^{-45}$ | $1.18 \times 10^{-38}$ | $3.40 \times 10^{38}$ | $\sim 7$ digits |
| **FP16** | 16 | $1 \,/\, 5 \,/\, 10$ | 15 | $5.96 \times 10^{-8}$ | $6.10 \times 10^{-5}$ | $\mathbf{65{,}504}$ | $\sim 3.3$ digits |
| **BF16** | 16 | $1 \,/\, 8 \,/\, 7$ | 127 | $9.18 \times 10^{-41}$ | $1.18 \times 10^{-38}$ | $\mathbf{3.39 \times 10^{38}}$ | $\sim 2.1$ digits |
| **FP8-E4M3** | 8 | $1 \,/\, 4 \,/\, 3$ | 7 | $1.95 \times 10^{-3}$ | $1.56 \times 10^{-2}$ | $\mathbf{448}$ | $\sim 1.0$ digit |
| **FP8-E5M2** | 8 | $1 \,/\, 5 \,/\, 2$ | 15 | $1.53 \times 10^{-5}$ | $6.10 \times 10^{-5}$ | $\mathbf{57{,}344}$ | $\sim 0.7$ digits |

---

## 4. The Gradient Underflow Catastrophe in FP16

In deep neural networks (e.g. 80-layer Transformers), backpropagation applies the chain rule repeatedly:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_1} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}_L} \prod_{l=2}^L \mathbf{W}_l^T \dots$$
Because weights and activation derivatives typically have magnitudes slightly below $1.0$, multiplying hundreds of matrices causes gradients in early layers to shrink exponentially into ranges like $10^{-8}$ to $10^{-6}$.

### The Hardware Trap:
1. **Normal Range Minimum**: In FP16, the smallest normalized number is $2^{-14} \approx 6.10 \times 10^{-5}$.
2. **Subnormal Range Minimum**: The smallest subnormal number is $2^{-24} \approx 5.96 \times 10^{-8}$.
3. **Flush-To-Zero (FTZ)**: On modern GPU Tensor Cores, subnormal numbers incur performance penalties, so GPUs run in Flush-To-Zero mode by default.

Any gradient with $|g| < 6.10 \times 10^{-5}$ (or $< 5.96 \times 10^{-8}$) **flushes to exact $0.0$**!
In an unscaled FP16 training run, **over $70\%$ to $80\%$ of all backward gradients underflow to zero**. The optimizer receives zero updates for the majority of the model weights, and training grinds to an immediate halt.

---

## 5. Automatic Mixed Precision (AMP) with Dynamic Loss Scaling

To prevent gradient underflow while exploiting fast FP16 Tensor Cores, Micikevicius et al. (2018) introduced **Automatic Mixed Precision (AMP)** with **Dynamic Loss Scaling**.

```
                           [ AMP TRAINING LOOP ]
                           
      [ Master Weights: FP32 ] ------------+
                 | (Downcast to FP16)      |
                 v                         |
        [ Weights: FP16 ]                  |
                 |                         |
          Forward Pass                     |
                 v                         |
         [ Loss L: FP32 ]                  |
                 |                         |
         Multiply by S                     |
                 v                         |
      [ Scaled Loss L_scaled ]             |
                 |                         |
        Backward Pass (FP16)               |
                 v                         |
    [ Scaled Gradients g_scaled ]          |
                 |                         |
           Check Inf / NaN                 |
          /               \                |
     (Has Inf/NaN)       (Clean)           |
        /                   \              |
   Skip Step!            Unscale: g / S    |
   Shrink S = S * 0.5       v              v
                     [ Optimizer Step: Adam (FP32) ]
                            |
                     Update Master Weights
```

### The Mathematical Mechanism:
By multiplying the loss $\mathcal{L}$ by a large scaling factor $S$ (e.g. $S = 2^{16} = 65{,}536$):
$$\mathcal{L}_{\text{scaled}} = S \cdot \mathcal{L}$$
By linearity of differentiation, the backward gradients are also scaled by $S$:
$$\mathbf{g}_{\text{scaled}} = \nabla_\theta \mathcal{L}_{\text{scaled}} = S \cdot \nabla_\theta \mathcal{L}$$

Multiplying by $65{,}536$ shifts the entire exponent distribution to the right by $\log_2(65{,}536) = 16\text{ bits}$! A tiny gradient of $10^{-8}$ becomes:
$$10^{-8} \times 65{,}536 \approx 6.55 \times 10^{-4}$$
which falls safely inside the normal representable range of FP16 ($> 6.10 \times 10^{-5}$). Zero underflow occurs!

### Dynamic Scaling Controller Algorithm:
1. **Initialize**: Set $S = 65{,}536$, counter $c = 0$, threshold $K = 2{,}000$.
2. **Execute Scaled Step**:
   $$\mathbf{g}_{\text{scaled}} = \text{backward}(S \cdot \mathcal{L})$$
3. **Inspect for Overflow**:
   If any element in $\mathbf{g}_{\text{scaled}}$ is $\pm \infty$ or $\text{NaN}$:
   - **Discard update**: Skip optimizer step entirely ($\theta_{t+1} = \theta_t$).
   - **Backoff scale factor**: $S \leftarrow \max(1, S \times 0.5)$.
   - **Reset counter**: $c \leftarrow 0$.
4. **Clean Update**:
   If all gradients are finite:
   - **Unscale**: $\mathbf{g}_{\text{FP32}} = \frac{\mathbf{g}_{\text{scaled}}}{S}$.
   - **Update Optimizer**: Run FP32 Adam using $\mathbf{g}_{\text{FP32}}$ on master FP32 weights.
   - **Increment Counter**: $c \leftarrow c + 1$.
   - **Growth Check**: If $c \ge K$:
     $$S \leftarrow S \times 2, \quad c \leftarrow 0$$

---

## 6. BF16: The Modern Industry Standard for LLMs

Google Brain recognized that the fundamental flaw of FP16 was its cramped 5-bit exponent. They created **BF16 (Bfloat16)**:
- Keeps the exact **8 exponent bits** of FP32.
- Shortens the mantissa to **7 bits**.

### Why BF16 Replaced FP16 in Modern LLM Pretraining:
1. **Identical Dynamic Range to FP32**: BF16 can represent values from $\approx 10^{-38}$ up to $\approx 3.39 \times 10^{38}$.
2. **Zero Underflow**: Tiny gradients never underflow.
3. **No Loss Scaling Needed**: Dynamic loss scaling code, scale checks, overflow skips, and Inf/NaN branches are completely eliminated.
4. **Tolerance of Coarse Precision**: Deep learning models act like stochastic ensembles; losing 3 bits of mantissa precision acts as a slight regularizer and does not degrade validation perplexity.
5. Supported natively on all modern hardware: Google TPU v2–v5, NVIDIA Ampere (A100), Hopper (H100), Blackwell (B200).

---

## 7. The FP8 Standard: OCP E4M3 vs. E5M2

The Open Compute Project (OCP) and NVIDIA Hopper architecture standardized two distinct 8-bit floating point formats:

```
+-----------------------------------------------------------------------------------+
| THE FP8 SPECIALIZED DUO                                                           |
+-----------------------------------------------------------------------------------+
| 1. FP8 E4M3 (1 sign, 4 exponent, 3 mantissa)                                      |
|    - Max value: 448                                                               |
|    - Higher precision (3 mantissa bits)                                           |
|    - USED FOR: Forward Pass Weights and Activations (needs high precision)        |
+-----------------------------------------------------------------------------------+
| 2. FP8 E5M2 (1 sign, 5 exponent, 2 mantissa)                                      |
|    - Max value: 57,344                                                            |
|    - Massive dynamic range (matches FP16's 5-bit exponent)                        |
|    - USED FOR: Backward Pass Gradients (needs wide dynamic range to prevent inf)  |
+-----------------------------------------------------------------------------------+
```

### Why Not Use E4M3 for Everything?
Gradients have high variance and large outlier spikes. The maximum representable number in E4M3 is only **448**; a gradient spike easily overflows 448, producing an immediate `inf` crash. Therefore, backward gradients require the 5-bit exponent of **E5M2** (max 57,344).

### Delayed Scaling in FP8:
Because 8-bit dynamic range is narrow, tensors are scaled by a per-tensor or per-block scaling factor $\alpha$:
$$\mathbf{X}_{\text{FP8}} = \text{quantize}(\alpha \cdot \mathbf{X})$$
To avoid scanning the entire tensor before every GEMM (which would incur memory bandwidth overhead), frameworks use **Delayed Scaling**: the maximum absolute value from step $t-1$ is used to compute the scale factor for step $t$.
