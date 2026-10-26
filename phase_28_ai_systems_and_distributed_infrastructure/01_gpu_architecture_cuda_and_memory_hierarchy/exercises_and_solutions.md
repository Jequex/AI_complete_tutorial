# Topic 01: Exercises & Solutions — GPU Architecture & Roofline Model

## Exercise 1: Arithmetic Intensity of the Softmax Operator
**Problem:**
Consider computing the standard Softmax function over a batch of attention vectors:
$$\mathbf{y} = \text{Softmax}(\mathbf{x}), \quad \mathbf{x} \in \mathbb{R}^{B \times S}$$
where $B = 32$ and $S = 4{,}096$. All values are stored in FP16 ($2\text{ bytes}$ per element).

The numerically stable 3-pass Softmax algorithm computes:
1. $m = \max_{j} x_j$ ($S$ comparisons)
2. $d = \sum_{j} \exp(x_j - m)$ ($S$ subtractions, $S$ exponentials, $S$ additions $\approx 3S\text{ FLOPs}$)
3. $y_j = \frac{\exp(x_j - m)}{d}$ ($S$ subtractions, $S$ exponentials, $S$ divisions $\approx 3S\text{ FLOPs}$)

Total FLOPs $\approx 7S$ per sequence.

1. Calculate the total FLOPs and total memory traffic (bytes read and written to HBM) for a naive, non-fused kernel that writes intermediate results back to HBM after each pass.
2. Calculate the total memory traffic for an optimally fused kernel that holds the vector in on-chip SRAM / Registers.
3. Compute the Arithmetic Intensity $I$ for both implementations. On an NVIDIA A100 SXM4 ($P_{\text{peak}} = 312\text{ TFLOP/s}$, $B_{\text{peak}} = 2{,}039\text{ GB/s}$, $I^* = 153.0\text{ FLOP/B}$), classify the execution regime and compute the attainable TFLOP/s.

<details>
<summary>Click to view Solution</summary>

### Solution:

#### 1. Naive (Unfused) Implementation:
- **Total Elements:** $N = B \times S = 32 \times 4{,}096 = 131{,}072\text{ elements}$.
- **Total FLOPs:** $7 \times N = 7 \times 131{,}072 = 917{,}504\text{ FLOPs} \approx 0.918\text{ MFLOPs}$.
- **Naive Memory Traffic:**
  - Pass 1 (max): Read $\mathbf{x}$ ($2N\text{ bytes}$).
  - Pass 2 (sum of exp): Read $\mathbf{x}$ ($2N\text{ bytes}$), write intermediate numerators ($2N\text{ bytes}$).
  - Pass 3 (normalize): Read intermediate numerators ($2N\text{ bytes}$), write $\mathbf{y}$ ($2N\text{ bytes}$).
  - Total Memory $= 2N + 2N + 2N + 2N + 2N = 10N\text{ bytes} = 10 \times 131{,}072 \times 2 = 1{,}310{,}720\text{ bytes} \approx 1.31\text{ MB}$.
- **Arithmetic Intensity (Naive):**
  $$I_{\text{naive}} = \frac{7N}{10N \times 2} = \frac{7}{20} = 0.35\text{ FLOPs/Byte}$$

#### 2. Fused Implementation:
- **Fused Memory Traffic:**
  - Read $\mathbf{x}$ once from HBM into SRAM/Registers: $2N\text{ bytes}$.
  - Write output $\mathbf{y}$ once back to HBM: $2N\text{ bytes}$.
  - Total Memory $= 4N\text{ bytes} = 4 \times 131{,}072 \times 2 = 524{,}288\text{ bytes} \approx 0.524\text{ MB}$.
- **Arithmetic Intensity (Fused):**
  $$I_{\text{fused}} = \frac{7N}{4N \times 2} = \frac{7}{8} = 0.875\text{ FLOPs/Byte}$$

#### 3. Roofline Classification & Attainable Performance (NVIDIA A100):
- A100 Ridge point: $I^* = 153.0\text{ FLOP/Byte}$.
- Since both $I_{\text{naive}} = 0.35$ and $I_{\text{fused}} = 0.875 \ll 153.0$, Softmax is **deeply memory-bound** in both cases.
- **Attainable Performance (Naive):**
  $$P_{\text{naive}} = I_{\text{naive}} \cdot B_{\text{peak}} = 0.35 \times 2{,}039\text{ GB/s} = 713.65\text{ GFLOP/s} = 0.714\text{ TFLOP/s}$$
  This is $\frac{0.714}{312} \approx \mathbf{0.23\%}$ of peak compute.
- **Attainable Performance (Fused):**
  $$P_{\text{fused}} = I_{\text{fused}} \cdot B_{\text{peak}} = 0.875 \times 2{,}039\text{ GB/s} = 1{,}784.1\text{ GFLOP/s} = 1.784\text{ TFLOP/s}$$
  Kernel fusion yields a $\frac{10N}{4N} = \mathbf{2.5\times}$ direct speedup purely by saving memory bandwidth.
</details>

---

## Exercise 2: Warp Divergence Penalty
**Problem:**
A CUDA kernel contains a branch:
```cuda
if (threadIdx.x % 2 == 0) {
    // Branch Even: takes 120 instructions
    path_even();
} else {
    // Branch Odd: takes 80 instructions
    path_odd();
}
```
Suppose the warp contains 32 threads (`threadIdx.x` from $0$ to $31$).

1. How many clock cycles does the warp require to complete execution?
2. What is the active SIMT thread slot efficiency (ratio of useful work to total instruction slots)?
3. How could you restructure the thread mapping so that warp divergence is completely eliminated?

<details>
<summary>Click to view Solution</summary>

### Solution:

#### 1. Required Execution Cycles:
Since adjacent threads (`0, 1, 2, ...`) alternate even and odd, every single warp has exactly 16 even threads and 16 odd threads.
- The warp cannot branch independently.
- Phase 1: Executes `path_even()` with 16 active threads, 16 masked off. Takes $120\text{ cycles}$.
- Phase 2: Executes `path_odd()` with 16 active threads, 16 masked off. Takes $80\text{ cycles}$.
- Total Cycles $= 120 + 80 = \mathbf{200\text{ cycles}}$.

#### 2. Active SIMT Efficiency:
- Useful instruction slots:
  $$(16\text{ threads} \times 120) + (16\text{ threads} \times 80) = 1{,}920 + 1{,}280 = 3{,}200\text{ thread-instructions}$$
- Total issued instruction slots:
  $$32\text{ threads} \times 200\text{ cycles} = 6{,}400\text{ slot-cycles}$$
- SIMT Efficiency:
  $$\eta = \frac{3{,}200}{6{,}400} = \mathbf{50.0\%}$$

#### 3. Restructuring to Eliminate Divergence:
Divergence occurs because threads within the same warp take different branches. We can restructure the mapping so that contiguous blocks of 32 threads take the identical branch:
$$\text{bool is\_even\_warp} = (\text{threadIdx.x} / 32) \% 2 == 0;$$
Or sort the input tasks so that all even tasks are processed by warps $0, 1, \dots, k$ and all odd tasks by warps $k+1, \dots$.
In this case:
- Even warps take $120\text{ cycles}$ with $100\%$ efficiency.
- Odd warps take $80\text{ cycles}$ with $100\%$ efficiency.
- Average execution time per warp pair is $\frac{120 + 80}{2} = 100\text{ cycles}$ (a **$2\times$ speedup** over $200\text{ cycles}$).
</details>

---

## Exercise 3: Tensor Core MMA Tile Calculation
**Problem:**
An NVIDIA H100 SXM5 GPU features 132 SMs running at a base clock of $1.83\text{ GHz}$ ($1.83 \times 10^9\text{ cycles/s}$).
Each SM contains 4 Fourth-Generation Tensor Cores.
Each Fourth-Gen Tensor Core can perform a $16 \times 16 \times 16$ FP16 matrix multiply-accumulate (MMA) tile per clock cycle.

1. How many scalar floating-point multiply-accumulate operations (FLOPs) are in a single $16 \times 16 \times 16$ MMA operation ($\mathbf{D} = \mathbf{A} \cdot \mathbf{B} + \mathbf{C}$)?
2. Derive the theoretical peak FP16 Tensor Core throughput in TFLOP/s for the entire H100 GPU from first principles.

<details>
<summary>Click to view Solution</summary>

### Solution:

#### 1. FLOPs in one $16 \times 16 \times 16$ tile:
- Multiplying $\mathbf{A} \in \mathbb{R}^{16 \times 16}$ by $\mathbf{B} \in \mathbb{R}^{16 \times 16}$ produces $\mathbf{C} \in \mathbb{R}^{16 \times 16}$ (which has $16 \times 16 = 256$ elements).
- Each output element is the dot product of a 16-element row and a 16-element column:
  - 16 multiplications
  - 15 additions (or 16 additions including accumulating with $\mathbf{C}_{i,j}$)
  - Total FLOPs $= 2 \times 16 \times 16 \times 16 = \mathbf{8{,}192\text{ FLOPs}}$ per tile.

#### 2. Derivation of Peak TFLOP/s:
- Total Tensor Cores on GPU $= 132\text{ SMs} \times 4\text{ Tensor Cores/SM} = 528\text{ Tensor Cores}$.
- Operations per cycle across the GPU:
  $$528 \times 8{,}192\text{ FLOPs} = 4{,}325{,}376\text{ FLOPs/cycle}$$
- At $1.83\text{ GHz}$:
  $$\text{Throughput} = 4{,}325{,}376\text{ FLOPs/cycle} \times 1.83 \times 10^9\text{ cycles/s} \approx 7.915 \times 10^{14}\text{ FLOPs/s}$$
  $$\approx \mathbf{791.5\text{ TFLOP/s}} \text{ (dense non-sparse without boost)}$$
- (With boost clock $\sim 1.98\text{ GHz}$ and architectural pipeline optimizations, official peak dense FP16 reaches $\mathbf{989\text{ TFLOP/s}}$, and with 2:4 structured sparsity reaches $\mathbf{1{,}978\text{ TFLOP/s}}$).
</details>
