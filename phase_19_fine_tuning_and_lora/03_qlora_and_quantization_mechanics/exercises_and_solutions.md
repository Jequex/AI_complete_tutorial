# Topic 19.3: Exercises & Solutions — QLoRA & Quantization Mechanics

---

### Problem 1: Manual Symmetric Uniform INT8 and INT4 Quantization (Easy)

Consider a small block of 4 weights from a neural network layer:
$$\mathbf{w} = [-0.12, \; 0.04, \; -0.28, \; 0.56]$$

**Tasks:**
1. Compute the symmetric INT8 scale factor $S_8$ (with range $[-127, 127]$).
2. Quantize $\mathbf{w}$ into discrete 8-bit integers $\mathbf{q}_8$.
3. Dequantize $\mathbf{q}_8$ to obtain reconstructed weights $\tilde{\mathbf{w}}_8$, and compute the Mean Squared Error (MSE).
4. Compute the symmetric INT4 scale factor $S_4$ (with range $[-8, 7]$).
5. Quantize $\mathbf{w}$ into discrete 4-bit integers $\mathbf{q}_4$, dequantize to $\tilde{\mathbf{w}}_4$, and compute the MSE.

<details>
<summary>View Complete Solution</summary>

**Step 1: Symmetric INT8 Quantization**
The maximum absolute value is:
$$\alpha_{\max} = \max(|-0.12|, |0.04|, |-0.28|, |0.56|) = 0.56$$

The scale factor $S_8$ is:
$$S_8 = \frac{\alpha_{\max}}{127} = \frac{0.56}{127} \approx 0.00440945$$

Quantization formula: $q_i = \operatorname{round}(w_i / S_8)$:
- $w_1 = -0.12 \implies -0.12 / 0.00440945 \approx -27.214 \implies q_1 = -27$
- $w_2 = 0.04 \implies 0.04 / 0.00440945 \approx 9.071 \implies q_2 = 9$
- $w_3 = -0.28 \implies -0.28 / 0.00440945 \approx -63.500 \implies q_3 = -64$ (or $-63$)
- $w_4 = 0.56 \implies 0.56 / 0.00440945 = 127.0 \implies q_4 = 127$

$$\mathbf{q}_8 = [-27, \; 9, \; -64, \; 127]$$

Dequantization $\tilde{w}_i = q_i \times S_8$:
- $\tilde{w}_1 = -27 \times 0.00440945 \approx -0.119055$ (error: $-0.000945$)
- $\tilde{w}_2 = 9 \times 0.00440945 \approx 0.039685$ (error: $+0.000315$)
- $\tilde{w}_3 = -64 \times 0.00440945 \approx -0.282205$ (error: $+0.002205$)
- $\tilde{w}_4 = 127 \times 0.00440945 = 0.560000$ (error: $0.000000$)

$$\text{MSE}_8 = \frac{(-0.000945)^2 + (0.000315)^2 + (0.002205)^2 + 0}{4} \approx \frac{5.855 \times 10^{-6}}{4} \approx \mathbf{1.46 \times 10^{-6}}$$

---

**Step 2: Symmetric INT4 Quantization**
For INT4, positive range is limited by $+7$:
$$S_4 = \frac{\alpha_{\max}}{7} = \frac{0.56}{7} = 0.080000$$

Quantization formula: $q_i = \operatorname{clip}(\operatorname{round}(w_i / S_4), -8, 7)$:
- $w_1 = -0.12 \implies -0.12 / 0.08 = -1.5 \implies q_1 = -2$ (or $-1$)
- $w_2 = 0.04 \implies 0.04 / 0.08 = 0.5 \implies q_2 = 1$
- $w_3 = -0.28 \implies -0.28 / 0.08 = -3.5 \implies q_3 = -4$
- $w_4 = 0.56 \implies 0.56 / 0.08 = 7.0 \implies q_4 = 7$

$$\mathbf{q}_4 = [-2, \; 1, \; -4, \; 7]$$

Dequantization $\tilde{w}_i = q_i \times S_4$:
- $\tilde{w}_1 = -2 \times 0.08 = -0.16$ (error: $-0.04$)
- $\tilde{w}_2 = 1 \times 0.08 = 0.08$ (error: $+0.04$)
- $\tilde{w}_3 = -4 \times 0.08 = -0.32$ (error: $-0.04$)
- $\tilde{w}_4 = 7 \times 0.08 = 0.56$ (error: $0.00$)

$$\text{MSE}_4 = \frac{(-0.04)^2 + (0.04)^2 + (-0.04)^2 + 0}{4} = \frac{0.0016 + 0.0016 + 0.0016}{4} = \mathbf{0.001200} = \mathbf{1.20 \times 10^{-3}}$$

*(Notice that INT4 error is roughly $800\times$ higher than INT8 due to having only 16 quantization bins instead of 256).*

</details>

---

### Problem 2: NormalFloat4 (NF4) Codebook Quantization (Medium)

Consider a normalized block where the absolute maximum is already scaled to $1.0$ ($c_1 = 1.0$).
Four normalized weight values are:
$$\hat{\mathbf{w}} = [0.120, \; -0.450, \; 0.700, \; -0.050]$$

Using the official 16-level NF4 codebook:
$$\mathcal{C} = \begin{bmatrix}
c_0 = -1.0, & c_1 = -0.6962, & c_2 = -0.5251, & c_3 = -0.3949, \\
c_4 = -0.2844, & c_5 = -0.1848, & c_6 = -0.0911, & c_7 = 0.0000, \\
c_8 = 0.0796, & c_9 = 0.1609, & c_{10} = 0.2461, & c_{11} = 0.3379, \\
c_{12} = 0.4407, & c_{13} = 0.5626, & c_{14} = 0.7230, & c_{15} = 1.0000
\end{bmatrix}$$

**Tasks:**
1. For each $\hat{w}_i$, determine the closest codebook entry $c_k$ and the corresponding 4-bit index $k \in \{0, \dots, 15\}$.
2. Calculate the reconstruction error for each weight and compute the total Mean Squared Error $\text{MSE}_{\text{NF4}}$.
3. Compare with uniform 4-bit quantization on the interval $[-1, 1]$ (step size $\Delta = 2 / 15 \approx 0.1333$), demonstrating why NF4 preserves higher fidelity near the center.

<details>
<summary>View Complete Solution</summary>

**Step 1: Finding Nearest NF4 Codebook Entries**
- $\hat{w}_1 = 0.120$:
  Compare distances to neighbors:
  - $|0.120 - c_8| = |0.120 - 0.0796| = 0.0404$
  - $|0.120 - c_9| = |0.120 - 0.1609| = 0.0409$
  Closest is $c_8 = 0.0796$ (Index 8).
  Reconstructed: $\tilde{w}_1 = 0.0796$. Error: $0.120 - 0.0796 = 0.0404$.

- $\hat{w}_2 = -0.450$:
  Compare distances:
  - $|-0.450 - c_2| = |-0.450 - (-0.5251)| = 0.0751$
  - $|-0.450 - c_3| = |-0.450 - (-0.3949)| = 0.0551$
  Closest is $c_3 = -0.3949$ (Index 3).
  Reconstructed: $\tilde{w}_2 = -0.3949$. Error: $-0.450 - (-0.3949) = -0.0551$.

- $\hat{w}_3 = 0.700$:
  Compare distances:
  - $|0.700 - c_{13}| = |0.700 - 0.5626| = 0.1374$
  - $|0.700 - c_{14}| = |0.700 - 0.7230| = 0.0230$
  Closest is $c_{14} = 0.7230$ (Index 14).
  Reconstructed: $\tilde{w}_3 = 0.7230$. Error: $0.700 - 0.7230 = -0.0230$.

- $\hat{w}_4 = -0.050$:
  Compare distances:
  - $|-0.050 - c_6| = |-0.050 - (-0.0911)| = 0.0411$
  - $|-0.050 - c_7| = |-0.050 - 0.0000| = 0.0500$
  Closest is $c_6 = -0.0911$ (Index 6).
  Reconstructed: $\tilde{w}_4 = -0.0911$. Error: $-0.050 - (-0.0911) = 0.0411$.

Indices: $[8, \; 3, \; 14, \; 6]$.

**Step 2: Compute MSE under NF4**
$$\text{MSE}_{\text{NF4}} = \frac{(0.0404)^2 + (-0.0551)^2 + (-0.0230)^2 + (0.0411)^2}{4}$$
$$= \frac{0.001632 + 0.003036 + 0.000529 + 0.001689}{4} = \frac{0.006886}{4} \approx \mathbf{0.001722}$$

**Step 3: Comparison with Uniform 4-bit**
In a uniform 16-level grid over $[-1, 1]$, the bin spacing is $\Delta = \frac{2}{15} \approx 0.1333$.
The maximum quantization error in any bin is $\frac{\Delta}{2} \approx 0.0667$.
In contrast, in the central region of NF4 (between $-0.2$ and $+0.2$, where $>60\%$ of Gaussian weights fall):
- The distance between $c_7 (0.0)$ and $c_8 (0.0796)$ is only $0.0796$.
- The maximum quantization error is $\frac{0.0796}{2} \approx 0.0398$!
NF4 offers **nearly double the resolution** in the high-density central region of neural network weights.

</details>

---

### Problem 3: Double Quantization Memory Budget for a 70B Model (Hard)

An enterprise wants to fine-tune a LLaMA-3-70B model ($N = 70.6 \times 10^9$ parameters) on a single server.
Compare three storage and fine-tuning approaches for the static model components:

1. **Approach 1: Full 16-bit LoRA (Unquantized Base):**
   - Base weights in 16-bit FP16 ($16\text{ bits/param}$).
   - LoRA adapter parameters: $0.2\%$ of base model ($M = 0.002 N$) trained with FP32 AdamW ($16\text{ bytes/param}$).

2. **Approach 2: Standard 4-bit QLoRA without Double Quantization:**
   - Base weights in 4-bit NF4 ($4\text{ bits/param}$).
   - Stage 1 scales in FP32 ($32\text{ bits}$) with block size $B_1 = 64$.
   - Same LoRA adapter ($0.2\%$, $16\text{ bytes/param}$).

3. **Approach 3: Modern QLoRA with Double Quantization:**
   - Base weights in 4-bit NF4 ($4\text{ bits/param}$).
   - Double Quantization with $B_1 = 64$ and $B_2 = 256$ ($0.127\text{ bits/param}$).
   - Same LoRA adapter ($0.2\%$, $16\text{ bytes/param}$).

Calculate the total static VRAM (in Gigabytes, $1\text{ GB} = 1024^3\text{ bytes}$) for all three approaches, and determine whether Approach 3 can fit within a single NVIDIA A100/H100 80GB GPU.

<details>
<summary>View Complete Solution</summary>

**Common LoRA Adapter Memory:**
$$M = 0.002 \times 70.6 \times 10^9 = 141.2 \times 10^6 \text{ parameters}$$
Adapter weights + FP16 grads + FP32 master weights + AdamW states $= 16\text{ bytes/param}$:
$$\text{Memory}_{\text{adapter}} = 141.2 \times 10^6 \times 16\text{ bytes} = 2.2592 \times 10^9\text{ bytes} \approx \frac{2.2592 \times 10^9}{1024^3} \approx \mathbf{2.10\text{ GB}}$$

---

**Approach 1: Unquantized FP16 Base + LoRA**
- Base Weights: $70.6 \times 10^9 \times 2\text{ bytes} = 141.2 \times 10^9\text{ bytes} \approx \mathbf{131.50\text{ GB}}$
- Total Static Memory:
  $$\text{Total} = 131.50 + 2.10 = \mathbf{133.60\text{ GB}}$$
*(Cannot fit on any single GPU; requires at least two 80GB GPUs).*

---

**Approach 2: 4-bit QLoRA without Double Quantization**
- Base 4-bit Weights:
  $$\text{Bits} = 70.6 \times 10^9 \times 4 = 282.4 \times 10^9\text{ bits} = 35.3 \times 10^9\text{ bytes} \approx \mathbf{32.88\text{ GB}}$$
- Single Quantization Scales ($B_1 = 64$, 32-bit float scale per block):
  $$\text{Bits} = 70.6 \times 10^9 \times \frac{32}{64} = 70.6 \times 10^9 \times 0.5 = 35.3 \times 10^9\text{ bits} = 4.4125 \times 10^9\text{ bytes} \approx \mathbf{4.11\text{ GB}}$$
- Total Static Memory:
  $$\text{Total} = 32.88 + 4.11 + 2.10 = \mathbf{39.09\text{ GB}}$$

---

**Approach 3: 4-bit QLoRA with Double Quantization**
- Base 4-bit Weights: $\mathbf{32.88\text{ GB}}$
- Double Quantization Scales ($0.12695\text{ bits/param}$):
  $$\text{Bits} = 70.6 \times 10^9 \times 0.126953 = 8.9629 \times 10^9\text{ bits} = 1.12036 \times 10^9\text{ bytes} \approx \mathbf{1.04\text{ GB}}$$
  *(Saves $4.11 - 1.04 = \mathbf{3.07\text{ GB}}$ solely on quantization scales!)*
- Total Static Memory:
  $$\text{Total} = 32.88 + 1.04 + 2.10 = \mathbf{36.02\text{ GB}}$$

---

**Hardware Feasibility Conclusion:**
Total static memory of **$36.02\text{ GB}$** leaves over **$43.98\text{ GB}$** of free VRAM on an 80GB GPU.
This abundant margin easily accommodates dynamic activations, KV cache, and batch size scaling for sequence lengths up to 4096 tokens.
Thus, **Approach 3 allows fine-tuning a 70B parameter model on a single 80GB GPU**, whereas standard full fine-tuning would have required a multi-node 16-GPU supercomputing cluster.

</details>
