# Topic 15.4: Exercises & Solutions — Seq2Seq & Bahdanau Attention

---

### Problem 1: Manual Calculation of Bahdanau Attention Distribution (Hard)

Consider a decoder generating target token $i=1$.
The previous decoder state is:
$$s_0 = [1.0, 0.0]^T \in \mathbb{R}^2$$

The source sentence has length $T_x = 3$, with encoder annotations:
$$h_1 = [1.0, 2.0]^T, \quad h_2 = [0.0, 1.0]^T, \quad h_3 = [-1.0, 0.0]^T$$

The Bahdanau Attention parameters are:
- Query projection: $W_a = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix}$
- Key projection: $U_a = \begin{bmatrix} 0.5 & 0.0 \\ 0.0 & 0.5 \end{bmatrix}$
- Bias: $b_a = [0.0, 0.0]^T$
- Scoring vector: $v_a = [1.0, 1.0]^T$

Useful values:
- $\tanh(1.5) \approx 0.9051, \quad \tanh(1.0) \approx 0.7616$
- $\tanh(0.5) \approx 0.4621, \quad \tanh(0.0) = 0.0$
- $e^{1.6667} \approx 5.2947, \quad e^{1.2237} \approx 3.3997, \quad e^{0.4621} \approx 1.5874$

**Questions:**
1. Compute the alignment energy scores $e_{1,1}, e_{1,2}, e_{1,3}$.
2. Compute the attention probability weights $\alpha_{1,1}, \alpha_{1,2}, \alpha_{1,3}$. Verify that they sum to $1.0$.
3. Compute the resulting dynamic context vector $c_1 \in \mathbb{R}^2$.

---

#### Solution

1. **Alignment Energy Calculation:**
   First compute the projected query $W_a s_0$:
   $$W_a s_0 = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix} \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix} = \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix}$$

   Now compute for each source position $j \in \{1, 2, 3\}$:
   - **For $j=1$ ($h_1 = [1.0, 2.0]^T$):**
     $$U_a h_1 = \begin{bmatrix} 0.5(1.0) \\ 0.5(2.0) \end{bmatrix} = \begin{bmatrix} 0.5 \\ 1.0 \end{bmatrix}$$
     $$a_{1,1} = W_a s_0 + U_a h_1 = \begin{bmatrix} 1.0 + 0.5 \\ 0.0 + 1.0 \end{bmatrix} = \begin{bmatrix} 1.5 \\ 1.0 \end{bmatrix}$$
     $$\tanh(a_{1,1}) = [\tanh(1.5), \tanh(1.0)]^T \approx [0.9051, 0.7616]^T$$
     $$e_{1,1} = v_a^T \tanh(a_{1,1}) = 1.0(0.9051) + 1.0(0.7616) \approx 1.6667$$

   - **For $j=2$ ($h_2 = [0.0, 1.0]^T$):**
     $$U_a h_2 = \begin{bmatrix} 0.0 \\ 0.5 \end{bmatrix} \implies a_{1,2} = \begin{bmatrix} 1.0 + 0.0 \\ 0.0 + 0.5 \end{bmatrix} = \begin{bmatrix} 1.0 \\ 0.5 \end{bmatrix}$$
     $$\tanh(a_{1,2}) = [\tanh(1.0), \tanh(0.5)]^T \approx [0.7616, 0.4621]^T$$
     $$e_{1,2} = 1.0(0.7616) + 1.0(0.4621) \approx 1.2237$$

   - **For $j=3$ ($h_3 = [-1.0, 0.0]^T$):**
     $$U_a h_3 = \begin{bmatrix} -0.5 \\ 0.0 \end{bmatrix} \implies a_{1,3} = \begin{bmatrix} 1.0 - 0.5 \\ 0.0 + 0.0 \end{bmatrix} = \begin{bmatrix} 0.5 \\ 0.0 \end{bmatrix}$$
     $$\tanh(a_{1,3}) = [\tanh(0.5), \tanh(0.0)]^T \approx [0.4621, 0.0]^T$$
     $$e_{1,3} = 1.0(0.4621) + 1.0(0.0) = 0.4621$$

2. **Softmax Attention Weights:**
   Denominator sum:
   $$\sum_{k=1}^3 \exp(e_{1,k}) = \exp(1.6667) + \exp(1.2237) + \exp(0.4621) \approx 5.2947 + 3.3997 + 1.5874 = 10.2818$$
   - $\alpha_{1,1} = \frac{5.2947}{10.2818} \approx 0.5149$
   - $\alpha_{1,2} = \frac{3.3997}{10.2818} \approx 0.3307$
   - $\alpha_{1,3} = \frac{1.5874}{10.2818} \approx 0.1544$
   Verification: $0.5149 + 0.3307 + 0.1544 = 1.0000$ (Exact sum to 1.0).

3. **Dynamic Context Vector $c_1$:**
   $$c_1 = \alpha_{1,1} h_1 + \alpha_{1,2} h_2 + \alpha_{1,3} h_3$$
   $$c_1 = 0.5149 \begin{bmatrix} 1.0 \\ 2.0 \end{bmatrix} + 0.3307 \begin{bmatrix} 0.0 \\ 1.0 \end{bmatrix} + 0.1544 \begin{bmatrix} -1.0 \\ 0.0 \end{bmatrix}$$
   First coordinate:
   $$c_{1,1} = 0.5149(1.0) + 0.3307(0.0) + 0.1544(-1.0) = 0.5149 - 0.1544 = 0.3605$$
   Second coordinate:
   $$c_{1,2} = 0.5149(2.0) + 0.3307(1.0) + 0.1544(0.0) = 1.0298 + 0.3307 = 1.3605$$
   $$\boxed{c_1 = \begin{bmatrix} 0.3605 \\ 1.3605 \end{bmatrix}}$$

---

### Problem 2: Additive Attention (Bahdanau) vs Dot-Product Attention (Luong) (Medium)

In 2015, Minh-Thang Luong et al. proposed Multiplicative (Dot-Product) Attention as an alternative to Bahdanau Additive Attention:
- **Bahdanau (Additive):** $e_{ij} = v_a^T \tanh(W_a s_{i-1} + U_a h_j)$
- **Luong (Dot-Product):** $e_{ij} = s_i^T h_j$ (when dimensions match) or $e_{ij} = s_i^T W_a h_j$ (general).

**Questions:**
1. Compare the computational complexity (FLOPs) of evaluating the attention energies across all $T_x$ source tokens for a single decoder step under both formulations (assuming $d_s = d_h = d$).
2. Why did modern Transformers (Vaswani et al. 2017) adopt scaled dot-product attention over additive attention?

#### Solution

1. **Computational Comparison:**
   - **Bahdanau Additive Attention:**
     Evaluating $W_a s_{i-1}$ costs $d^2$ operations.
     Evaluating $U_a h_j$ for all $T_x$ tokens costs $T_x \cdot d^2$ operations (can be precomputed once per sentence).
     At each decoder step, computing vector sum $W s + U h_j$, elementwise $\tanh$, and inner product with $v_a$ costs $\mathcal{O}(T_x \cdot d)$ transcendental and multiply-add operations.
   - **Luong General Multiplicative Attention:**
     Computing $s_i^T (W_a h_j) = (s_i^T W_a) h_j$.
     Evaluating $s_i^T W_a$ costs $d^2$ operations.
     The $T_x$ dot products with $h_j$ reduce to a **single matrix-vector multiplication**: $(s_i^T W_a) H$, costing $T_x \cdot d$ FLOPs.
     Most importantly, multiplicative attention requires **no nonlinear $\tanh$ evaluations**!

2. **Why Transformers Adopted Dot-Product Attention:**
   Dot-product attention can be implemented entirely using highly optimized General Matrix Multiply (`GEMM`) operations on GPU tensor cores:
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
   A single batch matrix multiplication computes all query-key dot products in parallel without branch divergences or elementwise transcendental activation calls.

---

### Problem 3: Information Bottleneck Capacity Calculation (Easy)

Suppose a Seq2Seq model without attention uses an LSTM with hidden size $d_h = 512$ (using 32-bit floating point numbers).
1. What is the total storage size of the bottleneck context vector $c$ in bytes?
2. If an input sentence contains 100 words, and each word contains roughly 50 bits of syntactic, semantic, and named-entity information, how much source information is lost if compressed into vector $c$?

#### Solution

1. Vector size:
   $$\text{Bytes} = 512 \times 4\text{ bytes} = 2,048\text{ bytes} = 2\text{ KB}$$
   In bits: $2,048 \times 8 = 16,384\text{ bits}$.

2. Source information:
   $$\text{Source Information} = 100\text{ words} \times 50\text{ bits/word} = 5,000\text{ bits}$$
   While $5,000$ bits is theoretically within the raw float32 capacity ($16,384$ bits), continuous neural representations cannot achieve lossless Shannon packing. As word count scales to 500 or 1,000 words ($25,000 - 50,000$ bits), the information mathematically exceeds the vector's capacity, making loss of facts unavoidable. Bahdanau attention solves this by maintaining all $T_x$ representations dynamically.
