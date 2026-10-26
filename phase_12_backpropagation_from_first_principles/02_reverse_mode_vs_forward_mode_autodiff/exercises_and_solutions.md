# Topic 12.2: Exercises & Solutions — Reverse-Mode vs. Forward-Mode Autodiff

---

### Problem 1: Dual Number Manual Differentiation (Easy)

Consider the rational function:
$$f(x) = \frac{x^2 + 1}{x - 2}$$

1. Evaluate $f(x)$ at $x = 3 + 1\epsilon$ using dual number arithmetic ($\epsilon^2 = 0$).
2. Compute the ordinary derivative $f'(x)$ using the calculus quotient rule and evaluate at $x = 3$.
3. Verify that the coefficient of $\epsilon$ in Part 1 matches $f'(3)$ exactly.

#### Solution

1. **Dual Number Evaluation:**
   Substitute $x = 3 + \epsilon$:
   - **Numerator:**
     $$N = (3 + \epsilon)^2 + 1 = (9 + 6\epsilon + \epsilon^2) + 1 = 10 + 6\epsilon \quad (\text{since } \epsilon^2 = 0)$$
   - **Denominator:**
     $$D = (3 + \epsilon) - 2 = 1 + \epsilon$$
   - **Quotient Division:**
     $$\frac{N}{D} = \frac{10 + 6\epsilon}{1 + \epsilon}$$
     Multiply numerator and denominator by the dual conjugate $(1 - \epsilon)$:
     $$\frac{N}{D} = \frac{(10 + 6\epsilon)(1 - \epsilon)}{(1 + \epsilon)(1 - \epsilon)} = \frac{10 - 10\epsilon + 6\epsilon - 6\epsilon^2}{1 - \epsilon^2} = \frac{10 - 4\epsilon}{1} = 10 - 4\epsilon$$
   Thus, $f(3 + \epsilon) = 10 - 4\epsilon$.
   - Real part: $10$
   - Dual part: $-4$

2. **Standard Calculus Quotient Rule:**
   $$f'(x) = \frac{(2x)(x - 2) - (x^2 + 1)(1)}{(x - 2)^2} = \frac{2x^2 - 4x - x^2 - 1}{(x - 2)^2} = \frac{x^2 - 4x - 1}{(x - 2)^2}$$
   Evaluate at $x = 3$:
   $$f'(3) = \frac{3^2 - 4(3) - 1}{(3 - 2)^2} = \frac{9 - 12 - 1}{1^2} = -4$$

3. **Verification:**
   The dual coefficient is $-4$, which is identical to the exact analytical derivative $f'(3) = -4$!

---

### Problem 2: Associativity & FLOP Count: JVP vs. VJP (Medium)

Consider a deep composition of functions:
$$y = f_3(f_2(f_1(x)))$$
with dimensions:
$$x \in \mathbb{R}^{1,000,000} \xrightarrow{f_1} h_1 \in \mathbb{R}^{1,000} \xrightarrow{f_2} h_2 \in \mathbb{R}^{10} \xrightarrow{f_3} y \in \mathbb{R}^1$$

The Jacobians have dimensions:
- $J_1 = \frac{\partial h_1}{\partial x} \in \mathbb{R}^{1000 \times 1,000,000}$
- $J_2 = \frac{\partial h_2}{\partial h_1} \in \mathbb{R}^{10 \times 1000}$
- $J_3 = \frac{\partial y}{\partial h_2} \in \mathbb{R}^{1 \times 10}$

By the chain rule, the total Jacobian is the matrix product:
$$J = J_3 \cdot J_2 \cdot J_1 \in \mathbb{R}^{1 \times 1,000,000}$$

**Questions:**
1. Compute the number of scalar multiplications required to compute the full gradient using **Forward-Mode Autodiff** (evaluating $J \cdot e_i$ for all $10^6$ basis vectors).
2. Compute the number of scalar multiplications required to compute the full gradient using **Reverse-Mode Autodiff** (evaluating $([1.0] \cdot J_3) \cdot J_2) \cdot J_1$).
3. What is the speedup factor of Reverse-Mode over Forward-Mode for this network?

#### Solution

1. **Forward-Mode FLOP Count:**
   For a single input basis vector $v \in \mathbb{R}^{1,000,000 \times 1}$:
   - $v_1 = J_1 v \implies (1000 \times 10^6) \times (10^6 \times 1) = 10^9$ multiplications.
   - $v_2 = J_2 v_1 \implies (10 \times 1000) \times (1000 \times 1) = 10^4$ multiplications.
   - $v_3 = J_3 v_2 \implies (1 \times 10) \times (10 \times 1) = 10$ multiplications.
   Total per direction $\approx 10^9$ FLOPs.
   To compute the gradient across all $d_{\text{in}} = 1,000,000$ inputs, we must repeat this $10^6$ times:
   $$\text{Total FLOPs}_{\text{forward}} = 10^6 \times 10^9 = 10^{15} \text{ multiplications (1 PetaFLOP!)}$$

2. **Reverse-Mode FLOP Count:**
   Start with cotangent seed $u = [1.0] \in \mathbb{R}^{1 \times 1}$:
   - Step 1: $u_1 = u \cdot J_3 \implies (1 \times 1) \times (1 \times 10) = 10$ multiplications. ($u_1 \in \mathbb{R}^{1 \times 10}$)
   - Step 2: $u_2 = u_1 \cdot J_2 \implies (1 \times 10) \times (10 \times 1000) = 10,000$ multiplications. ($u_2 \in \mathbb{R}^{1 \times 1000}$)
   - Step 3: $u_3 = u_2 \cdot J_1 \implies (1 \times 1000) \times (1000 \times 10^6) = 1,000,000,000 = 10^9$ multiplications. ($u_3 \in \mathbb{R}^{1 \times 10^6}$)
   $$\text{Total FLOPs}_{\text{reverse}} = 10 + 10,000 + 10^9 \approx 10^9 \text{ multiplications (1 GigaFLOP)}$$

3. **Speedup Factor:**
   $$\text{Speedup} = \frac{10^{15}}{10^9} = 1,000,000\times$$
   Reverse-Mode is **one million times faster** because it exploits the scalar output dimension ($d_{\text{out}} = 1$) to perform vector-matrix products from left to right instead of matrix-vector products from right to left!

---

### Problem 3: Activation Checkpointing Optimization (Hard)

Consider training an $L$-layer deep network where each layer's activation tensor consumes $M_{\text{act}}$ bytes of GPU memory.

1. What is the peak activation memory without checkpointing?
2. Suppose we divide the $L$ layers into segments of length $k$ and only store the activations at the start of each segment (storing $L/k$ checkpoints). During backprop, we recompute the $k$ forward activations inside each segment on-demand.
   Write an expression for the total peak memory as a function of $k$.
3. Find the optimal segment size $k^*$ that minimizes peak memory, and compute the resulting optimal memory complexity in terms of $L$.
4. What is the computational cost overhead introduced by this strategy?

#### Solution

1. **Standard Peak Memory:**
   Without checkpointing, every intermediate activation must be retained simultaneously:
   $$\text{Memory}_{\text{standard}} = L \cdot M_{\text{act}} = \mathcal{O}(L)$$

2. **Memory with Checkpointing:**
   Total peak memory consists of two components:
   - Checkpoint activations stored across the whole network: $\frac{L}{k} \cdot M_{\text{act}}$
   - Temporary activations within the active segment currently being backpropagated: $k \cdot M_{\text{act}}$
   $$\text{Peak Memory}(k) = \left( \frac{L}{k} + k \right) M_{\text{act}}$$

3. **Optimal Segment Size $k^*$:**
   To minimize $g(k) = \frac{L}{k} + k$, take the derivative with respect to $k$ and set to zero:
   $$\frac{dg}{dk} = -\frac{L}{k^2} + 1 = 0 \implies k^2 = L \implies k^* = \sqrt{L}$$
   Substituting $k^* = \sqrt{L}$:
   $$\text{Peak Memory}^* = \left( \frac{L}{\sqrt{L}} + \sqrt{L} \right) M_{\text{act}} = 2\sqrt{L} \cdot M_{\text{act}} = \mathcal{O}(\sqrt{L})$$
   For a 100-layer model, memory is reduced from $100 M_{\text{act}}$ to $2\sqrt{100} M_{\text{act}} = 20 M_{\text{act}}$—an **80% reduction in activation VRAM**!

4. **Compute Overhead:**
   Every segment's forward pass is executed twice: once in the initial forward pass, and once during rematerialization in the backward pass.
   - Standard forward pass: $1 \times \text{Cost}(F)$
   - Standard backward pass: $2 \times \text{Cost}(F)$
   - Recomputation forward: $+1 \times \text{Cost}(F)$
   - Total cost with checkpointing: $1 + 2 + 1 = 4 \times \text{Cost}(F)$ vs $3 \times \text{Cost}(F)$ standard.
   The overhead is exactly $\frac{4 - 3}{3} = \frac{1}{3} \approx 33.3\%$ additional compute time, which is a tiny price to pay for enabling models 5–10$\times$ larger to fit in GPU memory!
