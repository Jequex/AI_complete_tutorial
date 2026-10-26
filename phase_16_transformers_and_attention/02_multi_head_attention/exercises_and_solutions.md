# Topic 16.2: Exercises & Solutions — Multi-Head Attention

---

### Problem 1: Parameter Counting & Tensor Shapes in MHA (Easy)

In a standard Transformer architecture (e.g., *Vaswani et al. 2017* base model):
- Embedding dimension: $d_{\text{model}} = 512$
- Number of heads: $h = 8$
- Key/Query dimension: $d_k = d_{\text{model}} / h = 64$
- Value dimension: $d_v = d_{\text{model}} / h = 64$
- Batch size: $B = 16$
- Sequence length: $T = 128$

1. Compute the shape and number of scalar parameters (weights and biases) in:
   - The query projection layer $W^Q$
   - The key projection layer $W^K$
   - The value projection layer $W^V$
   - The final output projection layer $W^O$
   - Total parameters in this Multi-Head Attention block.
2. State the exact tensor shape at each intermediate step of the tensor reshaping pipeline:
   - After linear projection: $Q_{\text{flat}}$
   - After 4D reshape
   - After head permutation
   - The attention score matrix $S$
   - The attention output after head concatenation

<details>
<summary>View Complete Solution</summary>

**Step 1: Parameter Counting:**
Assuming standard linear layers with bias:
- $W^Q$: Maps from $d_{\text{model}} \to d_{\text{model}}$ ($512 \to 512$).
  - Weights: $512 \times 512 = 262,144$
  - Biases: $512$
  - Subtotal = $262,656$
- $W^K$: Maps from $512 \to 512$.
  - Weights: $512 \times 512 = 262,144$
  - Biases: $512$
  - Subtotal = $262,656$
- $W^V$: Maps from $512 \to 512$.
  - Weights: $512 \times 512 = 262,144$
  - Biases: $512$
  - Subtotal = $262,656$
- $W^O$: Maps from $h \cdot d_v = 512 \to 512$.
  - Weights: $512 \times 512 = 262,144$
  - Biases: $512$
  - Subtotal = $262,656$

$$\text{Total Trainable Parameters} = 4 \times 262,656 = 1,050,624 \quad (\approx 1.05\text{ Million})$$

*(Without bias, exactly $4 \times d_{\text{model}}^2 = 4 \times 512^2 = 1,048,576$ parameters).*

**Step 2: Tensor Shape Pipeline:**
- Input $X$: $[B, T, d_{\text{model}}] = [16, 128, 512]$
- Projected Query $Q$: $[B, T, d_{\text{model}}] = [16, 128, 512]$
- 4D Reshape: $[B, T, h, d_k] = [16, 128, 8, 64]$
- Head Permute: $[B, h, T, d_k] = [16, 8, 128, 64]$
- Attention Scores ($Q @ K^T$): $[B, h, T, T] = [16, 8, 128, 128]$
- Attention Context ($A @ V$): $[B, h, T, d_k] = [16, 8, 128, 64]$
- Transposed Context: $[B, T, h, d_k] = [16, 128, 8, 64]$
- Concatenated Output: $[B, T, h \cdot d_k] = [16, 128, 512]$
- Final Projection ($W^O$): $[B, T, d_{\text{model}}] = [16, 128, 512]$

</details>

---

### Problem 2: Manual 2-Head Attention Walkthrough (Medium)

Consider a sequence of $T = 2$ tokens with embedding dimension $d_{\text{model}} = 4$ and $h = 2$ heads ($d_k = d_v = 2$).
Let the input tokens be:
$$X = \begin{bmatrix} 1.0 & 0.0 & 0.0 & 1.0 \\ 0.0 & 1.0 & 1.0 & 0.0 \end{bmatrix} \in \mathbb{R}^{2 \times 4}$$

Head 1 operates on the first 2 dimensions of $X$, and Head 2 operates on the last 2 dimensions:
$$Q_1 = K_1 = V_1 = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix}, \quad Q_2 = K_2 = V_2 = \begin{bmatrix} 0.0 & 1.0 \\ 1.0 & 0.0 \end{bmatrix}$$

1. Compute the scaled attention weights $A^{(1)}$ for Head 1 and $A^{(2)}$ for Head 2 using $\sqrt{d_k} = \sqrt{2} \approx 1.4142$.
2. Compute the attention output matrices $\text{head}_1 = A^{(1)} V_1$ and $\text{head}_2 = A^{(2)} V_2$.
3. Concatenate $\mathbf{H} = [\text{head}_1 \; \text{head}_2] \in \mathbb{R}^{2 \times 4}$.
4. If the output matrix is the identity matrix $W^O = I_4$, what is the final Multi-Head Attention output?

<details>
<summary>View Complete Solution</summary>

**Step 1: Compute Attention for Head 1:**
$$S_1 = Q_1 K_1^T = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix}$$

Scale by $\sqrt{2} \approx 1.4142$:
$$Z_1 = \frac{S_1}{\sqrt{2}} = \begin{bmatrix} 0.7071 & 0.0 \\ 0.0 & 0.7071 \end{bmatrix}$$

Row-wise softmax for Row 1:
- $e^{0.7071 - 0.7071} = 1.0$; $e^{0.0 - 0.7071} = e^{-0.7071} \approx 0.4931$
- Sum $= 1.4931 \implies A_{11}^{(1)} = \frac{1.0}{1.4931} \approx 0.6698, \; A_{12}^{(1)} \approx 0.3302$
By symmetry, for Row 2: $A_{21}^{(1)} \approx 0.3302, \; A_{22}^{(1)} \approx 0.6698$.

$$A^{(1)} = \begin{bmatrix} 0.6698 & 0.3302 \\ 0.3302 & 0.6698 \end{bmatrix}$$

Now for Head 2:
$$S_2 = Q_2 K_2^T = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix}$$
Since $S_2 = S_1$, $Z_2 = Z_1$, so:
$$A^{(2)} = \begin{bmatrix} 0.6698 & 0.3302 \\ 0.3302 & 0.6698 \end{bmatrix}$$

**Step 2: Head Outputs:**
$$\text{head}_1 = A^{(1)} V_1 = \begin{bmatrix} 0.6698 & 0.3302 \\ 0.3302 & 0.6698 \end{bmatrix} \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix} = \begin{bmatrix} 0.6698 & 0.3302 \\ 0.3302 & 0.6698 \end{bmatrix}$$

$$\text{head}_2 = A^{(2)} V_2 = \begin{bmatrix} 0.6698 & 0.3302 \\ 0.3302 & 0.6698 \end{bmatrix} \begin{bmatrix} 0.0 & 1.0 \\ 1.0 & 0.0 \end{bmatrix} = \begin{bmatrix} 0.3302 & 0.6698 \\ 0.6698 & 0.3302 \end{bmatrix}$$

**Step 3: Head Concatenation:**
$$\mathbf{H} = [\text{head}_1 \; \text{head}_2] = \begin{bmatrix} 0.6698 & 0.3302 & 0.3302 & 0.6698 \\ 0.3302 & 0.6698 & 0.6698 & 0.3302 \end{bmatrix}$$

**Step 4: Output Projection ($W^O = I$):**
$$\text{Output} = \mathbf{H} I_4 = \begin{bmatrix} 0.6698 & 0.3302 & 0.3302 & 0.6698 \\ 0.3302 & 0.6698 & 0.6698 & 0.3302 \end{bmatrix}$$

Notice that Head 1 and Head 2 captured completely distinct feature mixtures without interfering with each other!

</details>

---

### Problem 3: Expressive Subspace Diversity Proof (Hard)

Let $\mathcal{H}_1, \dots, \mathcal{H}_h$ be $h$ independent heads, each projecting into a subspace of dimension $d_k$.

1. Explain why a single attention head ($h = 1, d_k = d_{\text{model}}$) that places 95% of its attention probability on token $j$ is incapable of extracting information about token $k$ ($k \ne j$) for that query.
2. Prove that Multi-Head Attention allows a query token to extract strong features from token $j$ via Head 1 and strong features from token $k$ via Head 2 simultaneously without compromising the attention fidelity of either head.

<details>
<summary>View Complete Solution</summary>

**Part 1: Single-Head Allocation Constraint:**
In a single head, the attention distribution is constrained to the probability simplex:
$$\sum_{m=1}^T A_{1m} = 1.0, \quad A_{1m} \ge 0$$
If the query at position 1 strongly attends to token $j$ such that $A_{1j} = 0.95$, the total remaining probability mass available for all other $T-1$ tokens is:
$$\sum_{m \ne j} A_{1m} = 1.0 - 0.95 = 0.05$$
Thus for any token $k \ne j$, $A_{1k} \le 0.05$.
The resulting output vector is:
$$\mathbf{o}_1 = 0.95 \mathbf{v}_j + \sum_{m \ne j} A_{1m} \mathbf{v}_m$$
The signal from token $k$ is attenuated by a factor of at least $\frac{0.05}{0.95} \approx \frac{1}{19}$, effectively erasing it from the contextual representation.

**Part 2: Multi-Head Subspace Decoupling:**
In Multi-Head Attention with $h \ge 2$, each head $i$ maintains an independent probability simplex:
$$\sum_{m=1}^T A_{1m}^{(i)} = 1.0 \quad \forall i \in \{1, \dots, h\}$$

Therefore, we can simultaneously set:
$$A_{1j}^{(1)} = 0.95 \implies \text{head}_1 \approx \mathbf{v}_{j}^{(1)}$$
$$A_{1k}^{(2)} = 0.95 \implies \text{head}_2 \approx \mathbf{v}_{k}^{(2)}$$

Concatenating both heads produces:
$$\mathbf{H}_1 = [\mathbf{v}_j^{(1)} \; \mathbf{v}_k^{(2)}]$$

The output projection $W^O$ can linearly combine these independent subspace features:
$$\mathbf{y}_1 = \mathbf{H}_1 W^O = \mathbf{v}_j^{(1)} W_{1:d_k}^O + \mathbf{v}_k^{(2)} W_{d_k+1:2d_k}^O$$

Thus, position 1 retains full-strength, unattenuated representations of both token $j$ and token $k$ simultaneously. This decouples the attention bottleneck across multiple feature channels!

</details>
