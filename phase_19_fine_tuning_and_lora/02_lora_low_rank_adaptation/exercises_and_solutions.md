# Topic 19.2: Exercises & Solutions — LoRA (Low-Rank Adaptation)

---

### Problem 1: Manual Step-by-Step LoRA Forward Pass & Weight Merging (Easy)

Consider a toy linear layer with input dimension $k = 2$ and output dimension $d = 2$.
The frozen base weight matrix $W_0$, LoRA down-projection $A$, and up-projection $B$ are given by:
$$W_0 = \begin{bmatrix} 1.0 & 2.0 \\ 3.0 & 4.0 \end{bmatrix}, \quad A = \begin{bmatrix} 1.0 & -1.0 \end{bmatrix} \in \mathbb{R}^{1 \times 2}, \quad B = \begin{bmatrix} 2.0 \\ -1.0 \end{bmatrix} \in \mathbb{R}^{2 \times 1}$$

The LoRA hyperparameters are rank $r = 1$ and scaling constant $\alpha = 2.0$.
The input feature vector is:
$$\mathbf{x} = \begin{bmatrix} 3.0 \\ 1.0 \end{bmatrix}$$

**Tasks:**
1. Compute the base forward transformation $\mathbf{h}_{\text{base}} = W_0 \mathbf{x}$.
2. Compute the intermediate LoRA activation $u = A \mathbf{x} \in \mathbb{R}$.
3. Compute the scaled LoRA low-rank delta $\Delta \mathbf{h} = \frac{\alpha}{r} B u$.
4. Compute the total adapted output $\mathbf{h} = \mathbf{h}_{\text{base}} + \Delta \mathbf{h}$.
5. Calculate the merged weight matrix $W_{\text{merged}} = W_0 + \frac{\alpha}{r} B A$, and verify that $W_{\text{merged}} \mathbf{x} = \mathbf{h}$.

<details>
<summary>View Complete Solution</summary>

**Step 1: Compute Base Output $\mathbf{h}_{\text{base}}$**
$$\mathbf{h}_{\text{base}} = W_0 \mathbf{x} = \begin{bmatrix} 1.0 & 2.0 \\ 3.0 & 4.0 \end{bmatrix} \begin{bmatrix} 3.0 \\ 1.0 \end{bmatrix} = \begin{bmatrix} (1.0)(3.0) + (2.0)(1.0) \\ (3.0)(3.0) + (4.0)(1.0) \end{bmatrix} = \begin{bmatrix} 3.0 + 2.0 \\ 9.0 + 4.0 \end{bmatrix} = \begin{bmatrix} 5.0 \\ 13.0 \end{bmatrix}$$

**Step 2: Compute Intermediate LoRA Activation $u$**
$$u = A \mathbf{x} = \begin{bmatrix} 1.0 & -1.0 \end{bmatrix} \begin{bmatrix} 3.0 \\ 1.0 \end{bmatrix} = (1.0)(3.0) + (-1.0)(1.0) = 3.0 - 1.0 = 2.0$$

**Step 3: Compute Scaled LoRA Delta $\Delta \mathbf{h}$**
The scaling factor is:
$$\frac{\alpha}{r} = \frac{2.0}{1} = 2.0$$
$$\Delta \mathbf{h} = \frac{\alpha}{r} B u = 2.0 \begin{bmatrix} 2.0 \\ -1.0 \end{bmatrix} (2.0) = 4.0 \begin{bmatrix} 2.0 \\ -1.0 \end{bmatrix} = \begin{bmatrix} 8.0 \\ -4.0 \end{bmatrix}$$

**Step 4: Compute Total Adapted Output $\mathbf{h}$**
$$\mathbf{h} = \mathbf{h}_{\text{base}} + \Delta \mathbf{h} = \begin{bmatrix} 5.0 \\ 13.0 \end{bmatrix} + \begin{bmatrix} 8.0 \\ -4.0 \end{bmatrix} = \begin{bmatrix} 13.0 \\ 9.0 \end{bmatrix}$$

**Step 5: Compute $W_{\text{merged}}$ and Verify**
First compute the low-rank outer product $B A$:
$$B A = \begin{bmatrix} 2.0 \\ -1.0 \end{bmatrix} \begin{bmatrix} 1.0 & -1.0 \end{bmatrix} = \begin{bmatrix} (2.0)(1.0) & (2.0)(-1.0) \\ (-1.0)(1.0) & (-1.0)(-1.0) \end{bmatrix} = \begin{bmatrix} 2.0 & -2.0 \\ -1.0 & 1.0 \end{bmatrix}$$

Scale by $\frac{\alpha}{r} = 2.0$:
$$\Delta W = \frac{\alpha}{r} B A = 2.0 \begin{bmatrix} 2.0 & -2.0 \\ -1.0 & 1.0 \end{bmatrix} = \begin{bmatrix} 4.0 & -4.0 \\ -2.0 & 2.0 \end{bmatrix}$$

Merge with base weight $W_0$:
$$W_{\text{merged}} = W_0 + \Delta W = \begin{bmatrix} 1.0 & 2.0 \\ 3.0 & 4.0 \end{bmatrix} + \begin{bmatrix} 4.0 & -4.0 \\ -2.0 & 2.0 \end{bmatrix} = \begin{bmatrix} 5.0 & -2.0 \\ 1.0 & 6.0 \end{bmatrix}$$

Verify forward pass through $W_{\text{merged}}$:
$$W_{\text{merged}} \mathbf{x} = \begin{bmatrix} 5.0 & -2.0 \\ 1.0 & 6.0 \end{bmatrix} \begin{bmatrix} 3.0 \\ 1.0 \end{bmatrix} = \begin{bmatrix} (5.0)(3.0) + (-2.0)(1.0) \\ (1.0)(3.0) + (6.0)(1.0) \end{bmatrix} = \begin{bmatrix} 15.0 - 2.0 \\ 3.0 + 6.0 \end{bmatrix} = \begin{bmatrix} 13.0 \\ 9.0 \end{bmatrix}$$

Check:
$$W_{\text{merged}} \mathbf{x} = \mathbf{h} = \begin{bmatrix} 13.0 \\ 9.0 \end{bmatrix} \quad \checkmark$$
The dynamic forward pass and merged linear transformation produce identically equal outputs.

</details>

---

### Problem 2: Full-Stack Parameter Reduction in LLaMA-3-8B (Medium)

In modern PEFT practice (such as QLoRA), adapters are applied to **all 7 linear projections** across all $L = 32$ Transformer layers of LLaMA-3-8B:
- Self-Attention Projections: $W_q, W_k, W_v, W_o \in \mathbb{R}^{d \times d}$ where $d = 4096$. (Note: In LLaMA-3 GQA, $W_k, W_v \in \mathbb{R}^{1024 \times 4096}$).
- Feed-Forward Projections: $W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{d_{\text{ffn}} \times d}$ and $W_{\text{down}} \in \mathbb{R}^{d \times d_{\text{ffn}}}$ where $d = 4096$ and $d_{\text{ffn}} = 14336$.

Assume rank $r = 16$.

**Tasks:**
1. Compute the total number of base parameters across all 7 linear projections across all 32 layers.
2. Compute the total number of LoRA adapter parameters ($A$ and $B$) across the same layers.
3. Determine the parameter reduction ratio (as a percentage).

<details>
<summary>View Complete Solution</summary>

**Step 1: Compute Base Linear Parameters per Layer**
In each layer:
- $W_q$: $4096 \times 4096 = 16,777,216$
- $W_k$: $1024 \times 4096 = 4,194,304$
- $W_v$: $1024 \times 4096 = 4,194,304$
- $W_o$: $4096 \times 4096 = 16,777,216$
- Attention Subtotal: $41,943,040$ parameters.

- $W_{\text{gate}}$: $14336 \times 4096 = 58,720,256$
- $W_{\text{up}}$: $14336 \times 4096 = 58,720,256$
- $W_{\text{down}}$: $4096 \times 14336 = 58,720,256$
- MLP Subtotal: $176,160,768$ parameters.

Total base linear parameters per layer:
$$P_{\text{layer, base}} = 41,943,040 + 176,160,768 = 218,103,808$$

Across all $L = 32$ layers:
$$P_{\text{total, base}} = 32 \times 218,103,808 = \mathbf{6,979,321,856 \text{ parameters}} \approx \mathbf{6.98\text{ Billion}}$$

---

**Step 2: Compute LoRA Parameters per Layer ($r = 16$)**
For a matrix with in-dimension $k$ and out-dimension $d$, LoRA uses $r(d + k)$ parameters:
- $W_q$: $16 \times (4096 + 4096) = 16 \times 8192 = 131,072$
- $W_k$: $16 \times (1024 + 4096) = 16 \times 5120 = 81,920$
- $W_v$: $16 \times (1024 + 4096) = 16 \times 5120 = 81,920$
- $W_o$: $16 \times (4096 + 4096) = 16 \times 8192 = 131,072$
- Attention LoRA Subtotal: $425,984$ parameters.

- $W_{\text{gate}}$: $16 \times (14336 + 4096) = 16 \times 18432 = 294,912$
- $W_{\text{up}}$: $16 \times (14336 + 4096) = 16 \times 18432 = 294,912$
- $W_{\text{down}}$: $16 \times (4096 + 14336) = 16 \times 18432 = 294,912$
- MLP LoRA Subtotal: $884,736$ parameters.

Total LoRA parameters per layer:
$$P_{\text{layer, LoRA}} = 425,984 + 884,736 = 1,310,720$$

Across all $L = 32$ layers:
$$P_{\text{total, LoRA}} = 32 \times 1,310,720 = \mathbf{41,943,040 \text{ parameters}} \approx \mathbf{41.94\text{ Million}}$$

---

**Step 3: Parameter Reduction Percentage**
$$\text{Ratio} = \frac{41,943,040}{6,979,321,856} \approx 0.0060096 \approx \mathbf{0.601\%}$$
$$\text{Reduction} = 100\% - 0.601\% = \mathbf{99.40\% \text{ reduction!}}$$

*(Instead of updating 6.98 billion parameters, LoRA only updates 41.9 million parameters, reducing disk checkpoints from 14 GB to just 84 MB).*

</details>

---

### Problem 3: Batch Analytical Gradient Derivation for LoRA (Hard)

Consider a mini-batch of $N$ token vectors arranged in matrix $X \in \mathbb{R}^{N \times k}$.
The LoRA forward pass is:
$$U = X A^T \in \mathbb{R}^{N \times r}$$
$$V = U B^T = X A^T B^T \in \mathbb{R}^{N \times d}$$
$$H = X W_0^T + \frac{\alpha}{r} V \in \mathbb{R}^{N \times d}$$

Let the upstream gradient of the scalar training objective $\mathcal{L}$ with respect to the output matrix $H$ be:
$$G = \frac{\partial \mathcal{L}}{\partial H} \in \mathbb{R}^{N \times d}$$

**Tasks:**
1. Derive the matrix expression for $\frac{\partial \mathcal{L}}{\partial B} \in \mathbb{R}^{d \times r}$.
2. Derive the matrix expression for $\frac{\partial \mathcal{L}}{\partial A} \in \mathbb{R}^{r \times k}$.
3. Derive the backpropagated gradient with respect to the input matrix $\frac{\partial \mathcal{L}}{\partial X} \in \mathbb{R}^{N \times k}$.

<details>
<summary>View Complete Solution</summary>

**Step 1: Gradient with respect to $B$**
From the chain rule:
$$\frac{\partial \mathcal{L}}{\partial V} = \frac{\alpha}{r} G \in \mathbb{R}^{N \times d}$$

The term $V = U B^T$. In scalar index notation:
$$V_{nj} = \sum_{p=1}^r U_{np} B_{jp}$$
Therefore, $\frac{\partial V_{nj}}{\partial B_{jp}} = U_{np}$.
By the matrix chain rule:
$$\frac{\partial \mathcal{L}}{\partial B_{jp}} = \sum_{n=1}^N \frac{\partial \mathcal{L}}{\partial V_{nj}} U_{np} = \frac{\alpha}{r} \sum_{n=1}^N G_{nj} U_{np} = \frac{\alpha}{r} (G^T U)_{jp}$$

In compact matrix form:
$$\boxed{\frac{\partial \mathcal{L}}{\partial B} = \frac{\alpha}{r} G^T U = \frac{\alpha}{r} G^T (X A^T) \in \mathbb{R}^{d \times r}}$$

---

**Step 2: Gradient with respect to $A$**
First compute the gradient with respect to the intermediate representation $U$:
$$V = U B^T \implies \frac{\partial \mathcal{L}}{\partial U} = \left(\frac{\partial \mathcal{L}}{\partial V}\right) B = \frac{\alpha}{r} G B \in \mathbb{R}^{N \times r}$$

Let $G_U = \frac{\partial \mathcal{L}}{\partial U} = \frac{\alpha}{r} G B$.
Since $U = X A^T$:
$$U_{np} = \sum_{m=1}^k X_{nm} A_{pm}$$
$$\frac{\partial \mathcal{L}}{\partial A_{pm}} = \sum_{n=1}^N (G_U)_{np} X_{nm} = (G_U^T X)_{pm}$$

In compact matrix form:
$$\boxed{\frac{\partial \mathcal{L}}{\partial A} = G_U^T X = \frac{\alpha}{r} (B^T G^T X) = \frac{\alpha}{r} (G B)^T X \in \mathbb{R}^{r \times k}}$$

---

**Step 3: Gradient with respect to input $X$**
The input $X$ flows through two paths:
1. Base path: $H_{\text{base}} = X W_0^T \implies \left(\frac{\partial \mathcal{L}}{\partial X}\right)_{\text{base}} = G W_0$
2. LoRA path: $U = X A^T \implies \left(\frac{\partial \mathcal{L}}{\partial X}\right)_{\text{LoRA}} = G_U A = \frac{\alpha}{r} (G B) A$

Summing both paths:
$$\boxed{\frac{\partial \mathcal{L}}{\partial X} = G W_0 + \frac{\alpha}{r} G B A = G \left( W_0 + \frac{\alpha}{r} B A \right) = G W_{\text{merged}} \in \mathbb{R}^{N \times k}}$$

Notice the mathematical beauty: the gradient backpropagated to previous layers is simply the upstream gradient multiplied by the merged weight matrix $W_{\text{merged}}$!

</details>
