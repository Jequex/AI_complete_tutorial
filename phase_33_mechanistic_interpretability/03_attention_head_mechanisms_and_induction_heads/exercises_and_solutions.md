# Topic 03: Exercises & Solutions — Attention Head Mechanisms & Induction Heads

## Exercise 1: Low-Rank Structure of $QK$ and $OV$ Circuits

In a Transformer model with hidden dimension $d = 4096$, attention head dimension $d_{\text{head}} = 64$, and projection matrices $W_Q, W_K \in \mathbb{R}^{d_{\text{head}} \times d}$ and $W_V \in \mathbb{R}^{d_{\text{head}} \times d}, W_O \in \mathbb{R}^{d \times d_{\text{head}}}$:

1. What are the maximum possible algebraic ranks of the circuit matrices $W_{QK} = W_Q^\top W_K$ and $W_{OV} = W_O W_V$?
2. What percentage of the full $d \times d$ ambient matrix space can a single attention head span?
3. How does this low-rank bottleneck protect the residual stream from destructive interference when dozens of attention heads write to it simultaneously?

### Solution

#### 1. Maximum Algebraic Rank
By Sylvester's rank inequality:
$$\text{rank}(A B) \le \min(\text{rank}(A), \text{rank}(B))$$
Since $W_Q^\top \in \mathbb{R}^{d \times d_{\text{head}}}$ and $W_K \in \mathbb{R}^{d_{\text{head}} \times d}$ have inner dimension $d_{\text{head}} = 64$:
$$\text{rank}(W_{QK}) = \text{rank}(W_Q^\top W_K) \le d_{\text{head}} = \mathbf{64}$$
Similarly, for $W_O \in \mathbb{R}^{d \times d_{\text{head}}}$ and $W_V \in \mathbb{R}^{d_{\text{head}} \times d}$:
$$\text{rank}(W_{OV}) = \text{rank}(W_O W_V) \le d_{\text{head}} = \mathbf{64}$$

#### 2. Percentage of Ambient Matrix Degrees of Freedom
The full ambient space $\mathbb{R}^{d \times d}$ has rank $d = 4096$.
The fraction of dimensions accessible to a single head's output is:
$$\text{Fraction} = \frac{64}{4096} = \frac{1}{64} \approx \mathbf{1.56\%}$$

#### 3. Residual Stream Preservation
Because each head writes along a low-dimensional subspace of rank at most 64, it leaves the remaining $4096 - 64 = 4032$ dimensions of the residual stream completely untouched. By assigning different heads approximately orthogonal write subspaces $W_O^{(h)}$, dozens of heads can concurrently read and write to the shared communication bus without corrupting each other's transmitted information.

---

## Exercise 2: Deriving the Composite Induction Form $W_{QK}^{(1)} W_{OV}^{(0)}$

Let the residual stream at Layer 0 be $\mathbf{h}_0^{(j)} = W_E \mathbf{t}_j$.
Suppose Layer 0 contains a Previous-Token Head with $OV$-matrix $W_{OV}^{(0)}$ such that at token position $j$, its output is:
$$\Delta \mathbf{h}_0^{(j)} = W_{OV}^{(0)} W_E \mathbf{t}_{j-1}$$
The representation entering Layer 1 at position $j$ is $\mathbf{h}_1^{(j)} = W_E \mathbf{t}_j + W_{OV}^{(0)} W_E \mathbf{t}_{j-1}$.

1. Expand the attention logit $\alpha_{i, j}^{(1)}$ of Layer 1 Head 1 between query position $i$ (with token $\mathbf{t}_i$) and key position $j$ into its 4 constituent bilinear terms.
2. Identify which specific term is responsible for the induction behavior $[A][B] \dots [A] \to [B]$.
3. For what algebraic condition on matrix $\mathbf{C} = \left(W_Q^{(1)} W_E\right)^\top \left(W_K^{(1)} W_{OV}^{(0)} W_E\right)$ will the attention weight $A_{i, j}^{(1)}$ reliably spike whenever $\mathbf{t}_i = \mathbf{t}_{j-1}$?

### Solution

#### 1. Expansion of Attention Logits
$$\alpha_{i, j}^{(1)} = \frac{1}{\sqrt{d_k}} \left(W_Q^{(1)} \mathbf{h}_1^{(i)}\right)^\top \left(W_K^{(1)} \mathbf{h}_1^{(j)}\right) = \frac{1}{\sqrt{d_k}} (\mathbf{h}_1^{(i)})^\top W_{QK}^{(1)} \mathbf{h}_1^{(j)}$$
Substituting $\mathbf{h}_1^{(i)} \approx W_E \mathbf{t}_i$ and $\mathbf{h}_1^{(j)} = W_E \mathbf{t}_j + W_{OV}^{(0)} W_E \mathbf{t}_{j-1}$:
$$\alpha_{i, j}^{(1)} = \frac{1}{\sqrt{d_k}} \Big[ \underbrace{\mathbf{t}_i^\top W_E^\top W_{QK}^{(1)} W_E \mathbf{t}_j}_{\text{Term 1: Token-Token}} + \underbrace{\mathbf{t}_i^\top W_E^\top W_{QK}^{(1)} W_{OV}^{(0)} W_E \mathbf{t}_{j-1}}_{\text{Term 2: Induction Cross-Term}} + \dots \Big]$$

#### 2. The Induction Term
**Term 2**:
$$\mathbf{t}_i^\top \left( W_E^\top W_{QK}^{(1)} W_{OV}^{(0)} W_E \right) \mathbf{t}_{j-1}$$
This term couples current token $\mathbf{t}_i$ directly with the *preceding* token $\mathbf{t}_{j-1}$ of position $j$.

#### 3. Algebraic Condition
Let $\mathbf{M}_{\text{ind}} = W_E^\top W_{QK}^{(1)} W_{OV}^{(0)} W_E \in \mathbb{R}^{V \times V}$.
For one-hot token representations $\mathbf{t}_i = \mathbf{e}_A$ and $\mathbf{t}_{j-1} = \mathbf{e}_A$, the score is:
$$\alpha_{i, j} = \mathbf{e}_A^\top \mathbf{M}_{\text{ind}} \mathbf{e}_A = (\mathbf{M}_{\text{ind}})_{A, A}$$
For the head to attend whenever $\mathbf{t}_i = \mathbf{t}_{j-1} = [A]$, the diagonal elements must be large and positive:
$$(\mathbf{M}_{\text{ind}})_{A, A} \gg (\mathbf{M}_{\text{ind}})_{A, B} \quad \forall B \neq A$$
That is, the matrix $\mathbf{M}_{\text{ind}}$ must have a **strongly positive trace and dominant positive diagonal**, ensuring that any repeated token fires a sharp attention peak to position $j$.

---

## Exercise 3: Direct Logit Attribution and Copy Head Eigenvalues

An attention head's $OV$-circuit acts on the residual stream as $\Delta \mathbf{h} = W_{OV} \mathbf{h}$.
When the final unembedding matrix $W_U$ is applied, the head's direct contribution to output logits for token $y$ given attended token $x$ is given by the matrix:
$$\mathbf{T} = W_U W_{OV} W_E \in \mathbb{R}^{V \times V}$$

1. Suppose $\mathbf{T}$ has eigendecomposition with eigenvectors $\mathbf{v}_k$ and eigenvalues $\lambda_k$. If an eigenvalue $\lambda_k > 0$ is large and positive, what does this imply about the head's behavior?
2. If an eigenvalue is large and negative ($\lambda_k < 0$), what behavior does the head execute?
3. In our experiment, the induction head attends to position $j = 1$ containing token $B$. If $\mathbf{T}_{B, B} = +7.0$ and $\mathbf{T}_{B, C} = 0.0$ for all $C \neq B$, by how much does the logit for token $B$ increase relative to other tokens?

### Solution

#### 1. Positive Eigenvalues: Copying
If $\lambda_k > 0$, representations that align with the corresponding eigenvector are amplified in the same direction. When $\mathbf{T}_{v, v} > 0$ along the diagonal, the head directly copies the identity of attended token $v$ into the next-token prediction logits.

#### 2. Negative Eigenvalues: Inhibition / Anti-Repetition
If $\lambda_k < 0$, attending to token $v$ actively suppresses the logit for token $v$. Such heads are known as **Inhibition Heads** or **Negative Name Movers**; they prevent the model from getting stuck in infinite repeating loops.

#### 3. Numerical Logit Attribution
The direct logit attribution for token $B$ is:
$$\Delta \text{logit}(B) = \mathbf{e}_B^\top \mathbf{T} \mathbf{e}_B = \mathbf{T}_{B, B} = \mathbf{+7.0}$$
For all other tokens $C \neq B$, $\Delta \text{logit}(C) = 0.0$.
Thus, the logit difference between target token $B$ and all competing tokens increases by **$+7.0$**, driving the post-softmax probability $P(B)$ close to $1.0$ (as observed in our experiment with $99.99\%$).
