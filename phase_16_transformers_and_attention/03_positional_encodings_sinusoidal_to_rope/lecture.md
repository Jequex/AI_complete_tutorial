# Topic 16.3: Positional Encodings — Sinusoidal to RoPE

---

## 1. Executive Summary & The 12-Year-Old Analogy

Self-attention possesses an extraordinary mathematical property that is simultaneously its greatest strength and its most dangerous flaw: **Permutation Equivariance**.

If you take a sentence:
> *"The dog bites the man."*

and permute its word order into:
> *"The man bites the dog."*

a vanilla self-attention layer produces the **exact same set of output vectors** (merely permuted in the same order)!
Because matrix multiplication $Q K^T$ calculates set-wise dot products without any concept of spatial coordinate, word order, or time, the model cannot distinguish between a subject performing an action and an object receiving it!

To inject word order, we must endow each token with a unique mathematical timestamp.
This lecture charts the evolution from the classical **Sinusoidal Positional Encodings** of Vaswani et al. (2017) to modern **Rotary Position Embeddings (RoPE)** (Su et al. 2021) that power state-of-the-art LLMs including LLaMA 3, Mistral, and Gemma.

```
                    PERMUTATION EQUIVARIANCE & POSITIONAL INJECTION
                    
   Without Positional Encoding:
   {"dog", "bites", "man"}  ───> [ Attention ] ───> Undifferentiated Set (No Word Order!)
   
   Vaswani Sinusoidal (Additive):
   Token Vector  x_pos ──┐
                         ├──> [ + ] ───> Input to Transformer Block
   Position Vector PE_pos ─┘
   
   Rotary Position Embedding (RoPE - Multiplicative in Attention):
   Query q_m ──> [ Rotate by m * θ ] ──> q̃_m ──┐
                                                 ├──> [ Inner Product <q̃_m, k̃_n> ] = g(q, k, m - n)
   Key   k_n ──> [ Rotate by n * θ ] ──> k̃_n ──┘
```

---

### The 12-Year-Old Analogy: Clocks Ticking at Different Speeds & Spinning Hands

Imagine you have 1,000 students standing in a line wearing identical blue school uniforms. From a helicopter, they all look identical. How can you tell who is at position 1, position 45, or position 800?

1. **The Grandfather Clock Gearbox (Sinusoidal Encoding):**
   - We give each student a wristwatch with 128 different hands ticking at different speeds!
   - The first hand (high frequency) spins around in 2 seconds.
   - The second hand spins in 10 seconds.
   - The 10th hand spins in 1 hour.
   - The 128th hand spins so slowly it takes 10,000 hours to make a single turn!
   - When student #45 stands up, their unique combination of clock hand angles is a completely distinct mathematical fingerprint that will never repeat.

2. **The Spinning Dancers (Rotary Position Embedding - RoPE):**
   - Instead of strapping heavy clocks to their wrists and adding clock numbers to their height, imagine the students are figure skaters spinning on ice.
   - Student at position $m$ spins by angle $m \times \theta$.
   - Student at position $n$ spins by angle $n \times \theta$.
   - When student $m$ reaches out to touch hands with student $n$, the angle between their outstretched hands depends **only on how many steps apart they are ($m - n$)**, regardless of whether they are at the beginning, middle, or end of the stadium!

---

## 2. Mathematical Symbol Decoder

| Symbol | Mathematical Domain | Description & Meaning |
| :--- | :--- | :--- |
| $pos$ or $m, n$ | Integer $\in \{0, 1, \dots, T-1\}$ | Absolute token position index in the sequence |
| $i$ | Integer $\in \{0, 1, \dots, d/2 - 1\}$ | Frequency channel index across the embedding dimension |
| $d$ or $d_{\text{model}}$ | Even Integer $\in \mathbb{N}$ | Dimensionality of the embedding / query-key vector |
| $\omega_i$ or $\theta_i$ | Scalar Real | Frequency scale: $\omega_i = 10000^{-2i / d}$ |
| $PE_{(pos, :)}$ | Vector $\in \mathbb{R}^d$ | Vaswani sinusoidal positional encoding vector for position $pos$ |
| $R_{\Theta, m}^d$ | Orthogonal Matrix $\in \mathbb{R}^{d \times d}$ | RoPE block-diagonal orthogonal rotation matrix for position $m$ |
| $k$ | Integer | Relative distance / offset between two tokens ($k = m - n$) |
| $\langle \tilde{\mathbf{q}}_m, \tilde{\mathbf{k}}_n \rangle$ | Scalar Real | Attention dot product between rotated query and rotated key |

---

## 3. The Permutation Equivariance Problem

A function $f: \mathbb{R}^{T \times d} \to \mathbb{R}^{T \times d}$ is **permutation equivariant** if for any permutation matrix $P \in \{0, 1\}^{T \times T}$:
$$f(P X) = P f(X)$$

Let us verify vanilla self-attention:
$$\text{Attention}(P X, P X, P X) = \text{softmax}\left(\frac{(PX)(PX)^T}{\sqrt{d_k}}\right) (PX)$$
Since $P$ is an orthogonal permutation matrix ($P P^T = I$):
$$(PX)(PX)^T = P X X^T P^T$$
$$\text{softmax}\left(\frac{P X X^T P^T}{\sqrt{d_k}}\right) P X = P \cdot \text{softmax}\left(\frac{X X^T}{\sqrt{d_k}}\right) P^T P X = P \cdot \text{softmax}\left(\frac{X X^T}{\sqrt{d_k}}\right) X = P \cdot \text{Attention}(X, X, X)$$

This proves self-attention is strictly permutation equivariant. It acts on an **unordered bag of vectors**. Without positional encodings, language models cannot understand syntax, verb tense, or narrative flow.

---

## 4. Vaswani Sinusoidal Positional Encodings (2017)

To break permutation symmetry, Vaswani et al. added a deterministic frequency vector $PE \in \mathbb{R}^{T \times d}$ directly to the token embedding matrix $X$:
$$\tilde{X}_{pos} = X_{pos} + PE_{pos}$$

The positional encoding is defined using sine and cosine functions of varying frequencies:
$$\boxed{PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right) = \sin(pos \cdot \omega_i)}$$
$$\boxed{PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right) = \cos(pos \cdot \omega_i)}$$
where $\omega_i = \frac{1}{10000^{2i/d}}$ forms a geometric progression from $1$ (for $i=0$) to $\frac{1}{10000}$ (for $i = d/2 - 1$).

### The Relative Position Shift Property & Complete Proof

The authors hypothesized that sinusoidal encodings would allow the model to easily learn to attend by relative positions, because for any fixed offset $k$, $PE_{pos + k}$ can be represented as a **linear transformation of $PE_{pos}$**.

#### Theorem: Linear Shift Property
For any fixed integer offset $k$, there exists a linear transformation matrix $M_i(k) \in \mathbb{R}^{2 \times 2}$ independent of $pos$ such that:
$$\begin{bmatrix} PE_{(pos + k, 2i)} \\ PE_{(pos + k, 2i+1)} \end{bmatrix} = M_i(k) \begin{bmatrix} PE_{(pos, 2i)} \\ PE_{(pos, 2i+1)} \end{bmatrix}$$

#### Proof:
Recall the classical trigonometric angle addition identities:
$$\sin(\alpha + \beta) = \sin\alpha \cos\beta + \cos\alpha \sin\beta$$
$$\cos(\alpha + \beta) = \cos\alpha \cos\beta - \sin\alpha \sin\beta$$

Let $\alpha = pos \cdot \omega_i$ and $\beta = k \cdot \omega_i$.
Then at position $pos + k$:
$$PE_{(pos + k, 2i)} = \sin((pos + k)\omega_i) = \sin(\alpha + \beta) = \cos(k\omega_i) \sin(pos \cdot \omega_i) + \sin(k\omega_i) \cos(pos \cdot \omega_i)$$
$$PE_{(pos + k, 2i+1)} = \cos((pos + k)\omega_i) = \cos(\alpha + \beta) = -\sin(k\omega_i) \sin(pos \cdot \omega_i) + \cos(k\omega_i) \cos(pos \cdot \omega_i)$$

Substitute $PE_{(pos, 2i)} = \sin(pos \cdot \omega_i)$ and $PE_{(pos, 2i+1)} = \cos(pos \cdot \omega_i)$:
$$\begin{bmatrix} PE_{(pos + k, 2i)} \\ PE_{(pos + k, 2i+1)} \end{bmatrix} = \begin{bmatrix} \cos(k\omega_i) & \sin(k\omega_i) \\ -\sin(k\omega_i) & \cos(k\omega_i) \end{bmatrix} \begin{bmatrix} PE_{(pos, 2i)} \\ PE_{(pos, 2i+1)} \end{bmatrix}$$

Let:
$$M_i(k) = \begin{bmatrix} \cos(k\omega_i) & \sin(k\omega_i) \\ -\sin(k\omega_i) & \cos(k\omega_i) \end{bmatrix}$$

$M_i(k)$ is an orthogonal 2D rotation matrix! Notice that $M_i(k)$ depends **only on the offset $k$ and frequency $\omega_i$**, and contains zero dependence on the absolute coordinate $pos$.
$$\blacksquare$$

---

## 5. Rotary Position Embedding (RoPE) (Su et al. 2021)

While sinusoidal encodings were elegant, they had a subtle drawback: they are **additive** at the input layer ($X + PE$). Through multiple layers of non-linear feedforward MLPs and LayerNorms, the additive positional signal becomes entangled and distorted.

Jianlin Su et al. (2021) asked:
> *Can we encode positional information directly into the attention dot product such that:*
> $$\langle \mathbf{q}_m, \mathbf{k}_n \rangle = g(\mathbf{x}_m, \mathbf{x}_n, m - n)$$
> *relying purely on relative distance $(m - n)$ throughout all layers of the network?*

### 5.1 Formulation in 2D Complex Plane
Consider a 2D vector $\mathbf{x} = [x_1, x_2]^T$, which can be represented as a complex number:
$$z = x_1 + i x_2 \in \mathbb{C}$$

To encode position $m$, we multiply $z$ by the complex rotation factor $e^{i m \theta}$:
$$\tilde{z}_m = z \cdot e^{i m \theta} = (x_1 + i x_2)(\cos(m\theta) + i \sin(m\theta)) = (x_1 \cos(m\theta) - x_2 \sin(m\theta)) + i (x_1 \sin(m\theta) + x_2 \cos(m\theta))$$

In matrix form:
$$\tilde{\mathbf{x}}_m = R_{\theta, m} \mathbf{x} = \begin{bmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$$

### 5.2 Preservation of Relative Distance in Dot Product
Now consider a Query at position $m$ and a Key at position $n$:
$$\tilde{\mathbf{q}}_m = R_{\theta, m} \mathbf{q}, \quad \tilde{\mathbf{k}}_n = R_{\theta, n} \mathbf{k}$$

In complex arithmetic, the inner product of two vectors is given by $\operatorname{Re}(\tilde{q} \tilde{k}^*)$:
$$\tilde{q}_m \tilde{k}_n^* = (q e^{i m \theta})(k e^{i n \theta})^* = q k^* e^{i m \theta} e^{-i n \theta} = (q k^*) e^{i (m - n)\theta}$$

Taking the real part:
$$\langle \tilde{\mathbf{q}}_m, \tilde{\mathbf{k}}_n \rangle = \operatorname{Re}\left( (q k^*) e^{i (m - n)\theta} \right)$$

$$\boxed{\langle R_{\theta, m} \mathbf{q}, R_{\theta, n} \mathbf{k} \rangle = \mathbf{q}^T R_{\theta, n - m} \mathbf{k} = g(\mathbf{q}, \mathbf{k}, m - n)}$$

The attention score between query $m$ and key $n$ depends **purely on the relative offset $(m - n)$**!
If we shift both tokens forward by 100 positions ($m + 100, n + 100$), their relative distance $(m + 100) - (n + 100) = m - n$ remains identical, so their attention affinity is perfectly preserved!

### 5.3 Generalization to $d$-Dimensional Vectors
For a $d$-dimensional space, RoPE breaks the vector into $d/2$ independent 2D pairs and rotates each pair by its own frequency $\theta_i = 10000^{-2(i-1)/d}$:

$$R_{\Theta, m}^d = \begin{bmatrix} 
\cos(m\theta_1) & -\sin(m\theta_1) & 0 & 0 & \dots \\
\sin(m\theta_1) & \cos(m\theta_1) & 0 & 0 & \dots \\
0 & 0 & \cos(m\theta_2) & -\sin(m\theta_2) & \dots \\
0 & 0 & \sin(m\theta_2) & \cos(m\theta_2) & \dots \\
\vdots & \vdots & \vdots & \vdots & \ddots
\end{bmatrix}$$

Because $R_{\Theta, m}$ is orthogonal ($R^T R = I$), vector norms are strictly preserved:
$$\|\tilde{\mathbf{q}}_m\|_2 = \|\mathbf{q}\|_2$$
RoPE introduces zero magnitude distortion; it acts purely as a spatial rotation!

---

## 6. Architectural Comparison

| Property | Learned Absolute Positional Embedding (BERT, GPT-2) | Vaswani Sinusoidal (Attention is All You Need) | Rotary Position Embedding (RoPE) (LLaMA 3, Gemma) |
| :--- | :--- | :--- | :--- |
| **Type** | Additive parameters at input | Additive fixed formula at input | Multiplicative rotation on Q & K |
| **Generalizes to Length $> T_{\text{train}}$?** | No (unseen indices have no trained weights) | Poorly (frequencies shift, but not trained) | Yes (relative rotations naturally extrapolate) |
| **Preserves Relative Offset?** | No guarantee | Yes (linear shift property) | Exact mathematical guarantee $\langle q_m, k_n \rangle = g(m-n)$ |
| **Vector Norm Preservation** | Distorts $\|x\|$ | Distorts $\|x\|$ | Exact ($\|R x\|_2 = \|x\|_2$) |
| **KV Cache Compatibility** | Requires position re-indexing | Requires position re-indexing | Perfectly caches rotated keys $k_n$ |
