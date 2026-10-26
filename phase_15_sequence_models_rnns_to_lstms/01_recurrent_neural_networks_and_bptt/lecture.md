# Topic 15.1: Recurrent Neural Networks & Backpropagation Through Time (BPTT)

---

## 1. Executive Summary & The 12-Year-Old Analogy

Standard Feedforward Neural Networks (MLPs) and Convolutional Networks (CNNs) operate under a foundational assumption: **all input feature vectors are statically fixed in dimension, independent, and identically distributed (i.i.d.)**. 

However, real-world data—such as natural language, audio waveforms, genomic DNA strings, financial stock tickers, and robot trajectory sensors—are **temporal sequences of variable length**. A word's meaning depends crucially on words spoken dozens of steps prior; a sudden price spike depends on the preceding trend.

To process sequences without blowing up the parameter space, we require an architecture that satisfies three core requirements:
1. **Variable-Length Processing:** The ability to ingest sequences of arbitrary temporal length $T$.
2. **Temporal Equivariance & Parameter Sharing:** A grammatical or physical pattern learned at timestep $t=2$ must apply identically if it occurs at timestep $t=50$.
3. **Internal State Memory:** The ability to maintain an evolving summary vector of all past observations.

The classical solution is the **Elman Recurrent Neural Network (RNN)** (Elman, 1990), which processes sequential inputs recurrence-by-recurrence using a hidden state vector $h_t$ that acts as the network's working memory. Training this network requires unrolling the computational graph across time, producing the algorithm known as **Backpropagation Through Time (BPTT)**.

```
       UNROLLED RECURRENT COMPUTATIONAL GRAPH ACROSS TIME:
       
         L_1              L_2              L_3                   L_T
          ▲                ▲                ▲                     ▲
          │                │                │                     │
       [ y_1 ]          [ y_2 ]          [ y_3 ]               [ y_T ]
          ▲                ▲                ▲                     ▲
      W_hy│            W_hy│            W_hy│                 W_hy│
          │     W_hh       │     W_hh       │          W_hh       │
[h_0] ──►[h_1] ───────► [h_2] ───────► [h_3] ─── ··· ───────► [h_T]
          ▲                ▲                ▲                     ▲
      W_xh│            W_xh│            W_xh│                 W_xh│
          │                │                │                     │
        [x_1]            [x_2]            [x_3]                 [x_T]
```

---

### The 12-Year-Old Analogy: The Conveyor Belt with a Fading Notepad

Imagine you are an inspector standing next to a long factory conveyor belt:
- Boxes of parts roll by you, one box per second ($x_1, x_2, x_3, \dots$).
- You hold a small pocket notepad with room for only **one single sentence** ($h_t$).
- You have an instruction rulebook ($W_{xh}, W_{hh}, b_h$) that tells you: *"Read the current box ($x_t$), read what you previously wrote on your notepad ($h_{t-1}$), erase the notepad, and write a new 1-sentence summary ($h_t$)."*
- At every second, your supervisor looks at your notepad and demands an immediate decision ($y_t = \text{inspect}(h_t)$).

**The Forward Pass:**
Box 1 comes in. You write note 1. Box 2 arrives. You combine Box 2 with note 1 to write note 2. Box 3 arrives. You combine Box 3 with note 2 to write note 3. The same rulebook is used on every single box!

**The Backward Pass (Blame Assignment):**
At second 20, you make a catastrophic mistake on Box 20 ($L_{20}$). Your boss yells: *"Why did you make that mistake?"*
You answer: *"I made the mistake because of what was written on my notepad at second 20."*
Your boss asks: *"Why was that on your notepad at second 20?"*
You answer: *"Because of what was on my notepad at second 19 and what was in Box 20!"*
Your boss keeps asking: *"Why was that on notepad 19? Notepad 18? ... Notepad 1?"*

This is **Backpropagation Through Time (BPTT)**. To assign blame to your rulebook, you must trace the chain of influence backwards in time through all 20 notepads.

**The Vanishing Notepad Problem:**
Every time you erase your notepad and write a new summary, you compress the old notes with a squishing marker ($\tanh$, which squeezes everything between $-1$ and $+1$). If your marker fades out 10% of the ink on every step:
- After 1 step: $90\%$ of the memory remains ($0.9$).
- After 10 steps: $0.9^{10} \approx 0.35$ remains.
- After 20 steps: $0.9^{20} \approx 0.12$ remains.
- After 50 steps: $0.9^{50} \approx 0.0051$ (almost zero!).

By the time you look back 20 steps, the ink has completely faded away! You cannot assign any blame to Box 1, even if Box 1 was the root cause of the error. This is the **Vanishing Gradient Dilemma**. Conversely, if your marker makes words 2x darker each step, after 20 steps the page is covered in massive black blotches ($2^{20} \approx 1,000,000$), tearing the paper. That is the **Exploding Gradient Dilemma**!

---

## 2. Mathematical Symbol Decoder

| Symbol | Mathematical Domain | Description & Dimensions |
| :--- | :--- | :--- |
| $T \in \mathbb{N}$ | Discrete Time Index | Total temporal sequence length |
| $x_t \in \mathbb{R}^{d_x}$ | Vector Space | Input feature vector at timestep $t \in \{1, \dots, T\}$ |
| $h_t \in \mathbb{R}^{d_h}$ | Vector Space | Hidden state vector at timestep $t$ (internal memory) |
| $h_0 \in \mathbb{R}^{d_h}$ | Vector Space | Initial hidden state boundary condition (typically $\mathbf{0}$) |
| $W_{xh} \in \mathbb{R}^{d_h \times d_x}$ | Linear Transformation | Input-to-hidden weight transition matrix |
| $W_{hh} \in \mathbb{R}^{d_h \times d_h}$ | Linear Transformation | Hidden-to-hidden recurrent transition matrix |
| $b_h \in \mathbb{R}^{d_h}$ | Vector Space | Recurrent hidden layer bias |
| $W_{hy} \in \mathbb{R}^{d_y \times d_h}$ | Linear Transformation | Hidden-to-output projection matrix |
| $b_y \in \mathbb{R}^{d_y}$ | Vector Space | Output layer bias |
| $\hat{y}_t \in \Delta^{d_y - 1}$ | Probability Simplex | Predicted probability distribution over $d_y$ classes at time $t$ |
| $y_t \in \{0, 1\}^{d_y}$ | Indicator / Distribution | Ground truth target vector at timestep $t$ |
| $L_t \in \mathbb{R}_{\ge 0}$ | Scalar Field | Cross-entropy loss at timestep $t$: $-\sum_k y_{t,k} \log \hat{y}_{t,k}$ |
| $L \in \mathbb{R}_{\ge 0}$ | Scalar Field | Total sequence objective loss: $L = \sum_{t=1}^T L_t$ |
| $J_{t, k} \equiv \frac{\partial h_t}{\partial h_k}$ | Jacobian Matrix $\mathbb{R}^{d_h \times d_h}$ | Sensitivity of hidden state $h_t$ with respect to state $h_k$ ($k \le t$) |
| $\rho(W_{hh})$ | Spectral Theory | Spectral radius: $\max \{|\lambda| : \det(W_{hh} - \lambda I) = 0\}$ |
| $\tau \in \mathbb{R}^+$ | Hyperparameter | Gradient clipping norm threshold |

---

## 3. The Elman RNN Cell & Parameter Sharing

### 3.1 Why Fixed Feedforward Networks Fail on Sequences
Consider feeding a sentence of text into a multi-layer perceptron (MLP). An MLP requires a fixed input vector of dimension $D$. If sentence 1 has 5 words and sentence 2 has 40 words, one must either truncate or zero-pad up to some maximum length $D_{\max}$. 

More critically, an MLP **does not share parameters across temporal positions**:
$$y = \sigma(W_1 x_1 + W_2 x_2 + \dots + W_T x_T + b)$$
If the word "not" appears at position $t=1$, the network uses weight matrix $W_1$. If "not" appears at position $t=4$, it uses completely distinct weight matrix $W_4$. The model cannot generalize that negation operates identically regardless of temporal placement.

### 3.2 The Elman RNN Forward Equations
The Elman RNN solves this by applying the **exact same parameter tensors** $(W_{xh}, W_{hh}, b_h, W_{hy}, b_y)$ recursively across every timestep:

$$\begin{aligned}
a_t &= W_{xh} x_t + W_{hh} h_{t-1} + b_h \\
h_t &= \tanh(a_t) \\
z_t &= W_{hy} h_t + b_y \\
\hat{y}_t &= \text{softmax}(z_t) = \frac{\exp(z_t)}{\sum_{k=1}^{d_y} \exp(z_{t, k})}
\end{aligned}$$

The total loss across the entire sequence is the sum of per-timestep losses:
$$L = \sum_{t=1}^T L_t = -\sum_{t=1}^T \sum_{k=1}^{d_y} y_{t, k} \log \hat{y}_{t, k}$$

Because parameters are shared across all timesteps, computing their gradients requires accumulating the gradient contributions from every timestep in the sequence.

---

## 4. Backpropagation Through Time (BPTT) Derivation

To train the RNN with gradient descent, we must compute:
$$\frac{\partial L}{\partial W_{hy}}, \quad \frac{\partial L}{\partial b_y}, \quad \frac{\partial L}{\partial W_{hh}}, \quad \frac{\partial L}{\partial W_{xh}}, \quad \frac{\partial L}{\partial b_h}$$

### 4.1 Output Layer Gradients
The output layer parameters affect loss $L$ directly through each step's output logit $z_t$:
$$\frac{\partial L_t}{\partial z_t} = \hat{y}_t - y_t \in \mathbb{R}^{d_y}$$

By the multivariable chain rule:
$$\frac{\partial L}{\partial W_{hy}} = \sum_{t=1}^T \frac{\partial L_t}{\partial W_{hy}} = \sum_{t=1}^T (\hat{y}_t - y_t) h_t^T$$
$$\frac{\partial L}{\partial b_y} = \sum_{t=1}^T \frac{\partial L_t}{\partial b_y} = \sum_{t=1}^T (\hat{y}_t - y_t)$$

### 4.2 Hidden State Adjoint Recursion
The hidden state $h_t$ influences the loss in two distinct ways:
1. **Directly:** via the current output prediction $\hat{y}_t$ and loss $L_t$.
2. **Indirectly:** via its propagation forward into future states $h_{t+1}, h_{t+2}, \dots, h_T$.

Therefore, the adjoint vector $\frac{\partial L}{\partial h_t}$ obeys the reverse recurrence relation:
$$\frac{\partial L}{\partial h_t} = \frac{\partial L_t}{\partial h_t} + \left( \frac{\partial h_{t+1}}{\partial h_t} \right)^T \frac{\partial L}{\partial h_{t+1}}$$

Where:
$$\frac{\partial L_t}{\partial h_t} = W_{hy}^T (\hat{y}_t - y_t)$$

And the local Jacobian between successive hidden states is:
$$\frac{\partial h_{t+1}}{\partial h_t} = \text{diag}(1 - h_{t+1}^2) W_{hh}$$

Substituting this in gives:
$$\frac{\partial L}{\partial h_t} = W_{hy}^T (\hat{y}_t - y_t) + W_{hh}^T \left( \frac{\partial L}{\partial h_{t+1}} \odot (1 - h_{t+1}^2) \right)$$

### 4.3 Recurrent Parameter Gradients
Let $\delta_t = \frac{\partial L}{\partial a_t} = \frac{\partial L}{\partial h_t} \odot (1 - h_t^2)$.
Since $W_{hh}$, $W_{xh}$, and $b_h$ are reused at every step:
$$\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^T \delta_t h_{t-1}^T$$
$$\frac{\partial L}{\partial W_{xh}} = \sum_{t=1}^T \delta_t x_t^T$$
$$\frac{\partial L}{\partial b_h} = \sum_{t=1}^T \delta_t$$

---

## 5. The Vanishing & Exploding Gradient Dilemma

### 5.1 The Jacobian Product Expansion
Consider an error signal occurring at timestep $t$ ($L_t$) and trace its gradient all the way back to an early hidden state $h_k$ ($k < t$):
$$\frac{\partial L_t}{\partial h_k} = \frac{\partial L_t}{\partial h_t} \frac{\partial h_t}{\partial h_k} = \frac{\partial L_t}{\partial h_t} \prod_{j=k+1}^t \frac{\partial h_j}{\partial h_{j-1}}$$

Expanding each step's transition Jacobian:
$$\frac{\partial h_j}{\partial h_{j-1}} = D_j W_{hh}$$
where $D_j = \text{diag}(1 - \tanh^2(a_j)) \in \mathbb{R}^{d_h \times d_h}$.

Thus, the composite Jacobian across $t - k$ steps is an alternating product of diagonal derivative matrices and the recurrent weight matrix:
$$\frac{\partial h_t}{\partial h_k} = \prod_{j=k+1}^t (D_j W_{hh}) = D_t W_{hh} D_{t-1} W_{hh} \cdots D_{k+1} W_{hh}$$

### 5.2 Mathematical Analysis of Exponential Decay / Explosion
Let us examine the operator norm (induced L2 matrix norm) of this product:
$$\left\| \frac{\partial h_t}{\partial h_k} \right\|_2 \le \prod_{j=k+1}^t \|D_j\|_2 \|W_{hh}\|_2$$

Observe the properties of the diagonal matrix $D_j$:
$$D_j(i, i) = 1 - \tanh^2(a_{j, i}) \in (0, 1]$$
Therefore, $\|D_j\|_2 = \max_i |1 - \tanh^2(a_{j, i})| \le 1.0$.

Let $\gamma = \|W_{hh}\|_2$ (the largest singular value of $W_{hh}$, bounded below by its spectral radius $\rho(W_{hh})$):

1. **Case 1: Vanishing Gradients ($\gamma < 1$):**
   $$\left\| \frac{\partial h_t}{\partial h_k} \right\|_2 \le \gamma^{t-k}$$
   As the temporal gap $\Delta t = t - k$ increases, $\gamma^{\Delta t} \to 0$ exponentially fast!
   For example, if $\gamma = 0.8$ and $\Delta t = 30$, then:
   $$\gamma^{30} = 0.8^{30} \approx 0.0012$$
   By the time gradients reach $h_0$, the signal has shrunk by 99.9%. **The network cannot learn long-term dependencies.**

2. **Case 2: Exploding Gradients ($\gamma > 1$):**
   If $\|W_{hh}\|_2 > 1$ and the neurons are operating in their linear regime ($a_j \approx 0 \implies D_j \approx I$):
   $$\left\| \frac{\partial h_t}{\partial h_k} \right\|_2 \approx \gamma^{t-k} \to \infty$$
   For $\gamma = 1.5$ and $\Delta t = 20$:
   $$1.5^{20} \approx 3325.25$$
   The gradients explode, producing floating-point overflow (`NaN` or `Inf`), causing weights to oscillate wildly and destroying training.

---

## 6. Gradient Clipping: Practical Safeguard Against Explosion

While vanishing gradients require fundamental architectural changes (such as the LSTM or GRU), **exploding gradients** can be effectively controlled using **Gradient Clipping** (Pascanu, Mikolov, & Bengio, 2013).

### 6.1 L2 Norm-Based Global Clipping
Instead of clipping weights elementwise (which distorts the gradient direction), norm-based clipping scales the entire gradient vector $\mathbf{g} = \nabla_\theta L$ proportionally if its Euclidean norm exceeds threshold $\tau$:

$$\mathbf{g} \leftarrow \begin{cases}
\mathbf{g} & \text{if } \|\mathbf{g}\|_2 \le \tau \\
\frac{\tau}{\|\mathbf{g}\|_2} \mathbf{g} & \text{if } \|\mathbf{g}\|_2 > \tau
\end{cases}$$

### 6.2 Key Geometric Properties of Norm Clipping
1. **Directional Invariance:** The cosine similarity between the original gradient $\mathbf{g}$ and the clipped gradient $\mathbf{g}_{\text{clip}}$ is strictly $1.0$:
   $$\cos(\mathbf{g}, \mathbf{g}_{\text{clip}}) = \frac{\mathbf{g} \cdot \left(\frac{\tau}{\|\mathbf{g}\|} \mathbf{g}\right)}{\|\mathbf{g}\| \left\|\frac{\tau}{\|\mathbf{g}\|} \mathbf{g}\right\|} = \frac{\tau \|\mathbf{g}\|^2}{\tau \|\mathbf{g}\|^2} = 1.0$$
2. **Bounded Step Size:** The maximum parameter update under learning rate $\eta$ is guaranteed not to exceed $\eta \cdot \tau$:
   $$\|\Delta \theta\|_2 = \eta \|\mathbf{g}_{\text{clip}}\|_2 \le \eta \tau$$

---

## 7. Numerical Verification & Summary

In `experiment.py`, we implemented an unrolled Elman RNN from scratch in pure Python and verified:
1. **GradCheck:** Analytical BPTT gradients matched two-sided finite difference numerical approximations with relative error **$< 7 \times 10^{-9}$**.
2. **Vanishing Gradient:** Over $T=20$ steps with $\|W_{hh}\| < 1$, gradient norm decayed from $0.256$ at step 19 down to $3.49 \times 10^{-7}$ at step 0 (a decay factor of $1.36 \times 10^{-6}$).
3. **Exploding Gradient Stabilization:** Under large recurrent weights, an exploding gradient of norm $798.9$ was clipped to exactly $\tau = 5.0$, with cosine similarity strictly equal to $1.000000$.
