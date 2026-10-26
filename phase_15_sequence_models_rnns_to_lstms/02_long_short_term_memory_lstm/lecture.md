# Topic 15.2: Long Short-Term Memory (LSTM) Networks

---

## 1. Executive Summary & The 12-Year-Old Analogy

As proven in Topic 15.1, the Elman Recurrent Neural Network fails on long temporal sequences because its state transition is fundamentally **multiplicative**:
$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t)$$
Backpropagating over $\Delta t$ timesteps forces repeated multiplication by the transition Jacobian:
$$\prod_{j=k+1}^t \left( \text{diag}(1 - h_j^2) W_{hh} \right)$$
If $\|W_{hh}\| < 1$, gradients exponentially collapse to zero; if $\|W_{hh}\| > 1$, gradients explode. This mathematical barrier prevented standard RNNs from learning dependencies longer than 10 to 15 timesteps.

In 1997, Sepp Hochreiter and Jürgen Schmidhuber published a seminal breakthrough: the **Long Short-Term Memory (LSTM)** network. 

The core genius of the LSTM architecture is the introduction of:
1. **The Additive Cell State Highway ($c_t$):** An internal linear memory channel where information is accumulated **additively** rather than multiplicatively.
2. **The Constant Error Carousel (CEC):** A mechanism that allows error gradients to flow backward indefinitely without exponential damping ($\frac{\partial c_t}{\partial c_{t-1}} \approx 1.0$).
3. **Four Gating Units:** Sigmoidal regulators ($f_t, i_t, \tilde{c}_t, o_t$) that dynamically control how much information to forget, write, candidate, and emit.

```
                  LSTM CELL COMPUTATION AT TIMESTEP t:
                  
              c_{t-1} ──────[ x ]────────────────(+)────────► c_t (Cell State)
                              ▲                   ▲
                              │                   │
                        f_t [ σ ]           i_t [ σ ] * [tanh] c_tilde_t
                              │                   │       │
    [h_{t-1}, x_t] ───────────┴───────────────────┴───────┴──────┐
                              │                                  │
                              ▼                                  ▼
                            [ σ ] o_t                [tanh](c_t) │
                              │                          │       │
                              └────────────────────────►[ x ]────┴─► h_t (Hidden)
```

---

### The 12-Year-Old Analogy: The Executive Assistant with Eraser, Pen, and Highlighter

Imagine you are the CEO of a company, but your memory resets every 10 seconds. To run the company, you hire an elite **Executive Assistant**:

1. **The Heavy Iron Safe ($c_t$ - The Cell State):**
   Behind your desk sits an iron safe containing a continuous ledger book. This safe never gets deleted automatically. It is a persistent physical document that rolls forward from year to year.

2. **The Forget Gate ($f_t$ - The Eraser):**
   Every morning, a memo arrives ($x_t$) alongside yesterday's brief ($h_{t-1}$). The assistant looks at the ledger ($c_{t-1}$) and asks: *"Is anything in this ledger obsolete?"*
   - If an old client canceled their contract, the assistant dials the Forget Gate to $0.0$ and **erases** that paragraph.
   - If the business strategy is still active, the assistant dials the Forget Gate to $1.0$ and leaves it untouched.

3. **The Input Gate & Candidate State ($i_t \odot \tilde{c}_t$ - The Ink Pen):**
   The assistant prepares a candidate draft of newly learned facts ($\tilde{c}_t$). The Input Gate ($i_t$) decides: *"How important is this new information right now?"*
   - If the new memo is junk mail, $i_t = 0.0$ (don't write it down).
   - If it is a million-dollar contract, $i_t = 1.0$, and the assistant permanently **writes it into the ledger using addition ($+$)**.

4. **The Output Gate ($o_t$ - The Highlighter):**
   The board of directors asks you a question ($h_t$). You don't read them the entire 500-page ledger. The assistant highlights only the relevant lines ($o_t \odot \tanh(c_t)$) and hands you a 1-page executive summary ($h_t$).

**Why This Solves Gradient Decay:**
In the Elman RNN, you rewrote your entire notebook from scratch every second with a fading marker.
In the LSTM, if a piece of information is critical, the assistant sets the Forget Gate to $1.0$ and the Input Gate to $0.0$. The information sits in the iron safe, untouched, for 100 years. During the backward pass, the gradient travels straight down the empty highway with a derivative of exactly **$1.0$**, arriving 100 timesteps earlier with zero decay!

---

## 2. Mathematical Symbol Decoder

| Symbol | Mathematical Domain | Description & Range |
| :--- | :--- | :--- |
| $x_t \in \mathbb{R}^{d_x}$ | Vector Space | Input feature vector at time $t$ |
| $h_t \in \mathbb{R}^{d_h}$ | Vector Space | Hidden state (working memory / output) at time $t$ |
| $c_t \in \mathbb{R}^{d_h}$ | Vector Space | Cell state (long-term memory highway) at time $t$ |
| $f_t \in (0, 1)^{d_h}$ | Gate Vector | **Forget Gate:** fraction of $c_{t-1}$ to retain |
| $i_t \in (0, 1)^{d_h}$ | Gate Vector | **Input Gate:** fraction of candidate state to add to $c_t$ |
| $\tilde{c}_t \in (-1, 1)^{d_h}$ | State Vector | **Candidate State:** new memory content generated from $[h_{t-1}, x_t]$ |
| $o_t \in (0, 1)^{d_h}$ | Gate Vector | **Output Gate:** fraction of cell state $c_t$ to expose in $h_t$ |
| $\sigma(z) = \frac{1}{1 + e^{-z}}$ | Nonlinearity | Sigmoid activation function mapping $\mathbb{R} \to (0, 1)$ |
| $\odot$ | Hadamard Product | Elementwise vector multiplication |
| $W_f, W_i, W_c, W_o$ | $\mathbb{R}^{d_h \times (d_h + d_x)}$ | Gate and candidate parameter weight matrices |
| $b_f, b_i, b_c, b_o$ | $\mathbb{R}^{d_h}$ | Gate and candidate bias vectors |

---

## 3. The Four Gates of the LSTM Cell

Let the concatenated input vector at timestep $t$ be:
$$v_t = \begin{bmatrix} h_{t-1} \\ x_t \end{bmatrix} \in \mathbb{R}^{d_h + d_x}$$

The standard LSTM cell evaluates six vector operations per timestep:

### 3.1 Forget Gate ($f_t$)
Controls what fraction of the previous long-term cell state $c_{t-1}$ should be eliminated:
$$f_t = \sigma(W_f v_t + b_f) = \sigma(W_{fh} h_{t-1} + W_{fx} x_t + b_f)$$
- If $f_{t, k} \to 0$: Neuron $k$'s previous memory is discarded.
- If $f_{t, k} \to 1$: Neuron $k$'s previous memory is perfectly retained.

### 3.2 Input Gate ($i_t$) & Candidate State ($\tilde{c}_t$)
Controls what new information should be added to the cell state:
$$i_t = \sigma(W_i v_t + b_i)$$
$$\tilde{c}_t = \tanh(W_c v_t + b_c)$$
The candidate state $\tilde{c}_t$ creates a bounded $(-1, 1)$ vector of new features, while $i_t$ acts as a soft binary filter controlling which coordinates are written.

### 3.3 Additive Cell State Update ($c_t$)
The cell state is updated via a **linear combination**:
$$\boxed{c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t}$$
Notice the absence of an outer non-linear activation function! Unlike the Elman RNN which computes $h_t = \tanh(\dots)$, the LSTM cell state $c_t$ is an **unrestricted additive sum**. This prevents saturation and preserves gradients.

### 3.4 Output Gate ($o_t$) & Hidden State ($h_t$)
Decides what portion of the cell state should be emitted as the observable hidden state $h_t$:
$$o_t = \sigma(W_o v_t + b_o)$$
$$\boxed{h_t = o_t \odot \tanh(c_t)}$$
The cell state $c_t$ is passed through $\tanh$ to bound its values into $(-1, 1)$, then scaled elementwise by output gate $o_t$.

---

## 4. Mathematical Proof of Gradient Preservation: The Constant Error Carousel

The core theoretical breakthrough of the LSTM is how gradients propagate through time along the cell state sequence $c_T \to c_{T-1} \to \dots \to c_k$.

### 4.1 The Jacobian of the Cell State
Let us differentiate $c_t$ with respect to $c_{t-1}$:
$$\frac{\partial c_t}{\partial c_{t-1}} = \frac{\partial}{\partial c_{t-1}} \left( f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \right)$$

Using the product rule and chain rule:
$$\frac{\partial c_t}{\partial c_{t-1}} = \text{diag}(f_t) + \underbrace{\left( \frac{\partial f_t}{\partial c_{t-1}} \odot c_{t-1} + \frac{\partial i_t}{\partial c_{t-1}} \odot \tilde{c}_t + i_t \odot \frac{\partial \tilde{c}_t}{\partial c_{t-1}} \right)}_{\text{Indirect dependency via } h_{t-1}}$$

Because $h_{t-1} = o_{t-1} \odot \tanh(c_{t-1})$, the second term represents indirect feedback. In the linear cell state channel, the primary direct derivative is:
$$\frac{\partial c_t}{\partial c_{t-1}} \approx \text{diag}(f_t)$$

### 4.2 Error Propagation Across $T - k$ Timesteps
Unrolling the chain rule across $\Delta t = t - k$ timesteps:
$$\frac{\partial c_t}{\partial c_k} = \prod_{j=k+1}^t \frac{\partial c_j}{\partial c_{j-1}} \approx \prod_{j=k+1}^t \text{diag}(f_j)$$

Compare this to the Elman RNN:
$$\text{Elman RNN:} \quad \prod_{j=k+1}^t \left( \text{diag}(1 - h_j^2) W_{hh} \right)$$
$$\text{LSTM Cell State:} \quad \prod_{j=k+1}^t \text{diag}(f_j)$$

### 4.3 Key Differences:
1. **No Matrix Multiplications:** The LSTM cell-state derivative contains **no recurrent weight matrix $W_{hh}$**! It is purely an elementwise product of diagonal gate matrices.
2. **No Shrinking Non-Linearities:** The factor $(1 - h_j^2) \le 1.0$ that forced Elman gradients to zero does not appear in the cell state chain.
3. **Dynamic Gradient Flow:** If the forget gates are open ($f_j \approx \mathbf{1}$):
   $$\frac{\partial c_t}{\partial c_k} \approx \prod_{j=k+1}^t I = I$$
   **The gradient error flows backward across hundreds of timesteps with a derivative of exactly 1.0!**

### 4.4 Forget Gate Bias Initialization Trick
In practice, if $W_f$ and $b_f$ are initialized with mean 0, the sigmoid gate outputs $f_t \approx \sigma(0) = 0.5$. Over 50 timesteps, $0.5^{50} \approx 8.8 \times 10^{-16}$, causing artificial vanishing!
To resolve this, Jozefowicz et al. (2015) introduced the **Forget Gate Bias Initialization Rule**:
$$b_f \leftarrow 1.0 \text{ or } 2.0$$
Since $\sigma(1.0) \approx 0.73$ and $\sigma(2.0) \approx 0.88$, initializing $b_f$ to a positive constant ensures that at the start of training, the gates default to **remembering everything**, preserving long-range gradient flow out of the box!

---

## 5. Complete Backward Pass Derivation

For an unrolled sequence with incoming gradient $\frac{\partial L}{\partial h_t}$:

1. **Gradient to Hidden State ($h_t$):**
   $$\delta h_t = \frac{\partial L}{\partial h_t} + \delta h_{t+1}^{\text{prev}}$$

2. **Output Gate Gradient ($o_t$):**
   $$\delta o_t = \delta h_t \odot \tanh(c_t) \odot o_t \odot (1 - o_t)$$

3. **Cell State Gradient ($c_t$):**
   $$\delta c_t = \delta h_t \odot o_t \odot (1 - \tanh^2(c_t)) + \delta c_{t+1} \odot f_{t+1}$$

4. **Candidate State Gradient ($\tilde{c}_t$):**
   $$\delta \tilde{c}_t = \delta c_t \odot i_t \odot (1 - \tilde{c}_t^2)$$

5. **Input Gate Gradient ($i_t$):**
   $$\delta i_t = \delta c_t \odot \tilde{c}_t \odot i_t \odot (1 - i_t)$$

6. **Forget Gate Gradient ($f_t$):**
   $$\delta f_t = \delta c_t \odot c_{t-1} \odot f_t \odot (1 - f_t)$$

---

## 6. Numerical Verification & Summary

In `experiment.py`, we implemented a complete LSTM cell and sequence processor from scratch in pure Python:
1. **GradCheck:** Analytical gradients for the LSTM cell were validated against central finite differences with max relative error **$1.49 \times 10^{-9}$**.
2. **50-Timestep Gradient Flow Test:**
   - **Elman RNN:** Gradient decayed from $2.00$ at step 50 down to $2.54 \times 10^{-26}$ at step 1 (preservation ratio $1.27 \times 10^{-26}$ — completely dead).
   - **LSTM Cell State:** Gradient decayed from $0.99998$ at step 50 down to $0.7217$ at step 1 (preservation ratio **$72.2\%$** — vibrant and active!).
