# Topic 15.2: Exercises & Solutions — Long Short-Term Memory (LSTM)

---

### Problem 1: Step-by-Step Manual Forward Pass of an LSTM Cell (Hard)

Consider a single-neuron LSTM cell ($d_x = 1, d_h = 1$) with the following parameters:
- Input vector at time $t$: $v_t = [h_{t-1}, x_t]^T$
- Forget gate: $W_f = [0.0, 0.0]$, $b_f = 2.0$
- Input gate: $W_i = [1.0, 2.0]$, $b_i = -1.0$
- Candidate state: $W_c = [0.5, 1.0]$, $b_c = 0.0$
- Output gate: $W_o = [0.0, 1.0]$, $b_o = 0.0$

Initial states at $t=0$:
- $h_0 = 0.5$
- $c_0 = 1.0$

At timestep $t = 1$, the input is $x_1 = 1.0$.
Useful numerical approximations:
- $\sigma(2.0) \approx 0.8808$
- $\sigma(1.5) \approx 0.8176$
- $\sigma(1.0) \approx 0.7311$
- $\tanh(1.25) \approx 0.8483$
- $\tanh(1.5744) \approx 0.9175$

**Questions:**
1. Compute the values of the four gates: $f_1, i_1, \tilde{c}_1, o_1$.
2. Compute the updated cell state $c_1$.
3. Compute the updated hidden state $h_1$.
4. What would happen to $c_1$ if the forget gate bias was $b_f = -10.0$ instead of $2.0$?

---

#### Solution

1. **Gate Evaluations:**
   - Concatenated input vector:
     $$v_1 = [h_0, x_1]^T = [0.5, 1.0]^T$$
   - **Forget Gate ($f_1$):**
     $$a_f = W_f v_1 + b_f = 0.0(0.5) + 0.0(1.0) + 2.0 = 2.0$$
     $$f_1 = \sigma(2.0) \approx 0.8808$$
   - **Input Gate ($i_1$):**
     $$a_i = W_i v_1 + b_i = 1.0(0.5) + 2.0(1.0) - 1.0 = 0.5 + 2.0 - 1.0 = 1.5$$
     $$i_1 = \sigma(1.5) \approx 0.8176$$
   - **Candidate State ($\tilde{c}_1$):**
     $$a_c = W_c v_1 + b_c = 0.5(0.5) + 1.0(1.0) + 0.0 = 0.25 + 1.0 = 1.25$$
     $$\tilde{c}_1 = \tanh(1.25) \approx 0.8483$$
   - **Output Gate ($o_1$):**
     $$a_o = W_o v_1 + b_o = 0.0(0.5) + 1.0(1.0) + 0.0 = 1.0$$
     $$o_1 = \sigma(1.0) \approx 0.7311$$

2. **Cell State Update ($c_1$):**
   $$c_1 = f_1 \cdot c_0 + i_1 \cdot \tilde{c}_1 = (0.8808)(1.0) + (0.8176)(0.8483)$$
   $$(0.8176)(0.8483) \approx 0.6936$$
   $$c_1 = 0.8808 + 0.6936 = 1.5744$$

3. **Hidden State Update ($h_1$):**
   $$h_1 = o_1 \cdot \tanh(c_1) = 0.7311 \cdot \tanh(1.5744) \approx 0.7311 \cdot 0.9175 \approx 0.6708$$

4. **Impact of Negative Forget Gate Bias:**
   If $b_f = -10.0$, then $a_f = -10.0$ and $f_1 = \sigma(-10.0) \approx 4.54 \times 10^{-5} \approx 0.0$.
   The cell state would become:
   $$c_1 = 0.0 \cdot c_0 + i_1 \cdot \tilde{c}_1 \approx 0.6936$$
   The previous memory $c_0 = 1.0$ would be completely purged, proving how the forget gate acts as an eraser on the long-term memory highway.

---

### Problem 2: Parameter Count Formula Derivation (Medium)

Given an LSTM layer with input feature dimension $d_x$ and hidden state dimension $d_h$:
1. Derive the formula for the total number of trainable parameters (weights and biases) in a standard LSTM layer.
2. Compare this against an Elman RNN with the same dimensions.
3. Calculate the exact parameter count for both architectures when $d_x = 128$ and $d_h = 512$.

#### Solution

1. **LSTM Parameter Count:**
   An LSTM has four distinct linear transforms (Forget, Input, Candidate, Output):
   - Each transform has a weight matrix of shape $d_h \times (d_h + d_x)$.
   - Each transform has a bias vector of shape $d_h$.
   - Parameters per transform: $d_h(d_h + d_x) + d_h = d_h(d_h + d_x + 1)$.
   - Since there are 4 gates/states:
     $$\boxed{P_{\text{LSTM}} = 4 \times \left[ d_h (d_h + d_x + 1) \right]}$$

2. **Elman RNN Parameter Count:**
   An Elman RNN has a single recurrent transform:
   - Weight $W_{hh}$: shape $d_h \times d_h$.
   - Weight $W_{xh}$: shape $d_h \times d_x$.
   - Bias $b_h$: shape $d_h$.
   - Total parameters:
     $$\boxed{P_{\text{RNN}} = d_h(d_h + d_x + 1)}$$
   Therefore, an LSTM contains **exactly 4 times** as many parameters as an Elman RNN of identical capacity.

3. **Numerical Comparison ($d_x = 128, d_h = 512$):**
   - $d_h + d_x + 1 = 512 + 128 + 1 = 641$.
   - $P_{\text{RNN}} = 512 \times 641 = 328,192$ parameters ($\approx 328\text{ K}$).
   - $P_{\text{LSTM}} = 4 \times 328,192 = 1,312,768$ parameters ($\approx 1.31\text{ M}$).

---

### Problem 3: Mathematical Proof of Gradient Highway in CEC (Medium)

Consider the cell state update $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$.
Suppose that for a sequence of $K$ timesteps, the inputs are uninformative noise such that the model sets $f_t = \mathbf{1}$ (all ones) and $i_t = \mathbf{0}$ (all zeros).

**Questions:**
1. Write the explicit relationship between $c_T$ and $c_{T-K}$.
2. Compute the exact Jacobian matrix $\frac{\partial c_T}{\partial c_{T-K}}$.
3. Explain why this proves that gradient signals cannot vanish under this condition.

#### Solution

1. For each timestep $t$ in the range:
   $$c_t = \mathbf{1} \odot c_{t-1} + \mathbf{0} \odot \tilde{c}_t = c_{t-1}$$
   Therefore, by unrolling across all $K$ steps:
   $$c_T = c_{T-1} = c_{T-2} = \dots = c_{T-K}$$

2. The derivative of an identity mapping is the identity matrix $I \in \mathbb{R}^{d_h \times d_h}$:
   $$\frac{\partial c_T}{\partial c_{T-K}} = \prod_{j=T-K+1}^T \frac{\partial c_j}{\partial c_{j-1}} = \prod_{j=1}^K I = I$$

3. Because $\frac{\partial c_T}{\partial c_{T-K}} = I$, for any error gradient $\frac{\partial L}{\partial c_T}$ arriving at step $T$:
   $$\frac{\partial L}{\partial c_{T-K}} = \left( \frac{\partial c_T}{\partial c_{T-K}} \right)^T \frac{\partial L}{\partial c_T} = I \cdot \frac{\partial L}{\partial c_T} = \frac{\partial L}{\partial c_T}$$
   The gradient vector passes back through $K$ timesteps with **magnitude decay ratio of exactly $1.000$** ($0\%$ loss of information). This is the mathematical definition of the **Constant Error Carousel (CEC)**.
