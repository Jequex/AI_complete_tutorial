# Topic 15.3: Exercises & Solutions — GRUs & Bidirectionality

---

### Problem 1: Manual Step-by-Step GRU Forward Pass (Hard)

Consider a 1-dimensional GRU cell ($d_x = 1, d_h = 1$) with the following parameters:
- Reset gate: $W_r = [0.0, 0.0]$, $b_r = -2.0$
- Update gate: $W_z = [0.0, 0.0]$, $b_z = 1.0$
- Candidate state weights: $W_{hh} = 2.0$, $W_{xh} = 1.0$, $b_h = 0.0$

Initial hidden state: $h_0 = 1.0$.
Input at $t = 1$: $x_1 = 0.5$.

Useful approximations:
- $\sigma(-2.0) \approx 0.1192$
- $\sigma(1.0) \approx 0.7311$
- $\tanh(0.7384) \approx 0.6282$

**Questions:**
1. Compute the reset gate activation $r_1$.
2. Compute the update gate activation $z_1$.
3. Compute the candidate hidden state $\tilde{h}_1$.
4. Compute the final updated hidden state $h_1$.
5. How does $r_1 \approx 0.1192$ affect the influence of $h_0$ on the candidate state?

---

#### Solution

1. **Reset Gate ($r_1$):**
   $$v_1 = [h_0, x_1]^T = [1.0, 0.5]^T$$
   $$a_r = 0.0(1.0) + 0.0(0.5) - 2.0 = -2.0$$
   $$r_1 = \sigma(-2.0) \approx 0.1192$$

2. **Update Gate ($z_1$):**
   $$a_z = 0.0(1.0) + 0.0(0.5) + 1.0 = 1.0$$
   $$z_1 = \sigma(1.0) \approx 0.7311$$

3. **Candidate Hidden State ($\tilde{h}_1$):**
   The reset gate scales the previous hidden state:
   $$r_1 \cdot h_0 = 0.1192 \times 1.0 = 0.1192$$
   The candidate pre-activation is:
   $$a_h = W_{hh}(r_1 \cdot h_0) + W_{xh} x_1 + b_h = 2.0(0.1192) + 1.0(0.5) + 0.0 = 0.2384 + 0.5000 = 0.7384$$
   $$\tilde{h}_1 = \tanh(0.7384) \approx 0.6282$$

4. **Final Hidden State Update ($h_1$):**
   Interpolate between $h_0$ and $\tilde{h}_1$ using $z_1$:
   $$h_1 = (1 - z_1) h_0 + z_1 \tilde{h}_1$$
   $$1 - z_1 = 1 - 0.7311 = 0.2689$$
   $$h_1 = (0.2689)(1.0) + (0.7311)(0.6282) = 0.2689 + 0.4593 = 0.7282$$

5. **Physical Role of Reset Gate:**
   Because $r_1 \approx 0.1192 \ll 1.0$, the prior memory $h_0$ was suppressed by over $88\%$ when evaluating candidate $\tilde{h}_1$. The candidate was dominated by the immediate input $x_1 = 0.5$ ($0.5000$ vs $0.2384$). This demonstrates the reset gate's role in clearing old context.

---

### Problem 2: Bidirectional Concatenation vs Hidden Size Trade-off (Medium)

Suppose you are designing a sequence classifier for clinical records. You are comparing:
- Architecture A: A Unidirectional LSTM with hidden size $d_h = 512$.
- Architecture B: A Bidirectional GRU where each direction has hidden size $d_h^{\text{dir}} = 256$, so that the final concatenated representation $[\vec{h}_t; \overleftarrow{h}_t]$ has total dimension $2 \times 256 = 512$.

Let input dimension be $d_x = 100$.

**Questions:**
1. Compute the total recurrent parameters for Architecture A.
2. Compute the total recurrent parameters for Architecture B (forward GRU + backward GRU).
3. Which architecture has fewer parameters, and by what percentage?

#### Solution

1. **Architecture A (Unidirectional LSTM, $d_h = 512, d_x = 100$):**
   $$P_{\text{LSTM}} = 4 \times d_h(d_h + d_x + 1) = 4 \times 512(512 + 100 + 1) = 2048 \times 613 = 1,255,424\text{ parameters}$$

2. **Architecture B (Bidirectional GRU, $d_h = 256, d_x = 100$):**
   Each direction is a GRU with hidden dimension $256$:
   $$P_{\text{GRU}}^{\text{dir}} = 3 \times 256(256 + 100 + 1) = 768 \times 357 = 274,176\text{ parameters}$$
   Since there are two independent directions (forward and backward):
   $$P_{\text{BiGRU}} = 2 \times 274,176 = 548,352\text{ parameters}$$

3. **Comparison:**
   Architecture B has only $548,352$ parameters compared to $1,255,424$ for Architecture A.
   $$\text{Reduction} = \frac{1,255,424 - 548,352}{1,255,424} \times 100\% = \frac{707,072}{1,255,424} \times 100\% \approx 56.32\%$$
   Architecture B cuts the parameter count by **$56.3\%$** while producing the exact same 512-dimensional output embedding and adding bidirectional future context!

---

### Problem 3: Why Can't We Use Bidirectional RNNs for Autoregressive Language Generation? (Easy)

A student proposes using a Bidirectional LSTM as the decoder for generating text one word at a time in a conversational chatbot.

**Questions:**
1. Explain why bidirectional recurrence is mathematically incompatible with autoregressive generation during inference.
2. Under what condition CAN a bidirectional model be used in language modeling? (Hint: masked language modeling).

#### Solution

1. **Autoregressive Generation Incompatibility:**
   At test time, the chatbot generates token $y_t$ given only previous tokens $y_1, \dots, y_{t-1}$. The future tokens $y_{t+1}, \dots, y_T$ **do not yet exist**—they have not been generated!
   Because a backward recurrent network requires unrolling from the terminal token $y_T$ down to $y_t$ ($\overleftarrow{h}_t = f(y_t, \overleftarrow{h}_{t+1})$), it is mathematically impossible to run the backward pass without knowing the future tokens in advance.

2. **Masked Language Modeling (BERT Paradigm):**
   Bidirectional models can be trained when the entire sequence is available simultaneously (e.g. BERT or BiLSTM encoders for classification, tagging, or embeddings). In this setup, random tokens are replaced with a `[MASK]` token, and the network uses both left and right context to predict the masked token without needing to generate sequences left-to-right.
