# Phase 15: Sequence Models: RNNs to LSTMs

---

## 🧭 Executive Overview

Standard feedforward networks and convolutional architectures operate under the foundational assumption that all input features are statically fixed in dimension, independent, and identically distributed ($i.i.d.$). 

However, natural human communication, biological systems, and physical processes unfold **temporally**:
- Natural Language (sentences of variable word counts, nested dependent clauses)
- Genomic Sequences (DNA base pairs, protein chains)
- Financial Markets & Sensor Streams (variable-duration time-series)
- Audio Waveforms & Speech Signals

**Phase 15: Sequence Models: RNNs to LSTMs** establishes the mathematical and architectural foundations of sequential modeling:
1. **Recurrent Neural Networks (RNNs) & Backpropagation Through Time (BPTT):** How parameter sharing across time steps allows variable-length sequence ingestion, and why unrolling the recurrence reveals the catastrophic **Vanishing and Exploding Gradient Dilemma**.
2. **Long Short-Term Memory (LSTM):** How Hochreiter & Schmidhuber (1997) bypassed the multiplicative Jacobian bottleneck by creating the **Constant Error Carousel (CEC)** and an **Additive Cell State Highway ($c_t$)** regulated by four specialized gates.
3. **Gated Recurrent Units (GRU) & Bidirectionality:** How Cho et al. (2014) reduced parameter counts by $25\%$ by fusing cell state and hidden state, and how Bidirectional processing breaks causal blindness to capture past and future context simultaneously.
4. **Sequence-to-Sequence (Seq2Seq) & Bahdanau Additive Attention:** How the fixed-length information bottleneck in encoder-decoder models was dismantled by introducing dynamic, content-based alignment—laying the direct foundation for the Transformer architecture.

---

## 🗺️ Phase Syllabus & Directory Architecture

```text
phase_15_sequence_models_rnns_to_lstms/
├── index.html                                            # Interactive Phase Portal Hub & Visual Syllabus
├── README.md                                             # Master Mathematical & Architectural Reference Guide
│
├── 01_recurrent_neural_networks_and_bptt/
│   ├── lecture.html / lecture.md                         # Temporal parameter sharing, Elman RNN cell, BPTT derivation,
│   │                                                     # Jacobian products, spectral radius rho(W_hh), gradient clipping
│   ├── exercises_and_solutions.html / .md                # Step-by-step 2-step manual BPTT, induction proof of vanishing bound,
│   │                                                     # L2 norm clipping geometry
│   └── experiment.py                                     # Pure Python 3 unrolled RNN, finite-difference GradCheck (<7e-9),
│                                                         # 20-step vanishing demonstration (1.36e-6), clipping stabilization
│
├── 02_long_short_term_memory_lstm/
│   ├── lecture.html / lecture.md                         # Constant Error Carousel, additive cell state c_t, the four gates,
│   │                                                     # proof of gradient preservation (f_t approx 1), forget bias init
│   ├── exercises_and_solutions.html / .md                # Step-by-step manual LSTM forward pass, parameter scaling law formula,
│   │                                                     # identity Jacobian gradient highway proof
│   └── experiment.py                                     # Pure Python 3 complete LSTM cell, analytical GradCheck (<1.5e-9),
│                                                         # 50-step gradient flow benchmark (Elman RNN 1.27e-26 vs LSTM 72.2%)
│
├── 03_gated_recurrent_units_and_bidirectionality/
│   ├── lecture.html / lecture.md                         # Cho et al. GRU architecture, Reset/Update gates, 25% parameter savings,
│   │                                                     # Bidirectional RNNs (forward + backward passes), causal limitations
│   ├── exercises_and_solutions.html / .md                # Manual GRU forward pass, BiGRU vs UniLSTM parameter trade-off,
│   │                                                     # why autoregressive generation prohibits bidirectionality
│   └── experiment.py                                     # Pure Python 3 GRU cell, parameter count comparison (25.0% reduction),
│                                                         # GradCheck (<7.8e-10), palindrome symmetry detection (50.0% vs 97.0%)
│
└── 04_seq2seq_and_bahdanau_attention/
    ├── lecture.html / lecture.md                         # Sutskever Seq2Seq, information bottleneck capacity flaw, Bahdanau
    │                                                     # additive attention, alignment energies e_ij, alpha_ij, dynamic context c_i
    ├── exercises_and_solutions.html / .md                # Manual Bahdanau attention calculations, Additive vs Dot-Product attention,
    │                                                     # Shannon channel capacity analysis
    └── experiment.py                                     # Pure Python 3 Encoder-Decoder Seq2Seq with Bahdanau attention,
                                                          # sum_j alpha_ij == 1.0 assertion (<1e-15), reversal attention heatmap
```

---

## 🔑 Core Mathematical Pillars Mastered in Phase 15

### 1. The Recurrent Jacobian Product & Gradient Vanishing
Tracing an error signal at timestep $t$ back to an early state $h_k$ ($k < t$) produces an alternating product of transition Jacobians:
$$\frac{\partial L_t}{\partial h_k} = \frac{\partial L_t}{\partial h_t} \prod_{j=k+1}^t \left( \text{diag}(1 - \tanh^2(a_j)) W_{hh} \right)$$
If the spectral norm $\|W_{hh}\|_2 \le \gamma < 1$, the gradient norm is strictly bounded by:
$$\left\| \frac{\partial h_t}{\partial h_k} \right\|_2 \le \gamma^{t-k}$$
As $t - k \to \infty$, the gradient vanishes exponentially fast ($\gamma^{\Delta t} \to 0$).

### 2. The LSTM Additive Highway & Constant Error Carousel
By splitting hidden state into an external output $h_t$ and an internal additive cell state $c_t$:
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$
The Jacobian along the cell state channel does not involve any recurrent weight matrix multiplication:
$$\frac{\partial c_t}{\partial c_{t-1}} \approx \text{diag}(f_t)$$
When the forget gate is saturated open ($f_t \approx \mathbf{1}$), the gradient propagates backward through hundreds of timesteps with a derivative of exactly $1.0$.

### 3. The GRU Formulation & Convex State Interpolation
The GRU merges cell and hidden states and couples memory update and forgetting into a single convex combination:
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$
Requiring only 3 weight sets instead of 4, the GRU achieves an exact **$25.0\%$ parameter reduction** over an LSTM of identical capacity ($3 d_h(d_h + d_x + 1)$ vs $4 d_h(d_h + d_x + 1)$).

### 4. Bidirectional Context Integration
For tasks requiring sequence understanding, causal models fail on early tokens due to future blindness. Bidirectional models run two counter-directional passes:
$$h_t^{\text{Bi}} = \begin{bmatrix} \vec{h}_t \\ \overleftarrow{h}_t \end{bmatrix} \in \mathbb{R}^{2 d_h}$$
guaranteeing that every token representation $h_t^{\text{Bi}}$ conditions on the entire sequence $\mathcal{F}(x_1, \dots, x_T)$.

### 5. Bahdanau Additive Attention & The Dynamic Context Vector
Vanilla Seq2Seq compresses $T_x$ tokens into a single static vector $c = h_{T_x}$, hitting a hard Shannon information bottleneck. Bahdanau Attention replaces this with a content-addressable memory:
$$e_{ij} = v_a^T \tanh(W_a s_{i-1} + U_a h_j + b_a)$$
$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}, \quad c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j$$
Memory capacity scales dynamically as $\mathcal{O}(T_x)$, permitting high-fidelity alignment and translation regardless of sentence length.

---

## 🧪 Verification & Empirical Benchmarks

All code in Phase 15 is written in **pure Python 3 with zero external dependencies**. Running every test suite verifies the theoretical principles:

| Topic | Primary Benchmark | Analytical GradCheck | Empirical Finding |
| :--- | :--- | :--- | :--- |
| **15.1 RNN & BPTT** | 20-step gradient vanishing & norm clipping | RelErr: $6.86 \times 10^{-9}$ | Gradient decayed by factor of $1.36 \times 10^{-6}$; norm clipped to $\tau=5.0$ with $\cos(\theta)=1.000000$ |
| **15.2 LSTM** | 50-step gradient flow: LSTM vs Elman RNN | RelErr: $1.49 \times 10^{-9}$ | Elman RNN decayed to $1.27 \times 10^{-26}$; LSTM cell state preserved $72.2\%$ of original gradient |
| **15.3 GRU & BiRNN** | Parameter tally & Sequence symmetry | RelErr: $7.71 \times 10^{-10}$ | GRU achieves exact $25.0\%$ parameter saving; BiRNN achieved $97.0\%$ accuracy vs $50.0\%$ (chance) for UniRNN |
| **15.4 Seq2Seq & Attn** | Attention normalization & Reversal task | Normalization: $< 10^{-15}$ | Attention sum strictly equals $1.0000000000$; generated alignment heatmap for sequence reversal |

---

## 🚀 Quickstart & Verification Command

To execute all phase experiments in sequence and verify clean exit code 0:

```bash
# Topic 15.1: Recurrent Neural Networks & BPTT
python3 phase_15_sequence_models_rnns_to_lstms/01_recurrent_neural_networks_and_bptt/experiment.py

# Topic 15.2: Long Short-Term Memory (LSTM)
python3 phase_15_sequence_models_rnns_to_lstms/02_long_short_term_memory_lstm/experiment.py

# Topic 15.3: Gated Recurrent Units (GRU) & Bidirectionality
python3 phase_15_sequence_models_rnns_to_lstms/03_gated_recurrent_units_and_bidirectionality/experiment.py

# Topic 15.4: Seq2Seq Architecture & Bahdanau Attention
python3 phase_15_sequence_models_rnns_to_lstms/04_seq2seq_and_bahdanau_attention/experiment.py
```
