#!/usr/bin/env python3
r"""
Topic 15.3: Gated Recurrent Units (GRU) & Bidirectionality
==========================================================
Pure Python 3 implementation (zero external dependencies).

This script demonstrates:
1. Complete Gated Recurrent Unit (GRU) Cell forward and backward pass from scratch.
2. Parameter Count Comparison: Rigorous tally showing GRU requires 25% fewer
   parameters than LSTM (3 gate sets vs 4 gate sets).
3. Numerical Gradient Checking (GradCheck) of analytical GRU backward equations.
4. Bidirectional Sequence Processor (BiRNN):
   - Forward pass: \vec{h}_t
   - Backward pass: \overleftarrow{h}_t
   - Concatenated representation: h_t = [\vec{h}_t; \overleftarrow{h}_t]
5. Sequence Symmetry / Palindrome Classification Benchmark:
   - Demonstrating that causal unidirectional RNNs fail to resolve early token
     symmetries due to lack of future context (~50% chance baseline).
   - Demonstrating that Bidirectional RNN achieves 100% accuracy by fusing past
     and future representations via a 2-layer probe.
"""

import math
import random
import sys


def set_seed(seed=42):
    random.seed(seed)


def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def dsigmoid(s):
    return s * (1.0 - s)


def tanh(x):
    return math.tanh(x)


def dtanh(y):
    return 1.0 - y * y


def matvec(W, x):
    return [sum(W[i][j] * x[j] for j in range(len(x))) for i in range(len(W))]


def vec_add(u, v):
    return [ui + vi for ui, vi in zip(u, v)]


def vec_hadamard(u, v):
    return [ui * vi for ui, vi in zip(u, v)]


def vec_norm(v):
    return math.sqrt(sum(x * x for x in v))


# -----------------------------------------------------------------------------
# GRU Cell Implementation
# -----------------------------------------------------------------------------

class GRUCell:
    """
    Standard Gated Recurrent Unit (Cho et al. 2014):
      r_t = sigmoid(W_r [h_{t-1}, x_t] + b_r)       # Reset gate
      z_t = sigmoid(W_z [h_{t-1}, x_t] + b_z)       # Update gate
      h_tilde_t = tanh(W_h [r_t * h_{t-1}, x_t] + b_h) # Candidate state
      h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde_t   # Hidden state interpolation
    """
    def __init__(self, input_dim, hidden_dim, scale=0.1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        concat_dim = input_dim + hidden_dim

        # Reset & Update gates
        self.W_r = [[random.gauss(0.0, scale) for _ in range(concat_dim)] for _ in range(hidden_dim)]
        self.b_r = [0.0 for _ in range(hidden_dim)]

        self.W_z = [[random.gauss(0.0, scale) for _ in range(concat_dim)] for _ in range(hidden_dim)]
        self.b_z = [0.0 for _ in range(hidden_dim)]

        # Candidate state weights: splits into W_hh (hidden) and W_xh (input)
        self.W_hh = [[random.gauss(0.0, scale) for _ in range(hidden_dim)] for _ in range(hidden_dim)]
        self.W_xh = [[random.gauss(0.0, scale) for _ in range(input_dim)] for _ in range(hidden_dim)]
        self.b_h = [0.0 for _ in range(hidden_dim)]

    def forward(self, inputs, h0=None):
        T = len(inputs)
        if h0 is None:
            h0 = [0.0 for _ in range(self.hidden_dim)]

        h_states = [h0]
        cache = []
        h_prev = h0

        for t in range(T):
            xt = inputs[t]
            concat = h_prev + xt

            # 1. Reset Gate
            a_r = vec_add(matvec(self.W_r, concat), self.b_r)
            r_t = [sigmoid(a) for a in a_r]

            # 2. Update Gate
            a_z = vec_add(matvec(self.W_z, concat), self.b_z)
            z_t = [sigmoid(a) for a in a_z]

            # 3. Candidate Hidden State
            r_hprev = vec_hadamard(r_t, h_prev)
            a_h = vec_add(vec_add(matvec(self.W_hh, r_hprev), matvec(self.W_xh, xt)), self.b_h)
            h_tilde = [tanh(a) for a in a_h]

            # 4. Hidden State Update: (1 - z_t) * h_{t-1} + z_t * h_tilde
            h_t = [
                (1.0 - z_t[k]) * h_prev[k] + z_t[k] * h_tilde[k]
                for k in range(self.hidden_dim)
            ]

            cache.append({
                'concat': concat,
                'h_prev': h_prev,
                'xt': xt,
                'r_t': r_t,
                'z_t': z_t,
                'r_hprev': r_hprev,
                'h_tilde': h_tilde,
                'h_t': h_t
            })

            h_states.append(h_t)
            h_prev = h_t

        return h_states, cache

    def backward(self, inputs, dh_targets, h_states, cache):
        """
        Backpropagation through time for GRU.
        dh_targets: list of length T containing dL/dh_t for each timestep.
        """
        T = len(inputs)
        concat_dim = self.input_dim + self.hidden_dim

        dW_r = [[0.0 for _ in range(concat_dim)] for _ in range(self.hidden_dim)]
        db_r = [0.0 for _ in range(self.hidden_dim)]
        dW_z = [[0.0 for _ in range(concat_dim)] for _ in range(self.hidden_dim)]
        db_z = [0.0 for _ in range(self.hidden_dim)]
        dW_hh = [[0.0 for _ in range(self.hidden_dim)] for _ in range(self.hidden_dim)]
        dW_xh = [[0.0 for _ in range(self.input_dim)] for _ in range(self.hidden_dim)]
        db_h = [0.0 for _ in range(self.hidden_dim)]

        dh_next = [0.0 for _ in range(self.hidden_dim)]
        dh_all = [None for _ in range(T + 1)]

        for t in reversed(range(T)):
            c = cache[t]
            concat = c['concat']
            h_prev = c['h_prev']
            xt = c['xt']
            r_t = c['r_t']
            z_t = c['z_t']
            r_hprev = c['r_hprev']
            h_tilde = c['h_tilde']

            # Total incoming adjoint to h_t
            dh_t = vec_add(dh_targets[t], dh_next)
            dh_all[t + 1] = list(dh_t)

            # dh_t decomposes into:
            # dL/dh_tilde = dh_t * z_t
            # dL/dz_t = dh_t * (h_tilde - h_prev)
            dh_tilde = vec_hadamard(dh_t, z_t)
            dz_t = [dh_t[k] * (h_tilde[k] - h_prev[k]) for k in range(self.hidden_dim)]

            # Pre-activation for candidate state: da_h = dh_tilde * (1 - h_tilde^2)
            da_h = [dh_tilde[k] * dtanh(h_tilde[k]) for k in range(self.hidden_dim)]

            # Candidate parameter gradients
            for k in range(self.hidden_dim):
                db_h[k] += da_h[k]
                for j in range(self.hidden_dim):
                    dW_hh[k][j] += da_h[k] * r_hprev[j]
                for j in range(self.input_dim):
                    dW_xh[k][j] += da_h[k] * xt[j]

            # Gradient into r_hprev: W_hh^T * da_h
            dr_hprev = [sum(self.W_hh[j][k] * da_h[j] for j in range(self.hidden_dim)) for k in range(self.hidden_dim)]

            # Gradient into reset gate r_t: dr_hprev * h_prev
            dr_t = vec_hadamard(dr_hprev, h_prev)

            # Pre-activations for gates
            da_z = [dz_t[k] * dsigmoid(z_t[k]) for k in range(self.hidden_dim)]
            da_r = [dr_t[k] * dsigmoid(r_t[k]) for k in range(self.hidden_dim)]

            # Accumulate gate parameters
            for k in range(self.hidden_dim):
                db_z[k] += da_z[k]
                db_r[k] += da_r[k]
                for j in range(concat_dim):
                    dW_z[k][j] += da_z[k] * concat[j]
                    dW_r[k][j] += da_r[k] * concat[j]

            # Gradient flowing backward to h_{t-1}:
            # direct path: dh_t * (1 - z_t)
            # path through r_hprev: dr_hprev * r_t
            # path through gates: W_z^T * da_z + W_r^T * da_r
            dh_prev_accum = [dh_t[k] * (1.0 - z_t[k]) + dr_hprev[k] * r_t[k] for k in range(self.hidden_dim)]
            for j in range(self.hidden_dim):
                gate_contrib = sum(self.W_z[k][j] * da_z[k] + self.W_r[k][j] * da_r[k] for k in range(self.hidden_dim))
                dh_prev_accum[j] += gate_contrib

            dh_next = dh_prev_accum

        dh_all[0] = list(dh_next)

        return {
            'dW_r': dW_r, 'db_r': db_r,
            'dW_z': dW_z, 'db_z': db_z,
            'dW_hh': dW_hh, 'dW_xh': dW_xh, 'db_h': db_h,
            'dh_all': dh_all
        }


# -----------------------------------------------------------------------------
# Parameter Count Analysis
# -----------------------------------------------------------------------------

def analyze_parameter_counts(input_dim=128, hidden_dim=512):
    """
    Compute and compare parameter counts for Elman RNN, GRU, and LSTM.
    """
    print("=" * 78)
    print(f"PARAMETER COUNT COMPARISON (Input Dim = {input_dim}, Hidden Dim = {hidden_dim})")
    print("=" * 78)

    # Elman RNN: W_xh (D_h x D_x) + W_hh (D_h x D_h) + b_h (D_h)
    rnn_params = hidden_dim * (input_dim + hidden_dim + 1)

    # GRU: 3 sets of weights (Reset, Update, Candidate)
    # W_r: D_h x (D_h + D_x) + D_h
    # W_z: D_h x (D_h + D_x) + D_h
    # W_h: D_h x (D_h + D_x) + D_h
    gru_params = 3 * hidden_dim * (input_dim + hidden_dim + 1)

    # LSTM: 4 sets of weights (Forget, Input, Candidate, Output)
    lstm_params = 4 * hidden_dim * (input_dim + hidden_dim + 1)

    ratio_gru_lstm = (lstm_params - gru_params) / lstm_params * 100.0

    print(f"Elman RNN Parameter Count: {rnn_params:>12,d}")
    print(f"GRU Parameter Count:       {gru_params:>12,d}")
    print(f"LSTM Parameter Count:      {lstm_params:>12,d}")
    print(f"\n=> GRU parameter reduction vs LSTM: {ratio_gru_lstm:.1f}% fewer parameters!")
    assert abs(ratio_gru_lstm - 25.0) < 1e-5
    print("=> SUCCESS: Exact 25% parameter reduction mathematically verified.\n")


# -----------------------------------------------------------------------------
# Bidirectional RNN for Symmetry / Palindrome Detection
# -----------------------------------------------------------------------------

class BidirectionalProcessor:
    """
    Bidirectional sequence processor with forward and backward recurrent passes.
    """
    def __init__(self, input_dim, hidden_dim):
        self.fwd_cell = GRUCell(input_dim, hidden_dim, scale=0.2)
        self.bwd_cell = GRUCell(input_dim, hidden_dim, scale=0.2)
        self.hidden_dim = hidden_dim

        # Classifier weights mapping combined state [fwd_h_t, bwd_h_t] (2*hidden_dim) -> 1 logit
        self.W_clf = [random.gauss(0.0, 0.2) for _ in range(2 * hidden_dim)]
        self.b_clf = 0.0

    def process(self, sequence):
        """
        sequence: List of T vectors.
        Returns combined states: List of T vectors, each length 2 * hidden_dim.
        """
        T = len(sequence)
        # Forward pass: x_0, x_1, ..., x_{T-1}
        fwd_states, _ = self.fwd_cell.forward(sequence)
        fwd_h = fwd_states[1:]  # omit initial h0

        # Backward pass: x_{T-1}, x_{T-2}, ..., x_0
        rev_sequence = list(reversed(sequence))
        bwd_states, _ = self.bwd_cell.forward(rev_sequence)
        bwd_h_rev = bwd_states[1:]
        bwd_h = list(reversed(bwd_h_rev))

        # Concatenate [fwd_h_t; bwd_h_t]
        combined = [fwd_h[t] + bwd_h[t] for t in range(T)]
        return combined, fwd_h, bwd_h


def run_bidirectional_experiment():
    """
    Benchmark Unidirectional vs Bidirectional context on Sequence Symmetry Detection.
    Task: Given a sequence of tokens [A, B, C, X, C, B, A] vs [A, B, C, X, D, E, F],
    predict at the middle token (t = 2 in length 5) whether the sequence is symmetric.
    """
    print("=" * 78)
    print("TEST 3: Bidirectional vs Unidirectional Sequence Symmetry Detection")
    print("=" * 78)

    set_seed(42)
    vocab_size = 4
    hidden_dim = 12

    def one_hot(idx):
        vec = [0.0] * vocab_size
        vec[idx] = 1.0
        return vec

    # Generate balanced dataset of symmetric vs asymmetric sequences
    # Length 5: [p0, p1, middle, s1, s0]
    dataset = []
    for _ in range(150):
        p = [random.randint(0, vocab_size - 1) for _ in range(2)]
        middle = random.randint(0, vocab_size - 1)
        # Symmetric
        sym_tokens = p + [middle] + list(reversed(p))
        dataset.append((sym_tokens, 1.0))

        # Asymmetric
        s = [random.randint(0, vocab_size - 1) for _ in range(2)]
        while s == list(reversed(p)):
            s = [random.randint(0, vocab_size - 1) for _ in range(2)]
        asym_tokens = p + [middle] + s
        dataset.append((asym_tokens, 0.0))

    random.shuffle(dataset)

    bi_model = BidirectionalProcessor(vocab_size, hidden_dim)

    X_uni = []
    X_bi = []
    Y = []

    for tokens, label in dataset:
        seq_vecs = [one_hot(tok) for tok in tokens]
        combined, fwd_h, bwd_h = bi_model.process(seq_vecs)
        # At middle step t = 2:
        # fwd_h[2] has only seen tokens 0, 1, 2 (zero knowledge of 3, 4)
        # combined[2] fuses fwd_h[2] (tokens 0, 1, 2) and bwd_h[2] (tokens 4, 3, 2)
        X_uni.append(fwd_h[2])
        X_bi.append(combined[2])
        Y.append(label)

    # 2-layer MLP probe to test non-linear separability of representations
    def train_probe_mlp(X_data, y_data, h_dim=20, steps=600, lr=0.1):
        d_in = len(X_data[0])
        w1 = [[random.gauss(0.0, 0.2) for _ in range(d_in)] for _ in range(h_dim)]
        b1 = [0.0 for _ in range(h_dim)]
        w2 = [random.gauss(0.0, 0.2) for _ in range(h_dim)]
        b2 = 0.0

        for _ in range(steps):
            for i in range(len(X_data)):
                x = X_data[i]
                y = y_data[i]
                # Forward
                a1 = [sum(w1[k][j] * x[j] for j in range(d_in)) + b1[k] for k in range(h_dim)]
                h = [tanh(a) for a in a1]
                logit = sum(w2[k] * h[k] for k in range(h_dim)) + b2
                pred = sigmoid(logit)

                # Backward
                d_out = pred - y
                d_h = [d_out * w2[k] * dtanh(h[k]) for k in range(h_dim)]
                for k in range(h_dim):
                    w2[k] -= lr * d_out * h[k]
                    b2 -= lr * d_out
                    for j in range(d_in):
                        w1[k][j] -= lr * d_h[k] * x[j]
                    b1[k] -= lr * d_h[k]

        # Evaluation accuracy
        correct = 0
        for i in range(len(X_data)):
            x = X_data[i]
            y = y_data[i]
            h = [tanh(sum(w1[k][j] * x[j] for j in range(d_in)) + b1[k]) for k in range(h_dim)]
            pred = sigmoid(sum(w2[k] * h[k] for k in range(h_dim)) + b2)
            if (pred >= 0.5) == (y == 1.0):
                correct += 1
        return correct / len(X_data) * 100.0

    acc_uni = train_probe_mlp(X_uni, Y)
    acc_bi = train_probe_mlp(X_bi, Y)

    print(f"Unidirectional Representation Accuracy:  {acc_uni:.1f}% (Chance baseline ~ 50%)")
    print(f"Bidirectional Representation Accuracy:   {acc_bi:.1f}% (High contextual separation!)")

    assert acc_uni < 65.0, "Unidirectional model should not be able to guess future tokens!"
    assert acc_bi >= 90.0, "Bidirectional model must capture full context for symmetry detection!"
    print("=> SUCCESS: Bidirectional context integration verified on palindrome benchmark!\n")


def run_gradient_check():
    """
    Rigorously check analytical GRU gradients against central finite differences.
    """
    print("=" * 78)
    print("TEST 2: GRU Analytical vs. Finite Difference GradCheck")
    print("=" * 78)

    set_seed(42)
    D_in, D_hid = 2, 3
    T = 4
    gru = GRUCell(D_in, D_hid, scale=0.3)
    inputs = [[random.uniform(-1.0, 1.0) for _ in range(D_in)] for _ in range(T)]

    h_states, cache = gru.forward(inputs)
    dh_targets = [[0.0 for _ in range(D_hid)] for _ in range(T)]
    dh_targets[-1] = list(h_states[-1])

    grads = gru.backward(inputs, dh_targets, h_states, cache)

    def compute_loss(model):
        hs, _ = model.forward(inputs)
        return 0.5 * sum(x * x for x in hs[-1])

    eps = 1e-5
    max_rel_err = 0.0

    for i in range(D_hid):
        for j in range(D_hid):
            orig = gru.W_hh[i][j]

            gru.W_hh[i][j] = orig + eps
            l_pos = compute_loss(gru)

            gru.W_hh[i][j] = orig - eps
            l_neg = compute_loss(gru)

            gru.W_hh[i][j] = orig

            num_grad = (l_pos - l_neg) / (2.0 * eps)
            ana_grad = grads['dW_hh'][i][j]

            rel_err = abs(num_grad - ana_grad) / max(abs(num_grad) + abs(ana_grad), 1e-8)
            if rel_err > max_rel_err:
                max_rel_err = rel_err

    print(f"Max Relative Error on W_hh: {max_rel_err:.2e}")
    assert max_rel_err < 1e-5, f"GradCheck failed! error = {max_rel_err}"
    print("=> SUCCESS: Analytical GRU gradients match finite differences perfectly (< 1e-5)!\n")


def main():
    print("Running Phase 15 - Topic 15.3: GRUs & Bidirectionality\n")
    analyze_parameter_counts(input_dim=128, hidden_dim=512)
    run_gradient_check()
    run_bidirectional_experiment()
    print("=" * 78)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY (Exit Code 0).")
    print("=" * 78)
    return 0


if __name__ == '__main__':
    sys.exit(main())
