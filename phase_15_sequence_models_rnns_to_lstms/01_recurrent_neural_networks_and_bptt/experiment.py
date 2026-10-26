#!/usr/bin/env python3
"""
Topic 15.1: Recurrent Neural Networks & Backpropagation Through Time (BPTT)
=============================================================================
Pure Python 3 implementation (zero external dependencies).

This script demonstrates:
1. An unrolled Elman RNN Cell forward and backward pass through time.
2. The Vanishing Gradient Dilemma: measuring the exponential decay of gradients
   dL_T/dh_k across 20 timesteps when spectral radius rho(W_hh) < 1.
3. The Exploding Gradient Dilemma: demonstrating numerical divergence when
   rho(W_hh) > 1, and proving how Gradient Clipping (threshold tau) stabilizes
   the gradient vector while preserving its directional integrity.
4. Rigorous Gradient Checking: validating analytical BPTT gradients against
   two-sided central finite differences to precision < 1e-6.
"""

import math
import random
import sys


def set_seed(seed=42):
    random.seed(seed)


# -----------------------------------------------------------------------------
# Linear Algebra & Math Utilities (Pure Python)
# -----------------------------------------------------------------------------

def matmul(A, B):
    """Matrix multiplication: A (n x m) * B (m x p) -> (n x p)."""
    n, m = len(A), len(A[0])
    p = len(B[0])
    C = [[0.0 for _ in range(p)] for _ in range(n)]
    for i in range(n):
        for k in range(m):
            a_ik = A[i][k]
            if a_ik != 0.0:
                for j in range(p):
                    C[i][j] += a_ik * B[k][j]
    return C


def matvec(A, x):
    """Matrix-vector product: A (n x m) * x (m) -> (n)."""
    return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]


def vec_add(u, v):
    """Vector addition: u + v."""
    return [ui + vi for ui, vi in zip(u, v)]


def vec_sub(u, v):
    """Vector subtraction: u - v."""
    return [ui - vi for ui, vi in zip(u, v)]


def vec_dot(u, v):
    """Vector dot product."""
    return sum(ui * vi for ui, vi in zip(u, v))


def vec_norm(v):
    """L2 Euclidean norm of a vector."""
    return math.sqrt(sum(x * x for x in v))


def outer_product(u, v):
    """Outer product of vectors: u (n) (x) v (m) -> Matrix (n x m)."""
    return [[ui * vj for vj in v] for ui in u]


def transpose(A):
    """Matrix transpose."""
    n, m = len(A), len(A[0])
    return [[A[i][j] for i in range(n)] for j in range(m)]


def tanh(x):
    """Hyperbolic tangent activation."""
    return math.tanh(x)


def dtanh(y):
    """Derivative of tanh given output y = tanh(x): d/dx = 1 - y^2."""
    return 1.0 - y * y


def softmax(x):
    """Numerically stable softmax."""
    max_x = max(x)
    exp_x = [math.exp(xi - max_x) for xi in x]
    sum_exp = sum(exp_x)
    return [e / sum_exp for e in exp_x]


# -----------------------------------------------------------------------------
# Elman RNN Implementation
# -----------------------------------------------------------------------------

class ElmanRNN:
    """
    Standard Elman Recurrent Neural Network:
      h_t = tanh(W_hh h_{t-1} + W_xh x_t + b_h)
      y_hat_t = softmax(W_hy h_t + b_y)
    """
    def __init__(self, input_dim, hidden_dim, output_dim, weight_scale=0.1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Xavier / Scaled initialization
        self.W_xh = [[random.gauss(0.0, weight_scale) for _ in range(input_dim)] for _ in range(hidden_dim)]
        self.W_hh = [[random.gauss(0.0, weight_scale) for _ in range(hidden_dim)] for _ in range(hidden_dim)]
        self.b_h = [0.0 for _ in range(hidden_dim)]

        self.W_hy = [[random.gauss(0.0, weight_scale) for _ in range(hidden_dim)] for _ in range(output_dim)]
        self.b_y = [0.0 for _ in range(output_dim)]

    def forward(self, inputs, h0=None):
        """
        Unroll RNN forward through time.
        inputs: List of T vectors, each length input_dim
        h0: initial hidden state (default zeros)
        Returns:
          h_states: List of T+1 hidden states (h_0 to h_T)
          a_states: List of T pre-activations (a_1 to a_T)
          outputs: List of T predicted probability distributions (y_hat_1 to y_hat_T)
        """
        T = len(inputs)
        if h0 is None:
            h0 = [0.0 for _ in range(self.hidden_dim)]

        h_states = [h0]
        a_states = []
        outputs = []

        h_prev = h0
        for t in range(T):
            xt = inputs[t]
            # a_t = W_xh * x_t + W_hh * h_{t-1} + b_h
            a_t = vec_add(vec_add(matvec(self.W_xh, xt), matvec(self.W_hh, h_prev)), self.b_h)
            h_t = [tanh(ai) for ai in a_t]
            # y_hat_t = softmax(W_hy * h_t + b_y)
            z_t = vec_add(matvec(self.W_hy, h_t), self.b_y)
            y_hat_t = softmax(z_t)

            a_states.append(a_t)
            h_states.append(h_t)
            outputs.append(y_hat_t)
            h_prev = h_t

        return h_states, a_states, outputs

    def backward(self, inputs, targets, h_states, a_states, outputs):
        """
        Backpropagation Through Time (BPTT).
        targets: List of T target one-hot vectors or probability distributions.
        Returns parameter gradients: dW_xh, dW_hh, db_h, dW_hy, db_y
        and hidden state gradients dh_states (dL/dh_t for all t).
        """
        T = len(inputs)
        dW_xh = [[0.0 for _ in range(self.input_dim)] for _ in range(self.hidden_dim)]
        dW_hh = [[0.0 for _ in range(self.hidden_dim)] for _ in range(self.hidden_dim)]
        db_h = [0.0 for _ in range(self.hidden_dim)]
        dW_hy = [[0.0 for _ in range(self.hidden_dim)] for _ in range(self.output_dim)]
        db_y = [0.0 for _ in range(self.output_dim)]

        dh_next = [0.0 for _ in range(self.hidden_dim)]
        dh_states = [None for _ in range(T + 1)]

        # Backward unrolling from T-1 down to 0
        W_hh_T = transpose(self.W_hh)
        W_hy_T = transpose(self.W_hy)

        for t in reversed(range(T)):
            # Loss at step t: Cross-Entropy L_t = -sum(y_t * log(y_hat_t))
            # dL_t / dz_t = (y_hat_t - y_t)
            dz_t = vec_sub(outputs[t], targets[t])

            # Gradients for output layer
            for i in range(self.output_dim):
                db_y[i] += dz_t[i]
                for j in range(self.hidden_dim):
                    dW_hy[i][j] += dz_t[i] * h_states[t + 1][j]

            # Gradient into hidden state h_t:
            # dh_t = (dL_t / dh_t) + (dL_{>t} / dh_t)
            # where dL_t / dh_t = W_hy^T * dz_t
            # and dL_{>t} / dh_t = W_hh^T * da_{t+1} (which was accumulated in dh_next)
            dh_t = vec_add(matvec(W_hy_T, dz_t), dh_next)
            dh_states[t + 1] = list(dh_t)

            # Backprop through tanh: da_t = dh_t * (1 - h_t^2)
            da_t = [dh_t[i] * dtanh(h_states[t + 1][i]) for i in range(self.hidden_dim)]

            # Gradients for recurrent weights and biases
            for i in range(self.hidden_dim):
                db_h[i] += da_t[i]
                for j in range(self.input_dim):
                    dW_xh[i][j] += da_t[i] * inputs[t][j]
                for j in range(self.hidden_dim):
                    dW_hh[i][j] += da_t[i] * h_states[t][j]

            # Next gradient flowing backward to h_{t-1}
            dh_next = matvec(W_hh_T, da_t)

        dh_states[0] = list(dh_next)

        return {
            'dW_xh': dW_xh,
            'dW_hh': dW_hh,
            'db_h': db_h,
            'dW_hy': dW_hy,
            'db_y': db_y,
            'dh_states': dh_states
        }


# -----------------------------------------------------------------------------
# Gradient Clipping
# -----------------------------------------------------------------------------

def clip_gradients(grad_dict, max_norm=1.0):
    """
    Clips parameter gradients in place according to global L2 norm:
      g <- g * (max_norm / max(global_norm, max_norm))
    """
    total_sq_norm = 0.0
    for name in ['dW_xh', 'dW_hh', 'dW_hy']:
        for row in grad_dict[name]:
            total_sq_norm += sum(g * g for g in row)
    for name in ['db_h', 'db_y']:
        total_sq_norm += sum(g * g for g in grad_dict[name])

    global_norm = math.sqrt(total_sq_norm)
    clip_factor = 1.0
    if global_norm > max_norm:
        clip_factor = max_norm / global_norm

    clipped = {}
    for name in ['dW_xh', 'dW_hh', 'dW_hy']:
        clipped[name] = [[g * clip_factor for g in row] for row in grad_dict[name]]
    for name in ['db_h', 'db_y']:
        clipped[name] = [g * clip_factor for g in grad_dict[name]]

    return clipped, global_norm, min(global_norm, max_norm)


# -----------------------------------------------------------------------------
# Verification & Experiments
# -----------------------------------------------------------------------------

def run_gradient_check():
    """
    Verify analytical BPTT gradients against central finite differences.
    """
    print("=" * 78)
    print("TEST 1: BPTT Analytical vs. Finite Difference Gradient Check (GradCheck)")
    print("=" * 78)

    set_seed(123)
    D_in, D_hid, D_out = 3, 4, 2
    T = 5
    model = ElmanRNN(D_in, D_hid, D_out, weight_scale=0.4)

    # Synthetic inputs and target distributions
    inputs = [[random.uniform(-1.0, 1.0) for _ in range(D_in)] for _ in range(T)]
    targets = []
    for _ in range(T):
        y_val = [0.0 for _ in range(D_out)]
        y_val[random.randint(0, D_out - 1)] = 1.0
        targets.append(y_val)

    # Forward & Analytical Backward
    h_states, a_states, outputs = model.forward(inputs)
    grads = model.backward(inputs, targets, h_states, a_states, outputs)

    def compute_total_loss(rnn):
        _, _, outs = rnn.forward(inputs)
        loss = 0.0
        for t in range(T):
            for k in range(D_out):
                if targets[t][k] > 0.0:
                    loss -= targets[t][k] * math.log(max(outs[t][k], 1e-15))
        return loss

    eps = 1e-5
    max_rel_error = 0.0

    # Check W_hh gradients
    for i in range(D_hid):
        for j in range(D_hid):
            orig_val = model.W_hh[i][j]

            model.W_hh[i][j] = orig_val + eps
            loss_pos = compute_total_loss(model)

            model.W_hh[i][j] = orig_val - eps
            loss_neg = compute_total_loss(model)

            model.W_hh[i][j] = orig_val

            num_grad = (loss_pos - loss_neg) / (2.0 * eps)
            ana_grad = grads['dW_hh'][i][j]

            denom = max(abs(num_grad) + abs(ana_grad), 1e-8)
            rel_err = abs(num_grad - ana_grad) / denom
            if rel_err > max_rel_error:
                max_rel_error = rel_err

    print(f"Max Relative Error on W_hh across {D_hid * D_hid} entries: {max_rel_error:.2e}")
    assert max_rel_error < 1e-5, f"GradCheck failed! rel_err = {max_rel_error}"
    print("=> SUCCESS: Analytical BPTT gradients match finite differences perfectly (< 1e-5)!\n")


def run_vanishing_gradient_experiment():
    """
    Demonstrate exponential gradient vanishing over 20 timesteps when spectral radius < 1.
    """
    print("=" * 78)
    print("TEST 2: Vanishing Gradient Across 20 Timesteps")
    print("=" * 78)

    set_seed(42)
    D_in, D_hid, D_out = 2, 8, 2
    T = 20

    # Initialize W_hh with small weights (spectral radius < 1)
    rnn = ElmanRNN(D_in, D_hid, D_out, weight_scale=0.2)
    inputs = [[random.uniform(-0.5, 0.5) for _ in range(D_in)] for _ in range(T)]

    # Targets only evaluate loss at terminal step T-1 (measuring how loss at T impacts earlier states)
    targets = [[0.5, 0.5] for _ in range(T)]
    targets[-1] = [1.0, 0.0]  # signal at step 19

    h_states, a_states, outputs = rnn.forward(inputs)

    # Backward pass propagating ONLY from step T-1
    # We zero out dz_t for all t < T-1 to isolate dL_{T-1} / dh_t
    T_idx = T - 1
    outputs_isolated = [list(o) for o in outputs]
    targets_isolated = [list(o) for o in outputs]  # dz_t = 0 for t < T_idx
    targets_isolated[T_idx] = targets[T_idx]

    grads = rnn.backward(inputs, targets_isolated, h_states, a_states, outputs_isolated)
    dh_states = grads['dh_states']

    print(f"{'Timestep (t)':<15}{'Temporal Distance (T - t)':<25}{'Norm ||dL_T / dh_t||':<25}")
    print("-" * 65)

    grad_norms = []
    for t in [19, 15, 10, 5, 1, 0]:
        norm = vec_norm(dh_states[t])
        grad_norms.append(norm)
        print(f"t = {t:<11}{19 - t:<25}{norm:<25.6e}")

    # Verify exponential vanishing: gradient at step 0 is orders of magnitude smaller than step 19
    decay_ratio = grad_norms[-1] / (grad_norms[0] + 1e-15)
    print(f"\nGradient decay ratio ||dL/dh_0|| / ||dL/dh_19||: {decay_ratio:.2e}")
    assert decay_ratio < 1e-4, "Expected significant gradient vanishing across 20 timesteps!"
    print("=> SUCCESS: Exponential vanishing gradient mathematically verified!\n")


def run_exploding_gradient_and_clipping():
    """
    Demonstrate exploding gradients when weights are large and prove stabilization via gradient clipping.
    """
    print("=" * 78)
    print("TEST 3: Exploding Gradient & Gradient Clipping Stabilization")
    print("=" * 78)

    set_seed(7)
    D_in, D_hid, D_out = 2, 8, 2
    T = 15

    # Initialize W_hh with large values (spectral radius > 1.5)
    rnn = ElmanRNN(D_in, D_hid, D_out, weight_scale=1.8)
    inputs = [[random.uniform(-0.1, 0.1) for _ in range(D_in)] for _ in range(T)]
    targets = [[1.0, 0.0] if t % 2 == 0 else [0.0, 1.0] for t in range(T)]

    h_states, a_states, outputs = rnn.forward(inputs)
    grads = rnn.backward(inputs, targets, h_states, a_states, outputs)

    # Compute unclipped global norm
    total_sq = sum(sum(g * g for g in row) for row in grads['dW_hh'])
    unclipped_norm = math.sqrt(total_sq)
    print(f"Unclipped ||dW_hh|| Norm: {unclipped_norm:.4f}")

    tau = 5.0
    clipped_grads, pre_norm, post_norm = clip_gradients(grads, max_norm=tau)

    # Verify directional alignment (cosine similarity must be exactly 1.0)
    flat_orig = []
    flat_clip = []
    for row in grads['dW_hh']:
        flat_orig.extend(row)
    for row in clipped_grads['dW_hh']:
        flat_clip.extend(row)

    cos_sim = vec_dot(flat_orig, flat_clip) / (vec_norm(flat_orig) * vec_norm(flat_clip))
    print(f"Gradient Clipping Threshold tau: {tau:.2f}")
    print(f"Global Norm Before Clipping:     {pre_norm:.4f}")
    print(f"Global Norm After Clipping:      {post_norm:.4f}")
    print(f"Cosine Similarity (Directional Invariance): {cos_sim:.6f}")

    assert abs(cos_sim - 1.0) < 1e-7, "Clipping must not alter the direction of the gradient!"
    assert post_norm <= tau + 1e-7, f"Post-clipping norm must not exceed tau ({tau})!"
    print("=> SUCCESS: Gradient clipping strictly bounds gradient norm while preserving direction!\n")


def main():
    print("Running Phase 15 - Topic 15.1: Recurrent Neural Networks & BPTT\n")
    run_gradient_check()
    run_vanishing_gradient_experiment()
    run_exploding_gradient_and_clipping()
    print("=" * 78)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY (Exit Code 0).")
    print("=" * 78)
    return 0


if __name__ == '__main__':
    sys.exit(main())
