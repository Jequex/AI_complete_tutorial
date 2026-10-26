#!/usr/bin/env python3
"""
Topic 24.1: Taxonomy of Generative Models & Autoregressive Models
================================================================

This experiment demonstrates the foundational dichotomy of generative modeling:
1. Autoregressive Generative Modeling:
   - Factorizing joint distributions via the chain rule: P(x) = prod_t P(x_t | x_{<t})
   - Step-by-step conditional sampling where every sampled token conditions future tokens.
2. Deterministic Autoencoders & The Empty Latent Space Failure:
   - Training a deterministic encoder-decoder bottleneck on structured data.
   - Demonstrating that while reconstruction on in-distribution data is near-perfect,
     sampling randomly from z ~ N(0, I) yields nonsensical hallucinations because
     the unconstrained latent space is riddled with "empty holes" (support voids).

Zero external dependencies: Pure Python 3 standard library.
"""

import math
import random

# Fix random seed for strict reproducibility
random.seed(42)


# =====================================================================
# 1. Autoregressive Sequence Generator (From Scratch)
# =====================================================================

class AutoregressiveCharModel:
    """
    A lightweight character-level autoregressive model using bigram/trigram
    conditional frequency tables with additive Laplace smoothing.
    P(x_1, ..., x_T) = prod_{t=1}^T P(x_t | x_{t-k}, ..., x_{t-1})
    """
    def __init__(self, context_size=2, smoothing=0.1):
        self.context_size = context_size
        self.smoothing = smoothing
        self.vocab = set()
        self.counts = {}  # context_tuple -> {next_char: count}
        self.totals = {}  # context_tuple -> sum of counts

    def fit(self, texts):
        # Build vocabulary
        for text in texts:
            for ch in text:
                self.vocab.add(ch)
        self.vocab.add("<BOS>")
        self.vocab.add("<EOS>")
        self.vocab_list = sorted(list(self.vocab))
        self.vocab_size = len(self.vocab_list)

        # Count n-grams
        for text in texts:
            tokens = ["<BOS>"] * self.context_size + list(text) + ["<EOS>"]
            for i in range(self.context_size, len(tokens)):
                context = tuple(tokens[i - self.context_size : i])
                target = tokens[i]
                if context not in self.counts:
                    self.counts[context] = {v: 0.0 for v in self.vocab}
                    self.totals[context] = 0.0
                self.counts[context][target] += 1.0
                self.totals[context] += 1.0

    def get_conditional_probs(self, context):
        context = tuple(context[-self.context_size:])
        probs = {}
        if context in self.counts:
            denom = self.totals[context] + self.smoothing * self.vocab_size
            for ch in self.vocab:
                probs[ch] = (self.counts[context].get(ch, 0.0) + self.smoothing) / denom
        else:
            # Uniform fallback with smoothing
            for ch in self.vocab:
                probs[ch] = 1.0 / self.vocab_size
        return probs

    def sample(self, max_length=30, temperature=1.0):
        current_context = ["<BOS>"] * self.context_size
        generated = []

        for _ in range(max_length):
            probs = self.get_conditional_probs(current_context)
            # Apply temperature scaling
            if temperature != 1.0:
                logits = {k: math.log(max(p, 1e-12)) / temperature for k, p in probs.items()}
                max_l = max(logits.values())
                exp_logits = {k: math.exp(v - max_l) for k, v in logits.items()}
                sum_exp = sum(exp_logits.values())
                probs = {k: v / sum_exp for k, v in exp_logits.items()}

            # Categorical roulette wheel sampling
            r = random.random()
            cum = 0.0
            chosen = "<EOS>"
            for ch, p in probs.items():
                cum += p
                if r <= cum:
                    chosen = ch
                    break

            if chosen == "<EOS>":
                break
            generated.append(chosen)
            current_context = current_context[1:] + [chosen]

        return "".join(generated)

    def sequence_log_likelihood(self, text):
        """Computes exact log P(x) = sum_t log P(x_t | x_{<t})"""
        tokens = ["<BOS>"] * self.context_size + list(text) + ["<EOS>"]
        total_ll = 0.0
        for i in range(self.context_size, len(tokens)):
            context = tuple(tokens[i - self.context_size : i])
            target = tokens[i]
            probs = self.get_conditional_probs(context)
            p = max(probs.get(target, 1e-12), 1e-12)
            total_ll += math.log(p)
        return total_ll


# =====================================================================
# 2. Deterministic Autoencoder (From Scratch)
# =====================================================================

class DeterministicAutoencoder:
    """
    A 1D MLP Autoencoder trained to compress 8-dimensional structured patterns
    into a 2-dimensional latent bottleneck z in R^2.
    Architecture:
      Encoder: x (dim 8) -> Linear(8->4) -> Tanh -> Linear(4->2) -> z
      Decoder: z (dim 2) -> Linear(2->4) -> Tanh -> Linear(4->8) -> x_hat
    Objective: MSE = (1/N) * sum ||x - x_hat||^2
    """
    def __init__(self, in_dim=8, hidden_dim=4, latent_dim=2, lr=0.05):
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lr = lr

        # Xavier initialization
        def init_weights(din, dout):
            limit = math.sqrt(6.0 / (din + dout))
            return [[random.uniform(-limit, limit) for _ in range(dout)] for _ in range(din)]

        # Encoder weights & biases
        self.W_enc1 = init_weights(in_dim, hidden_dim)
        self.b_enc1 = [0.0] * hidden_dim
        self.W_enc2 = init_weights(hidden_dim, latent_dim)
        self.b_enc2 = [0.0] * latent_dim

        # Decoder weights & biases
        self.W_dec1 = init_weights(latent_dim, hidden_dim)
        self.b_dec1 = [0.0] * hidden_dim
        self.W_dec2 = init_weights(hidden_dim, in_dim)
        self.b_dec2 = [0.0] * in_dim

    def encode(self, x):
        # Layer 1
        h1 = [0.0] * self.hidden_dim
        for j in range(self.hidden_dim):
            s = sum(x[i] * self.W_enc1[i][j] for i in range(self.in_dim)) + self.b_enc1[j]
            h1[j] = math.tanh(s)
        # Bottleneck Layer 2 (linear)
        z = [0.0] * self.latent_dim
        for j in range(self.latent_dim):
            z[j] = sum(h1[i] * self.W_enc2[i][j] for i in range(self.hidden_dim)) + self.b_enc2[j]
        return h1, z

    def decode(self, z):
        # Layer 1
        h_dec = [0.0] * self.hidden_dim
        for j in range(self.hidden_dim):
            s = sum(z[i] * self.W_dec1[i][j] for i in range(self.latent_dim)) + self.b_dec1[j]
            h_dec[j] = math.tanh(s)
        # Output Layer 2 (linear reconstruction)
        x_hat = [0.0] * self.in_dim
        for j in range(self.in_dim):
            x_hat[j] = sum(h_dec[i] * self.W_dec2[i][j] for i in range(self.hidden_dim)) + self.b_dec2[j]
        return h_dec, x_hat

    def train_step(self, x):
        h1, z = self.encode(x)
        h_dec, x_hat = self.decode(z)

        # Loss = sum_j (x_hat[j] - x[j])^2
        loss = sum((x_hat[j] - x[j]) ** 2 for j in range(self.in_dim))

        # Backpropagation:
        # 1. Output error: dL/dx_hat_j = 2 * (x_hat_j - x_j)
        d_xhat = [2.0 * (x_hat[j] - x[j]) for j in range(self.in_dim)]

        # 2. Decoder layer 2
        d_W_dec2 = [[0.0] * self.in_dim for _ in range(self.hidden_dim)]
        d_b_dec2 = [0.0] * self.in_dim
        d_hdec = [0.0] * self.hidden_dim

        for j in range(self.in_dim):
            d_b_dec2[j] = d_xhat[j]
            for i in range(self.hidden_dim):
                d_W_dec2[i][j] = h_dec[i] * d_xhat[j]
                d_hdec[i] += self.W_dec2[i][j] * d_xhat[j]

        # 3. Decoder tanh activation: dtanh(s) = 1 - tanh(s)^2 = 1 - h_dec^2
        d_s_dec = [d_hdec[i] * (1.0 - h_dec[i] ** 2) for i in range(self.hidden_dim)]

        # 4. Decoder layer 1
        d_W_dec1 = [[0.0] * self.hidden_dim for _ in range(self.latent_dim)]
        d_b_dec1 = [0.0] * self.hidden_dim
        d_z = [0.0] * self.latent_dim

        for j in range(self.hidden_dim):
            d_b_dec1[j] = d_s_dec[j]
            for i in range(self.latent_dim):
                d_W_dec1[i][j] = z[i] * d_s_dec[j]
                d_z[i] += self.W_dec1[i][j] * d_s_dec[j]

        # 5. Encoder layer 2 (linear)
        d_W_enc2 = [[0.0] * self.latent_dim for _ in range(self.hidden_dim)]
        d_b_enc2 = [0.0] * self.latent_dim
        d_h1 = [0.0] * self.hidden_dim

        for j in range(self.latent_dim):
            d_b_enc2[j] = d_z[j]
            for i in range(self.hidden_dim):
                d_W_enc2[i][j] = h1[i] * d_z[j]
                d_h1[i] += self.W_enc2[i][j] * d_z[j]

        # 6. Encoder tanh activation
        d_s_enc = [d_h1[i] * (1.0 - h1[i] ** 2) for i in range(self.hidden_dim)]

        # 7. Encoder layer 1
        d_W_enc1 = [[0.0] * self.hidden_dim for _ in range(self.in_dim)]
        d_b_enc1 = [0.0] * self.hidden_dim

        for j in range(self.hidden_dim):
            d_b_enc1[j] = d_s_enc[j]
            for i in range(self.in_dim):
                d_W_enc1[i][j] = x[i] * d_s_enc[j]

        # Parameter update (SGD)
        for i in range(self.in_dim):
            for j in range(self.hidden_dim):
                self.W_enc1[i][j] -= self.lr * d_W_enc1[i][j]
        for j in range(self.hidden_dim):
            self.b_enc1[j] -= self.lr * d_b_enc1[j]

        for i in range(self.hidden_dim):
            for j in range(self.latent_dim):
                self.W_enc2[i][j] -= self.lr * d_W_enc2[i][j]
        for j in range(self.latent_dim):
            self.b_enc2[j] -= self.lr * d_b_enc2[j]

        for i in range(self.latent_dim):
            for j in range(self.hidden_dim):
                self.W_dec1[i][j] -= self.lr * d_W_dec1[i][j]
        for j in range(self.hidden_dim):
            self.b_dec1[j] -= self.lr * d_b_dec1[j]

        for i in range(self.hidden_dim):
            for j in range(self.in_dim):
                self.W_dec2[i][j] -= self.lr * d_W_dec2[i][j]
        for j in range(self.in_dim):
            self.b_dec2[j] -= self.lr * d_b_dec2[j]

        return loss


# =====================================================================
# Main Experiment Execution & Demonstrations
# =====================================================================

def run_experiment():
    print("=" * 76)
    print("TOPIC 24.1: TAXONOMY OF GENERATIVE MODELS & AUTOREGRESSIVE MECHANICS")
    print("=" * 76)

    # -------------------------------------------------------------
    # Part 1: Autoregressive Sequence Generation
    # -------------------------------------------------------------
    print("\n--- [1] Autoregressive Generation via Chain Rule P(x) = prod_t P(x_t | x_{<t}) ---")
    training_corpus = [
        "quantum computing transforms cryptography",
        "quantum entanglement enables superdense coding",
        "quantum supremacy demonstrates computational speedups",
        "quantum algorithms solve integer factorization",
        "quantum information theory studies density matrices",
        "quantum circuits utilize unitary transformations",
        "cryptography relies on computational hardness assumptions",
        "cryptographic protocols secure distributed networks"
    ]

    ar_model = AutoregressiveCharModel(context_size=3, smoothing=0.05)
    ar_model.fit(training_corpus)

    print(f"  Training Corpus Size : {len(training_corpus)} phrases")
    print(f"  Vocabulary Size       : {ar_model.vocab_size} unique symbols")

    print("\n  Sampled Sequences from Autoregressive Prior:")
    for temp in [0.5, 0.8, 1.0]:
        sample_str = ar_model.sample(max_length=45, temperature=temp)
        ll = ar_model.sequence_log_likelihood(sample_str)
        print(f"    [Temp {temp:.1f}] \"{sample_str}\"  (Log-Likelihood: {ll:.2f})")

    # In-distribution vs out-of-distribution log-likelihood
    in_dist = "quantum algorithms utilize circuits"
    out_dist = "banana giraffe spaceship volcanos"
    ll_in = ar_model.sequence_log_likelihood(in_dist)
    ll_out = ar_model.sequence_log_likelihood(out_dist)
    print(f"\n  Exact Density Evaluation P(x):")
    print(f"    In-distribution text  : '{in_dist}' -> Log-Likelihood: {ll_in:.2f}")
    print(f"    Out-of-distribution   : '{out_dist}' -> Log-Likelihood: {ll_out:.2f}")
    assert ll_in > ll_out, "In-distribution log-likelihood must strictly exceed OOD"

    # -------------------------------------------------------------
    # Part 2: Deterministic Autoencoder Reconstruction vs Sampling
    # -------------------------------------------------------------
    print("\n--- [2] Deterministic Autoencoder Bottleneck vs Empty Latent Space ---")

    # Let data manifold be 3 discrete structured sinusoidal wave modes in 8 dimensions
    def make_pattern(freq, phase):
        return [math.sin(freq * i + phase) for i in range(8)]

    dataset = []
    # Class 1: Low frequency waves
    for _ in range(50):
        dataset.append(make_pattern(0.5, random.uniform(0, 0.2)))
    # Class 2: High frequency waves
    for _ in range(50):
        dataset.append(make_pattern(1.5, random.uniform(0, 0.2)))

    dae = DeterministicAutoencoder(in_dim=8, hidden_dim=6, latent_dim=2, lr=0.04)

    # Train DAE
    epochs = 400
    for epoch in range(epochs):
        random.shuffle(dataset)
        total_loss = 0.0
        for x in dataset:
            total_loss += dae.train_step(x)
        if (epoch + 1) % 100 == 0:
            avg_mse = total_loss / (len(dataset) * 8)
            print(f"    Epoch {epoch+1:3d}/{epochs} | Reconstruction MSE per dimension: {avg_mse:.6f}")

    # Evaluate reconstruction on training pattern
    test_x = make_pattern(0.5, 0.1)
    _, z_enc = dae.encode(test_x)
    _, rec_x = dae.decode(z_enc)
    rec_mse = sum((test_x[i] - rec_x[i]) ** 2 for i in range(8)) / 8.0

    print(f"\n  In-Distribution Reconstruction:")
    print(f"    Target x   : {[round(v, 2) for v in test_x[:4]]}...")
    print(f"    Recon x_hat: {[round(v, 2) for v in rec_x[:4]]}...")
    print(f"    Latent z   : {[round(v, 3) for v in z_enc]}")
    print(f"    Recon MSE  : {rec_mse:.6f} (Low error, successful compression)")
    assert rec_mse < 0.15, f"Reconstruction error too high: {rec_mse}"

    # Now inspect the latent coordinates of all training data
    latent_points = [dae.encode(x)[1] for x in dataset]
    z0_vals = [p[0] for p in latent_points]
    z1_vals = [p[1] for p in latent_points]
    z0_min, z0_max = min(z0_vals), max(z0_vals)
    z1_min, z1_max = min(z1_vals), max(z1_vals)

    print(f"\n  Latent Space Geometry of Deterministic Autoencoder:")
    print(f"    Latent Axis 0 Support: [{z0_min:.2f}, {z0_max:.2f}]")
    print(f"    Latent Axis 1 Support: [{z1_min:.2f}, {z1_max:.2f}]")

    # Pick an arbitrary point from standard normal space far from the manifold support
    # (The "Empty Void / Hole" in latent space)
    z_void = [z0_max + 3.0, z1_min - 3.0]
    _, x_hallucinated = dae.decode(z_void)

    # Calculate distance to valid sine wave templates
    def min_dist_to_templates(x_cand):
        dists = []
        for x_true in dataset:
            d = math.sqrt(sum((x_cand[i] - x_true[i]) ** 2 for i in range(8)))
            dists.append(d)
        return min(dists)

    dist_valid = min_dist_to_templates(rec_x)
    dist_hallucinated = min_dist_to_templates(x_hallucinated)

    print(f"\n  The 'Empty Latent Space' Diagnostic:")
    print(f"    Sampled from support z       : Distance to valid manifold = {dist_valid:.4f}")
    print(f"    Sampled from empty void {z_void}: Distance to valid manifold = {dist_hallucinated:.4f}")
    print(f"    Hallucinated Output Vector   : {[round(v, 2) for v in x_hallucinated]}")

    print("\n  Key Takeaway:")
    print("  * Autoregressive models model the exact data density P(x) and can generate valid sequences.")
    print("  * Deterministic Autoencoders only learn an isolated zero-loss manifold with no prior constraint.")
    print("  * Sampling random z vectors fails because empty latent space regions decode into garbage.")
    print("  * This motivates Variational Autoencoders (VAEs), which enforce a smooth prior p(z) = N(0, I).")
    print("=" * 76)
    print("EXPERIMENT COMPLETE: Exit code 0.")
    return True


if __name__ == "__main__":
    run_experiment()
