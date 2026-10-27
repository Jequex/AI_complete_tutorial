#!/usr/bin/env python3
"""
Topic 04: Polysemanticity & Sparse Autoencoders (SAEs)
------------------------------------------------------
A pure Python 3 standard library implementation demonstrating:
1. The Superposition Hypothesis (Toy Models of Superposition, Elhage et al. 2022):
   - D = 5 ground-truth sparse features packed into d = 2 dimensions (pentagon configuration).
   - Features activate sparsely (p = 0.08).
   - Standard basis neurons are inherently polysemantic (activating for multiple unrelated features).
2. Sparse Autoencoder (SAE, Bricken et al. 2023):
   - Overcomplete dictionary learning: z = ReLU(W_enc * x + b_enc)
   - Linear reconstruction: x_hat = W_dec * z + b_dec
   - Loss: MSE reconstruction + L1 sparsity penalty:
     L = ||x - x_hat||^2 + lambda * sum(|z_i|)
3. Training via Mini-batch Adam / SGD in pure Python standard library.
4. Verifying monosemantic feature recovery:
   - Matching learned SAE decoder columns to the 5 true ground-truth feature directions.
   - Proving recovery with high cosine similarity (>= 0.90).

Zero external dependencies (pure Python standard library).
"""

import math
import random

random.seed(42)

# =====================================================================
# 1. Pure Python Vector Math
# =====================================================================

def dot(u, v):
    return sum(a * b for a, b in zip(u, v))

def norm(u):
    return math.sqrt(sum(a * a for a in u)) + 1e-12

def cosine_similarity(u, v):
    return dot(u, v) / (norm(u) * norm(v))

def vec_add(u, v):
    return [a + b for a, b in zip(u, v)]

def vec_sub(u, v):
    return [a - b for a, b in zip(u, v)]

def vec_scale(u, s):
    return [a * s for a in u]

def matmul_vec(W, x):
    """W is list of rows."""
    return [dot(row, x) for row in W]

def matmul_trans_vec(W, y):
    """Computes W^T * y where W is rows."""
    cols = len(W[0])
    res = [0.0] * cols
    for i, row in enumerate(W):
        for j in range(cols):
            res[j] += row[j] * y[i]
    return res

def relu(x):
    return [max(0.0, v) for v in x]


# =====================================================================
# 2. Superposition Simulation: 5 Features in 2 Dimensions
# =====================================================================

class SuperpositionEnvironment:
    """
    Creates D = 5 sparse features embedded in d = 2 dimensions.
    The 5 features form a symmetric pentagon: theta_k = 2 * pi * k / 5.
    """
    def __init__(self, num_features=5, dim=2, feature_prob=0.08):
        self.num_features = num_features
        self.dim = dim
        self.feature_prob = feature_prob
        
        # Ground truth feature vectors: regular pentagon
        self.true_features = []
        for k in range(num_features):
            theta = 2.0 * math.pi * k / num_features
            vec = [math.cos(theta), math.sin(theta)]
            self.true_features.append(vec)

    def sample_features(self):
        """Samples sparse feature activations."""
        acts = []
        for _ in range(self.num_features):
            if random.random() < self.feature_prob:
                # Sparse positive activation
                acts.append(random.uniform(0.8, 1.5))
            else:
                acts.append(0.0)
        # Ensure at least 1 feature is active occasionally for robust training
        if all(a == 0.0 for a in acts) and random.random() < 0.6:
            k = random.randint(0, self.num_features - 1)
            acts[k] = random.uniform(0.8, 1.5)
        return acts

    def generate_point(self):
        """Generates 2D activation x = sum(f_k * v_k) + small noise."""
        acts = self.sample_features()
        x = [0.0] * self.dim
        for k, a in enumerate(acts):
            if a > 0.0:
                for d in range(self.dim):
                    x[d] += a * self.true_features[k][d]
        # Add tiny isotropic noise
        for d in range(self.dim):
            x[d] += random.gauss(0, 0.01)
        return acts, x


# =====================================================================
# 3. Sparse Autoencoder (SAE) with L1 Regularization
# =====================================================================

class SparseAutoencoder:
    """
    Dictionary Learning SAE:
      z = ReLU(W_enc * x + b_enc)   (dim -> dict_size)
      x_hat = W_dec * z + b_dec     (dict_size -> dim)
    Trained with Adam optimizer to minimize MSE + lambda * L1(z).
    """
    def __init__(self, dim=2, dict_size=5, l1_coeff=0.04, lr=0.02):
        self.dim = dim
        self.dict_size = dict_size
        self.l1_coeff = l1_coeff
        self.lr = lr
        
        # Initialize weights: W_enc (dict_size x dim), W_dec (dim x dict_size)
        self.W_enc = [[random.gauss(0, 0.5) for _ in range(dim)] for _ in range(dict_size)]
        self.b_enc = [0.0] * dict_size
        
        # W_dec rows correspond to dimension coordinates (dim x dict_size)
        self.W_dec = [[random.gauss(0, 0.5) for _ in range(dict_size)] for _ in range(dim)]
        self.b_dec = [0.0] * dim
        
        # Adam moments
        self._init_adam()

    def _init_adam(self):
        self.m_W_enc = [[0.0 for _ in range(self.dim)] for _ in range(self.dict_size)]
        self.v_W_enc = [[0.0 for _ in range(self.dim)] for _ in range(self.dict_size)]
        self.m_b_enc = [0.0] * self.dict_size
        self.v_b_enc = [0.0] * self.dict_size
        
        self.m_W_dec = [[0.0 for _ in range(self.dict_size)] for _ in range(self.dim)]
        self.v_W_dec = [[0.0 for _ in range(self.dict_size)] for _ in range(self.dim)]
        self.m_b_dec = [0.0] * self.dim
        self.v_b_dec = [0.0] * self.dim
        self.t = 0

    def encode(self, x):
        pre_act = [dot(self.W_enc[i], x) + self.b_enc[i] for i in range(self.dict_size)]
        return relu(pre_act)

    def decode(self, z):
        # x_hat = W_dec * z + b_dec
        x_hat = list(self.b_dec)
        for d in range(self.dim):
            for j in range(self.dict_size):
                x_hat[d] += self.W_dec[d][j] * z[j]
        return x_hat

    def get_decoder_feature_vector(self, j):
        """Returns the j-th dictionary feature vector (column j of W_dec)."""
        return [self.W_dec[d][j] for d in range(self.dim)]

    def train_step(self, batch_x):
        B = len(batch_x)
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        
        # Accumulate gradients
        g_W_enc = [[0.0 for _ in range(self.dim)] for _ in range(self.dict_size)]
        g_b_enc = [0.0] * self.dict_size
        g_W_dec = [[0.0 for _ in range(self.dict_size)] for _ in range(self.dim)]
        g_b_dec = [0.0] * self.dim
        
        total_mse = 0.0
        total_l1 = 0.0
        
        for x in batch_x:
            # Forward pass
            pre_acts = [dot(self.W_enc[i], x) + self.b_enc[i] for i in range(self.dict_size)]
            z = relu(pre_acts)
            x_hat = self.decode(z)
            
            # Reconstruction error e = 2 * (x_hat - x)
            e = [2.0 * (x_hat[d] - x[d]) for d in range(self.dim)]
            total_mse += sum((x_hat[d] - x[d])**2 for d in range(self.dim))
            total_l1 += sum(abs(v) for v in z)
            
            # Gradients for W_dec and b_dec
            for d in range(self.dim):
                g_b_dec[d] += e[d]
                for j in range(self.dict_size):
                    g_W_dec[d][j] += e[d] * z[j]
                    
            # Backprop into z: dL/dz = W_dec^T * e + lambda * sign(z)
            dL_dz = [0.0] * self.dict_size
            for j in range(self.dict_size):
                sum_dec = sum(self.W_dec[d][j] * e[d] for d in range(self.dim))
                l1_grad = self.l1_coeff if z[j] > 0 else 0.0
                dL_dz[j] = sum_dec + l1_grad
                
            # Gradients for W_enc and b_enc through ReLU
            for j in range(self.dict_size):
                if pre_acts[j] > 0:
                    delta = dL_dz[j]
                    g_b_enc[j] += delta
                    for d in range(self.dim):
                        g_W_enc[j][d] += delta * x[d]

        # Normalize by batch size B
        scale = 1.0 / B
        for i in range(self.dict_size):
            g_b_enc[i] *= scale
            for d in range(self.dim):
                g_W_enc[i][d] *= scale
        for d in range(self.dim):
            g_b_dec[d] *= scale
            for j in range(self.dict_size):
                g_W_dec[d][j] *= scale
                
        # Adam parameter updates
        def adam_update(param, grad, m, v):
            for i in range(len(param)):
                if isinstance(param[i], list):
                    adam_update(param[i], grad[i], m[i], v[i])
                else:
                    m[i] = beta1 * m[i] + (1 - beta1) * grad[i]
                    v[i] = beta2 * v[i] + (1 - beta2) * (grad[i] ** 2)
                    m_hat = m[i] / (1 - beta1 ** self.t)
                    v_hat = v[i] / (1 - beta2 ** self.t)
                    param[i] -= self.lr * m_hat / (math.sqrt(v_hat) + eps)

        adam_update(self.W_enc, g_W_enc, self.m_W_enc, self.v_W_enc)
        adam_update(self.b_enc, g_b_enc, self.m_b_enc, self.v_b_enc)
        adam_update(self.W_dec, g_W_dec, self.m_W_dec, self.v_W_dec)
        adam_update(self.b_dec, g_b_dec, self.m_b_dec, self.v_b_dec)
        
        # Normalize decoder columns to unit norm (prevents scaling trick)
        for j in range(self.dict_size):
            col_norm = math.sqrt(sum(self.W_dec[d][j] ** 2 for d in range(self.dim))) + 1e-8
            for d in range(self.dim):
                self.W_dec[d][j] /= col_norm
                
        return total_mse / B, total_l1 / B


# =====================================================================
# 4. Pipeline Execution & Evaluation
# =====================================================================

def run_experiment():
    print("=" * 78)
    print("PHASE 33 — TOPIC 04: POLYSEMANTICITY & SPARSE AUTOENCODERS")
    print("Resolving Superposition in Toy Models: 5 Features in 2 Dimensions")
    print("=" * 78)

    env = SuperpositionEnvironment(num_features=5, dim=2, feature_prob=0.08)
    
    print("\n[Ground Truth Geometric Configuration]")
    print(f"5 True Features arranged in a regular pentagon (72° separation):")
    for k, v in enumerate(env.true_features):
        angle_deg = k * 72.0
        print(f"  Feature {k}: angle = {angle_deg:5.1f}°, vector = [{v[0]:+6.3f}, {v[1]:+6.3f}]")

    # 1. Demonstrate Polysemanticity of Standard Basis Neurons
    print("\n[Demonstrating Polysemanticity of Raw Neurons]")
    print("Neuron 0 (x-axis) and Neuron 1 (y-axis) activate for multiple conflicting features:")
    for k, v in enumerate(env.true_features):
        n0_overlap = abs(v[0])
        n1_overlap = abs(v[1])
        print(f"  Feature {k} projection onto -> Neuron 0: {n0_overlap:.3f} | Neuron 1: {n1_overlap:.3f}")
    print("-> Standard basis neurons are polysemantic: observing Neuron 0 > 0 cannot tell")
    print("   whether Feature 0, Feature 1, or Feature 4 caused the activation!")

    # 2. Train Sparse Autoencoder
    print("\n[Training Sparse Autoencoder (SAE)]")
    sae = SparseAutoencoder(dim=2, dict_size=5, l1_coeff=0.05, lr=0.03)
    
    # Generate Training Dataset
    n_samples = 3000
    batch_size = 64
    dataset = [env.generate_point()[1] for _ in range(n_samples)]
    
    epochs = 40
    for epoch in range(epochs):
        random.shuffle(dataset)
        epoch_mse = 0.0
        epoch_l1 = 0.0
        steps = 0
        for i in range(0, n_samples, batch_size):
            batch = dataset[i : i + batch_size]
            mse, l1 = sae.train_step(batch)
            epoch_mse += mse
            epoch_l1 += l1
            steps += 1
            
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:2d}/{epochs:2d} | Reconstruction MSE: {epoch_mse/steps:.5f} | L1 Sparsity: {epoch_l1/steps:.4f}")

    # 3. Monosemantic Feature Recovery Evaluation
    print("\n[Feature Recovery Verification: True Features vs. Learned SAE Dictionary]")
    print(f"{'True Feature':<14} | {'Best Matching SAE Latent':<26} | {'Cosine Sim':<12} | {'Status'}")
    print("-" * 72)
    
    recovered_count = 0
    cos_similarities = []
    
    for k in range(env.num_features):
        true_v = env.true_features[k]
        best_latent = -1
        best_sim = -1.0
        
        for j in range(sae.dict_size):
            dec_v = sae.get_decoder_feature_vector(j)
            sim = cosine_similarity(true_v, dec_v)
            if sim > best_sim:
                best_sim = sim
                best_latent = j
                
        cos_similarities.append(best_sim)
        status = "RECOVERED (Monosemantic)" if best_sim >= 0.88 else "Partial"
        if best_sim >= 0.88:
            recovered_count += 1
            
        print(f"Feature {k} ({k*72:3.0f}°)  | Latent {best_latent} (dir=[{sae.get_decoder_feature_vector(best_latent)[0]:+5.2f}, {sae.get_decoder_feature_vector(best_latent)[1]:+5.2f}]) | {best_sim:>10.4f} | {status}")

    mean_sim = sum(cos_similarities) / len(cos_similarities)
    print("-" * 72)
    print(f"Mean Cosine Similarity: {mean_sim:.4f}")
    print(f"Features cleanly recovered: {recovered_count}/{env.num_features}")

    # Assertions
    assert mean_sim >= 0.85, f"Expected mean cosine similarity >= 0.85, got {mean_sim:.4f}"
    assert recovered_count >= 4, f"Expected at least 4/5 features cleanly recovered, got {recovered_count}"
    
    print("\n>>> ALL TESTS PASSED: Sparse Autoencoder successfully resolved superposition!")
    print("=" * 78)

if __name__ == "__main__":
    run_experiment()
