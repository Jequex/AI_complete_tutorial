#!/usr/bin/env python3
"""
Phase 23 - Topic 02: Sentence Embeddings & Sentence-BERT (SBERT)
Experiment: Siamese Network with Mean-Pooling & Anisotropy Cone Resolution

This experiment implements from first principles in pure Python 3:
1. Contextual token encoder with Mean Pooling over attention masks.
2. Demonstration of the Anisotropy Cone: showing why uncalibrated / raw mean embeddings
   collapse into a narrow cone with high average cosine similarity (> 0.85) for unrelated sentences.
3. Post-processing resolution: BERT-Whitening (centering and variance scaling) restoring isotropic geometry.
4. Siamese SBERT architecture with tied encoder weights and Mean-Pooling.
5. Analytical backpropagation using Cosine Similarity Regression & NLI distance objectives:
   grad(cos(u, v)) = (1 / ||u||) * (v_hat - cos * u_hat).
6. SBERT training resolving the anisotropy cone: spreading embeddings across the unit hypersphere S^(d-1).
7. Benchmark evaluation on semantic sentence pairs (Paraphrases vs Neutral vs Contradiction).

Zero external dependencies: pure standard Python 3.
"""

import math
import random
import sys
from typing import List, Dict, Tuple


def dot(u: List[float], v: List[float]) -> float:
    return sum(a * b for a, b in zip(u, v))


def norm(u: List[float]) -> float:
    return math.sqrt(sum(a * a for a in u))


def cosine_similarity(u: List[float], v: List[float]) -> float:
    nu = norm(u)
    nv = norm(v)
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return dot(u, v) / (nu * nv)


class SentenceEncoder:
    """
    Contextual sentence encoder mapping token sequences to fixed-size vectors
    via embedding lookup and Mean-Pooling over active attention masks.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 16, seed: int = 42):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        random.seed(seed)
        
        # Initialize word embeddings: V x d
        scale = 0.1 / math.sqrt(embed_dim)
        self.emb = [
            [(random.random() - 0.5) * scale for _ in range(embed_dim)]
            for _ in range(vocab_size)
        ]

    def encode(self, token_ids: List[int], mask: List[int]) -> List[float]:
        """
        Computes Mean-Pooled sentence representation:
        u = (1 / sum(m_t)) * sum_t (m_t * e_t)
        """
        mask_sum = sum(mask)
        if mask_sum == 0:
            return [0.0] * self.embed_dim
        
        pooled = [0.0] * self.embed_dim
        for t, tid in enumerate(token_ids):
            if mask[t] == 1:
                e_t = self.emb[tid]
                for j in range(self.embed_dim):
                    pooled[j] += e_t[j]
                    
        return [val / mask_sum for val in pooled]


class SiameseSBERT:
    """
    Siamese Sentence-BERT network with tied encoder weights, Mean-Pooling,
    and analytical Cosine-Similarity loss optimization.
    """
    def __init__(self, encoder: SentenceEncoder):
        self.encoder = encoder
        self.embed_dim = encoder.embed_dim

    def train_step(self, sA_ids: List[int], sA_mask: List[int],
                   sB_ids: List[int], sB_mask: List[int],
                   target_sim: float, lr: float) -> float:
        """
        Executes a Siamese update step minimizing:
        L = 0.5 * (cos(u, v) - target_sim)^2
        
        Uses exact analytical gradients:
        d(cos)/du = (1 / ||u||) * (v_hat - cos * u_hat)
        d(cos)/dv = (1 / ||v||) * (u_hat - cos * v_hat)
        """
        u = self.encoder.encode(sA_ids, sA_mask)
        v = self.encoder.encode(sB_ids, sB_mask)
        
        nu = norm(u) + 1e-12
        nv = norm(v) + 1e-12
        cos = dot(u, v) / (nu * nv)
        
        loss = 0.5 * ((cos - target_sim) ** 2)
        diff = cos - target_sim  # dL / d(cos)
        
        u_hat = [x / nu for x in u]
        v_hat = [x / nv for x in v]
        
        # Gradients w.r.t pooled vectors u and v
        grad_u = [diff * (1.0 / nu) * (v_hat[j] - cos * u_hat[j]) for j in range(self.embed_dim)]
        grad_v = [diff * (1.0 / nv) * (u_hat[j] - cos * v_hat[j]) for j in range(self.embed_dim)]
        
        # Backprop through Mean-Pooling into token embeddings
        len_A = max(sum(sA_mask), 1)
        len_B = max(sum(sB_mask), 1)
        
        for t, tid in enumerate(sA_ids):
            if sA_mask[t] == 1:
                for j in range(self.embed_dim):
                    self.encoder.emb[tid][j] -= lr * (grad_u[j] / len_A)
                    
        for t, tid in enumerate(sB_ids):
            if sB_mask[t] == 1:
                for j in range(self.embed_dim):
                    self.encoder.emb[tid][j] -= lr * (grad_v[j] / len_B)
                    
        return loss


def simulate_anisotropic_embeddings(num_sentences: int = 50, dim: int = 16, seed: int = 42) -> List[List[float]]:
    """
    Simulates raw BERT embeddings with a dominant shared mean vector mu,
    demonstrating the Anisotropy Cone Problem.
    """
    random.seed(seed)
    mu = [3.0] * dim
    embeddings = []
    for _ in range(num_sentences):
        eps = [(random.random() - 0.5) * 0.3 for _ in range(dim)]
        vec = [mu[i] + eps[i] for i in range(dim)]
        embeddings.append(vec)
    return embeddings


def apply_whitening(vectors: List[List[float]]) -> List[List[float]]:
    """
    Applies centering and variance normalization to eliminate anisotropy.
    """
    M = len(vectors)
    d = len(vectors[0])
    
    mu = [sum(vectors[i][j] for i in range(M)) / M for j in range(d)]
    centered = [[vectors[i][j] - mu[j] for j in range(d)] for i in range(M)]
    
    variances = [sum(c[j] ** 2 for c in centered) / M for j in range(d)]
    std_devs = [math.sqrt(max(v, 1e-8)) for v in variances]
    
    return [[c[j] / std_devs[j] for j in range(d)] for c in centered]


def main():
    print("=" * 80)
    print("PHASE 23 - TOPIC 02: SENTENCE EMBEDDINGS & SENTENCE-BERT (SBERT)")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # PART 1: The Anisotropy Cone Problem of Raw Transformer Embeddings
    # -------------------------------------------------------------------------
    print("\n[PART 1] Demonstrating the Anisotropy Cone Phenomenon...")
    aniso_vectors = simulate_anisotropic_embeddings(num_sentences=50, dim=16, seed=42)
    
    aniso_sims = []
    for i in range(len(aniso_vectors)):
        for j in range(i + 1, len(aniso_vectors)):
            aniso_sims.append(cosine_similarity(aniso_vectors[i], aniso_vectors[j]))
    avg_aniso_sim = sum(aniso_sims) / len(aniso_sims)
    min_aniso_sim = min(aniso_sims)
    max_aniso_sim = max(aniso_sims)

    print(f"[*] Raw / Uncalibrated Pairwise Cosine Similarities across 50 sentences:")
    print(f"    - Average Cosine Similarity : {avg_aniso_sim:.4f} (Expected > 0.85 in narrow cone)")
    print(f"    - Min / Max Range           : [{min_aniso_sim:.4f}, {max_aniso_sim:.4f}]")
    assert avg_aniso_sim > 0.85, f"Failed: Raw anisotropic embeddings should show avg cosine > 0.85, got {avg_aniso_sim:.4f}"
    print("    [!] Anisotropy Cone Confirmed: Completely distinct sentences collapsed into high cosine similarity!")

    # Apply Whitening transformation
    print("\n[*] Applying Whitening & Centering to restore isotropic geometry...")
    whitened_vectors = apply_whitening(aniso_vectors)
    white_sims = []
    for i in range(len(whitened_vectors)):
        for j in range(i + 1, len(whitened_vectors)):
            white_sims.append(cosine_similarity(whitened_vectors[i], whitened_vectors[j]))
    avg_white_sim = sum(white_sims) / len(white_sims)
    print(f"    - Whitened Average Cosine Similarity : {avg_white_sim:+.4f} (Centered near 0.0)")
    assert abs(avg_white_sim) < 0.15, f"Failed: Whitened embeddings should center near 0.0, got {avg_white_sim:.4f}"
    print("    [✓] Whitening successfully expanded the narrow cone across the full sphere!")

    # -------------------------------------------------------------------------
    # PART 2: Siamese SBERT Architecture & Training
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[PART 2] Training Siamese SBERT with Mean-Pooling & Cosine Similarity Loss")
    print("=" * 80)

    # Curated semantic sentence pairs:
    # 1. Paraphrases (Target Cosine = +1.0)
    # 2. Neutral / Cross-Domain (Target Cosine = 0.0)
    # 3. Contradictions / Opposites (Target Cosine = -0.5)
    training_data = [
        # Domain 1: Finance
        (["the", "stock", "market", "rose", "today"], ["equities", "gained", "value", "in", "trading"], 1.0),
        (["the", "bank", "raised", "interest", "rates"], ["the", "central", "bank", "increased", "rates"], 1.0),
        (["inflation", "surged", "rapidly", "this", "quarter"], ["prices", "climbed", "steeply", "this", "month"], 1.0),
        
        # Domain 2: Space Exploration
        (["the", "rocket", "launched", "into", "orbit"], ["the", "spacecraft", "ascended", "to", "space"], 1.0),
        (["astronauts", "walked", "on", "the", "moon"], ["explorers", "traversed", "the", "lunar", "surface"], 1.0),
        (["the", "satellite", "orbits", "the", "earth"], ["the", "probe", "circles", "the", "planet"], 1.0),

        # Domain 3: Culinary
        (["the", "chef", "baked", "delicious", "bread"], ["the", "baker", "made", "fresh", "pastries"], 1.0),
        (["cooking", "a", "healthy", "meal", "at", "home"], ["preparing", "nutritious", "dinner", "in", "kitchen"], 1.0),

        # Cross-Domain Neutral Pairs (Orthogonal concepts)
        (["the", "stock", "market", "rose", "today"], ["the", "rocket", "launched", "into", "orbit"], 0.0),
        (["the", "bank", "raised", "interest", "rates"], ["the", "chef", "baked", "delicious", "bread"], 0.0),
        (["astronauts", "walked", "on", "the", "moon"], ["equities", "gained", "value", "in", "trading"], 0.0),
        (["the", "satellite", "orbits", "the", "earth"], ["cooking", "a", "healthy", "meal", "at", "home"], 0.0),
        (["inflation", "surged", "rapidly", "this", "quarter"], ["the", "spacecraft", "ascended", "to", "space"], 0.0),

        # Contradictions / Opposites
        (["the", "stock", "market", "rose", "today"], ["the", "stock", "market", "crashed", "today"], -0.5),
        (["the", "bank", "raised", "rates"], ["the", "bank", "slashed", "rates", "to", "zero"], -0.5),
        (["the", "rocket", "launched", "successfully"], ["the", "rocket", "exploded", "on", "launch"], -0.5)
    ]

    # Build vocabulary
    vocab = {"<PAD>": 0}
    for sA, sB, _ in training_data:
        for w in sA + sB:
            if w not in vocab:
                vocab[w] = len(vocab)
    print(f"[+] Compiled vocabulary: {len(vocab)} unique tokens.")

    # Convert to padded token IDs and attention masks
    max_len = 8
    dataset = []
    for sA, sB, target_sim in training_data:
        ids_A = [vocab[w] for w in sA] + [0] * (max_len - len(sA))
        mask_A = [1] * len(sA) + [0] * (max_len - len(sA))
        ids_B = [vocab[w] for w in sB] + [0] * (max_len - len(sB))
        mask_B = [1] * len(sB) + [0] * (max_len - len(sB))
        dataset.append((ids_A, mask_A, ids_B, mask_B, target_sim))

    # Initialize Siamese SBERT
    encoder = SentenceEncoder(vocab_size=len(vocab), embed_dim=16, seed=42)
    sbert = SiameseSBERT(encoder=encoder)

    # Evaluate untrained similarities
    def evaluate_group(target_val: float) -> float:
        sims = []
        for ids_A, mask_A, ids_B, mask_B, target in dataset:
            if target == target_val:
                u = sbert.encoder.encode(ids_A, mask_A)
                v = sbert.encoder.encode(ids_B, mask_B)
                sims.append(cosine_similarity(u, v))
        return sum(sims) / len(sims)

    pre_para = evaluate_group(1.0)
    pre_neut = evaluate_group(0.0)
    pre_contra = evaluate_group(-0.5)
    print(f"[*] Pre-training Average Cosine Similarities:")
    print(f"    - Paraphrases (target +1.0) : {pre_para:+.4f}")
    print(f"    - Neutral     (target  0.0) : {pre_neut:+.4f}")
    print(f"    - Contradict  (target -0.5) : {pre_contra:+.4f}")

    # Train SBERT
    epochs = 120
    lr = 0.05
    random.seed(42)
    print(f"\n[+] Training SBERT across {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        random.shuffle(dataset)
        for ids_A, mask_A, ids_B, mask_B, target in dataset:
            loss = sbert.train_step(ids_A, mask_A, ids_B, mask_B, target, lr)
            total_loss += loss
        if epoch % 30 == 0 or epoch == epochs:
            print(f"    Epoch {epoch:3d}/{epochs} | Avg Cosine Loss: {total_loss / len(dataset):.4f}")

    # Post-training evaluation
    print("\n" + "=" * 80)
    print("SBERT BENCHMARK: POST-TRAINING COSINE SIMILARITIES")
    print("=" * 80)
    post_para = evaluate_group(1.0)
    post_neut = evaluate_group(0.0)
    post_contra = evaluate_group(-0.5)

    print(f"[*] Post-training Average Cosine Similarities:")
    print(f"    - Paraphrases (target +1.0) : {post_para:+.4f} (High Positive Sim)")
    print(f"    - Neutral     (target  0.0) : {post_neut:+.4f} (Orthogonal / Unrelated)")
    print(f"    - Contradict  (target -0.5) : {post_contra:+.4f} (Opposite / Repelled)")

    # Assertions
    assert post_para > 0.85, f"Failed: Paraphrase similarity ({post_para:.4f}) should be > 0.85"
    assert abs(post_neut) < 0.20, f"Failed: Neutral similarity ({post_neut:.4f}) should be near 0.0"
    assert post_contra < -0.20, f"Failed: Contradiction similarity ({post_contra:.4f}) should be negative"
    assert post_para > post_neut > post_contra, "Failed: Correct semantic ordering (Para > Neut > Contra) not satisfied"
    
    print("\n[✓] All SBERT Semantic Ranking and Anisotropy Resolution assertions PASSED!")
    print("\n[★] EXPERIMENT COMPLETE: ALL TESTS PASSED SUCCESSFULLY (Exit Code 0).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
