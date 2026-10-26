# Phase 23: Embeddings & Metric Learning

---

## 🧭 Executive Overview

In classical machine learning and early natural language processing, concepts were treated as discrete, orthogonal symbols. A word, image, or entity was represented as a one-hot basis vector of dimensionality $V \sim 10^6$ where the inner product between distinct entities was identically zero: $\mathbf{e}_i^\top \mathbf{e}_j = 0$. This discrete representation suffered from severe semantic isolation and combinatorial sparsity.

**Phase 23: Embeddings & Metric Learning** explores the mathematical and algorithmic foundations of continuous representation learning. Across this phase, discrete data points are mapped into dense vector spaces $\mathbb{R}^d$ ($d \ll V$) where geometric proximity (Euclidean distance and Cosine similarity) mirrors human semantic judgment.

Every single module across Phase 23 is implemented from first principles in pure Python 3 with **zero external dependencies** (no PyTorch, no Hugging Face, no NumPy, no external API keys).

Across four core modules, this phase establishes:
1. **Word Embeddings (Word2Vec to FastText):** The Distributional Hypothesis (Firth 1957), Continuous Bag-of-Words (CBOW) vs Continuous Skip-Gram, the Full Softmax Bottleneck, the Negative Sampling Objective (SGNS) and its analytical gradients, the 3/4-power unigram noise distribution, Levy & Goldberg's shifted PMI theorem, vector arithmetic ($\mathbf{v}_{\text{King}} - \mathbf{v}_{\text{Man}} + \mathbf{v}_{\text{Woman}} \approx \mathbf{v}_{\text{Queen}}$), and FastText subword character $n$-gram representations for typo resilience and OOV handling without `<UNK>`.
2. **Sentence Embeddings & SBERT:** The Cross-Encoder computational bottleneck ($\mathcal{O}(N^2)$), the Anisotropy Problem of raw BERT embeddings (narrow cone collapse with average cosine $> 0.90$), Sentence-BERT (SBERT) Siamese networks with tied weights, Mean-Pooling over attention masks, NLI triad classification with absolute difference vectors ($[\mathbf{u}, \mathbf{v}, |\mathbf{u} - \mathbf{v}|]$), cosine similarity regression, unit hypersphere projection $\mathbb{S}^{d-1}$, and BERT-Whitening.
3. **Metric Learning & Contrastive Losses:** Isometric metric spaces, Contrastive Loss (Hadsell et al. 2006) with Hooke's attractive spring and electrostatic repulsion, Triplet Loss (FaceNet) relative margin boundaries ($\max(0, D_{ap} - D_{an} + \alpha)$), and online negative mining regimes (Easy, Hard, and Semi-Hard negatives).
4. **InfoNCE & Multimodal CLIP:** The InfoNCE loss (CPC), step-by-step mathematical proof that InfoNCE maximizes a lower bound on Mutual Information ($I(X; Y) \ge \log(K) - \mathcal{L}_{\text{InfoNCE}}$), temperature parameter $\tau$ as an automatic continuous hard-negative selector, OpenAI's CLIP architecture with dual vision and text encoders, $N \times N$ symmetric cross-entropy loss, and open-vocabulary Zero-Shot Classification via natural language prompt prototypes.

---

## 🗺️ Phase Syllabus & Topic Breakdown

```text
phase_23_embeddings_and_metric_learning/
├── index.html                                                    # Phase Portal Hub & Visual Syllabus
├── README.md                                                     # Master Architecture Guide & Derivations
│
├── 01_word_embeddings_word2vec_to_fasttext/
│   ├── lecture.html / lecture.md                                 # Firth hypothesis, CBOW vs Skip-Gram, SGNS derivations, FastText n-grams
│   ├── exercises_and_solutions.html / .md                        # Softmax vs SGNS complexity, numerical updates, shifted PMI proof
│   └── experiment.py                                             # Pure Python SGNS training, cluster similarity, FastText OOV (Exit Code 0)
│
├── 02_sentence_embeddings_and_sbert/
│   ├── lecture.html / lecture.md                                 # Anisotropy cone, Cross vs Bi-encoders, SBERT Siamese, difference vectors, S^(d-1)
│   ├── exercises_and_solutions.html / .md                        # Anisotropy floor proof, NLI gradient derivation, BERT-whitening proof
│   └── experiment.py                                             # Pure Python anisotropy cone, SBERT Siamese training, Whitening (Exit Code 0)
│
├── 03_metric_learning_and_contrastive_losses/
│   ├── lecture.html / lecture.md                                 # Metric learning principles, Contrastive loss, Triplet loss, Semi-hard mining
│   ├── exercises_and_solutions.html / .md                        # Gradient steps, zero-loss geometric boundaries, mining classification
│   └── experiment.py                                             # Pure Python Contrastive & Triplet training in 2D with ASCII scatter (Exit Code 0)
│
└── 04_infonce_and_multimodal_clip/
    ├── lecture.html / lecture.md                                 # InfoNCE loss, MI lower bound proof, temperature tuning, CLIP, zero-shot transfer
    ├── exercises_and_solutions.html / .md                        # Full MI lower bound proof, analytical gradients, batch memory scaling
    └── experiment.py                                             # Pure Python InfoNCE, mini-CLIP dual encoder, zero-shot test (Exit Code 0)
```

---

## 🔑 Core Mathematical Pillars Mastered in Phase 23

1. **Skip-Gram with Negative Sampling (SGNS) Objective:**
   $$\boxed{\mathcal{L}_{\text{SGNS}} = \log \sigma(\mathbf{v}_{w_O}'^\top \mathbf{v}_{w_I}) + \sum_{i=1}^k \mathbb{E}_{w_i \sim P_n(w)} \left[ \log \sigma(-\mathbf{v}_{w_i}'^\top \mathbf{v}_{w_I}) \right]}$$
   Reduces computation per word update from $\mathcal{O}(V \cdot d)$ down to $\mathcal{O}((k + 1) \cdot d)$, delivering over $50,000\times$ acceleration.

2. **Shifted Pointwise Mutual Information Factorization (Levy & Goldberg, 2014):**
   $$\boxed{\mathbf{v}_w^\top \mathbf{v}_c' = \text{PMI}(w, c) - \log k}$$
   Proves that word vector inner products implicitly factorize the corpus log-odds co-occurrence matrix.

3. **Hypersphere Cosine-Euclidean Equivalence:**
   $$\boxed{\|\hat{\mathbf{u}} - \hat{\mathbf{v}}\|_2^2 = 2 - 2 \cos(\mathbf{u}, \mathbf{v}) \quad \forall \hat{\mathbf{u}}, \hat{\mathbf{v}} \in \mathbb{S}^{d-1}}$$
   Maximizing cosine similarity is mathematically identical to minimizing Euclidean distance on the unit sphere.

4. **Triplet Loss with Margin $\alpha$:**
   $$\boxed{\mathcal{L}_{\text{triplet}} = \max\left(0, \|f(a) - f(p)\|_2^2 - \|f(a) - f(n)\|_2^2 + \alpha\right)}$$
   Optimizes relative rank ordering; semi-hard mining ($D_{ap} < D_{an} < D_{ap} + \alpha$) is critical for gradient stability.

5. **InfoNCE Mutual Information Lower Bound (van den Oord et al., 2018):**
   $$\boxed{I(X; Y) \ge \log(K) - \mathcal{L}_{\text{InfoNCE}}}$$
   Minimizing InfoNCE loss directly maximizes a rigorous lower bound on the mutual information between representation spaces.

6. **CLIP Symmetric Multimodal Objective (Radford et al., 2021):**
   $$\boxed{\mathcal{L}_{\text{CLIP}} = -\frac{1}{2N} \sum_{i=1}^N \left[ \log \frac{\exp(S_{ii})}{\sum_{j=1}^N \exp(S_{ij})} + \log \frac{\exp(S_{ii})}{\sum_{j=1}^N \exp(S_{ji})} \right]}$$
   Aligns vision and text encoders into a shared isomorphic metric space enabling open-vocabulary zero-shot classification.

---

## 🧪 Verification & Execution

Every experiment script runs independently in standard Python 3 with zero external dependencies:

```bash
# 1. Word Embeddings (Word2Vec SGNS & FastText OOV Synthesizer)
python3 phase_23_embeddings_and_metric_learning/01_word_embeddings_word2vec_to_fasttext/experiment.py

# 2. Sentence Embeddings (Anisotropy Cone, SBERT Siamese & Whitening)
python3 phase_23_embeddings_and_metric_learning/02_sentence_embeddings_and_sbert/experiment.py

# 3. Metric Learning (Contrastive Loss, Triplet Loss & Semi-Hard Mining in 2D)
python3 phase_23_embeddings_and_metric_learning/03_metric_learning_and_contrastive_losses/experiment.py

# 4. InfoNCE Loss & Multimodal CLIP Dual-Encoder (Zero-Shot Classification)
python3 phase_23_embeddings_and_metric_learning/04_infonce_and_multimodal_clip/experiment.py
```
