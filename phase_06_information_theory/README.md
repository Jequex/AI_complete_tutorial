# Phase 06: Information Theory for Artificial Intelligence

---

## 🧭 Executive Overview

If **Calculus** (Phase 03) is the engine of optimization, and **Probability & Statistics** (Phases 04 & 05) provide the language of uncertainty, then **Information Theory** (Phase 06) is the **fundamental currency of artificial intelligence**.

Information Theory answers the profound questions:
- *How much knowledge is contained in a sentence or image?*
- *What is the absolute mathematical limit on data compression?*
- *How different are two probability distributions, and why does that difference drive modern LLM training (Cross-Entropy), Generative Models (VAEs), and AI Alignment (RLHF / DPO)?*

---

## 🗺️ Phase Syllabus & Topic Breakdown

```text
phase_06_information_theory/
├── index.html                                       # Visual Phase Hub
├── README.md                                        # Master Syllabus & Architecture Guide
│
├── 01_entropy_and_information/
│   ├── lecture.html / lecture.md                    # Surprisal, Shannon entropy, bits/nats, Shannon's source coding
│   ├── exercises_and_solutions.html / .md           # Binary entropy derivations, maximal entropy proofs
│   └── experiment.py                                # Pure Python Huffman coding & entropy compression bound engine
│
├── 02_joint_conditional_entropy/
│   ├── lecture.html / lecture.md                    # Joint entropy, conditional entropy, chain rule of entropy
│   ├── exercises_and_solutions.html / .md           # Proof that conditioning never increases entropy
│   └── experiment.py                                # Bigram vs unigram text entropy calculator on English sentences
│
├── 03_cross_entropy_and_perplexity/
│   ├── lecture.html / lecture.md                    # Cross-entropy, next-token prediction in LLMs, perplexity math
│   ├── exercises_and_solutions.html / .md           # Perplexity calculations, temperature scaling & cross-entropy
│   └── experiment.py                                # Mini LLM next-token cross-entropy & perplexity benchmark
│
├── 04_kl_divergence/
│   ├── lecture.html / lecture.md                    # Relative entropy, Gibbs' inequality, asymmetry, VAEs, RLHF policy drift
│   ├── exercises_and_solutions.html / .md           # Analytical KL divergence between two Gaussians & forward vs reverse KL
│   └── experiment.py                                # Forward vs reverse KL mode-covering vs mode-seeking visualizer
│
└── 05_mutual_information_and_information_bottleneck/
    ├── lecture.html / lecture.md                    # Mutual information, Venn diagrams, Tishby's Information Bottleneck
    ├── exercises_and_solutions.html / .md           # Mutual information in noisy communication channels & feature selection
    └── experiment.py                                # Neural layer mutual information estimator tracking representation compression
```

---

## 🔑 Core Mathematical Pillars Mastered in Phase 6

1. **Shannon Entropy:**
   The fundamental lower bound on information encoding:
   $$H(X) = -\sum_{x \in \mathcal{X}} P(x) \log_2 P(x)$$

2. **Cross-Entropy & LLM Perplexity:**
   The universal training objective of modern foundation models:
   $$H(P, Q) = -\sum_{x} P(x) \log_2 Q(x), \quad \text{Perplexity} = 2^{H(P, Q)}$$

3. **Kullback-Leibler (KL) Divergence:**
   Measuring distance between distributions in VAEs and RLHF alignment:
   $$D_{\text{KL}}(P \parallel Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)} \ge 0$$

4. **Mutual Information & Information Bottleneck:**
   Quantifying shared knowledge and optimal representation compression:
   $$I(X; Y) = H(X) - H(X \mid Y) = \min_\theta \Big[ I(X; T) - \beta I(T; Y) \Big]$$

---

## ⚡ Quickstart: Executing Phase 6 Labs

All experiments run in pure Python 3 without external dependencies:

```bash
# Topic 6.1: Entropy & Huffman Compression Limit
python3 phase_06_information_theory/01_entropy_and_information/experiment.py

# Topic 6.2: Joint & Conditional Entropy of Natural Language
python3 phase_06_information_theory/02_joint_conditional_entropy/experiment.py

# Topic 6.3: Cross-Entropy & Perplexity in Language Modeling
python3 phase_06_information_theory/03_cross_entropy_and_perplexity/experiment.py

# Topic 6.4: Forward vs Reverse KL Divergence
python3 phase_06_information_theory/04_kl_divergence/experiment.py

# Topic 6.5: Mutual Information & Representation Bottleneck
python3 phase_06_information_theory/05_mutual_information_and_information_bottleneck/experiment.py
```

