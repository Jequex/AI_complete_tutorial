# Phase 18: Large Language Models & Pretraining

---

## 🧭 Executive Overview

Scaling up from toy autoregressive GPTs to production-grade foundation models represents the defining transition in modern artificial intelligence engineering. At this frontier, models cease to be standalone academic experiments and become planetary-scale computational artifacts whose capabilities, memory footprints, and serving latencies are governed by fundamental laws of physics, statistical learning theory, and distributed systems.

**Phase 18: Large Language Models & Pretraining** establishes the rigorous theoretical and empirical engineering principles required to train and serve modern foundation models:

1. **Scaling Laws & Compute-Optimal Training:**
   How empirical loss surfaces follow strict power-law scaling $L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$; why early research (Kaplan et al., 2020) over-allocated parameters at the expense of data tokens; how the Chinchilla breakthrough (Hoffmann et al., 2022) established that parameters and tokens must scale equally ($N \propto \sqrt{C}, D \propto \sqrt{C}$); and why the modern LLaMA paradigm intentionally over-trains compact models across trillions of tokens to minimize lifetime inference latency and serving expenditure.

2. **Data Curation, Deduplication, and Quality Filtering:**
   How raw Common Crawl web scrapes containing tens of petabytes of noisy, machine-generated spam are refined into gold-standard pretraining corpora (RefinedWeb, Dolma, FineWeb); heuristic filtering (stopword density, symbol ratios, KenLM perplexity); fuzzy near-deduplication using MinHash and Locality-Sensitive Hashing (LSH); benchmark decontamination to protect evaluation integrity; and synthetic reasoning dataset synthesis (Cosmopedia, UltraChat).

3. **Distributed Training & Parallelism Strategies:**
   Overcoming the **Memory Wall** where a 70B parameter model requires $1,120\text{ GB}$ of static HBM during AdamW training; the 3D Parallelism taxonomy combining Fully Sharded Data Parallelism (ZeRO-1, ZeRO-2, ZeRO-3 / FSDP), intra-node Tensor Parallelism (Megatron-LM), and inter-node Pipeline Parallelism (GPipe vs. 1F1B); and mixed-precision dynamics (FP16 dynamic loss scaling vs. native BF16 training).

4. **Modern LLM Architectures (GQA, SwiGLU, FlashAttention):**
   The three architectural innovations powering contemporary open foundation models (LLaMA-3, Mistral, Gemma):
   - **Grouped-Query Attention (GQA):** Compressing the autoregressive KV-cache by up to $8\times$ during token generation to alleviate memory bandwidth saturation.
   - **SwiGLU Gated Activation:** Replacing traditional MLPs with continuous gated bilinear units, deriving the parameter-matched hidden dimension scaling rule $d_{ffn} \approx \frac{8}{3} d_{model}$.
   - **FlashAttention:** An exact IO-aware tiling algorithm utilizing on-chip SRAM and online softmax normalization to eliminate the $\mathcal{O}(T^2)$ HBM memory read/write bottleneck.

---

## 🗺️ Phase Syllabus & Directory Architecture

```text
phase_18_large_language_models_and_pretraining/
├── index.html                                                      # Interactive Phase Portal Hub & Visual Syllabus
├── README.md                                                       # Master Mathematical & Architectural Reference Guide
│
├── 01_scaling_laws_and_compute_optimal_training/
│   ├── lecture.html / lecture.md                                   # Power-law loss surface, Kaplan vs. Chinchilla shift,
│   │                                                               # Lagrange multipliers derivation, LLaMA over-training
│   ├── exercises_and_solutions.html / .md                          # GPT-3 allocation derivation, asymptotic scaling divergence,
│   │                                                               # lifetime inference breakeven volume Q*
│   └── experiment.py                                               # Pure Python 3 Chinchilla loss surface solver, analytical vs.
│                                                                   # numerical isoFLOP minimum, Hessian convexity verification
│
├── 02_data_curation_deduplication_and_filtering/
│   ├── lecture.html / lecture.md                                   # Data pipelines at scale, Gopher/C4 heuristic filters,
│   │                                                               # MinHash theorem, LSH banding & S-curve, decontamination
│   ├── exercises_and_solutions.html / .md                          # LSH parameter tuning (b, r), Shannon entropy filtering,
│   │                                                               # 13-gram benchmark collision probability
│   └── experiment.py                                               # Pure Python 3 MinHash + LSH indexer, Jaccard estimation,
│                                                                   # exact vs near-dedup rate, benchmark contamination check
│
├── 03_distributed_training_and_parallelism_strategies/
│   ├── lecture.html / lecture.md                                   # The Memory Wall (16N bytes), ZeRO-1/2/3 memory formulas,
│   │                                                               # Megatron-LM Column/Row TP, GPipe bubble vs 1F1B schedule
│   ├── exercises_and_solutions.html / .md                          # 175B model GPU sizing across clusters, Ring All-Reduce volume,
│   │                                                               # pipeline bubble fraction and activation memory bounds
│   └── experiment.py                                               # Pure Python 3 memory simulator (7B-70B), ZeRO cluster audit,
│                                                                   # Megatron-LM TP matrix multiplication verification (< 1e-12)
│
└── 04_modern_llm_architectures_gqa_swiglu_flashattention/
    ├── lecture.html / lecture.md                                   # GQA/MQA KV-cache bandwidth bounds, SwiGLU 8/3 dimension parity,
    │                                                               # FlashAttention SRAM tiling & Online Softmax recurrence
    ├── exercises_and_solutions.html / .md                          # KV-cache arithmetic intensity, analytical SwiGLU gradients,
    │                                                               # step-by-step manual FlashAttention online softmax trace
    └── experiment.py                                               # Pure Python 3 GQA execution simulator, SwiGLU parameter parity,
                                                                    # FlashAttention online softmax verification (< 1e-14 error)
```

---

## 🔑 Core Mathematical Pillars Mastered in Phase 18

### 1. The Chinchilla Equi-Marginal Principle
Minimizing $L(N, D) = E + A N^{-\alpha} + B D^{-\beta}$ subject to $C = 6ND$ reveals that at compute optimality, the marginal reducible loss contributed by model capacity must exactly balance the marginal reducible loss contributed by training tokens:
$$\alpha A N^{-\alpha} = \beta B D^{-\beta}$$
Leading to the closed-form optimal allocation:
$$N^* = G \cdot \left(\frac{C}{6}\right)^{\frac{\beta}{\alpha+\beta}}, \quad D^* = G^{-1} \cdot \left(\frac{C}{6}\right)^{\frac{\alpha}{\alpha+\beta}}$$
where $a = \frac{\beta}{\alpha+\beta} \approx 0.46$ and $b = \frac{\alpha}{\alpha+\beta} \approx 0.54$, proving that parameters and tokens should scale in virtually equal proportion ($N \propto \sqrt{C}, D \propto \sqrt{C}$).

### 2. The LLaMA Lifetime Inference Breakeven Boundary
When evaluating total organizational expenditure across pretraining and lifetime production serving:
$$C_{\text{lifecycle}} = 6 N D_{\text{train}} + 2 N Q_{\text{served}}$$
Training a smaller model $N_B < N_A$ with higher pretraining compute $\Delta C_{\text{pretrain}} = C_{pretrain, B} - C_{pretrain, A}$ delivers permanent compute and latency savings once served tokens exceed:
$$Q^* = \frac{\Delta C_{\text{pretrain}}}{2 (N_A - N_B)}$$

### 3. MinHash Theorem & LSH S-Curve Collision Probability
For shingle sets $S_A, S_B$ under random permutation $\pi$:
$$P(\min \pi(S_A) = \min \pi(S_B)) = \frac{|S_A \cap S_B|}{|S_A \cup S_B|} = J(A, B)$$
Grouping $M = b \cdot r$ signature slots into $b$ bands of $r$ rows yields the sigmoidal S-curve collision probability:
$$P_{\text{collision}}(s) = 1 - (1 - s^r)^b$$
creating a sharp threshold at $s^* \approx (1/b)^{1/r}$ that isolates near-duplicate documents with high recall while suppressing unrelated pairs to near-zero probability.

### 4. The 16N Memory Wall & ZeRO Hierarchy
During mixed-precision AdamW training, static state memory scales as $16N$ bytes ($1,120\text{ GB}$ for 70B):
$$\text{Memory}_{\text{DDP}} = 16N$$
$$\text{Memory}_{\text{ZeRO-1}} = 4N + \frac{12N}{N_d}$$
$$\text{Memory}_{\text{ZeRO-2}} = 2N + \frac{14N}{N_d}$$
$$\text{Memory}_{\text{ZeRO-3}} = \frac{16N}{N_d}$$
ZeRO-3 enables a 70B model to fit comfortably onto 16 standard 80GB GPUs ($70\text{ GB}$ per GPU) at the cost of a modest $50\%$ communication overhead ($3N$ words transferred vs $2N$ in DDP).

### 5. FlashAttention Online Softmax Normalization
By tiling attention computation inside fast on-chip SRAM, FlashAttention computes exact softmax incrementally across blocks:
$$m^{\text{new}} = \max(m_1, m_2)$$
$$d^{\text{new}} = d_1 e^{m_1 - m^{\text{new}}} + d_2 e^{m_2 - m^{\text{new}}}$$
$$O^{\text{new}} = O_1 e^{m_1 - m^{\text{new}}} + \left(\sum_j e^{x_j^{(2)} - m^{\text{new}}} V_j^{(2)}\right)$$
HBM traffic drops from $\mathcal{O}(T^2)$ to $\mathcal{O}(T)$ without any mathematical approximation.

---

## 🧪 Verification & Empirical Laboratory Benchmarks

Every laboratory script in Phase 18 is written in clean, dependency-free Python 3 using only the standard library:

```bash
# Topic 18.1: Scaling Laws & Compute-Optimal Frontier
python3 phase_18_large_language_models_and_pretraining/01_scaling_laws_and_compute_optimal_training/experiment.py

# Topic 18.2: Data Curation, MinHash & LSH Near-Deduplication
python3 phase_18_large_language_models_and_pretraining/02_data_curation_deduplication_and_filtering/experiment.py

# Topic 18.3: Distributed Training & ZeRO Memory Sharding Simulator
python3 phase_18_large_language_models_and_pretraining/03_distributed_training_and_parallelism_strategies/experiment.py

# Topic 18.4: Modern LLM Architectures (GQA, SwiGLU, FlashAttention Online Softmax)
python3 phase_18_large_language_models_and_pretraining/04_modern_llm_architectures_gqa_swiglu_flashattention/experiment.py
```
