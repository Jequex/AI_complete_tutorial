# Phase 17: Build a GPT From Scratch

---

## 🧭 Executive Overview

Large Language Models (LLMs) such as GPT-4, Claude, and LLaMA are fundamentally autoregressive Transformers trained to predict the next token in a sequence. While modern models scale to hundreds of billions of parameters across thousands of GPUs, the core mathematical architecture remains remarkably elegant, modular, and grounded in first principles.

**Phase 17: Build a GPT From Scratch** guides you through the complete end-to-end design, mathematical derivation, and pure Python implementation of a Generative Pre-trained Transformer.

Every single component is written completely from scratch in pure Python 3—with **zero external libraries** (no PyTorch, no NumPy, no Hugging Face, no TikToken).

Across four comprehensive modules, this phase covers:
1. **Byte-Pair Encoding (BPE) Tokenizer:** Tokenization taxonomy (character vs word vs subword), the Sennrich et al. (2016) BPE algorithm, frequency statistics, greedy pair merging, merge rule recording, lossless string encoding and decoding, UTF-8 byte fallbacks guaranteeing zero Out-Of-Vocabulary (`<UNK>`) tokens, and special control tokens (`<|endoftext|>`).
2. **Causal Self-Attention & Pre-LN Transformer Blocks:** Enforcing the arrow of time in generative models, upper-triangular negative infinity causal masking ($M_{ij} = -\infty$ for $j > i$), multi-head query-key-value projections, the Pre-LayerNorm architecture (Radford et al., 2019), unhindered residual stream gradient highways, and the $4\times$ expanded feed-forward network (MLP) with smooth Gaussian Error Linear Unit (GELU) activations.
3. **Full GPT Model Architecture & Weight Tying:** Unifying token embedding tables ($W_{\text{token}} \in \mathbb{R}^{V \times d_{\text{model}}}$), learned positional embeddings ($W_{\text{pos}} \in \mathbb{R}^{T_{\max} \times d_{\text{model}}}$), $N$ stacked Transformer blocks, final LayerNorm, and the output Language Model (LM) Head. Weight Tying (Press & Wolf, 2017) mechanics ($W_{\text{head}} = W_{\text{token}}^T$), reducing parameter count by 24% while aligning geometric representations, and exact closed-form parameter counting for GPT-2 Small (124M), Medium (355M), Large (774M), and XL (1.5B).
4. **Autoregressive Generation, Sampling Strategies & KV-Caching:** The sequential next-token feedback loop, greedy search (argmax) pathologies, temperature scaling, top-$k$ truncation, top-$p$ (nucleus) sampling (Holtzman et al., 2020), and the Key-Value (KV) Cache mechanism—eliminating redundant historical projections and reducing per-token generation complexity from $\mathcal{O}(T^2)$ to $\mathcal{O}(T)$.

---

## 🗺️ Phase Syllabus & Topic Breakdown

```text
phase_17_build_a_gpt_from_scratch/
├── index.html                                            # Phase Portal Hub & Quickstart
├── README.md                                             # Master Syllabus & Architecture Guide
│
├── 01_byte_pair_encoding_bpe_tokenizer/
│   ├── lecture.html / lecture.md                         # Subword taxonomy, BPE merge algorithm, byte-level fallback, <|endoftext|>
│   ├── exercises_and_solutions.html / .md                # Manual BPE merge tracing, quadratic FLOPs comparison, emoji byte fallback
│   └── experiment.py                                     # Pure Python BPE Tokenizer class, train, merge table, lossless round-trip test (Exit Code 0)
│
├── 02_causal_self_attention_and_transformer_blocks/
│   ├── lecture.html / lecture.md                         # Causal masking (tril), scaled dot-product attention, Pre-LN, residual stream, GELU MLP
│   ├── exercises_and_solutions.html / .md                # Manual causal softmax derivation, block parameter counting, Pre-LN gradient highway proof
│   └── experiment.py                                     # Pure Python CausalAttention, MLP, TransformerBlock, future non-leakage test (Exit Code 0)
│
├── 03_gpt_model_architecture_and_weight_tying/
│   ├── lecture.html / lecture.md                         # Full model pipeline (wte, wpe, blocks, ln_f, head), weight tying (Press & Wolf 2017), GPT-2 family
│   ├── exercises_and_solutions.html / .md                # GPT-2 Small manual calculation, VRAM analysis for untied heads, Log-Sum-Exp derivation
│   └── experiment.py                                     # Pure Python GPT assembly with weight tying, forward pass, CE loss, closed-form count (Exit Code 0)
│
└── 04_autoregressive_generation_and_sampling_strategies/
    ├── lecture.html / lecture.md                         # Autoregressive generation loop, greedy, temperature, top-k, top-p nucleus, KV-cache
    ├── exercises_and_solutions.html / .md                # Manual top-k/top-p filtering, KV-cache VRAM formula, asymptotic 2/3 M speedup proof
    └── experiment.py                                     # Pure Python generation suite, greedy/temp/top-k/top-p, exact KV-cache match & benchmark (Exit Code 0)
```

---

## 🔑 Core Mathematical Pillars Mastered in Phase 17

1. **Greedy Subword Pair Merging (BPE):**
   $$\boxed{(u^*, v^*) = \arg\max_{(u, v)} \sum_{c \in \mathcal{C}} \sum_{j=1}^{|c|-1} \mathbb{I}[c_j = u \land c_{j+1} = v]}$$
   Iteratively contracts sequence length while maintaining full byte-level coverage over all unicode characters.

2. **Causal Scaled Dot-Product Attention:**
   $$\boxed{A = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} + M \right), \quad M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}}$$
   Ensures that every token's attention distribution is strictly non-anticipative, preventing information leakage from future positions.

3. **Pre-LayerNorm Transformer Block & Residual Recurrence:**
   $$\boxed{\mathbf{x}^{(1)} = \mathbf{x} + \text{Attention}(\text{LN}_1(\mathbf{x})), \qquad \mathbf{x}_{\text{out}} = \mathbf{x}^{(1)} + \text{MLP}(\text{LN}_2(\mathbf{x}^{(1)}))}$$
   Guarantees an unhindered identity highway $\mathbf{x}_L = \mathbf{x}_0 + \sum_{l=0}^{L-1} F_l(\text{LN}(\mathbf{x}_l))$ through arbitrary network depths.

4. **Weight Tying & Logit Prediction:**
   $$\boxed{\mathbf{z}_t = \mathbf{h}_t^{(\text{final})} \cdot W_{\text{token}}^T \in \mathbb{R}^V}$$
   Reuses the input token embedding lookup matrix as the output classification projection, saving $V \times d_{\text{model}}$ parameters and enforcing geometric alignment.

5. **Closed-Form Parameter Formula for GPT Models:**
   $$\boxed{\text{Params} = (V + T_{\max})d + L(12 d^2 + 13 d) + 2d + \mathbb{I}[\text{untied}] \cdot V d}$$
   Accurately predicts parameter counts across the entire GPT-2 model hierarchy (124M, 355M, 774M, 1.5B).

6. **KV-Cache Complexity Reduction:**
   $$\boxed{\text{Naive: } \mathcal{O}\left(\sum_{t=1}^M t^2\right) = \mathcal{O}(M^3) \quad \Longrightarrow \quad \text{KV-Cached: } \mathcal{O}\left(\sum_{t=1}^M t\right) = \mathcal{O}(M^2)}$$
   Reduces per-token computational latency to a constant number of new vector projections per generation step.

---

## 🧪 Quickstart: Running the Labs

All lab scripts run in standard Python 3 with zero external dependencies:

```bash
# Lab 17.1: Byte-Pair Encoding (BPE) Tokenizer training, merge rules & lossless round-trip
python3 phase_17_build_a_gpt_from_scratch/01_byte_pair_encoding_bpe_tokenizer/experiment.py

# Lab 17.2: Causal Self-Attention, Pre-LN Transformer Blocks & future non-leakage test
python3 phase_17_build_a_gpt_from_scratch/02_causal_self_attention_and_transformer_blocks/experiment.py

# Lab 17.3: Full GPT Model Assembly, Weight Tying & exact parameter verification (GPT-2 Small to XL)
python3 phase_17_build_a_gpt_from_scratch/03_gpt_model_architecture_and_weight_tying/experiment.py

# Lab 17.4: Autoregressive Generation, Decoding Policies (Greedy, Temp, Top-k, Top-p) & KV-Cache
python3 phase_17_build_a_gpt_from_scratch/04_autoregressive_generation_and_sampling_strategies/experiment.py
```
