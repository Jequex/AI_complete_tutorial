# Phase 19: Fine-Tuning & PEFT / LoRA

---

## 🧭 Executive Overview

Pre-training equips Large Language Models (LLMs) with general-purpose representation manifolds across trillions of tokens. However, adapting these models to specialized downstream domains—such as legal drafting, medical reasoning, SQL generation, and instruction-following—requires fine-tuning.

Historically, model adaptation relied on **Full Fine-Tuning (FFT)**: computing backpropagation gradients across every single weight parameter in the network. For modern foundation models ($7\text{B}$ to $405\text{B}$ parameters), Full Fine-Tuning is crippled by two fatal barriers:
1. **The FFT Memory Wall:** Mixed-precision training with AdamW requires approximately **$16$ bytes of static GPU memory per model parameter** ($16 \times N$), requiring multi-GPU server clusters just to update a modest 7-billion parameter model before allocating a single activation byte.
2. **Catastrophic Forgetting:** Unconstrained gradient descent on specialized task distributions overwrites general reasoning, multilingual capabilities, and safety guardrails due to destructive gradient interference ($\langle \mathbf{g}_A, \mathbf{g}_B \rangle < 0$).

**Phase 19: Fine-Tuning & PEFT / LoRA** deconstructs the mathematical, algorithmic, and systems engineering solutions to these bottlenecks. Built completely from first principles in pure Python 3—with **zero external libraries** (no PyTorch, no HuggingFace PEFT, no bitsandbytes)—this phase provides an uncompromising, production-grade foundation in parameter-efficient adaptation.

Across four comprehensive modules, this phase covers:
1. **Full Fine-Tuning vs. PEFT & Catastrophic Forgetting:** The pretraining vs. fine-tuning paradigm, the 16 bytes/param static memory wall breakdown, and mathematical proofs of catastrophic forgetting via second-order Taylor expansion and gradient conflict dynamics.
2. **LoRA: Low-Rank Adaptation:** The Intrinsic Rank Hypothesis, low-rank factorization ($W = W_0 + \frac{\alpha}{r} B A$), why $B = 0$ initialization is mathematically essential, learning rate stabilization via $\frac{\alpha}{r}$, analytical backprop derivations, and zero-latency inference weight merging.
3. **QLoRA & Quantization Mechanics:** Floating-point vs integer representations, why uniform 4-bit quantization wastes precision on bell-shaped weights, the 16-level NormalFloat4 (NF4) information-theoretic quantile codebook, Double Quantization compressing scales from $0.5$ to $0.127$ bits/param, and Paged Optimizers using CUDA Unified Memory.
4. **PEFT Adapters, Prompt Tuning, and Prefix Tuning:** Houlsby series bottleneck adapters vs. parallel adapters, continuous soft Prompt Tuning and the power of scale law, cross-layer Key-Value Prefix Tuning injection, and a master comparative trade-off matrix.

---

## 🗺️ Phase Syllabus & Topic Breakdown

```text
phase_19_fine_tuning_and_lora/
├── index.html                                                      # Phase Portal Hub & Interactive Syllabus
├── README.md                                                       # Master Architectural Syllabus & Formula Guide
│
├── 01_full_fine_tuning_vs_peft/
│   ├── lecture.html / lecture.md                                   # Brain surgery analogy, 16N bytes wall, Taylor gradient interference
│   ├── exercises_and_solutions.html / .md                          # 14B VRAM footprint, gradient interference, Bayesian EWC derivation
│   └── experiment.py                                               # Pure Python memory wall breakdown & catastrophic forgetting simulation (Exit Code 0)
│
├── 02_lora_low_rank_adaptation/
│   ├── lecture.html / lecture.md                                   # Sticky note analogy, intrinsic rank, B=0 proof, alpha/r scaling, weight merging
│   ├── exercises_and_solutions.html / .md                          # Manual forward pass, LLaMA-3-8B 99.4% param reduction, batch backprop gradients
│   └── experiment.py                                               # Pure Python LoRALinear layer, analytical vs finite-diff grads, weight merging (Exit Code 0)
│
├── 03_qlora_and_quantization_mechanics/
│   ├── lecture.html / lecture.md                                   # High-res photo analogy, uniform vs NF4 quantile math, Double Quantization, Paged Optimizers
│   ├── exercises_and_solutions.html / .md                          # INT8/INT4 hand quantization, NF4 codebook lookup, 70B Double Quantization VRAM budget
│   └── experiment.py                                               # Pure Python INT8/INT4 vs NF4 benchmark, Double Quantization & Paged Optimizers (Exit Code 0)
│
└── 04_peft_adapters_prompt_tuning_and_prefix_tuning/
    ├── lecture.html / lecture.md                                   # Swiss Army knife analogy, Series vs Parallel Adapters, Prompt Tuning, Prefix KV-injection
    ├── exercises_and_solutions.html / .md                          # Prefix attention hand calculation, multi-method parameter budgeting, context length FLOPs tax
    └── experiment.py                                               # Pure Python Prefix-Tuning KV injection, taxonomy benchmark, prefix steering test (Exit Code 0)
```

---

## 🔑 Core Mathematical Equations Mastered in Phase 19

1. **The Full Fine-Tuning Static Memory Wall:**
   $$\boxed{M_{\text{static}} = 2N + 2N + 4N + 4N + 4N = \mathbf{16N \text{ bytes}}}$$
   $$(FP16 \text{ weights}) + (FP16 \text{ grads}) + (FP32 \text{ master}) + (AdamW \; m_t) + (AdamW \; v_t)$$

2. **First- and Second-Order Catastrophic Gradient Interference:**
   $$\boxed{\Delta \mathcal{L}_A \approx -\eta \langle \mathbf{g}_A, \mathbf{g}_B \rangle + \frac{\eta^2}{2} \mathbf{g}_B^T \mathbf{H}_A \mathbf{g}_B}$$
   $$\text{Destructive Interference: } \langle \mathbf{g}_A, \mathbf{g}_B \rangle < 0 \implies -\eta \langle \mathbf{g}_A, \mathbf{g}_B \rangle > 0$$

3. **LoRA Low-Rank Decomposition:**
   $$\boxed{W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} B A}$$
   $$W_0 \in \mathbb{R}^{d \times k} \text{ (Frozen)}, \quad A \in \mathbb{R}^{r \times k} \sim \mathcal{N}(0, 1/r), \quad B \in \mathbb{R}^{d \times r} = \mathbf{0}$$

4. **Zero-Latency Inference Weight Merging:**
   $$\boxed{W_{\text{merged}} = W_0 + \frac{\alpha}{r} B A \in \mathbb{R}^{d \times k}, \quad h = W_{\text{merged}} x}$$

5. **Double Quantization Memory Overhead:**
   $$\boxed{\text{DQ Overhead} = \frac{8\text{ bits}}{B_1} + \frac{32\text{ bits}}{B_1 \times B_2} = \frac{8}{64} + \frac{32}{64 \times 256} = 0.125 + 0.00195 = \mathbf{0.127 \text{ bits/param}}}$$

6. **Prefix-Tuned Attention Concatenation:**
   $$\boxed{K_{\text{full}} = [P_K; \; K_{\text{seq}}], \quad V_{\text{full}} = [P_V; \; V_{\text{seq}}]}$$
   $$\text{Attention}(Q, K_{\text{full}}, V_{\text{full}}) = \operatorname{softmax}\left(\frac{Q K_{\text{full}}^T}{\sqrt{d_k}}\right) V_{\text{full}}$$

---

## ⚡ Quickstart: Running the Pure Python Experiments

Verify all four topics from your terminal using standard Python 3 (no dependencies required):

```bash
# 1. Topic 19.1: FFT Memory Wall Breakdown & Catastrophic Forgetting
python3 phase_19_fine_tuning_and_lora/01_full_fine_tuning_vs_peft/experiment.py

# 2. Topic 19.2: LoRA Layer, Analytical Backprop Gradients & Weight Merging
python3 phase_19_fine_tuning_and_lora/02_lora_low_rank_adaptation/experiment.py

# 3. Topic 19.3: Uniform INT4 vs. NF4 Quantization & Double Quantization
python3 phase_19_fine_tuning_and_lora/03_qlora_and_quantization_mechanics/experiment.py

# 4. Topic 19.4: Prefix Tuning Key-Value Injection & PEFT Parameter Benchmarks
python3 phase_19_fine_tuning_and_lora/04_peft_adapters_prompt_tuning_and_prefix_tuning/experiment.py
```
