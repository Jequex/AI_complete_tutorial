# Phase 20: Instruction Tuning & Supervised Fine-Tuning (SFT)

---

## 🧭 Executive Overview

Pre-training an autoregressive Large Language Model across trillions of tokens transforms random weights into a statistical mirror of internet text. However, a raw base pre-trained model is an **internet auto-completer**, not an assistant. When asked a direct question, it frequently hallucinates subsequent exam questions, forum headers, or homework prompts rather than providing an answer.

**Phase 20: Instruction Tuning & Supervised Fine-Tuning (SFT)** covers the mathematical, algorithmic, and engineering foundations that bridge the chasm between raw document prediction and aligned, conversational intelligence.

Every single component across this phase is implemented from first principles in pure Python 3 with **zero external dependencies** (no PyTorch, no Hugging Face, no NumPy, no external API keys).

Across four core modules, this phase establishes:
1. **Base Models vs. Instruct Models & Chat Templates:** The HHH Triad (Helpfulness, Honesty, Harmlessness), plaintext prompt formats (Stanford Alpaca) and their prompt-injection vulnerabilities, special control token standards (OpenAI ChatML `<|im_start|>`, `<|im_end|>`, Meta LLaMA 3 `<|start_header_id|>`, `<|eot_id|>`), and generation prompt turn-initiation mechanics.
2. **Supervised Fine-Tuning & Loss Masking:** The mathematical formulation of the SFT objective, the critical necessity of loss masking on prompt tokens (`ignore_index = -100`), analytical proof that unmasked loss forces prompt memorization and degrades capacity, and next-token shift dynamics.
3. **Instruction Datasets & Multi-Turn Conversations:** Algorithmic dataset generation via Self-Instruct (Wang et al., 2022) with ROUGE-L deduplication, Evol-Instruct (WizardLM, Xu et al., 2023) depth and breadth mutations, multi-turn conversation sequence compilation, tool use / function calling modeling (`<tool_call>` supervised, `<tool_response>` environmental output masked), and sequence packing with segment isolation.
4. **Instruction Evaluation & LLM-as-a-Judge:** Ground-truth deterministic benchmarks (MMLU, GSM8K, HumanEval pass@k) vs open-ended conversational suites (MT-Bench, AlpacaEval), the LLM-as-a-Judge framework (Zheng et al., 2023), cognitive judge biases (Position Bias, Verbosity Bias, Self-Enhancement Bias), bidirectional order-swapping mitigation, length-penalized win rates, and Bradley-Terry tournament Elo rating estimation.

---

## 🗺️ Phase Syllabus & Topic Breakdown

```text
phase_20_instruction_tuning_and_sft/
├── index.html                                                    # Phase Portal Hub & Visual Syllabus
├── README.md                                                     # Master Architecture Guide & Derivations
│
├── 01_base_models_vs_instruct_models/
│   ├── lecture.html / lecture.md                                 # Base vs Instruct shift, HHH Triad, Alpaca, ChatML, LLaMA 3 tokens
│   ├── exercises_and_solutions.html / .md                        # Web continuation vs SFT shift, ChatML token tracing, prompt injection proof
│   └── experiment.py                                             # Pure Python ChatTemplate engine, tokenizer with special tokens (Exit Code 0)
│
├── 02_supervised_fine_tuning_and_loss_masking/
│   ├── lecture.html / lecture.md                                 # SFT objective, target = -100 masking, zero-gradient proof, capacity allocation
│   ├── exercises_and_solutions.html / .md                        # Numerical masked cross-entropy, analytical gradient proof, token efficiency ratio
│   └── experiment.py                                             # Pure Python masked loss engine, mini LM training: masked vs unmasked (Exit Code 0)
│
├── 03_instruction_datasets_and_multi_turn_conversations/
│   ├── lecture.html / lecture.md                                 # Self-Instruct, Evol-Instruct, multi-turn dialogues, tool calling lifecycle
│   ├── exercises_and_solutions.html / .md                        # ROUGE-L deduplication trace, multi-turn mask vector, tool response leakage proof
│   └── experiment.py                                             # Multi-turn dialogue compiler, tool call / response masking, packing (Exit Code 0)
│
└── 04_instruction_evaluation_and_llm_as_a_judge/
    ├── lecture.html / lecture.md                                 # MMLU/GSM8K/HumanEval, MT-Bench, LLM judge biases & mitigations, Bradley-Terry
    ├── exercises_and_solutions.html / .md                        # Bidirectional consensus math, BT MLE gradient derivation, length penalty utility
    └── experiment.py                                             # LLM judge simulator, order swapping, verbosity penalty, Elo solver (Exit Code 0)
```

---

## 🔑 Core Mathematical Pillars Mastered in Phase 20

1. **The Masked SFT Objective:**
   $$\boxed{\mathcal{L}_{\text{SFT}}(\theta) = -\frac{1}{|\mathcal{T}_{\text{resp}}|} \sum_{t \in \mathcal{T}_{\text{resp}}} \log P_\theta(w_t \mid w_{<t})}$$
   Guarantees that gradient updates $\nabla_\theta \mathcal{L}$ are driven strictly by the assistant's answer generation rather than memorizing user questions.

2. **Analytical Proof of Zero Gradient under `ignore_index = -100`:**
   $$\boxed{\frac{\partial \ell_t}{\partial z_{t, v}} = \begin{cases} p_{t, v} - \mathbb{I}[v = y_t] & \text{if } y_t \ne -100 \\ 0 & \text{if } y_t = -100 \end{cases} \quad \implies \quad \sum_{t \in \mathcal{T}_{\text{prompt}}} \frac{\partial \ell_t}{\partial W} = \mathbf{0}}$$
   Eliminates gradient interference between prompt tokens and response tokens.

3. **Self-Instruct Deduplication Threshold (ROUGE-L):**
   $$\boxed{\max_{s \in \mathcal{D}} F_{\text{LCS}}(s_{\text{candidate}}, s) < 0.70, \qquad F_{\text{LCS}} = \frac{2 \cdot R_{\text{LCS}} \cdot P_{\text{LCS}}}{R_{\text{LCS}} + P_{\text{LCS}}}}$$
   Prevents synthetic instruction datasets from collapsing into redundant lexical clusters.

4. **Bidirectional Order-Swapping Consensus Decision:**
   $$\boxed{\text{Result}(M_1, M_2) = \begin{cases} M_1 \text{ Wins} & \text{if } \text{Judge}(M_1, M_2) = A \land \text{Judge}(M_2, M_1) = B \\ M_2 \text{ Wins} & \text{if } \text{Judge}(M_1, M_2) = B \land \text{Judge}(M_2, M_1) = A \\ \text{Tie / Bias} & \text{otherwise} \end{cases}}$$
   Neutralizes first-position primacy bias and last-position recency bias.

5. **Bradley-Terry Preference Probability & Elo Rating:**
   $$\boxed{P(i \succ j) = \frac{e^{r_i}}{e^{r_i} + e^{r_j}} = \frac{1}{1 + e^{-(r_i - r_j)}}, \qquad \text{Elo}_i = 1000 + \frac{400}{\ln 10} r_i}$$
   Solves global tournament rankings across arbitrary pairwise evaluation matchups.

---

## 🧪 Verification & Experiment Suite

All four experimental scripts are self-testing, require zero third-party packages, and exit with code 0:

```bash
# Topic 20.1: Chat Templates & Special Token Encoding
python3 phase_20_instruction_tuning_and_sft/01_base_models_vs_instruct_models/experiment.py

# Topic 20.2: Supervised Fine-Tuning & Loss Masking Dynamics
python3 phase_20_instruction_tuning_and_sft/02_supervised_fine_tuning_and_loss_masking/experiment.py

# Topic 20.3: Multi-Turn Dataset Processor & Function Calling Masking
python3 phase_20_instruction_tuning_and_sft/03_instruction_datasets_and_multi_turn_conversations/experiment.py

# Topic 20.4: LLM-as-a-Judge Evaluation & Bias Mitigation Engine
python3 phase_20_instruction_tuning_and_sft/04_instruction_evaluation_and_llm_as_a_judge/experiment.py
```
