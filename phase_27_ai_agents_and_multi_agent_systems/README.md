# Phase 27: AI Agents & Multi-Agent Systems

Welcome to **Phase 27: AI Agents & Multi-Agent Systems** of the Complete Artificial Intelligence & Machine Learning Engineering Curriculum.

While prior phases established foundational neural representations, large language model pretraining, and reinforcement learning, Phase 27 shifts the paradigm from passive, single-turn text completion to **active, autonomous, closed-loop cybernetic agency**. Here, we bridge the chasm between autoregressive language models and goal-directed intelligent agents capable of perceiving partially observable environments, manipulating external tools via grammar-constrained schemas, deliberating over branching search trees, managing multi-tiered cognitive memory architectures, and collaborating within multi-agent consensus societies.

---

## Master Architectural Roadmap

```
                          PHASE 27: AI AGENTS & MULTI-AGENT SYSTEMS
                                              │
              ┌───────────────────────────────┴───────────────────────────────┐
              ▼                                                               ▼
   [01] Perception-Action Loops                                    [02] Tool Use & Function Calling
• Model vs Agent Paradigm                                       • JSON Schema Formalization (RFC 8927)
• Cybernetic Sense-Plan-Act Feedback                            • Grammar-Constrained Decoding (CFG/DFA)
• POMDP 6-Tuple ⟨S, A, T, R, Ω, O, γ⟩                          • Logit Masking: z_tilde = z + log I(v in M(q))
• Bayesian Belief State b'(s') Derivation                       • Sandboxed Execution & Privilege Separation
• LLM Context as Empirical Belief State                         • Multi-Turn Self-Correction Error Feedback
              │                                                               │
              └───────────────────────────────┬───────────────────────────────┘
                                              │
                                              ▼
                             [03] Strategic Planning & Search
                          • Chain-of-Thought (Linear Factorization)
                          • ReAct: Interleaved Thought-Action-Observation
                          • Tree-of-Thoughts (ToT): Generator G(s) & Evaluator V(s)
                          • Beam Pruning & DFS with State Backtracking
                          • Reflexion: Episodic Verbal Reinforcement Learning
                                              │
              ┌───────────────────────────────┴───────────────────────────────┐
              ▼                                                               ▼
   [04] Cognitive Memory Systems                                   [05] Multi-Agent Societies
• Working Memory (Context Window Buffer)                        • Coordination Topologies (Hub, Hierarchy, Blackboard)
• Episodic Memory (Interaction History Log)                     • FIPA-ACL Structured Communication Standards
• Semantic Memory (Vector Store Knowledge)                      • Speech Act Performatives (PROPOSE, CRITIQUE, INFORM)
• Procedural Memory (System Prompts & Schemas)                  • Multi-Agent Debate & Conflict Resolution
• Tri-Factor Scoring: S = w_r R + w_i I + w_s Sim               • Condorcet's Jury Theorem & Consensus Ratification
• Exponential Recency Decay & Generative Reflection             • Verification of Emergent Collective Intelligence
```

---

## Mathematical Pillars & Core Derivations

### 1. The POMDP Formulation & Bayesian Belief Update
In real-world environments, the underlying state $S_t \in \mathcal{S}$ is partially observable. The agent receives noisy observations $O_t \in \Omega$ governed by observation emission probability $\mathcal{O}(o \mid s', a)$.

The **Belief State** $b_t(s) \triangleq \mathbb{P}(S_t = s \mid h_t)$ is a sufficient statistic for history $h_t = (a_0, o_1, \dots, a_{t-1}, o_t)$. The recursive Bayesian filter update rule is:
$$\boxed{b_{t+1}(s') = \frac{\mathcal{O}(o_{t+1} \mid s', a_t) \sum_{s \in \mathcal{S}} \mathcal{T}(s' \mid s, a_t) b_t(s)}{\sum_{s'' \in \mathcal{S}} \mathcal{O}(o_{t+1} \mid s'', a_t) \sum_{s \in \mathcal{S}} \mathcal{T}(s'' \mid s, a_t) b_t(s)}}}$$

In modern LLM agents, the autoregressive context buffer $\tau_t = [\sigma_0, o_1, a_1, \dots, o_t]$ serves as an empirical neural belief state $\hat{b}_t \approx \phi(\tau_t)$.

### 2. Grammar-Constrained Decoding via Finite-State Automata
To ensure 100% adherence to tool JSON schemas without syntactical hallucinations, inference engines construct a Deterministic Finite Automaton (DFA) $\mathcal{A} = (Q, \Sigma, \delta, q_0, F)$ tracking the valid prefix. At step $t$, the logit vector $\mathbf{z}_t \in \mathbb{R}^{|\mathcal{V}|}$ is projected via vocabulary mask $\mathcal{M}(q)$:
$$\tilde{z}_{t, v} = \begin{cases} z_{t, v} & \text{if } v \in \mathcal{M}(q) \\ -\infty & \text{if } v \notin \mathcal{M}(q) \end{cases}$$
Sampling from $\mathbb{P}(w_t = v) = \frac{\exp(\tilde{z}_{t, v})}{\sum_{v' \in \mathcal{M}(q)} \exp(z_{t, v'})}$ mathematically guarantees valid JSON and schema parameter conformance.

### 3. Tree-of-Thoughts (ToT) State Valuation & Backtracking
Linear Chain-of-Thought (CoT) suffers from exponential error accumulation: $\mathbb{P}(\text{Success}) = \prod_{t=1}^T (1 - \epsilon_t) \xrightarrow{T \to \infty} 0$.

Tree-of-Thoughts overcomes this by formulating deliberation as search over tree $\mathcal{S}_{\text{tree}}$:
- **Thought Generator**: $G(p_\theta, s, k)$ samples $k$ candidate next reasoning thoughts.
- **State Evaluator**: $V(p_\theta, s) \in [0, 1]$ heuristics rate branch feasibility.
- **Search & Backtracking**: Depth-First Search prunes branches when $V(s) = \text{"impossible"}$ and unwinds the call stack to explore alternate frontiers, elevating multi-step problem solving success from $<15\%$ to $>85\%$.

### 4. Tri-Factor Cognitive Memory Scoring
To maintain long-term memory across finite token windows, agents score candidate memories $m$ against query $q$ at clock time $t$:
$$\boxed{S(m, q, t) = w_r \cdot R(m, t) + w_i \cdot I(m) + w_s \cdot \text{Sim}(m, q)}$$
where:
- **Recency**: $R(m, t) = \exp(-\lambda \cdot (t - t_{\text{accessed}}))$ with half-life $T_{1/2} = \frac{\ln 2}{\lambda}$.
- **Importance**: $I(m) \in [0, 1]$ computed via cognitive criticality rating.
- **Relevance**: $\text{Sim}(m, q) = \frac{\mathbf{e}_m \cdot \mathbf{e}_q}{\|\mathbf{e}_m\| \|\mathbf{e}_q\|}$ (cosine similarity).

Generative Reflection periodically consolidates low-level episodic event logs into high-level semantic principles.

### 5. Multi-Agent Consensus & Condorcet's Jury Theorem
When $N$ independent specialized agents (Proposer, Critic, Judge) debate an objective decision under majority voting, and each agent has accuracy $p > 0.5$, the ensemble accuracy $P_N$ is:
$$\boxed{P_N = \sum_{k=\frac{N+1}{2}}^N \binom{N}{k} p^k (1 - p)^{N - k}}$$
As $N \to \infty$, $\lim_{N \to \infty} P_N = 1.0$. Adversarial critiques eliminate silent single-agent reasoning failures before code execution or decision ratification.

---

## Phase Navigation & Directory Index

| Topic | Directory | Core Concepts | Resources |
| :--- | :--- | :--- | :--- |
| **01** | `01_agent_architectures_and_perception_action_loops/` | Model vs Agent, Cybernetic Sense-Plan-Act, POMDP 6-tuple, Bayesian belief updates, Shannon entropy. | [Lecture (HTML)](01_agent_architectures_and_perception_action_loops/lecture.html) &bull; [Exercises](01_agent_architectures_and_perception_action_loops/exercises_and_solutions.html) &bull; [Experiment](01_agent_architectures_and_perception_action_loops/experiment.py) |
| **02** | `02_tool_use_and_function_calling/` | JSON Schema definitions, grammar-constrained decoding, FSM logit masking, sandboxed dispatch, self-correction. | [Lecture (HTML)](02_tool_use_and_function_calling/lecture.html) &bull; [Exercises](02_tool_use_and_function_calling/exercises_and_solutions.html) &bull; [Experiment](02_tool_use_and_function_calling/experiment.py) |
| **03** | `03_planning_strategies_react_cot_tot/` | Chain-of-Thought, ReAct closed loop, Tree-of-Thoughts (ToT) DFS search & backtracking, Reflexion episodic buffers. | [Lecture (HTML)](03_planning_strategies_react_cot_tot/lecture.html) &bull; [Exercises](03_planning_strategies_react_cot_tot/exercises_and_solutions.html) &bull; [Experiment](03_planning_strategies_react_cot_tot/experiment.py) |
| **04** | `04_agent_memory_systems/` | Cognitive tiers (working, episodic, semantic, procedural), tri-factor scoring, exponential decay, eviction, reflection. | [Lecture (HTML)](04_agent_memory_systems/lecture.html) &bull; [Exercises](04_agent_memory_systems/exercises_and_solutions.html) &bull; [Experiment](04_agent_memory_systems/experiment.py) |
| **05** | `05_multi_agent_orchestration_and_protocols/` | Coordination topologies, FIPA-ACL message envelopes, debate convergence, Condorcet jury theorem, consensus. | [Lecture (HTML)](05_multi_agent_orchestration_and_protocols/lecture.html) &bull; [Exercises](05_multi_agent_orchestration_and_protocols/exercises_and_solutions.html) &bull; [Experiment](05_multi_agent_orchestration_and_protocols/experiment.py) |

---

## Running the Zero-Dependency Pure Python Experiments

Every experiment in Phase 27 is built from first principles using **pure Python 3 standard library** with zero external dependencies (no PyTorch, no LangChain, no NumPy, no external network APIs):

```bash
# Topic 01: Sense-Plan-Act Loop & POMDP Bayesian Belief Tracking
python3 phase_27_ai_agents_and_multi_agent_systems/01_agent_architectures_and_perception_action_loops/experiment.py

# Topic 02: JSON Schema Validation & Self-Correcting Tool Feedback
python3 phase_27_ai_agents_and_multi_agent_systems/02_tool_use_and_function_calling/experiment.py

# Topic 03: ReAct & Tree-of-Thoughts (ToT) Constraint Solver with Backtracking
python3 phase_27_ai_agents_and_multi_agent_systems/03_planning_strategies_react_cot_tot/experiment.py

# Topic 04: Multi-Tiered Cognitive Memory Manager with Reflection & Eviction
python3 phase_27_ai_agents_and_multi_agent_systems/04_agent_memory_systems/experiment.py

# Topic 05: Multi-Agent FIPA-ACL Protocol & Consensus Debate Simulation
python3 phase_27_ai_agents_and_multi_agent_systems/05_multi_agent_orchestration_and_protocols/experiment.py
```
