# Topic 04: Agent Memory Systems

## 1. Introduction & 12-Year-Old Intuition

Think about how your own brain handles information throughout a busy school day:

1. **Working Memory (Your Desk)**: You sit at your school desk. The desk is small—it can only hold a notebook, a pencil, and an eraser. This is whatever you are thinking about *right this second*. If someone dumps twenty heavy textbooks on your desk, everything falls off.
2. **Episodic Memory (Your Diary)**: When you get home, you remember specific events that happened today: *"At 10:15 AM, the fire alarm rang during chemistry class; at 12:30 PM, Marcus spilled tomato soup on his sneakers."* It is a chronological movie of your life.
3. **Semantic Memory (Your Mental Encyclopedia)**: You don't remember the exact Tuesday in third grade when you learned that Paris is the capital of France, or that touching a hot stovetop burns your finger. That knowledge has been **consolidated** into timeless, abstract facts.
4. **Procedural Memory (Muscle Memory)**: You don't have to consciously remember how to pedal a bicycle or tie your shoelaces; the habit is baked into your brain's circuitry.
5. **Sleep & Consolidation (The Midnight Cleaner)**: While you sleep, your brain doesn't keep every single piece of useless trivia. It discards what color socks the bus driver was wearing (low importance), keeps the chemistry formula needed for Friday's exam (high importance), and synthesizes your day's experiences into wiser habits.

In AI Agent engineering:
- **Working Memory** is the LLM's active Context Window (strictly bounded by token limits).
- **Episodic Memory** is the append-only event stream of past tool calls, interactions, and user messages.
- **Semantic Memory** is a vector database / knowledge graph holding timeless consolidated facts.
- **Procedural Memory** consists of the system prompts and tool schemas defining how the agent behaves.
- **Memory Consolidation & Eviction** is the algorithmic process that prevents the context window from overflowing by summarizing and ranking memories using **Recency**, **Importance**, and **Relevance**.

---

## 2. Mathematical Symbol Decoder

| Symbol | Name | Formal Type | Intuitive Interpretation |
| :--- | :--- | :--- | :--- |
| $\mathcal{M}_{\text{working}}$ | Working Memory | Token Buffer $L \le L_{\max}$ | The active prompt context currently being processed by Transformer self-attention. |
| $\mathcal{M}_{\text{episodic}}$ | Episodic Memory | Event Stream $\{m_1, m_2, \dots\}$ | Ordered chronological log of historical agent-environment interactions. |
| $\mathcal{M}_{\text{semantic}}$ | Semantic Memory | Knowledge Store / Vector DB | Vector indexed collection of timeless principles, summarized insights, and domain knowledge. |
| $\mathcal{M}_{\text{procedural}}$ | Procedural Memory | System Instructions & Schemas | Static rules, few-shot demonstrations, and tool signatures guiding agent behavior. |
| $t \in \mathbb{N}$ | Current Time | Discrete Clock Step | Monotonically increasing clock tracking interaction cycles. |
| $t_m \in \mathbb{N}$ | Memory Timestamp | Discrete Time | The time step at which memory $m$ was created or last retrieved. |
| $\lambda > 0$ | Decay Rate | Positive Real Scalar | Rate of exponential temporal decay for recency scoring. |
| $R(m, t)$ | Recency Score | Real $\in (0, 1]$ | Temporal freshness: $\exp(-\lambda (t - t_m))$. |
| $I(m)$ | Importance Score | Real $\in [0, 1]$ | Inherent operational criticality of memory $m$. |
| $\text{Sim}(m, q)$ | Semantic Relevance | Real $\in [-1, 1]$ | Cosine similarity between query embedding $\mathbf{e}_q$ and memory embedding $\mathbf{e}_m$. |
| $S(m, q, t)$ | Composite Retrieval Score | Real $\in \mathbb{R}$ | Weighted score determining which memories are retrieved into working context. |

---

## 3. Cognitive Architecture of Agent Memory

```
  ┌────────────────────────────────────────────────────────────────────────┐
  │                           AGENT COGNITION                              │
  │                                                                        │
  │  ┌──────────────────────┐              ┌────────────────────────────┐  │
  │  │  PROCEDURAL MEMORY   │              │       WORKING MEMORY       │  │
  │  │  • System Prompt     │─────────────▶│  • Active Prompt Context   │  │
  │  │  • Tool Schemas      │              │  • Capacity: L <= L_max    │  │
  │  └──────────────────────┘              └─────────────┬──────────────┘  │
  │                                                      │                 │
  │                         Eviction & Event Logging     │ Context Injection
  │                                                      ▼                 │
  │  ┌───────────────────────────────────────────────────┴──────────────┐  │
  │  │                  EPISODIC MEMORY (Event Archive)                 │  │
  │  │  • Raw tool calls, user queries, runtime error logs               │  │
  │  │  • Each item: (id, text, created_at, last_accessed, importance)  │  │
  │  └───────────────────────────────┬──────────────────────────────────┘  │
  │                                  │                                     │
  │                     Generative Reflection Loop                         │
  │                                  ▼                                     │
  │  ┌──────────────────────────────────────────────────────────────────┐  │
  │  │                  SEMANTIC MEMORY (Vector Store)                  │  │
  │  │  • Consolidated abstract insights & user preference models       │  │
  │  │  • Indexed by dense vector embeddings e_m                        │  │
  │  └──────────────────────────────────────────────────────────────────┘  │
  └────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Multi-Factor Memory Scoring & Retrieval Dynamics

When an agent receives a user prompt or encounters an environment obstacle, it must retrieve the most pertinent memories from long-term storage ($\mathcal{M}_{\text{episodic}} \cup \mathcal{M}_{\text{semantic}}$) and inject them into its active working context.

Following the Generative Agents framework (Park et al. 2023), each memory candidate $m$ is evaluated against the current query $q$ and time $t$ via a **tri-factor scoring objective**:

$$\boxed{S(m, q, t) = w_r \cdot R(m, t) + w_i \cdot I(m) + w_s \cdot \text{Sim}(m, q)}$$

where the weights are non-negative and sum to one: $w_r + w_i + w_s = 1$.

### 1. Recency Decay Function $R(m, t)$
Memory accessibility decays exponentially as a function of elapsed time since the memory was last retrieved:
$$R(m, t) = \exp\left(-\lambda \cdot (t - t_{\text{last\_accessed}})\right)$$
- If accessed this turn ($t = t_{\text{last\_accessed}}$), $R = 1.0$.
- As $\Delta t \to \infty$, $R \to 0$.

### 2. Intrinsic Importance Function $I(m)$
Not all observations are created equal. A user declaring *"I am allergic to peanuts"* has vastly higher cognitive importance than *"I wore black socks today"*.
Importance is computed by prompting an evaluation model (or applying rule-based heuristics) on a normalized $[0.0, 1.0]$ scale:
$$I(m) = \frac{\text{LLM\_Rate}(m) - 1}{9}, \quad \text{where rating } \in \{1, 2, \dots, 10\}$$

### 3. Semantic Relevance Function $\text{Sim}(m, q)$
Relevance measures topical affinity using the cosine similarity between the dense vector embedding of the query $\mathbf{e}_q$ and memory $\mathbf{e}_m$:
$$\text{Sim}(m, q) = \frac{\mathbf{e}_m \cdot \mathbf{e}_q}{\|\mathbf{e}_m\| \|\mathbf{e}_q\|} = \frac{\sum_{k=1}^d e_{m, k} e_{q, k}}{\sqrt{\sum_{k=1}^d e_{m, k}^2} \sqrt{\sum_{k=1}^d e_{q, k}^2}}$$

---

## 5. Generative Reflection: Synthesizing Episodic Logs into Semantic Insights

Raw episodic memories accumulate rapidly: 100 interaction steps generate thousands of noisy event logs. If an agent only searches raw logs, it will fail to grasp overarching patterns.

To solve this, agents execute **Generative Reflection**:
1. **Trigger Condition**: When the cumulative importance of recent episodic memories exceeds a threshold $\sum I(m_k) \ge \Theta_{\text{reflect}}$, the reflection cycle triggers.
2. **High-Level Question Generation**: The agent reviews the last $N$ episodic memories and prompts itself:
   *"Given only the information above, what are 3 high-level questions we can answer about the user or system?"*
3. **Evidence Retrieval**: For each generated question, the agent retrieves the top relevant episodic traces.
4. **Synthesis & Storage**: The agent synthesizes an abstract principle:
   $$\text{Insight} = \text{LLM}(\text{Question}, \{m_{k_1}, m_{k_2}, \dots\})$$
   *"Insight: The user prioritizes strict unit test coverage and refuses PR merges without passing CI checks."*
5. The insight is stored in **Semantic Memory** with a high intrinsic importance ($I \ge 0.85$).

---

## 6. Eviction Dynamics: Preserving Context Window Invariants

The working memory buffer has a strict hardware / model token limit $L_{\max}$. When total tokens exceed $L_{\max}$, an eviction policy must drop items.

Standard FIFO (First-In, First-Out) sliding windows are disastrous because they evict critical early instructions. Instead, high-performance agents apply **Priority-Score Eviction**:
$$\text{Eviction Priority}(m) = I(m) \cdot R(m, t)$$
The item with the lowest composite score is evicted from the active working context. Because it is permanently preserved in the episodic archive, it can be seamlessly paged back into context via the retrieval operator whenever a future query demands it.

---

## 7. Summary & Next Steps

In this topic, we explored:
- The four cognitive tiers of agent memory: Working, Episodic, Semantic, and Procedural.
- Multi-factor memory scoring combining Recency $R(m, t)$, Importance $I(m)$, and Relevance $\text{Sim}(m, q)$.
- Generative Reflection synthesizing low-level event logs into timeless semantic principles.
- Priority eviction guaranteeing bounded context window size without losing critical facts.

In **Topic 05: Multi-Agent Orchestration and Protocols**, we will transition from single-agent cognition to multi-agent collective intelligence: coordination topologies, FIPA-ACL communication protocols, debate mechanics, and consensus formation.
