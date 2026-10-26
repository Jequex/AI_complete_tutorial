# Topic 03: Planning Strategies — ReAct, Chain-of-Thought & Tree-of-Thoughts

## 1. Introduction & 12-Year-Old Intuition

Imagine you are trying to solve a massive hedge maze in an amusement park.

There are three ways you could try to escape:
1. **The Blind Sprinter (Standard Chain-of-Thought)**: You close your eyes, imagine a sequence of 50 turns in your head (*"Left, Right, Left, Straight, Right..."*), and then run as fast as you can with your eyes shut. What happens? If your very second turn was slightly wrong, you slam face-first into a hedge on turn 3, and all the remaining 47 turns are completely useless.
2. **The Observant Hiker (ReAct: Reason + Act)**: You take three steps, open your eyes, check the signpost, and read your compass: *"Aha! The sign says 'Exit to the North', but there is a mud puddle in front of me. Therefore, I will step Right around the puddle."* You interleave **Thinking** with **Acting** and **Observing**.
3. **The Chess Grandmaster (Tree-of-Thoughts - ToT)**: You reach a major crossroad with three branching paths. Instead of committing blindly, you pause. You send a little drone down Path A (it hits a dead end), you look down Path B (it leads to a locked iron gate), and down Path C (you see sunlight and open fields). You mentally prune Paths A and B, choose Path C, and if Path C later turns out to be blocked, you calmly **backtrack** to your last known fork and try another branch.
4. **The Wise Journaler (Reflexion)**: Before you leave the maze, you write in your notebook: *"Note for next time: The blue tunnel has hidden thorns; always take the stone archway instead."* When you attempt the maze again tomorrow, you read your past notes and never repeat the same mistake.

In AI Agent systems, these represent the four fundamental tiers of autonomous planning.

---

## 2. Mathematical Symbol Decoder

| Symbol | Name | Formal Type | Intuitive Interpretation |
| :--- | :--- | :--- | :--- |
| $x \in \mathcal{X}$ | Task Input / Goal | Token Sequence | The problem statement, query, or objective prompt. |
| $y_t$ | Intermediate Thought | Token Sequence | A discrete reasoning step or cognitive deliberation token. |
| $a_t \in \mathcal{A}$ | Environment Action | Tool Call / API Execution | An external action executed in the environment. |
| $o_t \in \Omega$ | Observation | Perceptual Payload / Text | The environmental result returned after action $a_t$. |
| $\tau_t$ | Execution Trajectory | Sequence $(t_1, a_1, o_1, \dots)$ | The interleaved history of thoughts, actions, and observations. |
| $s \in \mathcal{S}_{\text{tree}}$ | ToT State | State Node | An intermediate frontier state in the reasoning tree. |
| $G(p_\theta, s, k)$ | Thought Generator | Function: $\mathcal{S} \to 2^{\mathcal{S}}$ | Samples $k$ candidate next reasoning steps from the LLM. |
| $V(p_\theta, s)$ | State Evaluator / Heuristic | Function: $\mathcal{S} \to [0, 1]$ or Enum | Evaluates the feasibility/promisingness of a reasoning branch. |
| $b \in \mathbb{N}$ | Beam Width | Positive Integer | The number of top-rated candidate thought nodes retained per depth level. |
| $\mathcal{M}_{\text{ref}}$ | Episodic Memory Buffer | List of Verbal Strings | Collection of self-reflective post-mortems from prior attempts. |

---

## 3. The Taxonomy of Agent Planning

```
  1. Standard Chain-of-Thought (Linear)
     Input ──▶ Thought 1 ──▶ Thought 2 ──▶ Thought 3 ──▶ Output

  2. ReAct (Interleaved Closed-Loop)
     Thought 1 ──▶ Action 1 ──▶ Observation 1 ──▶ Thought 2 ──▶ Action 2 ──▶ Output

  3. Tree-of-Thoughts (Branching + Backtracking)
                  ┌──▶ [Thought 1A] (Score: 0.1) ── [PRUNED]
     [Input State]├──▶ [Thought 1B] (Score: 0.8) ──▶ [Thought 2A] (Solved! 1.0)
                  └──▶ [Thought 1C] (Score: 0.4) ── [BACKTRACK]

  4. Reflexion (Episodic Verbal Reinforcement)
     Attempt 1 (Failed) ──▶ Self-Reflection Summary ──▶ Memory Buffer ──▶ Attempt 2 (Success)
```

---

## 4. Formal Analysis: Chain-of-Thought vs. ReAct

### 4.1 Chain-of-Thought (CoT, Wei et al. 2022)
In Chain-of-Thought, reasoning is modeled as an autoregressive linear factorization:
$$\mathbb{P}(y_{1:T} \mid x) = \prod_{t=1}^T \mathbb{P}(y_t \mid x, y_{<t})$$
**The Compounding Error Vulnerability:**
Let $\epsilon_t \in (0, 1)$ be the probability of a reasoning hallucination at step $t$. The probability of an uncorrupted multi-step trajectory decays exponentially:
$$\mathbb{P}(\text{Success}) = \prod_{t=1}^T (1 - \epsilon_t) \le (1 - \epsilon_{\min})^T \xrightarrow{T \to \infty} 0$$
Once a false premise is generated, autoregressive conditioning guarantees that subsequent reasoning steps build upon the error.

### 4.2 ReAct: Synergizing Reasoning and Acting (Yao et al. 2022)
ReAct expands the agent's action space to include internal cognitive acts $\mathcal{T}_{\text{thought}}$:
$$\hat{\mathcal{A}} = \mathcal{A}_{\text{env}} \cup \mathcal{T}_{\text{thought}}$$
At each step $t$:
1. **Thought generation**: $y_t \sim \pi_\theta(\cdot \mid \tau_{t-1})$ decomposes goals, tracks progress, or formulates hypotheses.
2. **Action execution**: $a_t \sim \pi_\theta(\cdot \mid \tau_{t-1}, y_t)$ invokes an external tool.
3. **Observation reception**: $o_t \sim \text{Env}(\cdot \mid a_t)$ grounds the model in ground truth.

Because observations $o_t$ originate from real software compilers, APIs, or databases, they inject **objective external reality** back into the attention context, breaking hallucination cascades.

---

## 5. Tree-of-Thoughts (ToT): Exploration, Evaluation & Backtracking

For complex constraint satisfaction, combinatorial search, and mathematical problem-solving, both CoT and ReAct fall short because they lack **strategic lookahead** and **backtracking**.

Tree-of-Thoughts (Yao et al. 2023) generalizes reasoning into search over a tree:

### The Four Pillars of ToT:
1. **Thought Decomposition**: Partitioning the problem into discrete thought units (e.g., in the Game of 24, a thought is combining two numbers with an arithmetic operator).
2. **Thought Generator $G(p_\theta, s, k)$**: Samples $k$ candidate thoughts:
   $$G(s) = \{ y^{(1)}, y^{(2)}, \dots, y^{(k)} \}$$
3. **State Evaluator $V(p_\theta, s)$**: Computes the value of intermediate state $s$:
   - Categorical Classification: `{"sure", "likely", "impossible"}`
   - Value Scoring: $V(s) \in [0, 1]$ via few-shot self-evaluations.
4. **Search Algorithms**:
   - **Breadth-First Search (BFS) / Beam Search**: Retains the top-$b$ promising states at depth $d$:
     $$S_{d+1} = \operatorname{arg\,top-b}_{s' \in \bigcup_{s \in S_d} G(s)} V(s')$$
   - **Depth-First Search (DFS) with Backtracking**: Explores the most promising branch until $V(s) = \text{"impossible"}$, then unwinds the call stack to the parent frontier.

---

## 6. Reflexion: Verbal Reinforcement Learning

Introduced by Shinn et al. (2023), **Reflexion** gives agents human-like metacognitive reflection without requiring weight fine-tuning or gradient descent.

When an agent trajectory $\tau$ fails an automated test suite or unit test metric $M(\tau) = 0$:
1. A **Self-Reflection Model** generates verbal feedback:
   $$r_i \sim \pi_{\text{reflect}}(\cdot \mid x, \tau_i, \text{Error})$$
   *"I failed because I assumed table 'users' had column 'name' instead of 'first_name'. In the next attempt, I must query the schema first."*
2. The feedback $r_i$ is appended to an episodic memory buffer $\mathcal{M}_{\text{ref}}$.
3. In subsequent attempt $i+1$, the prompt context includes:
   $$\text{Context}_{i+1} = [x, \mathcal{M}_{\text{ref}}]$$
4. Empirical studies demonstrate that episodic verbal self-reflection increases code generation solve rates on HumanEval from $68\%$ to $>91\%$ across 3 trials.

---

## 7. Summary & Next Steps

In this topic, we mastered:
- The four tiers of agent planning: CoT, ReAct, Tree-of-Thoughts, and Reflexion.
- Why linear CoT suffers from exponential error accumulation $\prod (1 - \epsilon_t)$.
- The mechanics of Tree-of-Thoughts state generators $G(s)$, heuristics $V(s)$, and DFS backtracking.
- Verbal reinforcement learning via Reflexion memory buffers.

In **Topic 04: Agent Memory Systems**, we will explore how agents store, consolidate, score, and evict long-term memories across context limits using recency, importance, and semantic relevance.
