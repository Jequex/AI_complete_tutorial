# Topic 03: Exercises & Solutions — Planning Strategies: ReAct, CoT & ToT

---

### Exercise 1: Search Space Complexity: CoT vs. BFS-ToT vs. Beam-ToT

Consider a reasoning task of depth $D$ (requiring $D$ sequential reasoning steps).
At each step, an LLM can generate $k$ distinct plausible thought candidates.

**Question:**
1. How many total states are generated in standard linear Chain-of-Thought (CoT)?
2. How many total states are evaluated in an exhaustive Breadth-First Search (BFS) Tree-of-Thoughts to depth $D$?
3. How many states are evaluated in a Beam Search Tree-of-Thoughts with beam width $b$?
4. If $k = 5, D = 4, b = 3$, calculate the concrete numerical count of state evaluations for all three methods.

---

#### Solution to Exercise 1

**1. Chain-of-Thought (CoT):**
CoT samples exactly 1 branch greedily at each step without branching:
$$N_{\text{CoT}} = D$$

**2. Exhaustive BFS Tree-of-Thoughts:**
At depth 0, there is 1 root state.
At depth 1, there are $k$ states.
At depth $d$, there are $k^d$ states.
The total number of states generated and evaluated is the sum of the geometric series:
$$N_{\text{BFS}} = \sum_{d=1}^D k^d = \frac{k(k^D - 1)}{k - 1}$$

**3. Beam Search Tree-of-Thoughts (Beam width $b$):**
At depth 1: $k$ states are generated and evaluated, and the top $b$ are retained.
At each subsequent depth $d \in \{2, \dots, D\}$:
Each of the $b$ survivors generates $k$ candidates, resulting in $b \cdot k$ evaluations.
The total evaluations across all $D$ steps:
$$N_{\text{Beam}} = k + (D - 1) \cdot b \cdot k$$

**4. Numerical Evaluation for $k = 5, D = 4, b = 3$:**
- **CoT**: $N_{\text{CoT}} = 4$ evaluations.
- **Exhaustive BFS**:
  $$N_{\text{BFS}} = 5^1 + 5^2 + 5^3 + 5^4 = 5 + 25 + 125 + 625 = \mathbf{780} \text{ evaluations.}$$
- **Beam Search ($b=3$)**:
  $$N_{\text{Beam}} = 5 + (4 - 1) \cdot (3) \cdot (5) = 5 + 3 \cdot 15 = 5 + 45 = \mathbf{50} \text{ evaluations.}$$

Beam Search reduces the search space by over $93.5\%$ relative to exhaustive BFS ($50$ vs $780$) while retaining lookahead and multi-hypothesis tracking.

---

### Exercise 2: Mathematical Proof of Error Recovery in DFS Backtracking vs. Linear CoT

Suppose a planning problem requires $T$ consecutive correct decisions.
At any step, a greedy linear decision has independent failure probability $\epsilon \in (0, 1)$.
In a Tree-of-Thoughts Depth-First Search with backtracking, the agent can explore up to $B$ alternate candidate branches at any decision node if an evaluation heuristic flags a branch as a dead end. Assume the evaluation heuristic has perfect precision (flags dead ends accurately).

**Question:**
1. Derive the probability that linear Chain-of-Thought successfully solves the problem.
2. Derive the probability that the DFS Tree-of-Thoughts solves the problem.
3. For $T = 6, \epsilon = 0.30$, compare the success probability of CoT versus ToT with $B = 3$ alternate branches.

---

#### Solution to Exercise 2

**1. Linear CoT Success Probability:**
Because CoT is linear without backtracking, every single step $1, \dots, T$ must succeed on the first attempt:
$$\mathbb{P}(\text{Success}_{\text{CoT}}) = (1 - \epsilon)^T$$

**2. DFS Backtracking Success Probability:**
At any step $t$, the step fails if and only if **all $B$ candidate branches fail**.
The probability that all $B$ branches fail at step $t$ is:
$$\mathbb{P}(\text{Step } t \text{ fails}) = \epsilon^B$$
Therefore, the probability that step $t$ successfully finds an admissible branch is:
$$\mathbb{P}(\text{Step } t \text{ succeeds}) = 1 - \epsilon^B$$
For the full sequence of $T$ independent decision stages:
$$\mathbb{P}(\text{Success}_{\text{ToT}}) = (1 - \epsilon^B)^T$$

**3. Concrete Calculation for $T = 6, \epsilon = 0.30, B = 3$:**
- **Chain-of-Thought:**
  $$\mathbb{P}(\text{Success}_{\text{CoT}}) = (1 - 0.30)^6 = (0.70)^6 \approx \mathbf{0.1176} \quad (11.76\%)$$
- **Tree-of-Thoughts ($B = 3$):**
  $$\epsilon^B = 0.30^3 = 0.027$$
  $$1 - \epsilon^B = 1 - 0.027 = 0.973$$
  $$\mathbb{P}(\text{Success}_{\text{ToT}}) = (0.973)^6 \approx \mathbf{0.8488} \quad (84.88\%)$$

Backtracking with 3 candidate thoughts increases the overall mission success rate from **$11.8\%$ to $84.9\%$**, demonstrating why search and backtracking are foundational for complex reasoning.
