# Topic 05: Exercises & Solutions — Multi-Agent Orchestration & Protocols

---

### Exercise 1: Quantitative Condorcet Jury Amplification in Multi-Agent Consensus

Consider an ensemble of $N$ independent debating agents voting on whether a complex pull request contains a critical concurrency bug.
Assume each agent has an individual accuracy of $p = 0.70$ (i.e., a $70\%$ chance of correctly identifying bug presence or absence).
Consensus is decided by strict majority rule.

**Question:**
1. Write the general binomial formula for the ensemble success probability $P_N$ for odd $N$.
2. Compute the exact ensemble accuracy for:
   - $N = 1$ (Single-agent baseline)
   - $N = 3$ (Three-agent society: Proposer, Critic, Judge)
   - $N = 5$ (Five-agent specialized panel)
3. How much does increasing from $N = 1$ to $N = 5$ reduce the system's probability of error?

---

#### Solution to Exercise 1

**1. General Binomial Formula:**
For an odd number of agents $N = 2m + 1$, majority requires at least $m + 1 = \frac{N+1}{2}$ correct votes:
$$P_N = \sum_{k=\frac{N+1}{2}}^N \binom{N}{k} p^k (1 - p)^{N - k}$$

**2. Exact Evaluations for $p = 0.70$ ($1 - p = 0.30$):**

- **For $N = 1$:**
  $$P_1 = p = \mathbf{0.7000} \quad (70.0\%)$$

- **For $N = 3$:** Majority requires at least 2 correct votes ($k = 2, 3$):
  $$P_3 = \binom{3}{2} p^2 (1 - p)^1 + \binom{3}{3} p^3 (1 - p)^0$$
  $$\binom{3}{2} = 3, \quad p^2 (1 - p) = (0.49)(0.30) = 0.147$$
  $$\binom{3}{3} = 1, \quad p^3 = 0.343$$
  $$P_3 = 3(0.147) + 1(0.343) = 0.441 + 0.343 = \mathbf{0.7840} \quad (78.4\%)$$

- **For $N = 5$:** Majority requires at least 3 correct votes ($k = 3, 4, 5$):
  $$P_5 = \binom{5}{3} p^3 (1 - p)^2 + \binom{5}{4} p^4 (1 - p)^1 + \binom{5}{5} p^5$$
  $$\binom{5}{3} = 10, \quad p^3 (1 - p)^2 = (0.343)(0.09) = 0.03087 \implies 10(0.03087) = 0.3087$$
  $$\binom{5}{4} = 5, \quad p^4 (1 - p) = (0.2401)(0.30) = 0.07203 \implies 5(0.07203) = 0.36015$$
  $$\binom{5}{5} = 1, \quad p^5 = 0.16807 \implies 1(0.16807) = 0.16807$$
  $$P_5 = 0.3087 + 0.36015 + 0.16807 = \mathbf{0.83692} \quad (83.69\%)$$

**3. Error Reduction:**
- Error for $N = 1$: $E_1 = 1 - 0.70 = 0.3000$ ($30.0\%$).
- Error for $N = 5$: $E_5 = 1 - 0.83692 = 0.16308$ ($16.31\%$).
- Relative error reduction:
  $$\frac{E_1 - E_5}{E_1} = \frac{0.3000 - 0.16308}{0.3000} = \frac{0.13692}{0.3000} \approx \mathbf{45.64\%}$$
Moving from a single agent to a 5-agent debate panel cuts the error rate by nearly **half** ($45.6\%$), without retraining the underlying model.

---

### Exercise 2: Communication Message Complexity in Contract Net Protocol (CNP)

In the classical FIPA Contract Net Protocol (CNP), an orchestrator Manager wants to allocate a subtask to the most capable Worker among $N$ available agents.

The protocol proceeds in 4 phases:
1. **Call for Proposals (CFP)**: Manager broadcasts task description to all $N$ workers.
2. **Propose / Refuse**: Each of the $N$ workers replies either with a `PROPOSE(cost, estimated_time)` or `REFUSE`.
3. **Accept / Reject**: Manager evaluates proposals, sends `ACCEPT_PROPOSAL` to the single winning worker, and `REJECT_PROPOSAL` to the other $N - 1$ workers.
4. **Inform**: The winning worker executes the task and sends `INFORM_RESULT` back to the Manager.

**Question:**
1. Express the total number of network messages exchanged across the entire 4-phase protocol as a function of $N$.
2. Compare this to an uncoordinated peer-to-peer gossip protocol where every agent communicates with every other agent to negotiate task assignment.

---

#### Solution to Exercise 2

**1. Total Message Count for FIPA Contract Net Protocol:**
- Phase 1 (CFP broadcast): $N$ messages (Manager $\to$ each of $N$ workers).
- Phase 2 (Propose / Refuse replies): $N$ messages (each of $N$ workers $\to$ Manager).
- Phase 3 (Awards & Rejections): $N$ messages (Manager sends 1 Accept + $N-1$ Rejections).
- Phase 4 (Result return): $1$ message (Winning worker $\to$ Manager).

Total messages:
$$M_{\text{CNP}}(N) = N + N + N + 1 = \mathbf{3N + 1}$$
The message complexity scales **linearly**: $\mathcal{O}(N)$.

**2. Comparison to Peer-to-Peer All-to-All Gossip:**
In an uncoordinated peer-to-peer network without a designated manager, every agent must negotiate pairwise with all other $N-1$ agents to establish global consensus:
$$M_{\text{P2P}}(N) = 2 \cdot \binom{N}{2} = N(N - 1) = \mathcal{O}(N^2)$$
For $N = 20$ agents:
- CNP requires $3(20) + 1 = \mathbf{61}$ messages.
- P2P requires $20 \times 19 = \mathbf{380}$ messages.
The structured hierarchical protocol reduces message overhead by over $83.9\%$.
