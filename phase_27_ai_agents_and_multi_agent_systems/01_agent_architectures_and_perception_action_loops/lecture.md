# Topic 01: Agent Architectures & Perception-Action Loops

## 1. Introduction & 12-Year-Old Intuition

Imagine you are captaining a submarine 2,000 meters beneath the surface of the Arctic Ocean. Through the thick steel hull, you cannot see anything with your eyes. Pitch-black water surrounds you. 

How do you keep the submarine from crashing into jagged underwater mountains?
You have three things:
1. **Sensors (Sense)**: Every few seconds, your sonar emits a high-pitched "ping" and listens for echoes. The dials register water pressure, engine temperature, and magnetic orientation. These readings are incomplete and noisy—a school of fish might look like an iceberg for a split second!
2. **A Nautical Map & Mental Model (Plan)**: You do not just react blindly to a single ping. You have a nautical chart and a track record of past pings. In your head (or on the navigation table), you maintain a **belief** about where the submarine probably is: *"Based on the last three pings and our speed, we are 90% likely in Sector 4, but there is a 10% chance ocean currents pushed us toward Sector 5."*
3. **Propellers & Rudders (Act)**: You order the helm: *"Turn rudder 15 degrees starboard, engines half ahead!"* The submarine cuts through the water, changing its physical position.

A few seconds later, you ping again. The new echo confirms or refutes your belief, and the cycle repeats.

This is the **Cybernetic Sense-Plan-Act Feedback Loop**.

### What Separates an "AI Model" from an "AI Agent"?
A standard Large Language Model (like raw GPT-4 or Gemini) is like a world-class encyclopedia sitting on a shelf. If you open it to page 412 and read a sentence, it predicts the next word. It does not decide anything, it possesses no continuous memory, it cannot touch the world, and it receives no feedback from the consequences of its outputs. It is a **passive predictive engine**.

An **AI Agent** takes that predictive engine, places it in the captain's chair, connects it to sensors (terminal outputs, web search results, sensor readings), connects it to rudders and motors (code execution, database queries, robotic arms), and loops its outputs back into its inputs over time. An agent does not just output text—it executes actions, observes environmental feedback, updates its internal belief of the world, and persists until a goal is achieved.

---

## 2. Mathematical Symbol Decoder

| Symbol | Name | Formal Domain / Type | Intuitive Interpretation |
| :--- | :--- | :--- | :--- |
| $\mathcal{S}$ | Environmental State Space | Set $\{s_1, s_2, \dots\}$ or $\mathbb{R}^d$ | The true, underlying physical configuration of the world (often hidden from the agent). |
| $\mathcal{A}$ | Action Space | Set $\{a_1, a_2, \dots\}$ or $\mathbb{R}^k$ | The set of valid commands or actions the agent can project onto the environment. |
| $\mathcal{T}$ (or $P$) | Transition Function | $\mathcal{S} \times \mathcal{A} \times \mathcal{S} \to [0, 1]$ | Environmental dynamics: $\mathbb{P}(s_{t+1}=s' \mid s_t=s, a_t=a)$, the law governing how states mutate. |
| $\mathcal{R}$ | Reward Function | $\mathcal{S} \times \mathcal{A} \to \mathbb{R}$ | Scalar feedback signal scoring agent progress toward an objective. |
| $\Omega$ | Observation Space | Set $\{o_1, o_2, \dots\}$ or $\mathbb{R}^m$ | The set of signals, tokens, or perceptual readings the agent can directly observe. |
| $\mathcal{O}$ | Observation Function | $\mathcal{S} \times \mathcal{A} \times \Omega \to [0, 1]$ | Sensor emissions: $\mathbb{P}(o_{t+1}=o \mid s_{t+1}=s', a_t=a)$, the noisy lens over true reality. |
| $b(s)$ | Belief State | Probability distribution $\Delta(\mathcal{S})$ | Sufficient statistic: $b_t(s) \triangleq \mathbb{P}(S_t = s \mid h_t)$, the agent's probability distribution over world states. |
| $h_t$ | History / Trajectory | $(a_0, o_1, a_1, o_2, \dots, a_{t-1}, o_t)$ | The complete historical trace of actions taken and sensory observations gathered. |
| $\pi_\theta(a \mid b)$ | Agent Policy | $\Delta(\mathcal{S}) \times \mathcal{A} \to [0, 1]$ | The agent's decision rule mapping its current belief state (or prompt context) to an action. |
| $H(b)$ | Belief Entropy | Scalar $\ge 0$ (in bits / nats) | Information-theoretic measure of epistemic uncertainty: $-\sum_{s} b(s) \log_2 b(s)$. |
| $\gamma$ | Discount Factor | $\gamma \in [0, 1)$ | Temporal horizon parameter weighting immediate versus long-term returns. |

---

## 3. The Cybernetic Loop: Sense-Plan-Act & OODA

Autonomous agency is rooted in **Cybernetics** (Norbert Wiener, 1948) and control theory:
1. **Sense (Perception)**: Transducing raw environmental emissions $\omega \in \Omega$ into internal symbolic representations or token embeddings.
2. **Plan (Cognition & Deliberation)**: Fusing new observations with past episodic memory to update the internal belief state $b_t(s)$, evaluating utility functions, and deciding on the optimal action.
3. **Act (Execution & Effectuation)**: Projecting discrete symbolic action plans into environmental mutations via tool APIs, effectors, or system calls.

```
       ┌────────────────────────────────────────────────────────┐
       │                      ENVIRONMENT                       │
       │                   True State S_t                       │
       └──────────────┬──────────────────────────▲──────────────┘
                      │                          │
           Sensory Emission (O)           Action Effectuation (A)
                      │                          │
       ┌──────────────▼──────────────────────────┴──────────────┐
       │                         AGENT                          │
       │  ┌───────────────┐   ┌───────────────┐   ┌──────────┐  │
       │  │     SENSE     │──▶│     PLAN      │──▶│   ACT    │  │
       │  │ (Observation) │   │ (Belief/State)│   │ (Policy) │  │
       │  └───────────────┘   └───────────────┘   └──────────┘  │
       └────────────────────────────────────────────────────────┘
```

### Ashby's Law of Requisite Variety (1956)
A foundational law of agent systems states:
$$\mathcal{V}(\text{Controller}) \ge \mathcal{V}(\text{Disturbances})$$
*Only variety in the agent's internal policy can destroy variety in the environment.* If an environment can exhibit $K$ distinct operational failures or configurations, an agent architecture lacking at least $K$ distinct sensing and remedial action pathways will inevitably enter unrecoverable failure states.

---

## 4. The Formal POMDP Mathematical Framework

When an agent cannot directly inspect the true environmental state $S_t \in \mathcal{S}$ (which is the case in nearly all real-world software, web, and physical environments), the decision problem is formalized as a **Partially Observable Markov Decision Process (POMDP)**.

A discrete-time POMDP is defined by the 6-tuple:
$$\mathcal{M}_{\text{POMDP}} = \langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \Omega, \mathcal{O}, \gamma \rangle$$

1. $\mathcal{S}$ is the underlying state space.
2. $\mathcal{A}$ is the action space.
3. $\mathcal{T}(s' \mid s, a) = \mathbb{P}(S_{t+1} = s' \mid S_t = s, A_t = a)$ is the Markovian state transition probability.
4. $\mathcal{R}(s, a)$ is the reward function mapping state-action pairs to real values.
5. $\Omega$ is the observation space.
6. $\mathcal{O}(o \mid s', a) = \mathbb{P}(O_{t+1} = o \mid S_{t+1} = s', A_t = a)$ is the observation probability distribution.
7. $\gamma \in [0, 1)$ is the temporal discount factor.

Because the true state $S_t$ is hidden, the agent cannot base its policy $\pi(a \mid s)$ directly on $s$. Instead, the agent tracks its **interaction history**:
$$h_t = (a_0, o_1, a_0, \dots, a_{t-1}, o_t)$$

---

## 5. Mathematical Derivation: Exact Bayesian Belief State Updating

Tracking arbitrary-length history $h_t$ is computationally intractable ($|h_t| \to \infty$ as $t \to \infty$). Fortunately, the **Belief State** $b_t \in \Delta(\mathcal{S})$—where $b_t(s) \triangleq \mathbb{P}(S_t = s \mid h_t)$—is a **sufficient statistic** for the history $h_t$.

Let us rigorously derive the recursive update rule $b_{t+1} = \text{BayesFilter}(b_t, a_t, o_{t+1})$.

### Step 1: Definition of the Posterior Belief
By definition:
$$b_{t+1}(s') = \mathbb{P}(S_{t+1} = s' \mid h_{t+1}) = \mathbb{P}(S_{t+1} = s' \mid h_t, a_t, o_{t+1})$$

### Step 2: Applying Bayes' Rule
Conditioning on the new observation $o_{t+1}$:
$$\mathbb{P}(S_{t+1} = s' \mid h_t, a_t, o_{t+1}) = \frac{\mathbb{P}(o_{t+1} \mid S_{t+1} = s', h_t, a_t) \cdot \mathbb{P}(S_{t+1} = s' \mid h_t, a_t)}{\mathbb{P}(o_{t+1} \mid h_t, a_t)}$$

### Step 3: Conditional Independence of Observations
Under the POMDP assumption, the sensor observation $o_{t+1}$ depends only on the state just entered $S_{t+1} = s'$ and the action taken $a_t$, and is conditionally independent of past history $h_t$:
$$\mathbb{P}(o_{t+1} \mid S_{t+1} = s', h_t, a_t) = \mathcal{O}(o_{t+1} \mid s', a_t)$$

### Step 4: Marginalizing Over Prior States (Prediction Step)
By the law of total probability over all possible prior states $s \in \mathcal{S}$:
$$\mathbb{P}(S_{t+1} = s' \mid h_t, a_t) = \sum_{s \in \mathcal{S}} \mathbb{P}(S_{t+1} = s' \mid S_t = s, h_t, a_t) \cdot \mathbb{P}(S_t = s \mid h_t, a_t)$$

Since transition dynamics depend only on $(S_t = s, a_t)$ and action $a_t$ is independent of $S_t$ given $h_t$:
$$\mathbb{P}(S_{t+1} = s' \mid h_t, a_t) = \sum_{s \in \mathcal{S}} \mathcal{T}(s' \mid s, a_t) \cdot b_t(s)$$

### Step 5: Normalization Factor (Evidence)
The denominator $Z \triangleq \mathbb{P}(o_{t+1} \mid h_t, a_t)$ is obtained by summing the numerator over all candidate destination states $s'' \in \mathcal{S}$:
$$Z = \sum_{s'' \in \mathcal{S}} \mathcal{O}(o_{t+1} \mid s'', a_t) \sum_{s \in \mathcal{S}} \mathcal{T}(s'' \mid s, a_t) b_t(s)$$

### Final Formula
Combining Steps 2–5 gives the exact recursive Bayesian filter:
$$\boxed{b_{t+1}(s') = \frac{\mathcal{O}(o_{t+1} \mid s', a_t) \sum_{s \in \mathcal{S}} \mathcal{T}(s' \mid s, a_t) b_t(s)}{\sum_{s'' \in \mathcal{S}} \mathcal{O}(o_{t+1} \mid s'', a_t) \sum_{s \in \mathcal{S}} \mathcal{T}(s'' \mid s, a_t) b_t(s)}}$$

---

## 6. The LLM Context Window as an Empirical Belief State

In modern LLM-based agent systems, the agent does not maintain a closed-form probability vector across billions of physical world states. Instead:
- The **Observation Space $\Omega$** consists of UTF-8 text tokens (JSON payloads, shell stdout/stderr, HTML DOM trees, tool execution returns).
- The **Action Space $\mathcal{A}$** consists of structured tool invocations (JSON function calls, SQL commands, terminal strings).
- The **Belief State $b_t$** is implicitly parameterized by the LLM's autoregressive prompt context:
  $$\tau_t = [\text{System Prompt}, o_1, a_1, o_2, a_2, \dots, o_t]$$
- When the LLM processes $\tau_t$, its internal Transformer self-attention layers compute latent state representations $\mathbf{h}_L \in \mathbb{R}^{d_{\text{model}}}$, which serve as a rich, non-linear neural approximation of the true belief state:
  $$\hat{b}_t \approx \text{Transformer}(\tau_t)$$

```
  Observation o_t (Text/JSON) ──▶ [Tokenizer] ──▶ Tokens x_i
                                                      │
                                                      ▼
  Prompt History: [ System Prompt | Tool Definitions | o_1 | a_1 | ... | o_t ]
                                                      │
                                                      ▼
                                       Autoregressive Transformer
                                         Self-Attention Engine
                                                      │
                                                      ▼
  Next Action Prediction a_t ◀── [Action Parser] ◀── Generated Tokens y_k
```

---

## 7. Environmental Taxonomies in Agent Engineering

When architecting an AI agent, its design pattern depends fundamentally on the properties of its target environment:

| Environmental Axis | Case A | Case B | Impact on Agent Design |
| :--- | :--- | :--- | :--- |
| **Observability** | **Fully Observable** (Chess, Go) | **Partially Observable** (Web browsing, Software debugging) | Fully observable environments require no belief tracking; partially observable environments require memory buffers, Bayesian filters, or context aggregation. |
| **Determinism** | **Deterministic** (Pure math, unit tests) | **Stochastic** (Network latency, stock markets) | Stochastic environments require retries, fallback plans, and confidence-weighted deliberation. |
| **Dynamics** | **Static** (Querying a fixed database) | **Dynamic** (Real-time auctions, live multiplayer games) | In dynamic environments, the world changes while the agent is "thinking", penalizing slow deliberation. |
| **Discreteness** | **Discrete** (Token strings, chess moves) | **Continuous** (Robotic joint velocities, steering angles) | Discrete actions match LLM token vocabularies directly; continuous actions require discretization or specialized policy heads. |
| **Horizon** | **Episodic** (Single-turn classification) | **Sequential** (Multi-turn refactoring, multi-day research) | Sequential environments require credit assignment, long-term memory, and backtracking mechanisms. |

---

## 8. Summary & Next Steps

In this topic, we established:
- Why an autonomous **agent** is distinguished from a passive **model** by closed-loop feedback and environmental state mutation.
- The cybernetic **Sense-Plan-Act** loop and Ashby's Law of Requisite Variety.
- The formal mathematical 6-tuple of **POMDPs** and the rigorous step-by-step derivation of the **Bayesian Belief State update**.
- How LLM context windows implement empirical belief state tracking.

In **Topic 02: Tool Use and Function Calling**, we will explore how agents break out of pure text generation to interface deterministically with external APIs, compilers, and databases via grammar-constrained decoding and JSON Schema dispatch.
