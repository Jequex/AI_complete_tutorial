# Topic 02: Tool Use & Function Calling

## 1. Introduction & 12-Year-Old Intuition

Imagine the world's most brilliant chef locked inside a windowless library. The chef knows every recipe in human history, from French soufflés to authentic ramen. 

However, if someone enters and asks: *"Can you bake me a chocolate cake?"*, the chef has a problem. The chef has no oven, no flour, no eggs, no mixer, and no timer. In the library, all the chef can do is write a vivid essay describing what the cake would taste like.

Now, imagine giving that chef access to a modern automated kitchen equipped with appliances:
- A digital mixing bowl that takes instructions: `mix(speed=3, duration_sec=60)`
- A smart oven: `bake(temperature_c=180, time_min=35)`
- A digital pantry: `dispense(ingredient="flour", amount_grams=250)`

Each machine comes with a strict manual (a **Schema**). If the chef shouts at the oven: *"Bake at temperature 'very hot'!"*, the oven doesn't understand. A red screen flashes: `Error: 'temperature_c' must be an integer, got string 'very hot'`. 

The chef immediately looks at the manual, corrects the command to `bake(temperature_c=200, time_min=30)`, and presses Start. The cake bakes successfully.

In AI engineering:
- The **Chef** is the Large Language Model.
- The **Appliances** are external software tools (compilers, SQL databases, calculators, search engines, web APIs).
- The **Manuals** are JSON Schemas.
- The **Red Screen & Self-Correction** is the Tool Error Feedback Loop.

Without tools, an LLM is a closed-world text generator. With tools, an LLM becomes an **agent** capable of inspecting databases, executing code, sending emails, and altering the external world.

---

## 2. Mathematical Symbol Decoder

| Symbol | Name | Formal Type | Intuition / Meaning |
| :--- | :--- | :--- | :--- |
| $\mathcal{T}_{\text{tools}}$ | Tool Library / Registry | Set $\{f_1, f_2, \dots, f_M\}$ | The collection of executable functions exposed to the agent. |
| $\Sigma_i$ | Tool Schema | JSON Schema $\in \mathcal{J}$ | The formal specification of parameter names, data types, and constraints for tool $f_i$. |
| $\mathcal{G}_{\text{CFG}}$ | Context-Free Grammar | 4-Tuple $(V_N, V_T, P, S)$ | Formal grammar specifying valid structural syntax (e.g. JSON grammar). |
| $q \in Q$ | FSM State | Finite Automaton State | Current state of the parser tracking valid next characters/tokens during generation. |
| $\mathcal{M}(q)$ | Allowed Token Mask | Subset of Vocabulary $\mathcal{V}$ | The set of tokens whose emission leaves the prefix legally completable under grammar $\mathcal{G}$. |
| $z_t \in \mathbb{R}^{|\mathcal{V}|}$ | Logit Vector | Real Vector | Raw unnormalized output predictions of the LLM for token position $t$. |
| $\tilde{z}_t$ | Masked Logits | Real Vector | Logits filtered by grammar mask: $\tilde{z}_{t, v} = z_{t, v}$ if $v \in \mathcal{M}(q)$, else $-\infty$. |
| $e_t$ | Error Observation | Diagnostic String / Object | Structured exception message returned by schema validator or sandbox runtime. |

---

## 3. Tool Schema Architecture (JSON Schema Standard)

Tools are presented to LLMs via standardized JSON Schemas (formalized in RFC 8927 and OpenAI/Anthropic/Gemini function calling APIs).

A schema defines:
1. `name`: Unique identifier for function dispatch.
2. `description`: Crucial semantic description used by the LLM's attention mechanism to route queries to the appropriate tool.
3. `parameters`: Object schema containing:
   - `type`: Data type (`string`, `number`, `integer`, `boolean`, `array`, `object`).
   - `properties`: Dictionary mapping parameter names to their types and descriptions.
   - `required`: List of mandatory arguments that cannot be omitted.
   - Constraints: `enum`, `minimum`, `maximum`, `pattern` (regex).

### Example JSON Schema Definition:
```json
{
  "name": "execute_sql_query",
  "description": "Executes a read-only SQL query against the customer database.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "The SELECT query to execute. Must not contain mutation statements."
      },
      "max_rows": {
        "type": "integer",
        "description": "Maximum rows to return.",
        "minimum": 1,
        "maximum": 500
      }
    },
    "required": ["query"]
  }
}
```

---

## 4. Grammar-Constrained Decoding: Guarantees via Finite-State Machines

A pervasive failure mode of naive LLM generation is **hallucinatory syntax corruption**: the model emits missing closing brackets `{"foo": 12`, illegal keys, or invalid types.

To achieve 100% deterministic conformance, production inference engines (e.g., Guidance, Outlines, llama.cpp BNF grammar, SGLang) utilize **Grammar-Constrained Decoding**.

### Mathematical Formulation of Logit Masking
Let $\mathcal{V}$ be the model's token vocabulary. At generation step $t$, the transformer produces logit vector $\mathbf{z}_t \in \mathbb{R}^{|\mathcal{V}|}$.

Let $w_{1:t-1} = (w_1, w_2, \dots, w_{t-1})$ be the sequence of tokens generated so far.
A formal Context-Free Grammar $\mathcal{G}$ or Deterministic Finite Automaton (DFA) $\mathcal{A} = (Q, \Sigma, \delta, q_0, F)$ tracks the parse state $q = \delta^*(q_0, w_{1:t-1})$.

The engine computes the **admissible token set**:
$$\mathcal{M}(q) = \{ v \in \mathcal{V} \mid \exists w \in \mathcal{V}^* \text{ such that } (w_{1:t-1} \cdot v \cdot w) \in \mathcal{L}(\mathcal{G}) \}$$

The output logits are projected via an indicator mask:
$$\tilde{z}_{t, v} = \begin{cases} z_{t, v} & \text{if } v \in \mathcal{M}(q) \\ -\infty & \text{if } v \notin \mathcal{M}(q) \end{cases}$$

Applying softmax yields the constrained probability distribution:
$$\mathbb{P}(w_t = v \mid w_{1:t-1}) = \frac{\exp(\tilde{z}_{t, v})}{\sum_{v' \in \mathcal{M}(q)} \exp(z_{t, v'})}$$

Because all invalid tokens have probability identically zero ($\exp(-\infty) = 0$), the LLM is **mathematically incapable of generating malformed JSON**.

```
  [Unconstrained Logits z_t] ──▶ [Grammar Mask M(q)] ──▶ [Masked Logits z_tilde]
                                          ▲                       │
                                          │                       ▼
                                 [JSON Schema DFA] ◀────── [Softmax Sample]
```

---

## 5. Tool Dispatch & Sandboxed Execution Architecture

Once a structured tool call $a_t = (\text{tool\_name}, \boldsymbol{\theta})$ is parsed, the agent architecture dispatches it through a secure runtime:

```
  LLM Tool Call ──▶ [Schema Validator] ──▶ (Passed) ──▶ [Sandboxed Dispatcher]
                            │                                     │
                        (Failed)                              (Execution)
                            │                                     │
                            ▼                                     ▼
                  SchemaValidationError                  Stdout / Exception
                            │                                     │
                            └───────────────┬─────────────────────┘
                                            ▼
                              [Error Feedback Injection]
                                            │
                                            ▼
                               [LLM Context Window: Turn t+1]
```

### Key Sandboxing Principles:
1. **Principle of Least Privilege**: Grant only read access where mutation is not requested.
2. **Process Isolation & Resource Quotas**: Execute shell scripts or compilers in disposable containers (e.g. Docker, gVisor, or restricted subprocesses) with CPU timeouts ($t_{\text{limit}} \le 30\text{s}$) and memory caps ($M_{\text{limit}} \le 512\text{MB}$).
3. **AST Safety Auditing**: Static analysis inspecting code trees to reject destructive calls (`os.system("rm -rf /")`, unauthorized sockets).

---

## 6. Self-Correction Feedback Loops: Recovering from Runtime Errors

When a tool call fails (due to schema validation or runtime exceptions like `FileNotFoundError`, `DatabaseLockedError`, `ZeroDivisionError`), naive agents crash or halt.

High-performance autonomous agents implement **Self-Correction Feedback Loops**:
1. Intercept the exception object $E$.
2. Format $E$ into a structured, human-readable error observation:
   $$\mathcal{O}_{\text{error}} = \text{fmt}("ToolExecutionError: " + \text{str}(E) + "\nHint: Expected integer >= 0")$$
3. Append $(\text{Action}, \mathcal{O}_{\text{error}})$ directly to the conversation trajectory:
   $$\tau_{t+1} = [\tau_t, a_t, \mathcal{O}_{\text{error}}]$$
4. The agent re-conditions on $\tau_{t+1}$ and generates a repaired parameter assignment $a_{t+1}$.

### Convergence of Self-Correction
Let $p_e \in (0, 1)$ be the single-turn error probability. With informative error feedback, the probability of failure decays exponentially with retry horizon $K$:
$$\mathbb{P}(\text{Failure after } K \text{ turns}) \le \prod_{k=1}^K p_e^{(k)} \ll (p_e)^K \to 0$$

---

## 7. Summary & Next Steps

In this topic, we explored:
- How JSON Schemas formalize tool parameters and types.
- Grammar-constrained decoding via FSM logit masking $\tilde{z}_t = z_t + \log \mathbb{I}(v \in \mathcal{M}(q))$.
- Sandboxed tool dispatch and isolation.
- Structured self-correction loops recovering from validation and runtime errors.

In **Topic 03: Planning Strategies — ReAct, CoT, and Tree-of-Thoughts**, we will scale up from single tool calls to complex multi-step reasoning, exploring how agents deliberate, branch, evaluate intermediate states, and backtrack.
