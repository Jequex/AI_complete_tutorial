# Topic 05: Building MCP Servers & Multi-Server Clients

---

## 1. The 12-Year-Old Intuitive Analogy: The Grand Symphony Conductor

Imagine you are the conductor of an international orchestra. 

On your left sits the violinist section (**The Database Resource Server**)—they hold the sheet music, history books, and archived compositions. In the center sit the percussionists and brass players (**The Tool Server**)—they have the loud drums and cymbals that make physical sound reverberate through the hall. On your right sit the flutists and singers (**The Prompt Server**).

If each musician tried to play independently whenever they felt like it, the concert hall would be chaotic noise! 

The **Host Orchestrator** is the conductor:
1. Before the concert starts, the conductor greets each section, checks what instruments they brought, and confirms they are tuned to the same pitch (**Capability Aggregation**).
2. During the performance, when a quiet melody is needed, the conductor points at the violinists to turn the page (`resources/read`).
3. When dramatic impact is needed, the conductor signals the percussionist to strike the bass drum (`tools/call`).
4. At the end of the concert, the conductor lowers their baton, all instruments fall silent together, and the musicians pack up safely (**Graceful Subprocess Teardown**).

In modern AI engineering, the most capable agents do not connect to just one tool. They run as **Multi-Server Host Orchestrators**, unifying dozens of independent MCP servers into a single coherent reasoning loop.

---

## 2. Complete Mathematical & Protocol Symbol Glossary

| Symbol / Term | Formal Mathematical Signature | Description / Role |
| :--- | :--- | :--- |
| $\mathcal{S}_i$ | Server Instance $i \in \{1, \dots, N\}$ | Isolated MCP server process (e.g. PostgreSQL server, Git server, Slack server). |
| $\mathcal{O}_{\text{host}}$ | Multi-Server Host Orchestrator | Central manager process holding the agent loop and subprocess handles. |
| $\Gamma_{\text{tools}}$ | $\bigcup_{i=1}^N \mathcal{T}(\mathcal{S}_i)$ | Unified tool catalog aggregated across all connected servers. |
| $\Gamma_{\text{res}}$ | $\bigcup_{i=1}^N \mathcal{R}(\mathcal{S}_i)$ | Unified resource catalog aggregated across all connected servers. |
| $\mathcal{R}_{\text{route}}$ | $\text{name} \mapsto \mathcal{S}_i$ | Routing lookup table mapping tool names or URI schemes to their owner server. |
| $\mathcal{L}_{\text{agent}}$ | ReAct / Plan-and-Solve Loop | The multi-step LLM inference and tool dispatch execution loop. |
| $\mathcal{T}_{\text{teardown}}$ | $\lim_{t \to T} \text{SIGTERM} \to \text{SIGKILL}$ | Ordered process termination protecting database transactions and open files. |

---

## 3. End-to-End Multi-Server Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                      HOST CLIENT / AGENT RUNTIME                       │
│                                                                        │
│  User Goal: "Analyze Q3 sales in postgres and send email report"       │
│                                                                        │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │ Unified Tool & Resource Routing Table                          │   │
│   │  • URI "db://sales/q3"          ──► Routed to Server 1 (DB)    │   │
│   │  • Tool "calculate_aggregates"  ──► Routed to Server 2 (Compute)│  │
│   │  • Tool "send_email"            ──► Routed to Server 3 (Email) │   │
│   └────────────────────────────────┬───────────────────────────────┘   │
└────────────────────────────────────┼───────────────────────────────────┘
                                     │
           ┌─────────────────────────┼─────────────────────────┐
           │ Stdio Pipe              │ Stdio Pipe              │ Stdio Pipe
           ▼                         ▼                         ▼
┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
│  MCP Server 1 (DB)   │  │ MCP Server 2 (Stats) │  │ MCP Server 3 (Email) │
│  - Database queries  │  │ - Python math numpy  │  │ - SMTP dispatch      │
│  - Table schemas     │  │ - Financial formulas │  │ - Draft generation   │
└──────────────────────┘  └──────────────────────┘  └──────────────────────┘
```

### 3.1. Capability-Gated Catalog Discovery
When orchestrating $N$ servers, the host must not blindly query every method on every server. Instead:
$$\text{Query Tools}(\mathcal{S}_i) \iff \text{"tools"} \in \text{Capabilities}(\mathcal{S}_i)$$
$$\text{Query Resources}(\mathcal{S}_i) \iff \text{"resources"} \in \text{Capabilities}(\mathcal{S}_i)$$
If a server only declares `resources`, sending `tools/list` will trigger a `-32601 Method not found` error or block an unbuffered pipe.

---

## 4. Building a Production Stdio MCP Server from Scratch

A production-grade Python stdio server must follow five non-negotiable rules:
1. **Unbuffered I/O**: Launch with Python's `-u` flag (`python3 -u server.py`) or call `sys.stdout.flush()` after writing every single line.
2. **Strict Stderr Isolation**: All human-readable debug logs must be routed to `sys.stderr` (`print(..., file=sys.stderr)`). Any stray character on `sys.stdout` corrupts the host's JSON-RPC line parser.
3. **Robust Signal Handling**: Handle `SIGINT` and `SIGTERM` cleanly so that database connections and child threads close properly.
4. **Unhandled Method Fallback**: Any method not recognized by the server must return `-32601 Method Not Found` with the client's `id` preserved.
5. **Tool Error Segregation**: Trap Python exceptions within tool handlers and return them with `isError: true` rather than crashing the subprocess.

---

## 5. The Multi-Server Autonomous Agent Decision Loop

The host coordinates a multi-turn agent execution trajectory:

$$\text{State } s_t = \langle \mathcal{G}_{\text{goal}}, \mathcal{H}_{\text{history}}, \Gamma_{\text{tools}}, \Gamma_{\text{res}} \rangle$$
$$\text{Action } a_t = \pi_{\text{LLM}}(s_t) \in \{\text{ReadResource}(u), \text{CallTool}(n, \theta), \text{Finish}(y)\}$$

```
Iter 1: LLM selects ReadResource("db://sales/q3")
        Host routes to Server 1 -> Returns sales numbers
Iter 2: LLM selects CallTool("calculate_aggregates", {"data": [...]})
        Host routes to Server 2 -> Returns sum & growth percentages
Iter 3: LLM selects CallTool("send_email", {"to": "exec@co.com", "body": "..."})
        Host routes to Server 3 -> Returns delivery receipt
Iter 4: LLM emits Final Summary -> Loop terminates successfully
```

---

## 6. Process Teardown & Lifecycle Reliability

Host orchestrators must guarantee no orphaned child processes remain after user exit:
1. **Orderly Shutdown Request**: Send `{"method": "shutdown", "id": ...}` over stdin.
2. **Pipe Closure**: Close child `stdin` to signal EOF.
3. **Grace Period**: Wait up to $\tau_{\text{grace}} = 2.0\text{s}$ using `process.wait(timeout=2)`.
4. **Escalation**: If the subprocess hangs, send `process.kill()` (`SIGKILL`) to force operating system cleanup.
