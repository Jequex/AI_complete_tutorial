# Topic 01: MCP Architecture & JSON-RPC 2.0 Wire Format

---

## 1. The 12-Year-Old Intuitive Analogy: The Universal USB-C Cable for AI

Imagine you have a cutting-edge smartphone or game console, but every accessory in the world had a completely different, proprietary connector. Your headphones needed a 7-pin triangular plug, your controller needed a circular 12-pin plug, your external storage needed a flat 30-pin ribbon cable, and your camera needed a tiny screw-in socket. If you bought five gadgets from five different manufacturers, you needed five custom converters, tangled cables, and unique drivers. One small hardware update from the phone maker, and half your adapters stopped working!

That was exactly the state of Artificial Intelligence applications before late 2024:
Every AI agent framework (LangChain, LlamaIndex, custom company scripts) had to write custom, ad-hoc wrapper code for every single data source and tool—PostgreSQL, GitHub, Slack, local file systems, Brave Search, Google Drive. If there were $N$ AI host applications and $M$ tools, the software industry had to maintain $N \times M$ fragile custom adapters.

```
Without MCP (The N x M Adapter Nightmare):
[Claude Desktop] ──┬──► Custom GitHub Connector ────► GitHub API
                   ├──► Custom Postgres Connector ──► Postgres DB
                   └──► Custom Slack Connector ─────► Slack API

[Cursor IDE]    ──┬──► Another GitHub Connector ──► GitHub API
                   ├──► Another Postgres Conn ─────► Postgres DB
                   └──► Another Slack Conn ────────► Slack API

With Model Context Protocol (1 x N + 1 x M Elegance):
[Host / Client] ─── (Universal MCP Protocol over JSON-RPC 2.0) ───► [Any MCP Server]
 (Claude, Cursor,                                                   (Postgres, Git,
  Custom Agents)                                                     Filesystem, Slack)
```

The **Model Context Protocol (MCP)** is the **USB-C standard for AI applications**. Just like USB-C allows any computer to plug into any monitor, keyboard, hard drive, or microphone over a single standardized hardware and packet specification, MCP defines a single open protocol over **JSON-RPC 2.0** so any AI model host can communicate with any data source, tool catalog, or environment seamlessly.

---

## 2. Complete Mathematical & Protocol Symbol Glossary

| Symbol / Term | Formal Definition / Wire Representation | Role in MCP & JSON-RPC 2.0 |
| :--- | :--- | :--- |
| $\mathcal{H}$ | **Host / Client** | The orchestrator application (e.g., Claude Desktop, IDE, Autonomous Agent) holding the LLM context. |
| $\mathcal{S}$ | **Server** | Lightweight program exposing data (Resources), actions (Tools), or prompt templates (Prompts). |
| $\mathcal{T}_{\text{stdio}}$ | **Standard I/O Transport** | Subprocess communication channel where messages are framed via newline characters (`\n`) over `stdin`/`stdout`. |
| $\mathcal{T}_{\text{sse}}$ | **HTTP + SSE Transport** | Remote transport where client sends HTTP POST requests and server streams events via Server-Sent Events (`text/event-stream`). |
| $\text{jsonrpc}$ | `"2.0"` (Strict String Literal) | Protocol version identifier; requests without exactly `"2.0"` trigger error code `-32600`. |
| $\text{id}$ | $\text{String} \cup \text{Integer} \cup \text{null}$ | Unique correlation token matching an asynchronous response back to the originating client request. |
| $\text{method}$ | String identifier (e.g. `tools/list`, `resources/read`) | The Remote Procedure Call (RPC) operation requested on the server. |
| $\text{params}$ | Object (`{...}`) or Array (`[...]`) | Structured payload passed to the RPC method. |
| $\text{result}$ | Any valid JSON entity | Returned upon successful execution of an RPC method (mutually exclusive with $\text{error}$). |
| $\text{error}$ | Object: `{"code": int, "message": str, "data"?: any}` | Returned upon failed execution; conforms to the standardized JSON-RPC error registry. |
| $\mathcal{N}$ | Notification (Message with no `id`) | One-way fire-and-forget message requiring **no response** from the receiver. |
| $E_{\text{code}}$ | Standard Integer Error Codes | Standard range defined by JSON-RPC 2.0 specification ($-32700$ to $-32600$). |

---

## 3. The Client-Host-Server Topology

The MCP ecosystem defines three explicit participants in the runtime topology:

```
┌────────────────────────────────────────────────────────────────────────┐
│                              HOST                                      │
│  (e.g., IDE, Desktop Agent Environment, Custom Python Agent Runtime)   │
│                                                                        │
│   ┌──────────────────────────┐         ┌───────────────────────────┐   │
│   │         LLM Core         │         │        MCP Client         │   │
│   │  (Generates reasoning &  │◄───────►│  (Manages connections,    │   │
│   │   tool call intents)     │         │   routes calls, security) │   │
│   └──────────────────────────┘         └─────────────┬─────────────┘   │
└──────────────────────────────────────────────────────┼─────────────────┘
                                                       │
                           Standardized JSON-RPC 2.0   │  (Bidirectional)
                           over Stdio or HTTP/SSE      │
                                                       │
         ┌─────────────────────────────────────────────┴──────────────────┐
         │                                                                │
         ▼                                                                ▼
┌─────────────────────────────────┐              ┌─────────────────────────────────┐
│        MCP Server: Local        │              │       MCP Server: Remote        │
│  (Subprocess spawned via stdio) │              │   (External service via SSE)    │
│  - Reads local git repo         │              │  - Queries enterprise database  │
│  - Executes file edits          │              │  - Invokes CRM/ERP APIs         │
└─────────────────────────────────┘              └─────────────────────────────────┘
```

1. **Host ($\mathcal{H}_{\text{runtime}}$)**: The outer boundary process (such as an IDE or multi-agent orchestrator). It manages authentication, displays UI elements to users, enforces human-in-the-loop permission prompts, and houses the core LLM inference loop.
2. **Client ($\mathcal{C}_{\text{mcp}}$)**: An internal component within the Host that maintains a 1-to-1 connection to an MCP server. A Host with three connected servers runs three separate MCP Client instances.
3. **Server ($\mathcal{S}$)**: An isolated process that exposes contextual primitives. Crucially, **MCP servers do not execute LLM inference themselves**—they are lightweight protocol endpoints exposing resources, tools, and prompts to the host client.

---

## 4. The JSON-RPC 2.0 Wire Specification

Every MCP communication exchange is structured as a JSON-RPC 2.0 message. The specification enforces four distinct message schemas:

### 4.1. Request Object
A client requesting an action or data from a server (or vice-versa in server sampling):
$$\mathcal{M}_{\text{req}} = \langle \text{jsonrpc}: \text{"2.0"}, \; \text{id}: k \in (\mathbb{Z} \cup \Sigma^*), \; \text{method}: s \in \Sigma^*, \; \text{params}?: \mathcal{P} \rangle$$

```json
{
  "jsonrpc": "2.0",
  "id": "req-1042",
  "method": "tools/call",
  "params": {
    "name": "calculate_tax",
    "arguments": {
      "subtotal": 1250.00,
      "region": "CA"
    }
  }
}
```

### 4.2. Successful Response Object
Sent in reply to a Request when execution completes without fault. The `id` must match the request's `id`:
```json
{
  "jsonrpc": "2.0",
  "id": "req-1042",
  "result": {
    "content": [
      {
        "type": "text",
        "text": "103.125"
      }
    ],
    "isError": false
  }
}
```

### 4.3. Error Response Object
Sent when parsing, validation, or internal execution fails:
```json
{
  "jsonrpc": "2.0",
  "id": "req-1042",
  "error": {
    "code": -32602,
    "message": "Invalid params: Missing required field 'region'",
    "data": {
      "expected": ["subtotal", "region"],
      "received": ["subtotal"]
    }
  }
}
```

### 4.4. Notification Object
A fire-and-forget message without an `id`. The receiver **must never emit a response** for a notification:
```json
{
  "jsonrpc": "2.0",
  "method": "notifications/resources/updated",
  "params": {
    "uri": "file:///workspace/src/app.py"
  }
}
```

---

## 5. Standard Error Codes Specification

JSON-RPC 2.0 reserves specific integer error codes. In MCP, error propagation must strictly adhere to these constants:

$$\begin{array}{|l|l|l|}
\hline
\textbf{Code} & \textbf{Constant Name} & \textbf{Trigger Condition} \\
\hline
-32700 & \text{Parse Error} & \text{Invalid JSON syntax received by the parser.} \\
-32600 & \text{Invalid Request} & \text{JSON is valid, but fails schema (e.g., missing } \texttt{"jsonrpc": "2.0"}\text{).} \\
-32601 & \text{Method Not Found} & \text{The requested method is not registered on the receiver.} \\
-32602 & \text{Invalid Params} & \text{Parameter arguments do not match expected types or requirements.} \\
-32603 & \text{Internal Error} & \text{Unhandled runtime exception or server failure during execution.} \\
-32000 \dots -32099 & \text{Server-defined Error} & \text{Application-specific extension errors reserved for MCP implementations.} \\
\hline
\end{array}$$

---

## 6. Batch Request Processing Mathematics

Clients may pack multiple requests and notifications into a single JSON array:
$$\mathcal{B} = \left[ m_1, m_2, \dots, m_K \right]$$

The dispatch engine evaluates each element independently:
$$\mathcal{R}(\mathcal{B}) = \left\{ \mathcal{E}(m_i) \mid m_i \in \mathcal{B} \land \text{is\_request}(m_i) \right\}$$

Key batch invariants:
1. If $\mathcal{B} = []$ (empty array), the server **must** return an Error Object with code $-32600$ (Invalid Request).
2. Notifications ($m_i$ without `id`) produce **no response item**.
3. If all items in $\mathcal{B}$ are notifications, the server returns **no response at all** (null/empty output).
4. Order of response elements in $\mathcal{R}(\mathcal{B})$ does not have to match the order in $\mathcal{B}$; clients must correlate responses using the matching `id`.

---

## 7. Transport Framing Mechanisms

JSON is inherently a structured data format, but physical network interfaces and OS pipe descriptors transmit flat byte streams. Therefore, framing is required.

### 7.1. Standard I/O (stdio) Framing
In the stdio transport, the host client launches the MCP server as a child process:
- Server reads JSON requests from `stdin`.
- Server writes JSON responses and notifications to `stdout`.
- Every message is strictly framed by a newline character (`\n` or ASCII `0x0A`).
- **Critical rule for stdio servers**: A server must **never print arbitrary debug statements to `stdout`** (e.g. `print("connecting...")`), because non-JSON characters corrupt the line parser. All server logging must be directed to `stderr` or emitted via MCP's `notifications/message` protocol!

### 7.2. Server-Sent Events (SSE) Framing
In remote deployments over HTTP:
- The client opens a persistent HTTP GET connection with header `Accept: text/event-stream`.
- The server streams events using chunking:
  ```http
  event: endpoint
  data: http://localhost:8080/messages?session_id=abc123xyz

  event: message
  data: {"jsonrpc":"2.0","method":"ping","id":1}
  ```
- Subsequent client-to-server messages are dispatched via standard HTTP `POST` requests targeted at the provided session endpoint.

---

## 8. Summary & Mental Checklist

1. **Topology**: Host manages LLM and security; Client manages 1-to-1 connection; Server exposes tools and data.
2. **Wire Format**: Strict JSON-RPC 2.0 with `jsonrpc: "2.0"`.
3. **Four Message Types**: Request (with `id`), Response (result), Error (code + message), Notification (no `id`).
4. **Standard Error Range**: $-32700$ (Parse), $-32600$ (Invalid Request), $-32601$ (Method Not Found), $-32602$ (Invalid Params), $-32603$ (Internal Error).
5. **Framing**: Newline-delimited for stdio (`stdout`), HTTP event stream for SSE.
