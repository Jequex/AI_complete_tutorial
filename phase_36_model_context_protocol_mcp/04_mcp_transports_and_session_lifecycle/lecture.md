# Topic 04: MCP Transports & Session Lifecycle

---

## 1. The 12-Year-Old Intuitive Analogy: The Formal Handshake & Secret Knocks

Imagine you are arriving at an international embassy to do important business. 

You cannot simply kick the door open and start shouting demands like *"Give me your financial ledger!"* or *"Sign this treaty!"*. If you do that, security guards will immediately escort you off the premises.

Instead, there is a strict formal protocol:
1. **The Introduction (Passport Exchange)**: You walk up to the desk, state your identity, the language you speak (Protocol Version), and the credentials you carry. The embassy officer inspects your credentials, verifies compatibility, and hands you an official badge detailing what rooms you are permitted to enter (**Capabilities Negotiation**).
2. **The Acknowledgment (The Seal)**: You bow or say *"Acknowledged and ready"* (`notifications/initialized`). Only after this second step do the doors unlock!
3. **The Heartbeat Check**: Every ten minutes, a security officer asks *"Are you still present and well?"* (`ping`), and you respond with a nod (`pong`).
4. **The Graceful Departure**: When your business is finished, you say goodbye, return your security badge, and the embassy locks the gates.

In MCP, this exact diplomatic sequence is the **Session Lifecycle**. It guarantees that an AI client and external server agree on version compatibility, feature sets, and connection health before a single line of data or tool code is exchanged.

---

## 2. Complete Mathematical & Protocol Symbol Glossary

| Symbol / Term | Formal Protocol Signature | Description / Role |
| :--- | :--- | :--- |
| $\mathcal{S}_{\text{state}}$ | $\Sigma_{\text{state}} \in \{\text{UNINIT}, \text{INITING}, \text{INITED}, \text{TERM}\}$ | Discrete lifecycle state of an active MCP session connection. |
| $\mathcal{V}_{\text{proto}}$ | ISO Date Format String (e.g. `"2024-11-05"`) | Protocol version string governing wire format semantics and capabilities. |
| $\mathcal{C}_{\text{client}}$ | Object: `{"roots"?: {...}, "sampling"?: {...}}` | Feature set supported and advertised by the host client. |
| $\mathcal{C}_{\text{server}}$ | Object: `{"resources"?: {...}, "tools"?: {...}, "prompts"?: {...}}` | Feature set supported and advertised by the MCP server. |
| $\text{initialize}$ | Request: $\{\text{protocolVersion}, \text{capabilities}, \text{clientInfo}\}$ | Handshake Initiation: First message sent by client to server. |
| $\mathcal{N}_{\text{initialized}}$ | Notification: `notifications/initialized` | Handshake Finalization: Sent by client confirming receipt of server capabilities. |
| $\text{ping}$ | Request: `ping` $\to \{\}$ | Bidirectional liveness probe ensuring connection has not silently dropped. |
| $\Delta t_{\text{ping}}$ | Duration ($\mathbb{R}^+$ seconds) | Liveness timeout threshold; exceeding $\Delta t_{\text{ping}}$ triggers session termination. |

---

## 3. Session Lifecycle State Machine

The session lifecycle follows a strict deterministic finite automaton (DFA):

```
                        ┌───────────────────┐
                        │   UNINITIALIZED   │
                        └─────────┬─────────┘
                                  │
                                  │ Client sends `initialize`
                                  │ Server responds with capabilities
                                  ▼
                        ┌───────────────────┐
                        │    INITIALIZING   │
                        └─────────┬─────────┘
                                  │
                                  │ Client sends `notifications/initialized`
                                  ▼
                        ┌───────────────────┐
       ┌───────────────►│    INITIALIZED    │◄──────────────┐
       │                └─────────┬─────────┘               │
       │                          │                         │
       │ ping / pong              │ Disconnect / SIGTERM    │ Resource & Tool
       │ Heartbeat Loop           │ or `shutdown`           │ Operations
       │                          ▼                         │
       └────────────────►┌──────────────────┐───────────────┘
                         │    TERMINATED    │
                         └──────────────────┘
```

### 3.1. Handshake State Invariants:
1. **Prior to `initialize`**: Any functional RPC call (e.g., `tools/list`, `resources/read`, `prompts/get`) **must be rejected** with JSON-RPC error code `-32600` (`Invalid Request: Server not initialized`).
2. **During `INITIALIZING`**: The server is waiting for the client's confirmation notification. Any RPC calls received during this window are rejected.
3. **Transition to `INITIALIZED`**: The notification `notifications/initialized` requires no response. Once processed by the server, the connection enters full operational mode.

---

## 4. Capability Negotiation Deep Dive

During `initialize`, both parties exchange capability objects detailing supported primitives:

### 4.1. Client Capabilities
```json
{
  "capabilities": {
    "roots": {
      "listChanged": true
    },
    "sampling": {}
  }
}
```
- `roots`: Signals whether the client can supply workspace root directories and whether it supports dynamic root change notifications.
- `sampling`: Signals whether the server is permitted to request LLM completions back from the host client (human-in-the-loop or agentic sub-querying).

### 4.2. Server Capabilities
```json
{
  "capabilities": {
    "resources": {
      "subscribe": true,
      "listChanged": true
    },
    "tools": {
      "listChanged": true
    },
    "prompts": {
      "listChanged": true
    },
    "logging": {}
  }
}
```
- `subscribe`: Indicates whether clients may subscribe to individual resource URIs.
- `listChanged`: Indicates whether the server will emit notifications when items are dynamically added or removed from its catalogs.

---

## 5. Stdio Transport vs. HTTP with Server-Sent Events (SSE)

MCP standardizes two primary transport layers:

```
┌──────────────────────────────────────┐  ┌──────────────────────────────────────┐
│       Stdio Transport (Local)        │  │      HTTP + SSE Transport (Remote)   │
├──────────────────────────────────────┤  ├──────────────────────────────────────┤
│ • Subprocess spawned by Host Client  │  │ • Hosted on remote web/cloud server  │
│ • Zero network ports or firewalls    │  │ • Client connects via HTTP GET /sse   │
│ • Input over stdin, Output on stdout │  │ • Server streams SSE events down     │
│ • Framed strictly by \n              │  │ • Client posts RPCs via HTTP POST    │
│ • Process death = Session terminated │  │ • Session ID identifies state        │
└──────────────────────────────────────┘  └──────────────────────────────────────┘
```

### 5.1. The SSE Protocol Dance
1. **Initial Connect**: Client performs `GET /sse` with header `Accept: text/event-stream`.
2. **Endpoint Discovery**: Server emits an SSE `endpoint` event containing the unique URI to which subsequent client messages must be posted:
   ```http
   event: endpoint
   data: /messages?session_id=8923a10f
   ```
3. **RPC Invocations**: Client sends standard HTTP POST requests to `/messages?session_id=8923a10f` containing JSON-RPC requests.
4. **Server Streaming**: Server streams responses and push notifications back across the persistent SSE channel.

---

## 6. Liveness Detection & Graceful Teardown

To detect stale connections (such as hung worker threads or severed networks):
$$\text{Liveness Check}: \quad \Delta t = t_{\text{current}} - t_{\text{last\_activity}}$$
If $\Delta t > \tau_{\text{threshold}}$, the host dispatches:
```json
{"jsonrpc": "2.0", "id": "hb-1", "method": "ping"}
```
The receiver must immediately respond with an empty result:
```json
{"jsonrpc": "2.0", "id": "hb-1", "result": {}}
```
If no pong arrives within a timeout window, the transport is force-killed and resources (database connections, file locks, child subprocesses) are cleaned up.
