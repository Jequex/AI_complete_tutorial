# Topic 04: Exercises & Solutions — MCP Transports & Session Lifecycle

---

## Exercise 1: Handshake State Machine Invariants

### Problem
A newly spawned MCP server receives the following message as its very first input on `stdin`:
```json
{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
```
1. How must the server respond according to the MCP specification?
2. What state transition occurs, if any?
3. What sequence of messages must occur before `tools/list` can be successfully handled?

### Solution
1. **Server Response**:
   The server must reject the request with a JSON-RPC Error Object:
   ```json
   {
     "jsonrpc": "2.0",
     "id": 1,
     "error": {
       "code": -32600,
       "message": "Invalid Request: Server not initialized. Received 'tools/list' before 'initialize'."
     }
   }
   ```
2. **State Transition**:
   No state transition occurs. The server remains in the `UNINITIALIZED` state.
3. **Required Handshake Sequence**:
   - **Step A**: Client sends `initialize` request with `protocolVersion`, `capabilities`, and `clientInfo`.
   - **Step B**: Server responds with its `protocolVersion`, `capabilities`, and `serverInfo`. (Server enters `INITIALIZING` state).
   - **Step C**: Client sends `notifications/initialized` notification. (Server enters `INITIALIZED` state).
   - **Step D**: Client can now send `tools/list`, which will be processed and returned.

---

## Exercise 2: Server-Sent Events (SSE) Reconnection & State Recovery

### Problem
In an HTTP + SSE transport architecture, a client experiences a temporary network blip and drops its SSE connection.
1. How does the client re-establish communication without losing session context?
2. Why is the session ID maintained in the query parameter of the POST endpoint?

### Solution
1. **Reconnection Mechanism**:
   - The client re-opens an HTTP GET request to `/sse` providing either its prior session ID or headers (e.g. `Last-Event-ID`).
   - If supported by the server, the server resumes streaming from the event log for that `session_id`. If the session timed out or was garbage-collected, the server returns 404 or an endpoint indicating a fresh initialization is required.
2. **Session ID in Endpoint**:
   - HTTP POST requests are inherently stateless.
   - By including `?session_id=...` in the POST URL, the server can route the JSON-RPC request to the exact in-memory worker or message queue handling that client's state machine.

---

## Exercise 3: Heartbeat Timeout Calculation & Deadlock Prevention

### Problem
Suppose an MCP client connects to a remote compute server over SSE. The client sends a `ping` request every 30 seconds.
1. What should the client do if 3 consecutive `ping` requests receive no response within 10 seconds?
2. Why should tools that perform long-running computations (e.g., training a machine learning model for 10 minutes) use asynchronous progress notifications rather than blocking the main RPC thread?

### Solution
1. **Liveness Timeout Action**:
   - If 3 consecutive heartbeats time out, the client must mark the transport as `DEAD` or `DISCONNECTED`.
   - It should terminate child workers or network sockets, purge pending request promises with a timeout exception, and attempt an orderly reconnect.
2. **Long-Running Computation Architecture**:
   - If a tool blocks the single worker thread synchronously for 10 minutes, the server will be unable to process incoming `ping` requests or client cancellation notifications (`notifications/cancelled`).
   - The host will assume the server has deadlocked and kill it.
   - Therefore, long-running tools must either execute asynchronously on a background worker thread, periodically emitting `notifications/progress` while keeping the main loop reactive to `ping` messages.
