# Topic 01: Exercises & Solutions — MCP Architecture & JSON-RPC 2.0 Wire Format

---

## Exercise 1: Standard Error Code Resolution

### Problem
An MCP server receives the following raw payloads over its stdio transport:
1. `{"jsonrpc": "2.0", "method": "database/query", "params": {"sql": 12345}, "id": "q1"}` (where the SQL parameter must be a string)
2. `{"jsonrpc": "2.0", "method": "tools/non_existent_tool", "id": "q2"}`
3. `{"method": "ping", "id": "q3"}`
4. `{"jsonrpc": "2.0", "method": "calculate", "params": [1, 2], broken`

For each payload, specify:
- Whether an error response must be returned.
- The standard JSON-RPC 2.0 error code integer.
- The exact `id` field value in the response object.

### Solution
1. **Payload 1**:
   - Error code: `-32602` (`Invalid params`). The SQL parameter received an integer instead of the expected string type.
   - `id`: `"q1"` (preserved from the request).
   - Response: `{"jsonrpc": "2.0", "id": "q1", "error": {"code": -32602, "message": "Invalid params: 'sql' must be string"}}`.

2. **Payload 2**:
   - Error code: `-32601` (`Method not found`). The method `tools/non_existent_tool` is not in the server's method table.
   - `id`: `"q2"`.
   - Response: `{"jsonrpc": "2.0", "id": "q2", "error": {"code": -32601, "message": "Method not found"}}`.

3. **Payload 3**:
   - Error code: `-32600` (`Invalid Request`). Missing the required `"jsonrpc": "2.0"` protocol attribute.
   - `id`: `"q3"` (even though the request is malformed, JSON parsing succeeded and recovered `id`).
   - Response: `{"jsonrpc": "2.0", "id": "q3", "error": {"code": -32600, "message": "Invalid Request: missing jsonrpc version"}}`.

4. **Payload 4**:
   - Error code: `-32700` (`Parse error`). The string is corrupted syntax and fails JSON deserialization.
   - `id`: `null` (since the JSON parser cannot parse the object, it cannot safely extract the `id`).
   - Response: `{"jsonrpc": "2.0", "id": null, "error": {"code": -32700, "message": "Parse error"}}`.

---

## Exercise 2: Batch Processing Protocol Rules

### Problem
A client submits a batch containing four operations:
```json
[
  {"jsonrpc": "2.0", "method": "ping", "id": 1},
  {"jsonrpc": "2.0", "method": "notifications/progress", "params": {"pct": 50}},
  {"jsonrpc": "2.0", "method": "invalid_method", "id": 2},
  {"jsonrpc": "2.0", "method": "notifications/log", "params": {"msg": "done"}}
]
```
1. How many objects should be in the array returned by the server?
2. Which methods correspond to each returned response object?
3. What happens if the client sends an empty batch array `[]`?

### Solution
1. Exactly **2** response objects are returned.
2. The second and fourth operations are **Notifications** (they lack an `id` property). According to the JSON-RPC 2.0 specification, notifications **must never produce a response**, even when sent inside a batch.
   - The returned array contains:
     - The successful result for `ping` (`id: 1`).
     - The error object (`-32601 Method not found`) for `invalid_method` (`id: 2`).
   ```json
   [
     {"jsonrpc": "2.0", "id": 1, "result": {"status": "pong"}},
     {"jsonrpc": "2.0", "id": 2, "error": {"code": -32601, "message": "Method not found"}}
   ]
   ```
3. If an empty array `[]` is received, the server must return a single JSON-RPC Error Object with code `-32600` (`Invalid Request`):
   ```json
   {"jsonrpc": "2.0", "id": null, "error": {"code": -32600, "message": "Invalid Request: Batch cannot be empty"}}
   ```

---

## Exercise 3: Stdio Subprocess Contamination Prevention

### Problem
A junior engineer writes an MCP Python server using stdio transport and adds:
```python
print("Connecting to database...")
```
Why does this break the MCP protocol client, and how must logging be handled in stdio-based servers?

### Solution
1. In stdio transport, the host client parses `stdout` line-by-line expecting pure JSON strings separated by `\n`.
2. When `print("Connecting to database...")` is executed, the raw string is written directly to `stdout`. The host's JSON parser tries to parse `"Connecting to database..."` as JSON-RPC, throws a `-32700 Parse error`, and typically terminates the transport connection.
3. **Correct logging strategies**:
   - Redirect non-protocol terminal logs to `sys.stderr` (`print("...", file=sys.stderr)`), which the host captures separately for developer debugging.
   - Use the official MCP notification primitive:
     ```json
     {"jsonrpc": "2.0", "method": "notifications/message", "params": {"level": "info", "data": "Connecting to database..."}}
     ```
