# Topic 03: Exercises & Solutions — MCP Tools & Prompts

---

## Exercise 1: Distinguishing Protocol Errors from Tool Execution Failures

### Problem
For each of the following scenarios in an MCP environment, specify whether the server should return a **JSON-RPC Error Object** (with code $-32xxx$) or a **Standard Result with `isError: true`**:
1. The client sends `{"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "non_existent_tool"}, "id": 1}`.
2. The client calls tool `calculator` with parameters `{"operation": "divide", "a": 10, "b": 0}`, causing a `ZeroDivisionError`.
3. The client calls tool `database_query` with SQL `SELECT * FROM non_existent_table`, causing an SQL execution error.
4. The client sends a broken JSON string: `{"jsonrpc": "2.0", "method": "tools/call", ...`.

### Solution
1. **Scenario 1 (Unknown Tool Name)**:
   - Response: JSON-RPC Error Object with code `-32601` (`Method not found`) or `-32602` (`Invalid params` indicating tool name does not exist), OR tool call result with `isError: true`. Standard practice in MCP SDKs returns a tool call error result or `-32601` if tools router checks names.
2. **Scenario 2 (Division by Zero)**:
   - Response: **Standard Result with `isError: true`**.
   ```json
   {
     "jsonrpc": "2.0",
     "id": 1,
     "result": {
       "content": [{"type": "text", "text": "ZeroDivisionError: division by zero"}],
       "isError": true
     }
   }
   ```
   *Rationale*: This is a domain failure. The model must receive this feedback to understand why its arithmetic attempt failed and self-correct.
3. **Scenario 3 (SQL Table Missing)**:
   - Response: **Standard Result with `isError: true`**.
   *Rationale*: The transport and RPC protocol functioned perfectly; the external SQL engine rejected the query.
4. **Scenario 4 (Malformed JSON String)**:
   - Response: **JSON-RPC Error Object with code `-32700` (`Parse error`)** and `id: null`. The message cannot be parsed into a structured object.

---

## Exercise 2: JSON Schema Parameter Validation Contract

### Problem
A tool `create_user` specifies the following `inputSchema`:
```json
{
  "type": "object",
  "properties": {
    "username": {"type": "string"},
    "age": {"type": "integer"},
    "role": {"type": "string", "enum": ["viewer", "editor", "admin"]}
  },
  "required": ["username", "role"]
}
```
Which of the following argument dictionaries will pass validation? If they fail, specify the reason:
- Argument A: `{"username": "jdoe", "role": "editor"}`
- Argument B: `{"username": "alice", "age": 28}`
- Argument C: `{"username": "charlie", "role": "superadmin"}`
- Argument D: `{"username": "bob", "role": "admin", "age": "twenty-five"}`

### Solution
- **Argument A**: **PASS**. Both required fields (`username` and `role`) are present, types are valid string, and `"editor"` is in the allowed enum.
- **Argument B**: **FAIL**. Missing required parameter `'role'`.
- **Argument C**: **FAIL**. Field `'role'` value `'superadmin'` is not in allowed enum `['viewer', 'editor', 'admin']`.
- **Argument D**: **FAIL**. Field `'age'` expected integer, got `string` (`"twenty-five"`).

---

## Exercise 3: Designing Multi-Turn MCP Prompts

### Problem
Design an MCP prompt named `troubleshoot_network` that accepts two arguments:
- `host` (required string)
- `protocol` (optional string, default `"TCP"`)

Write the `prompts/get` handler in Python that returns a two-turn conversation providing diagnostic instructions to the model.

### Solution
```python
def handle_troubleshoot_network(args: dict) -> dict:
    host = args.get("host")
    if not host:
        raise ValueError("Missing required prompt argument: 'host'")
    protocol = args.get("protocol", "TCP")

    messages = [
        {
            "role": "user",
            "content": {
                "type": "text",
                "text": f"I am diagnosing network connectivity issues to host '{host}' using {protocol}."
            }
        },
        {
            "role": "assistant",
            "content": {
                "type": "text",
                "text": (
                    f"Understood. Let's run diagnostic steps for {protocol} against {host}.\n"
                    "Step 1: Check DNS resolution.\n"
                    "Step 2: Probe ICMP ping latency.\n"
                    f"Step 3: Test {protocol} port socket handshake."
                )
            }
        },
        {
            "role": "user",
            "content": {
                "type": "text",
                "text": "Please execute the appropriate MCP diagnostic tools to begin Step 1."
            }
        }
    ]
    return {
        "description": "Multi-turn network troubleshooting sequence",
        "messages": messages
    }
```
This structured sequence bootstraps the AI agent with clear, multi-step problem solving discipline.
