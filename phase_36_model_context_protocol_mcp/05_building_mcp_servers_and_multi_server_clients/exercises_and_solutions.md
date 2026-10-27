# Topic 05: Exercises & Solutions — Building MCP Servers & Multi-Server Clients

---

## Exercise 1: Multi-Server Tool Name Collisions & Namespacing

### Problem
An AI orchestrator connects to two independent third-party MCP servers:
- Server A (GitHub Integration) exposes a tool named `search`.
- Server B (Elasticsearch Integration) also exposes a tool named `search`.

1. What failure occurs if the host attempts a naive flat merge into its tool registry?
2. How should the host client resolve this naming collision while preserving LLM tool comprehension?

### Solution
1. **The Collision Failure**:
   In a flat routing dictionary `tool_routes[tool_name] = server_client`, the second server overwrites the first. If the LLM generates a function call `{"name": "search", "arguments": {"query": "auth bug"}}`, the host cannot know whether to query GitHub issues or Elasticsearch indexes.
2. **Namespacing Strategies**:
   - **Prefix Namespacing**: The host prefixes each tool with the server's unique identifier when presenting tools to the model:
     - `github__search` $\to$ routed to Server A
     - `elasticsearch__search` $\to$ routed to Server B
   - When the LLM emits a call to `github__search`, the host client strips the prefix `github__` and dispatches `{"name": "search", ...}` to Server A. This maintains transparency for both the model and the underlying MCP server.

---

## Exercise 2: Unbuffered Pipe Deadlocks in Subprocess MCP Servers

### Problem
A developer implements an MCP server in Python without `sys.stdout.flush()`. When launched by a host client, the handshake hangs indefinitely on `self.process.stdout.readline()`.
1. Explain the low-level operating system reason for this hang.
2. What two distinct modifications prevent this bug?

### Solution
1. **Low-Level Root Cause**:
   Standard C/Python runtimes use block buffering (typically 4KB or 8KB) when `stdout` is connected to a pipe or file rather than an interactive TTY terminal. Because the small JSON-RPC response `{"jsonrpc": "2.0", ...}` is only a few hundred bytes, it sits in Python's internal memory buffer and is never transmitted down the OS pipe. The host client blocks waiting on `readline()` because no bytes are in the pipe buffer.
2. **Preventative Modifications**:
   - **Modification A**: Run Python with unbuffered mode using the `-u` command line flag: `python3 -u server.py`.
   - **Modification B**: Explicitly call `sys.stdout.flush()` immediately after every `sys.stdout.write(line + "\n")`.

---

## Exercise 3: Designing an Autonomous Multi-Server Agent Pipeline

### Problem
You are building an autonomous DevOps agent connected to:
- A `GitServer` (Resources: git diffs, commit history; Tools: `git_commit`, `create_branch`)
- A `TestServer` (Tools: `run_pytest`, `check_coverage`)
- A `TicketServer` (Resources: Jira tickets; Tools: `update_ticket_status`)

Sketch the step-by-step state sequence the agent loop follows to resolve the user request: *"Fix bug PROJ-42, verify tests pass, and mark ticket as In Review"*.

### Solution
1. **Step 1 (Read Ticket Context)**:
   - Tool/Resource: `resources/read {"uri": "jira://tickets/PROJ-42"}`
   - Route: `TicketServer`
   - Outcome: Extracts bug description and reproduction steps.
2. **Step 2 (Create Branch)**:
   - Tool: `tools/call {"name": "create_branch", "arguments": {"branch": "fix-proj-42"}}`
   - Route: `GitServer`
3. **Step 3 (Run Test Suite)**:
   - Tool: `tools/call {"name": "run_pytest", "arguments": {"test_path": "tests/test_auth.py"}}`
   - Route: `TestServer`
   - Outcome: Identifies test failure.
4. **Step 4 (Apply Fix & Re-test)**:
   - Agent synthesizes patch and calls `run_pytest` again until `isError: false` and all assertions pass.
5. **Step 5 (Commit & Update Ticket)**:
   - Tool: `tools/call {"name": "git_commit", "arguments": {"message": "Fix authentication bug PROJ-42"}}` on `GitServer`.
   - Tool: `tools/call {"name": "update_ticket_status", "arguments": {"ticket": "PROJ-42", "status": "In Review"}}` on `TicketServer`.
6. **Step 6 (Completion Response)**:
   - Agent outputs final summary to user with commit hash and ticket verification link.
