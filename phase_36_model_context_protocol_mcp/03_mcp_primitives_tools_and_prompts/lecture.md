# Topic 03: MCP Primitives — Tools & Prompts

---

## 1. The 12-Year-Old Intuitive Analogy: The Robot's Hands vs. The Recipe Cards

Imagine you build a smart home robot. 

In Topic 02, we gave the robot eyes to read books in the library (**Resources**). But reading about baking a cake doesn't actually bake the cake! To make things happen in the real world, your robot needs two different things:

1. **Tools (The Robotic Hands)**:
   The robot has robotic hands that can turn on the oven, stir the batter, and measure flour. But handing a robot hot tools is dangerous! Before the robot touches the oven, it must check a strict safety manual (the **JSON Schema**):
   *"You may only set the temperature between 100°F and 500°F, and you must specify Fahrenheit or Celsius."*
   If the robot tries to set the oven to "purple", the safety circuit trips before the kitchen catches fire. Furthermore, if the egg carton is empty, the robot doesn't explode—it reports back calmly: *"Tool error: Out of eggs"* (`isError: true`).

2. **Prompts (The Recipe Cards)**:
   A prompt is a reusable recipe card stored in a kitchen binder. Instead of having to explain to your robot every single day: *"Please wake up, clean the counter, verify the ingredients, preheat the oven, and call me when done"*, you simply pull out Recipe Card #4 (`/bake_cake`), fill in the blank `flavor="chocolate"`, and hand it to the robot.

In MCP:
- **Tools** are model-controlled executable actions that produce side effects in external systems.
- **Prompts** are user-controlled interactive workflow templates with slash-command style arguments.

---

## 2. Complete Mathematical & Protocol Symbol Glossary

| Symbol / Term | Formal Protocol Signature | Description / Role |
| :--- | :--- | :--- |
| $\mathcal{T}_{\text{tool}}$ | $\langle \text{name}, \text{desc}, \mathcal{S}_{\text{json}}\rangle$ | Tool definition schema including JSON Schema parameter specifications. |
| $\mathcal{S}_{\text{json}}$ | JSON Schema Draft-07 / 2020-12 | Formal schema defining expected argument types, required fields, and constraints. |
| $\text{tools/list}$ | $\emptyset \to \{\text{tools}: [\mathcal{T}_{\text{tool}}]\}$ | Catalog query returning all executable functions exposed by the MCP server. |
| $\text{tools/call}$ | $\{\text{name}: \Sigma^*, \text{arguments}: \mathcal{A}\} \to \mathcal{R}_{\text{tool}}$ | Invokes a specific tool with validated structured parameters. |
| $\mathcal{R}_{\text{tool}}$ | $\{\text{content}: [\text{ContentBlock}], \text{isError}: \mathbb{B}\}$ | Tool execution outcome. Note that errors do NOT trigger JSON-RPC crash codes. |
| $\text{isError}$ | $\text{true} \mid \text{false}$ | Protocol boolean flag distinguishing domain execution failure from server failure. |
| $\mathcal{P}_{\text{prompt}}$ | $\langle \text{name}, \text{desc}, \text{args}: [\mathcal{A}_{\text{desc}}]\rangle$ | User workflow template descriptor exposed via `prompts/list`. |
| $\text{prompts/get}$ | $\{\text{name}: \Sigma^*, \text{arguments}?: \mathcal{A}\} \to \{\text{messages}: [\mathcal{M}]\}$ | Expands a template into multi-turn messages ready for LLM consumption. |

---

## 3. Tools Architecture & Execution Pipeline

Unlike resources (which are read-only and passive), tools execute arbitrary computation, interact with operating system APIs, send emails, or commit database mutations.

```
┌────────────────────────────────────────────────────────────────────────┐
│                              HOST RUNTIME                              │
│                                                                        │
│   1. Tool Discovery:                                                   │
│      Client ─── tools/list ───► Server                                 │
│      Client ◄── {tools: [{name: "exec_sql", inputSchema: ...}]} ───────│
│                                                                        │
│   2. Model Reasoning:                                                  │
│      Host passes tool JSON schemas into LLM function calling context.  │
│      LLM emits tool invocation intent: {"name": "exec_sql", ...}       │
│                                                                        │
│   3. Host Permission Gate:                                             │
│      [HUMAN-IN-THE-LOOP APPROVAL / SECURITY POLICY CHECK]              │
│                                                                        │
│   4. Execution Dispatch:                                               │
│      Client ─── tools/call {name: "exec_sql", arguments: {...}} ──────►│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ JSON-RPC 2.0
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                              MCP SERVER                                │
│                                                                        │
│   Step A: Validate arguments against JSON Schema:                      │
│           type checks, required field checks, enum checks.             │
│   Step B: If invalid:                                                  │
│           return {content: [{text: "Schema error"}], isError: true}    │
│   Step C: Execute tool inside sandboxed boundary.                      │
│   Step D: Catch any runtime exception:                                 │
│           return {content: [{text: str(exc)}], isError: true}          │
│   Step E: Success:                                                     │
│           return {content: [{type: "text", text: result}],             │
│                   isError: false}                                      │
└────────────────────────────────────────────────────────────────────────┘
```

### 3.1. The Critical Distinction: Protocol Error vs. Tool `isError`
A fundamental architectural principle of MCP is the separation of **protocol failure** vs. **execution failure**:
- **Protocol Failure (JSON-RPC Error Object)**:
  Used *only* when the wire message is unparseable (`-32700`), the request is malformed (`-32600`), or the tool name does not exist on the server (`-32601`).
- **Tool Domain Failure (`isError: true` inside result)**:
  When an argument violates schema validation, an external API returns 404, or the python function raises an exception, the JSON-RPC response is **successful** (`"result": {...}`), but contains `"isError": true`.
  
  *Why?* Because an LLM needs to see the tool error message in its context window to correct its reasoning (e.g., self-healing parameter correction) without severing the host-server transport pipe!

```json
{
  "jsonrpc": "2.0",
  "id": 42,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "DivisionByZeroError: denominator cannot be zero. Please adjust parameters."
      }
    ],
    "isError": true
  }
}
```

---

## 4. Prompts Architecture: User-Driven Workflows

While Tools are chosen and triggered autonomously by the LLM, **Prompts** are designed for the **end user**. They represent slash commands, interactive macros, and structured playbooks (e.g. `/review-code`, `/explain-bug`, `/generate-sql`).

```
User (via Chat UI)
    │
    │ 1. Selects slash command: "/code_review code='x = 1' language='python'"
    ▼
Host Client
    │
    │ 2. Dispatches prompts/get:
    │    {"name": "code_review", "arguments": {"code": "x = 1", "language": "python"}}
    ▼
MCP Server
    │
    │ 3. Generates structured multi-turn message payload
    ▼
Host Client
    │
    │ 4. Receives message sequence:
    │    [{"role": "user", "content": {"type": "text", "text": "Please review..."}}]
    ▼
LLM Inference Loop
```

### 4.1. The `prompts/get` Wire Contract
```json
{
  "jsonrpc": "2.0",
  "id": "p-10",
  "method": "prompts/get",
  "params": {
    "name": "incident_triage",
    "arguments": {
      "service": "billing-api",
      "severity": "SEV-1"
    }
  }
}
```

Server response:
```json
{
  "jsonrpc": "2.0",
  "id": "p-10",
  "result": {
    "description": "Orchestrates an active production incident investigation",
    "messages": [
      {
        "role": "user",
        "content": {
          "type": "text",
          "text": "Active SEV-1 incident declared on service billing-api. Inspect recent deployment logs and alert metrics immediately."
        }
      }
    ]
  }
}
```

---

## 5. Security Boundaries & Sandboxing

Because tools have the power to alter external environments, MCP implements a strict defense-in-depth model:

1. **Host-Side Sandboxing**: Servers run in restricted child subprocesses or network namespaces with minimal privilege.
2. **Schema Sanitization**: Tool arguments are validated prior to execution, preventing command injection and directory traversal.
3. **Explicit User Consent**: Host applications typically enforce an interactive confirmation modal before executing high-impact tools (e.g. `delete_database`, `git_push`, `send_email`).
