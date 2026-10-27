# Phase 36: Model Context Protocol (MCP) & Extensible Agent Ecosystems

Welcome to **Phase 36** of the AI Complete Tutorial! This module provides an end-to-end, mathematically grounded, and rigorously verified exploration of the **Model Context Protocol (MCP)**—the open standard protocol connecting AI models to external tools, databases, and environments.

---

## Architecture Overview

```text
┌────────────────────────────────────────────────────────────────────────┐
│                              Host Client                               │
│  (Claude Desktop, Cursor IDE, Custom Python Agent Runtime)             │
│                                                                        │
│   ┌───────────────────────────┐         ┌──────────────────────────┐   │
│   │         LLM Core          │◄───────►│        MCP Client        │   │
│   │   (Reasoning & Intent)    │         │ (Multi-Server Manager)   │   │
│   └───────────────────────────┘         └────────────┬─────────────┘   │
└──────────────────────────────────────────────────────┼─────────────────┘
                                                       │
                      JSON-RPC 2.0 (stdio / HTTP+SSE)  │
                                                       │
         ┌─────────────────────────────────────────────┼─────────────────────────────┐
         ▼                                             ▼                             ▼
┌──────────────────────────────┐        ┌──────────────────────────────┐      ┌──────────────────────────────┐
│          MCP Server          │        │          MCP Server          │      │          MCP Server          │
│       (PostgreSQL DB)        │        │      (Local Filesystem)      │      │        (Compute Engine)      │
├──────────────────────────────┤        ├──────────────────────────────┤      ├──────────────────────────────┤
│ Resources: db://tables/...   │        │ Resources: file:///repo/...  │      │ Tools: run_regression        │
│ Tools: execute_query         │        │ Tools: write_file, git_diff  │      │ Prompts: /analyze_model      │
└──────────────────────────────┘        └──────────────────────────────┘      └──────────────────────────────┘
```

---

## Curriculum Structure

This phase is organized into five progressive, self-contained topics. Each topic contains a lecture note, interactive lecture web page, exercise problem sets with solutions, and a pure Python 3 standard library verification experiment:

| # | Topic Directory | Focus Areas & Primitives | Experiment Highlights |
|---|---|---|---|
| **01** | [`01_mcp_architecture_and_json_rpc_wire_format/`](01_mcp_architecture_and_json_rpc_wire_format/) | Client-Host-Server topology, JSON-RPC 2.0 framing, standard error codes ($-32700 \dots -32600$), batching | Pure Python JSON-RPC parser & router, stdio/SSE framing |
| **02** | [`02_mcp_primitives_resources_and_dynamic_context/`](02_mcp_primitives_resources_and_dynamic_context/) | URI schemes (`file:///`, `db://`), RFC 6570 templates, dynamic pub-sub notifications | Resource server, RFC 6570 variable matching, push updates |
| **03** | [`03_mcp_primitives_tools_and_prompts/`](03_mcp_primitives_tools_and_prompts/) | JSON Schema parameter contracts, tool execution error isolation (`isError`), prompt workflows | Parameter validator, execution sandbox, multi-turn prompt templates |
| **04** | [`04_mcp_transports_and_session_lifecycle/`](04_mcp_transports_and_session_lifecycle/) | Handshake sequence (`initialize` $\to$ `notifications/initialized`), capabilities, ping liveness | Lifecycle DFA state machine, capability negotiation, SSE stream |
| **05** | [`05_building_mcp_servers_and_multi_server_clients/`](05_building_mcp_servers_and_multi_server_clients/) | Stdio subprocess servers, multi-server orchestrator, catalog aggregation, agent reasoning loops | Dual-subprocess orchestration, cross-server routing, autonomous agent |

---

## Interactive Portal

Open `index.html` in any modern web browser or run a local HTTP server:

```bash
# From workspace root
open phase_36_model_context_protocol_mcp/index.html
```

The portal provides:
- Dark/Light theme toggle with system default preference detection.
- KaTeX mathematical and protocol rendering.
- Collapsible exercise solutions.
- Clickable links to all lectures, HTML pages, and runnable experiments.

---

## Verification & Test Suite

All experiments are implemented exclusively with the Python 3 standard library (zero external pip dependencies) and run with exit code 0:

```bash
# Run Topic 01: Wire Format & JSON-RPC 2.0 Parser
python3 phase_36_model_context_protocol_mcp/01_mcp_architecture_and_json_rpc_wire_format/experiment.py

# Run Topic 02: Resources & Dynamic Subscriptions
python3 phase_36_model_context_protocol_mcp/02_mcp_primitives_resources_and_dynamic_context/experiment.py

# Run Topic 03: Tools & Prompts Registry
python3 phase_36_model_context_protocol_mcp/03_mcp_primitives_tools_and_prompts/experiment.py

# Run Topic 04: Transports & Lifecycle State Machine
python3 phase_36_model_context_protocol_mcp/04_mcp_transports_and_session_lifecycle/experiment.py

# Run Topic 05: Multi-Server Subprocess Orchestrator
python3 phase_36_model_context_protocol_mcp/05_building_mcp_servers_and_multi_server_clients/experiment.py
```
