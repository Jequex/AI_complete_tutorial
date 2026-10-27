#!/usr/bin/env python3
"""
Phase 36 - Topic 05: Building MCP Servers & Multi-Server Clients
Pure Python 3 Standard Library Implementation

Demonstrates:
- Spawning and managing multiple isolated MCP stdio subprocess servers
- Capability handshake over real OS pipes (stdin/stdout)
- Host-level tool and resource catalog aggregation based on negotiated capabilities
- Fallback handler returning -32601 Method Not Found for unhandled requests
- Autonomous agent decision loop routing intents across distinct servers
- Graceful shutdown and process lifecycle cleanup
"""

import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple


# Code for Subprocess Server 1: Resource Provider
RESOURCE_SERVER_CODE = """
import sys
import json

def send(msg):
    sys.stdout.write(json.dumps(msg) + "\\n")
    sys.stdout.flush()

RESOURCES = {
    "file:///reports/q3_revenue.json": {
        "name": "Q3 Revenue Report",
        "mimeType": "application/json",
        "text": json.dumps({"q1": 150000, "q2": 185000, "q3": 210000, "units": "USD"})
    },
    "file:///company/policies.md": {
        "name": "Corporate Policies",
        "mimeType": "text/markdown",
        "text": "# Data Security Policy\\nAll aggregates must be rounded to nearest integer."
    }
}

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
    except Exception:
        continue

    method = req.get("method")
    req_id = req.get("id")

    if method == "initialize":
        send({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"resources": {"subscribe": True}},
                "serverInfo": {"name": "SubprocessResourceServer", "version": "1.0.0"}
            }
        })
    elif method == "notifications/initialized":
        pass  # Acknowledged
    elif method == "resources/list":
        items = [{"uri": k, "name": v["name"], "mimeType": v["mimeType"]} for k, v in RESOURCES.items()]
        send({"jsonrpc": "2.0", "id": req_id, "result": {"resources": items}})
    elif method == "resources/read":
        uri = (req.get("params") or {}).get("uri")
        if uri in RESOURCES:
            send({"jsonrpc": "2.0", "id": req_id, "result": {"contents": [{"uri": uri, "mimeType": RESOURCES[uri]["mimeType"], "text": RESOURCES[uri]["text"]}]}})
        else:
            send({"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": "Resource not found"}})
    elif method == "ping":
        send({"jsonrpc": "2.0", "id": req_id, "result": {}})
    elif method == "shutdown":
        send({"jsonrpc": "2.0", "id": req_id, "result": {"status": "ok"}})
        break
    else:
        if req_id is not None:
            send({"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Method {method} not found"}})
"""

# Code for Subprocess Server 2: Compute & Tool Provider
TOOL_SERVER_CODE = """
import sys
import json

def send(msg):
    sys.stdout.write(json.dumps(msg) + "\\n")
    sys.stdout.flush()

TOOLS = [
    {
        "name": "aggregate_stats",
        "description": "Calculates sum, mean, and growth rate across numerical series",
        "inputSchema": {
            "type": "object",
            "properties": {
                "series": {"type": "array"}
            },
            "required": ["series"]
        }
    },
    {
        "name": "currency_converter",
        "description": "Converts currency value between USD and EUR",
        "inputSchema": {
            "type": "object",
            "properties": {
                "amount": {"type": "number"},
                "rate": {"type": "number"}
            },
            "required": ["amount", "rate"]
        }
    }
]

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
    except Exception:
        continue

    method = req.get("method")
    req_id = req.get("id")

    if method == "initialize":
        send({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "SubprocessToolServer", "version": "1.0.0"}
            }
        })
    elif method == "notifications/initialized":
        pass  # Acknowledged
    elif method == "tools/list":
        send({"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}})
    elif method == "tools/call":
        params = req.get("params") or {}
        tool_name = params.get("name")
        args = params.get("arguments") or {}

        if tool_name == "aggregate_stats":
            series = args.get("series", [])
            total = sum(series)
            mean = total / len(series) if series else 0
            growth = ((series[-1] - series[0]) / series[0]) * 100 if len(series) > 1 and series[0] != 0 else 0
            stats = {"sum": total, "mean": mean, "growth_percent": round(growth, 2)}
            send({"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": json.dumps(stats)}], "isError": False}})
        elif tool_name == "currency_converter":
            amount = args.get("amount", 0)
            rate = args.get("rate", 1.0)
            converted = round(amount * rate, 2)
            send({"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": str(converted)}], "isError": False}})
        else:
            send({"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Unknown tool {tool_name}"}})
    elif method == "ping":
        send({"jsonrpc": "2.0", "id": req_id, "result": {}})
    elif method == "shutdown":
        send({"jsonrpc": "2.0", "id": req_id, "result": {"status": "ok"}})
        break
    else:
        if req_id is not None:
            send({"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Method {method} not found"}})
"""


class StdioMCPClient:
    """Manages an active stdio connection to a single MCP subprocess server."""

    def __init__(self, name: str, script_code: str):
        self.name = name
        self.script_code = script_code
        self.process: Optional[subprocess.Popen] = None
        self._request_counter = 0
        self.server_capabilities: Dict[str, Any] = {}

    def start(self) -> Dict[str, Any]:
        """Spawns the child process and performs the MCP initialize handshake."""
        self.process = subprocess.Popen(
            [sys.executable, "-u", "-c", self.script_code],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        # 1. Send initialize request
        init_res = self.send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "MultiServerHostClient", "version": "1.0.0"}
        })

        # 2. Store negotiated capabilities
        res_payload = init_res.get("result", {})
        self.server_capabilities = res_payload.get("capabilities", {})

        # 3. Send notifications/initialized
        self.send_notification("notifications/initialized")
        return res_payload

    def send_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.process or self.process.poll() is not None:
            raise RuntimeError(f"Server '{self.name}' process is not running.")

        self._request_counter += 1
        req_id = f"{self.name}-req-{self._request_counter}"
        payload = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            payload["params"] = params

        wire_line = json.dumps(payload) + "\n"
        self.process.stdin.write(wire_line)
        self.process.stdin.flush()

        # Read line response from stdout
        resp_line = self.process.stdout.readline()
        if not resp_line:
            raise EOFError(f"Server '{self.name}' closed stdout stream unexpectedly.")
        return json.loads(resp_line.strip())

    def send_notification(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        if not self.process or self.process.poll() is not None:
            raise RuntimeError(f"Server '{self.name}' is not running.")
        payload = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()

    def close(self) -> None:
        if self.process and self.process.poll() is None:
            try:
                self.send_request("shutdown")
            except Exception:
                pass
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()


class MultiServerHostOrchestrator:
    """
    Host Client managing multiple distinct MCP servers.
    Aggregates resource & tool catalogs and provides a unified agent execution router.
    """

    def __init__(self):
        self.clients: Dict[str, StdioMCPClient] = {}
        self.tool_routes: Dict[str, str] = {}      # tool_name -> server_name
        self.resource_routes: Dict[str, str] = {}  # uri -> server_name

    def register_server(self, server_name: str, script_code: str) -> None:
        client = StdioMCPClient(server_name, script_code)
        init_data = client.start()
        self.clients[server_name] = client
        print(f"  ✓ Connected to '{server_name}' [Version: {init_data.get('serverInfo', {}).get('version', 'unknown')}]")

    def aggregate_catalogs(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        all_tools = []
        all_resources = []

        for server_name, client in self.clients.items():
            # Query tools if capability declared
            if "tools" in client.server_capabilities:
                try:
                    t_resp = client.send_request("tools/list")
                    tools = t_resp.get("result", {}).get("tools", [])
                    for t in tools:
                        self.tool_routes[t["name"]] = server_name
                        all_tools.append({**t, "server": server_name})
                except Exception as e:
                    print(f"    Warning: Could not list tools for {server_name}: {e}")

            # Query resources if capability declared
            if "resources" in client.server_capabilities:
                try:
                    r_resp = client.send_request("resources/list")
                    resources = r_resp.get("result", {}).get("resources", [])
                    for r in resources:
                        self.resource_routes[r["uri"]] = server_name
                        all_resources.append({**r, "server": server_name})
                except Exception as e:
                    print(f"    Warning: Could not list resources for {server_name}: {e}")

        return all_tools, all_resources

    def read_resource(self, uri: str) -> Dict[str, Any]:
        server_name = self.resource_routes.get(uri)
        if not server_name:
            raise KeyError(f"No registered server owns resource URI: {uri}")
        client = self.clients[server_name]
        resp = client.send_request("resources/read", {"uri": uri})
        return resp.get("result", {})

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        server_name = self.tool_routes.get(tool_name)
        if not server_name:
            raise KeyError(f"No registered server provides tool: {tool_name}")
        client = self.clients[server_name]
        resp = client.send_request("tools/call", {"name": tool_name, "arguments": arguments})
        return resp.get("result", {})

    def shutdown_all(self) -> None:
        for client in self.clients.values():
            client.close()


# ---------------- Autonomous Agent Simulation ----------------
def simulate_autonomous_agent(orchestrator: MultiServerHostOrchestrator, user_prompt: str) -> Dict[str, Any]:
    """
    Simulates an autonomous multi-server reasoning loop:
    1. Inspects query and locates authoritative Q3 revenue report from Resource Server.
    2. Reads and parses JSON data.
    3. Selects 'aggregate_stats' tool on Tool Server.
    4. Converts aggregated result to EUR using 'currency_converter' tool on Tool Server.
    5. Returns synthesized answer.
    """
    print(f"\n[Agent Goal] '{user_prompt}'")

    # Step 1: Agent finds resource
    target_uri = "file:///reports/q3_revenue.json"
    print(f"  Step 1: Reading data from Resource Server ({target_uri})...")
    res_data = orchestrator.read_resource(target_uri)
    raw_text = res_data["contents"][0]["text"]
    parsed_json = json.loads(raw_text)
    print(f"    Raw Context Retrieved: {parsed_json}")

    # Step 2: Agent extracts quarterly series [150000, 185000, 210000]
    series = [parsed_json["q1"], parsed_json["q2"], parsed_json["q3"]]
    print(f"  Step 2: Dispatching 'aggregate_stats' tool to Tool Server with series={series}...")
    tool_res1 = orchestrator.call_tool("aggregate_stats", {"series": series})
    stats_output = json.loads(tool_res1["content"][0]["text"])
    print(f"    Tool Aggregation Output: {stats_output}")

    # Step 3: Convert USD total to EUR (exchange rate 0.92)
    usd_total = stats_output["sum"]
    print(f"  Step 3: Dispatching 'currency_converter' tool (Amount: ${usd_total}, Rate: 0.92)...")
    tool_res2 = orchestrator.call_tool("currency_converter", {"amount": usd_total, "rate": 0.92})
    eur_total = float(tool_res2["content"][0]["text"])
    print(f"    Converted Total in EUR: €{eur_total:,.2f}")

    return {
        "revenue_usd": usd_total,
        "revenue_eur": eur_total,
        "growth_percent": stats_output["growth_percent"],
        "mean_quarterly": stats_output["mean"]
    }


# ---------------- Verification & Demonstration ----------------
def run_tests():
    print("=" * 70)
    print("Phase 36 Topic 05: Multi-Server Host Orchestrator Verification")
    print("=" * 70)

    orchestrator = MultiServerHostOrchestrator()

    try:
        print("\n[1] Launching and Initializing Two Subprocess MCP Servers...")
        orchestrator.register_server("resource_server", RESOURCE_SERVER_CODE)
        orchestrator.register_server("tool_server", TOOL_SERVER_CODE)

        print("\n[2] Aggregating Cross-Server Catalogs...")
        tools, resources = orchestrator.aggregate_catalogs()
        assert len(tools) == 2, f"Expected 2 tools, found {len(tools)}"
        assert len(resources) == 2, f"Expected 2 resources, found {len(resources)}"

        tool_names = [t["name"] for t in tools]
        assert "aggregate_stats" in tool_names
        assert "currency_converter" in tool_names

        resource_uris = [r["uri"] for r in resources]
        assert "file:///reports/q3_revenue.json" in resource_uris
        print(f"  ✓ Unified Tools Catalog: {tool_names} (Routed to tool_server)")
        print(f"  ✓ Unified Resources Catalog: {resource_uris} (Routed to resource_server)")

        print("\n[3] Executing Multi-Server Autonomous Agent Loop...")
        result = simulate_autonomous_agent(
            orchestrator,
            user_prompt="Summarize Q3 financial revenue and convert total earnings to EUR"
        )

        assert result["revenue_usd"] == 545000  # 150000 + 185000 + 210000
        assert result["revenue_eur"] == 501400.0  # 545000 * 0.92
        assert result["growth_percent"] == 40.0   # ((210000 - 150000) / 150000) * 100
        print(f"\n  ✓ Agent successfully synthesized multi-server outputs:")
        print(f"    - USD Total: ${result['revenue_usd']:,}")
        print(f"    - EUR Total: €{result['revenue_eur']:,}")
        print(f"    - 3-Quarter Growth: {result['growth_percent']}%")

        print("\n[4] Testing Graceful Subprocess Shutdown...")
        orchestrator.shutdown_all()
        for name, client in orchestrator.clients.items():
            assert client.process.poll() is not None
            print(f"  ✓ Server '{name}' cleanly terminated with exit code {client.process.poll()}")

    finally:
        orchestrator.shutdown_all()

    print("\n" + "=" * 70)
    print("ALL MULTI-SERVER MCP ORCHESTRATOR TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
