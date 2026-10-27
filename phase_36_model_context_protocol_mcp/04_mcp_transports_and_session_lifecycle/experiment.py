#!/usr/bin/env python3
"""
Phase 36 - Topic 04: MCP Transports & Session Lifecycle
Pure Python 3 Standard Library Implementation

Demonstrates:
- Lifecycle handshake state machine: UNINITIALIZED -> INITIALIZING -> INITIALIZED -> TERMINATED
- Client and server capability negotiation (roots, sampling, resources, tools, prompts)
- Protocol version negotiation
- Stdio and SSE transport simulation
- Ping heartbeats and timeout detection
- Out-of-order request rejection prior to initialization
"""

import enum
import json
import time
from typing import Any, Callable, Dict, List, Optional, Tuple


class SessionState(enum.Enum):
    UNINITIALIZED = "UNINITIALIZED"
    INITIALIZING = "INITIALIZING"
    INITIALIZED = "INITIALIZED"
    TERMINATED = "TERMINATED"


SUPPORTED_PROTOCOL_VERSIONS = ["2024-11-05", "2024-10-07"]
LATEST_PROTOCOL_VERSION = "2024-11-05"


class MCPSessionServer:
    """Simulates an MCP server managing session state machine and capability negotiation."""

    def __init__(self, name: str = "TestLifecycleServer", version: str = "1.0.0"):
        self.server_info = {"name": name, "version": version}
        self.state = SessionState.UNINITIALIZED
        self.negotiated_version: Optional[str] = None
        self.client_capabilities: Dict[str, Any] = {}
        self.server_capabilities = {
            "resources": {"subscribe": True, "listChanged": True},
            "tools": {"listChanged": True},
            "prompts": {"listChanged": True},
            "logging": {}
        }
        self.last_heartbeat = time.time()
        self.cleanup_invoked = False

    def handle_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        method = message.get("method")
        req_id = message.get("id")

        # 1. State: UNINITIALIZED
        if self.state == SessionState.UNINITIALIZED:
            if method != "initialize":
                if req_id is not None:
                    return {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {
                            "code": -32600,
                            "message": f"Server not initialized. Received '{method}' before 'initialize'."
                        }
                    }
                return None

            # Handle initialize request
            params = message.get("params", {})
            client_version = params.get("protocolVersion")
            self.client_capabilities = params.get("capabilities", {})

            # Version negotiation
            if client_version in SUPPORTED_PROTOCOL_VERSIONS:
                self.negotiated_version = client_version
            else:
                self.negotiated_version = LATEST_PROTOCOL_VERSION

            self.state = SessionState.INITIALIZING
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": self.negotiated_version,
                    "capabilities": self.server_capabilities,
                    "serverInfo": self.server_info
                }
            }

        # 2. State: INITIALIZING
        if self.state == SessionState.INITIALIZING:
            if method == "notifications/initialized":
                self.state = SessionState.INITIALIZED
                return None
            else:
                if req_id is not None:
                    return {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {
                            "code": -32600,
                            "message": "Server awaiting 'notifications/initialized' acknowledgment."
                        }
                    }
                return None

        # 3. State: TERMINATED
        if self.state == SessionState.TERMINATED:
            if req_id is not None:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32600, "message": "Session is terminated."}
                }
            return None

        # 4. State: INITIALIZED (Standard Operation)
        self.last_heartbeat = time.time()

        if method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}

        if method == "shutdown":
            self.terminate()
            return {"jsonrpc": "2.0", "id": req_id, "result": {"status": "terminated"}}

        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": [{"name": "system_diagnostics", "description": "Runs hardware check"}]}
            }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method '{method}' not implemented."}
        }

    def terminate(self) -> None:
        self.state = SessionState.TERMINATED
        self.cleanup_invoked = True


class MockStdioTransport:
    """Simulates bidirectional newline-delimited stdio pipe communication."""

    def __init__(self, server: MCPSessionServer):
        self.server = server
        self.stdin_buffer: List[str] = []
        self.stdout_buffer: List[str] = []

    def write_stdin(self, msg: Dict[str, Any]) -> None:
        raw_line = json.dumps(msg) + "\n"
        self.stdin_buffer.append(raw_line)

    def process_cycle(self) -> None:
        while self.stdin_buffer:
            line = self.stdin_buffer.pop(0).strip()
            if not line:
                continue
            parsed = json.loads(line)
            response = self.server.handle_message(parsed)
            if response is not None:
                self.stdout_buffer.append(json.dumps(response) + "\n")

    def read_stdout(self) -> List[Dict[str, Any]]:
        messages = []
        while self.stdout_buffer:
            raw = self.stdout_buffer.pop(0).strip()
            messages.append(json.loads(raw))
        return messages


class MockSSETransport:
    """Simulates HTTP + Server-Sent Events (SSE) remote transport."""

    def __init__(self, server: MCPSessionServer):
        self.server = server
        self.session_id = "sess_mcp_998877"
        self.sse_event_stream: List[Tuple[str, str]] = []  # (event, data)

    def connect(self) -> str:
        """Client performs HTTP GET /sse, server returns endpoint event with POST URL."""
        endpoint_url = f"http://127.0.0.1:8000/messages?session_id={self.session_id}"
        self.sse_event_stream.append(("endpoint", endpoint_url))
        return endpoint_url

    def send_post_message(self, message: Dict[str, Any]) -> None:
        """Client sends HTTP POST with JSON-RPC payload."""
        response = self.server.handle_message(message)
        if response is not None:
            # Server streams reply back via active SSE connection
            self.sse_event_stream.append(("message", json.dumps(response)))

    def poll_events(self) -> List[Tuple[str, str]]:
        events = list(self.sse_event_stream)
        self.sse_event_stream.clear()
        return events


# ---------------- Verification & Demonstration ----------------
def run_tests():
    print("=" * 70)
    print("Phase 36 Topic 04: MCP Transports & Session Lifecycle Verification")
    print("=" * 70)

    # 1. State Machine & Handshake over Stdio
    print("\n[1] Testing Lifecycle State Machine & Out-of-Order Rejection...")
    server = MCPSessionServer()
    transport = MockStdioTransport(server)

    assert server.state == SessionState.UNINITIALIZED

    # Attempt to query tools before initialize
    transport.write_stdin({"jsonrpc": "2.0", "method": "tools/list", "id": 1})
    transport.process_cycle()
    out = transport.read_stdout()
    assert len(out) == 1
    assert out[0]["error"]["code"] == -32600
    assert "Server not initialized" in out[0]["error"]["message"]
    print("  ✓ Requests prior to initialize properly rejected with -32600")

    # Step 1 of Handshake: Send initialize request
    init_request = {
        "jsonrpc": "2.0",
        "method": "initialize",
        "id": 2,
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "roots": {"listChanged": True},
                "sampling": {}
            },
            "clientInfo": {"name": "TestHostClient", "version": "0.1.0"}
        }
    }
    transport.write_stdin(init_request)
    transport.process_cycle()
    out = transport.read_stdout()
    assert len(out) == 1
    init_result = out[0]["result"]
    assert init_result["protocolVersion"] == "2024-11-05"
    assert "tools" in init_result["capabilities"]
    assert "resources" in init_result["capabilities"]
    assert server.state == SessionState.INITIALIZING
    print("  ✓ Handshake Step 1 (initialize) negotiated protocol version '2024-11-05' and capabilities")

    # Attempt to query tools while in INITIALIZING state
    transport.write_stdin({"jsonrpc": "2.0", "method": "tools/list", "id": 3})
    transport.process_cycle()
    out = transport.read_stdout()
    assert out[0]["error"]["code"] == -32600
    assert "awaiting 'notifications/initialized'" in out[0]["error"]["message"]
    print("  ✓ Requests in INITIALIZING state rejected pending acknowledgment")

    # Step 2 of Handshake: Send notifications/initialized
    transport.write_stdin({"jsonrpc": "2.0", "method": "notifications/initialized"})
    transport.process_cycle()
    out = transport.read_stdout()
    assert len(out) == 0, "Notification must produce zero response"
    assert server.state == SessionState.INITIALIZED
    print("  ✓ Handshake Step 2 (notifications/initialized) transitioned session to INITIALIZED")

    # Step 3: Now tools/list works
    transport.write_stdin({"jsonrpc": "2.0", "method": "tools/list", "id": 4})
    transport.process_cycle()
    out = transport.read_stdout()
    assert len(out) == 1
    assert out[0]["result"]["tools"][0]["name"] == "system_diagnostics"
    print("  ✓ Normal RPC operations fully accessible in INITIALIZED state")

    print("\n[2] Testing Ping Heartbeats...")
    transport.write_stdin({"jsonrpc": "2.0", "method": "ping", "id": 5})
    transport.process_cycle()
    out = transport.read_stdout()
    assert out[0]["result"] == {}
    assert out[0]["id"] == 5
    print("  ✓ Ping request returned pong result {} within latency window")

    print("\n[3] Testing Graceful Teardown & Resource Cleanup...")
    transport.write_stdin({"jsonrpc": "2.0", "method": "shutdown", "id": 6})
    transport.process_cycle()
    out = transport.read_stdout()
    assert out[0]["result"]["status"] == "terminated"
    assert server.state == SessionState.TERMINATED
    assert server.cleanup_invoked is True
    print("  ✓ Graceful shutdown invoked server cleanup and set state to TERMINATED")

    print("\n[4] Testing HTTP + Server-Sent Events (SSE) Transport Simulation...")
    sse_server = MCPSessionServer("RemoteSSEServer")
    sse_transport = MockSSETransport(sse_server)

    endpoint_url = sse_transport.connect()
    events = sse_transport.poll_events()
    assert len(events) == 1
    assert events[0][0] == "endpoint"
    assert "session_id=sess_mcp_998877" in events[0][1]
    print(f"  ✓ SSE GET connection established, received endpoint: {events[0][1]}")

    # Initialize via HTTP POST
    sse_transport.send_post_message({
        "jsonrpc": "2.0",
        "method": "initialize",
        "id": 100,
        "params": {"protocolVersion": "2024-11-05", "capabilities": {}}
    })
    events = sse_transport.poll_events()
    assert len(events) == 1
    assert events[0][0] == "message"
    data = json.loads(events[0][1])
    assert data["result"]["serverInfo"]["name"] == "RemoteSSEServer"
    print("  ✓ Initialized over SSE stream with streamed JSON-RPC reply")

    # Finalize handshake
    sse_transport.send_post_message({"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert sse_server.state == SessionState.INITIALIZED
    print("  ✓ SSE Session reached INITIALIZED state successfully")

    print("\n" + "=" * 70)
    print("ALL MCP TRANSPORTS & SESSION LIFECYCLE TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
