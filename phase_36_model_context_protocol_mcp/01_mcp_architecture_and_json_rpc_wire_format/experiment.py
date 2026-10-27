#!/usr/bin/env python3
"""
Phase 36 - Topic 01: MCP Architecture & JSON-RPC 2.0 Wire Format
Pure Python 3 Standard Library Implementation

Demonstrates:
- JSON-RPC 2.0 protocol specifications (request, response, error, notification)
- Standard error codes (-32700, -32600, -32601, -32602, -32603)
- Batch request parsing and response aggregation
- Stdio line framing (\n) and SSE chunk framing
- Bidirectional message routing and dispatching
"""

import json
import sys
from typing import Any, Callable, Dict, List, Optional, Union, Tuple


# Standard JSON-RPC 2.0 Error Codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

ERROR_MESSAGES = {
    PARSE_ERROR: "Parse error: invalid JSON received.",
    INVALID_REQUEST: "Invalid Request: The JSON sent is not a valid Request object.",
    METHOD_NOT_FOUND: "Method not found: The method does not exist / is not available.",
    INVALID_PARAMS: "Invalid params: Invalid method parameter(s).",
    INTERNAL_ERROR: "Internal error: Internal JSON-RPC error.",
}


def make_error_object(code: int, message: Optional[str] = None, data: Any = None) -> Dict[str, Any]:
    err = {
        "code": code,
        "message": message if message is not None else ERROR_MESSAGES.get(code, "Unknown error")
    }
    if data is not None:
        err["data"] = data
    return err


def make_response(req_id: Any, result: Any = None, error: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    resp: Dict[str, Any] = {"jsonrpc": "2.0", "id": req_id}
    if error is not None:
        resp["error"] = error
    else:
        resp["result"] = result
    return resp


class StdioFramer:
    """Newline-delimited JSON framing for standard I/O communication."""

    @staticmethod
    def frame(payload: Union[Dict[str, Any], List[Dict[str, Any]]]) -> str:
        return json.dumps(payload, separators=(',', ':')) + "\n"

    @staticmethod
    def unframe(raw_bytes_or_str: str) -> List[str]:
        lines = [line.strip() for line in raw_bytes_or_str.strip().split("\n") if line.strip()]
        return lines


class SSEFramer:
    """Server-Sent Events (SSE) message chunking and parsing."""

    @staticmethod
    def frame(payload: Union[Dict[str, Any], List[Dict[str, Any]]], event: str = "message") -> str:
        data_str = json.dumps(payload, separators=(',', ':'))
        return f"event: {event}\ndata: {data_str}\n\n"

    @staticmethod
    def unframe(raw_sse: str) -> List[Tuple[str, str]]:
        chunks = raw_sse.strip().split("\n\n")
        events: List[Tuple[str, str]] = []
        for chunk in chunks:
            if not chunk.strip():
                continue
            event_type = "message"
            data_parts = []
            for line in chunk.split("\n"):
                if line.startswith("event:"):
                    event_type = line[len("event:"):].strip()
                elif line.startswith("data:"):
                    data_parts.append(line[len("data:"):].strip())
            if data_parts:
                events.append((event_type, "\n".join(data_parts)))
        return events


class JSONRPCRouter:
    """
    JSON-RPC 2.0 compliant router and dispatcher.
    Supports method registration, parameter validation, error mapping, and batch processing.
    """

    def __init__(self):
        self._handlers: Dict[str, Callable[[Any], Any]] = {}
        self._notification_handlers: Dict[str, Callable[[Any], None]] = {}

    def register_method(self, method_name: str, handler: Callable[[Any], Any]) -> None:
        self._handlers[method_name] = handler

    def register_notification(self, method_name: str, handler: Callable[[Any], None]) -> None:
        self._notification_handlers[method_name] = handler

    def dispatch_raw(self, raw_input: str) -> Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]:
        """Parses raw string and executes single or batch JSON-RPC request."""
        try:
            parsed = json.loads(raw_input)
        except json.JSONDecodeError as exc:
            return make_response(None, error=make_error_object(PARSE_ERROR, data=str(exc)))

        if isinstance(parsed, list):
            if len(parsed) == 0:
                return make_response(None, error=make_error_object(INVALID_REQUEST, "Batch array cannot be empty"))
            responses = []
            for item in parsed:
                res = self._process_single_object(item)
                if res is not None:  # Notifications return None
                    responses.append(res)
            return responses if responses else None
        elif isinstance(parsed, dict):
            return self._process_single_object(parsed)
        else:
            return make_response(None, error=make_error_object(INVALID_REQUEST, "Payload must be object or array"))

    def _process_single_object(self, obj: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(obj, dict):
            return make_response(None, error=make_error_object(INVALID_REQUEST, "Request must be a JSON object"))

        # Protocol version check
        if obj.get("jsonrpc") != "2.0":
            req_id = obj.get("id") if isinstance(obj.get("id"), (str, int)) else None
            return make_response(req_id, error=make_error_object(INVALID_REQUEST, "Missing or invalid 'jsonrpc' version. Must be '2.0'."))

        method = obj.get("method")
        if not isinstance(method, str) or not method:
            req_id = obj.get("id") if isinstance(obj.get("id"), (str, int)) else None
            return make_response(req_id, error=make_error_object(INVALID_REQUEST, "Missing or invalid 'method' field."))

        is_notification = "id" not in obj
        req_id = obj.get("id")

        params = obj.get("params")
        if params is not None and not isinstance(params, (dict, list)):
            if is_notification:
                return None
            return make_response(req_id, error=make_error_object(INVALID_PARAMS, "'params' must be structured object or array."))

        # Notification handling
        if is_notification:
            if method in self._notification_handlers:
                try:
                    self._notification_handlers[method](params)
                except Exception:
                    pass  # Notifications suppress errors per spec
            elif method in self._handlers:
                try:
                    self._handlers[method](params)
                except Exception:
                    pass
            return None

        # Regular request handling
        if method not in self._handlers:
            return make_response(req_id, error=make_error_object(METHOD_NOT_FOUND, f"Method '{method}' not found."))

        handler = self._handlers[method]
        try:
            result = handler(params)
            return make_response(req_id, result=result)
        except TypeError as te:
            return make_response(req_id, error=make_error_object(INVALID_PARAMS, f"Parameter mismatch: {te}"))
        except ValueError as ve:
            return make_response(req_id, error=make_error_object(INVALID_PARAMS, f"Value validation failed: {ve}"))
        except Exception as e:
            return make_response(req_id, error=make_error_object(INTERNAL_ERROR, f"Internal execution fault: {e}"))


# ---------------- Verification & Demonstration ----------------
def run_tests():
    print("=" * 70)
    print("Phase 36 Topic 01: MCP JSON-RPC 2.0 Wire Format Verification")
    print("=" * 70)

    router = JSONRPCRouter()

    # Register sample RPC methods
    def handle_ping(params):
        return {"status": "pong", "protocol": "mcp/2024-11-05"}

    def handle_echo(params):
        if not isinstance(params, dict) or "message" not in params:
            raise ValueError("Expected parameter 'message' in params dictionary")
        return {"echo": params["message"]}

    def handle_add(params):
        if isinstance(params, list):
            if len(params) != 2:
                raise ValueError("Expected exactly 2 operands in array")
            return params[0] + params[1]
        elif isinstance(params, dict):
            if "a" not in params or "b" not in params:
                raise ValueError("Expected keys 'a' and 'b'")
            return params["a"] + params["b"]
        raise TypeError("Params must be array or object")

    def handle_buggy(params):
        raise RuntimeError("Simulated crash in server business logic")

    notifications_received: List[Dict[str, Any]] = []

    def handle_log(params):
        notifications_received.append(params)

    router.register_method("ping", handle_ping)
    router.register_method("echo", handle_echo)
    router.register_method("add", handle_add)
    router.register_method("buggy", handle_buggy)
    router.register_notification("notifications/log", handle_log)

    print("\n[1] Testing Standard Request & Response...")
    req1 = json.dumps({"jsonrpc": "2.0", "method": "ping", "id": 1})
    resp1 = router.dispatch_raw(req1)
    assert resp1 == {"jsonrpc": "2.0", "id": 1, "result": {"status": "pong", "protocol": "mcp/2024-11-05"}}
    print("  ✓ ping request responded correctly")

    req2 = json.dumps({"jsonrpc": "2.0", "method": "add", "params": [15, 27], "id": "req-add-1"})
    resp2 = router.dispatch_raw(req2)
    assert resp2 == {"jsonrpc": "2.0", "id": "req-add-1", "result": 42}
    print("  ✓ positional parameter addition succeeded: 15 + 27 = 42")

    req3 = json.dumps({"jsonrpc": "2.0", "method": "add", "params": {"a": 100, "b": 250}, "id": "req-add-2"})
    resp3 = router.dispatch_raw(req3)
    assert resp3 == {"jsonrpc": "2.0", "id": "req-add-2", "result": 350}
    print("  ✓ named parameter addition succeeded: 100 + 250 = 350")

    print("\n[2] Testing Standard Error Conditions...")
    # Parse Error (-32700)
    resp_parse = router.dispatch_raw('{"jsonrpc": "2.0", "method": "ping", "broken_json')
    assert resp_parse["error"]["code"] == PARSE_ERROR
    print("  ✓ Error -32700 (Parse Error) properly returned for invalid JSON syntax")

    # Invalid Request (-32600)
    resp_inv1 = router.dispatch_raw('{"method": "ping", "id": 2}')  # missing jsonrpc
    assert resp_inv1["error"]["code"] == INVALID_REQUEST
    resp_inv2 = router.dispatch_raw('{"jsonrpc": "1.0", "method": "ping", "id": 3}')  # bad version
    assert resp_inv2["error"]["code"] == INVALID_REQUEST
    print("  ✓ Error -32600 (Invalid Request) properly returned for missing/invalid 'jsonrpc' version")

    # Method Not Found (-32601)
    resp_notfound = router.dispatch_raw('{"jsonrpc": "2.0", "method": "unknown_tool", "id": 4}')
    assert resp_notfound["error"]["code"] == METHOD_NOT_FOUND
    print("  ✓ Error -32601 (Method Not Found) returned for unregistered method")

    # Invalid Params (-32602)
    resp_invparams = router.dispatch_raw('{"jsonrpc": "2.0", "method": "echo", "params": {"wrong_field": 123}, "id": 5}')
    assert resp_invparams["error"]["code"] == INVALID_PARAMS
    print("  ✓ Error -32602 (Invalid Params) returned for missing parameter fields")

    # Internal Error (-32603)
    resp_internal = router.dispatch_raw('{"jsonrpc": "2.0", "method": "buggy", "id": 6}')
    assert resp_internal["error"]["code"] == INTERNAL_ERROR
    print("  ✓ Error -32603 (Internal Error) returned for unhandled internal exceptions")

    print("\n[3] Testing Notification Processing (No ID)...")
    notif = json.dumps({"jsonrpc": "2.0", "method": "notifications/log", "params": {"level": "info", "msg": "Client ready"}})
    resp_notif = router.dispatch_raw(notif)
    assert resp_notif is None, "Notifications must produce no response payload"
    assert len(notifications_received) == 1
    assert notifications_received[0]["msg"] == "Client ready"
    print("  ✓ Notification processed silently without returning response object")

    print("\n[4] Testing Batch Requests & Mixed Notifications...")
    batch = json.dumps([
        {"jsonrpc": "2.0", "method": "ping", "id": 10},
        {"jsonrpc": "2.0", "method": "notifications/log", "params": {"batch": True}},
        {"jsonrpc": "2.0", "method": "add", "params": [10, 20], "id": 11},
        {"jsonrpc": "2.0", "method": "missing_method", "id": 12}
    ])
    batch_resp = router.dispatch_raw(batch)
    assert isinstance(batch_resp, list)
    assert len(batch_resp) == 3, f"Expected 3 responses (notification excluded), got {len(batch_resp)}"
    ids = [r["id"] for r in batch_resp]
    assert ids == [10, 11, 12]
    assert batch_resp[0]["result"]["status"] == "pong"
    assert batch_resp[1]["result"] == 30
    assert batch_resp[2]["error"]["code"] == METHOD_NOT_FOUND
    print("  ✓ Batch request properly returned responses for requests and omitted notification")

    empty_batch_resp = router.dispatch_raw("[]")
    assert empty_batch_resp["error"]["code"] == INVALID_REQUEST
    print("  ✓ Empty batch array rejected with -32600")

    print("\n[5] Testing Wire Framing Protocols...")
    # Stdio Newline Framing
    framer_obj = {"jsonrpc": "2.0", "method": "ping", "id": 99}
    stdio_framed = StdioFramer.frame(framer_obj)
    assert stdio_framed.endswith("\n")
    unframed_stdio = StdioFramer.unframe(stdio_framed)
    assert len(unframed_stdio) == 1
    assert json.loads(unframed_stdio[0]) == framer_obj
    print("  ✓ Stdio newline framing and unframing verified")

    # SSE Chunk Framing
    sse_framed = SSEFramer.frame(framer_obj, event="endpoint")
    assert "event: endpoint\n" in sse_framed
    assert "data: " in sse_framed
    assert sse_framed.endswith("\n\n")
    events = SSEFramer.unframe(sse_framed)
    assert len(events) == 1
    assert events[0][0] == "endpoint"
    assert json.loads(events[0][1]) == framer_obj
    print("  ✓ Server-Sent Events (SSE) framing and chunk un-chunking verified")

    print("\n" + "=" * 70)
    print("ALL JSON-RPC 2.0 PROTOCOL & WIRE FORMAT TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
