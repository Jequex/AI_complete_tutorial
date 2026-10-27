#!/usr/bin/env python3
"""
Phase 36 - Topic 02: MCP Primitives - Resources & Dynamic Context
Pure Python 3 Standard Library Implementation

Demonstrates:
- URI schemes and MIME types (file:///, db://, custom://)
- Static resource registry & pagination (resources/list)
- RFC 6570 URI template routing (resources/templates/list)
- Resource retrieval (resources/read) with text and base64 blob content
- Real-time dynamic subscriptions (resources/subscribe, notifications/resources/updated)
- Host-side context window injection formatting
"""

import base64
import json
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


class RFC6570Matcher:
    """Simple RFC 6570 URI Template variable parser and matcher."""

    def __init__(self, template: str):
        self.template = template
        # Extract {var_name} placeholders
        self.var_names = re.findall(r"\{([a-zA-Z0-9_]+)\}", template)
        # Convert template into regex pattern
        regex_pattern = re.escape(template)
        for var in self.var_names:
            escaped_var = re.escape(f"{{{var}}}")
            regex_pattern = regex_pattern.replace(escaped_var, f"(?P<{var}>[^/]+)")
        self.regex = re.compile(f"^{regex_pattern}$")

    def match(self, uri: str) -> Optional[Dict[str, str]]:
        m = self.regex.match(uri)
        if m:
            return m.groupdict()
        return None

    def expand(self, **kwargs) -> str:
        res = self.template
        for k, v in kwargs.items():
            res = res.replace(f"{{{k}}}", str(v))
        return res


class ResourceServer:
    """MCP Server providing Resource capabilities and dynamic subscriptions."""

    def __init__(self, name: str = "MockResourceServer"):
        self.name = name
        self.resources: Dict[str, Dict[str, Any]] = {}
        self.templates: List[Dict[str, Any]] = []
        self._template_matchers: List[Tuple[RFC6570Matcher, Callable[[Dict[str, str]], Dict[str, Any]]]] = []
        self.subscriptions: Dict[str, Set[str]] = {}  # uri -> set of client_ids
        self.client_notifiers: Dict[str, Callable[[str, Dict[str, Any]], None]] = {}

    def register_static_resource(self, uri: str, name: str, mime_type: str,
                                 description: str = "", text: Optional[str] = None,
                                 blob: Optional[bytes] = None) -> None:
        item: Dict[str, Any] = {
            "uri": uri,
            "name": name,
            "mimeType": mime_type,
            "description": description
        }
        if text is not None:
            item["text"] = text
        elif blob is not None:
            item["blob"] = base64.b64encode(blob).decode("utf-8")
        else:
            item["text"] = ""
        self.resources[uri] = item

    def register_template(self, uri_template: str, name: str, mime_type: str,
                          description: str,
                          resolver: Callable[[Dict[str, str]], Dict[str, Any]]) -> None:
        matcher = RFC6570Matcher(uri_template)
        self.templates.append({
            "uriTemplate": uri_template,
            "name": name,
            "mimeType": mime_type,
            "description": description
        })
        self._template_matchers.append((matcher, resolver))

    def register_client(self, client_id: str, notify_fn: Callable[[str, Dict[str, Any]], None]) -> None:
        self.client_notifiers[client_id] = notify_fn

    # Protocol RPC: resources/list
    def handle_resources_list(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        cursor = (params or {}).get("cursor", 0)
        page_size = 2
        items = list(self.resources.values())

        start = int(cursor)
        end = start + page_size
        slice_items = items[start:end]

        result: Dict[str, Any] = {
            "resources": [
                {
                    "uri": r["uri"],
                    "name": r["name"],
                    "description": r.get("description", ""),
                    "mimeType": r.get("mimeType", "text/plain")
                }
                for r in slice_items
            ]
        }
        if end < len(items):
            result["nextCursor"] = str(end)
        return result

    # Protocol RPC: resources/templates/list
    def handle_resources_templates_list(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {"resourceTemplates": self.templates}

    # Protocol RPC: resources/read
    def handle_resources_read(self, params: Dict[str, Any]) -> Dict[str, Any]:
        uri = params.get("uri")
        if not uri:
            raise ValueError("Parameter 'uri' is required")

        # 1. Check static resources
        if uri in self.resources:
            item = self.resources[uri]
            content = {"uri": uri, "mimeType": item["mimeType"]}
            if "text" in item:
                content["text"] = item["text"]
            elif "blob" in item:
                content["blob"] = item["blob"]
            return {"contents": [content]}

        # 2. Check dynamic template matchers
        for matcher, resolver in self._template_matchers:
            matched_vars = matcher.match(uri)
            if matched_vars is not None:
                resolved = resolver(matched_vars)
                content = {
                    "uri": uri,
                    "mimeType": resolved.get("mimeType", "text/plain")
                }
                if "text" in resolved:
                    content["text"] = resolved["text"]
                elif "blob" in resolved:
                    blob_data = resolved["blob"]
                    content["blob"] = base64.b64encode(blob_data).decode("utf-8") if isinstance(blob_data, bytes) else blob_data
                return {"contents": [content]}

        raise KeyError(f"Resource not found: {uri}")

    # Protocol RPC: resources/subscribe
    def handle_resources_subscribe(self, client_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        uri = params.get("uri")
        if not uri:
            raise ValueError("Parameter 'uri' is required")
        if uri not in self.subscriptions:
            self.subscriptions[uri] = set()
        self.subscriptions[uri].add(client_id)
        return {}

    # Protocol RPC: resources/unsubscribe
    def handle_resources_unsubscribe(self, client_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        uri = params.get("uri")
        if uri and uri in self.subscriptions:
            self.subscriptions[uri].discard(client_id)
        return {}

    # Mutation trigger (e.g. file modified or DB row updated)
    def update_resource(self, uri: str, new_text: str) -> None:
        if uri in self.resources:
            self.resources[uri]["text"] = new_text
        # Dispatch notification to all subscribed clients
        subscribers = self.subscriptions.get(uri, set())
        for client_id in subscribers:
            notifier = self.client_notifiers.get(client_id)
            if notifier:
                notifier("notifications/resources/updated", {"uri": uri})


class ResourceClient:
    """MCP Host Client consuming resources and integrating dynamic context."""

    def __init__(self, client_id: str, server: ResourceServer):
        self.client_id = client_id
        self.server = server
        self.server.register_client(client_id, self.on_notification)
        self.notifications_received: List[Dict[str, Any]] = []

    def on_notification(self, method: str, params: Dict[str, Any]) -> None:
        self.notifications_received.append({"method": method, "params": params})

    def list_all_resources(self) -> List[Dict[str, Any]]:
        all_items = []
        cursor = 0
        while True:
            resp = self.server.handle_resources_list({"cursor": cursor})
            all_items.extend(resp["resources"])
            if "nextCursor" in resp:
                cursor = int(resp["nextCursor"])
            else:
                break
        return all_items

    def read_resource(self, uri: str) -> Dict[str, Any]:
        return self.server.handle_resources_read({"uri": uri})

    def subscribe(self, uri: str) -> None:
        self.server.handle_resources_subscribe(self.client_id, {"uri": uri})

    def build_prompt_context(self, user_query: str, attached_uris: List[str]) -> List[Dict[str, str]]:
        """
        Formats MCP Resources into standard LLM prompt messages (Context Window Injection).
        """
        system_blocks = ["You are an intelligent AI assistant equipped with context resources."]
        for uri in attached_uris:
            resp = self.read_resource(uri)
            for c in resp["contents"]:
                mime = c.get("mimeType", "text/plain")
                body = c.get("text", f"[Binary blob: {len(c.get('blob', ''))} base64 chars]")
                system_blocks.append(
                    f"\n--- BEGIN RESOURCE: {uri} (Type: {mime}) ---\n"
                    f"{body}\n"
                    f"--- END RESOURCE: {uri} ---"
                )

        messages = [
            {"role": "system", "content": "\n".join(system_blocks)},
            {"role": "user", "content": user_query}
        ]
        return messages


# ---------------- Verification & Demonstration ----------------
def run_tests():
    print("=" * 70)
    print("Phase 36 Topic 02: MCP Resources & Dynamic Context Verification")
    print("=" * 70)

    server = ResourceServer("EnterpriseDataServer")

    # 1. Register static resources with different schemes
    server.register_static_resource(
        uri="file:///workspace/config.json",
        name="Project Config",
        mime_type="application/json",
        description="Core deployment configuration",
        text=json.dumps({"env": "production", "workers": 8, "rate_limit": 1000}, indent=2)
    )

    server.register_static_resource(
        uri="file:///workspace/README.md",
        name="Documentation",
        mime_type="text/markdown",
        description="Project instructions",
        text="# Global Enterprise Service\nRuns mission-critical processing pipelines."
    )

    server.register_static_resource(
        uri="custom://telemetry/summary",
        name="System Metrics",
        mime_type="text/plain",
        description="Live CPU and Memory stats",
        text="CPU: 32% | Mem: 4.2GB / 16GB | Uptime: 99.98%"
    )

    # Binary resource (e.g. icon/raw data)
    server.register_static_resource(
        uri="file:///assets/logo.png",
        name="App Icon",
        mime_type="image/png",
        description="Binary PNG banner",
        blob=b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    )

    # 2. Register dynamic RFC 6570 templates
    mock_db = {
        "101": {"name": "Alice Chen", "role": "Senior Distributed Systems Engineer", "team": "Infrastructure"},
        "102": {"name": "Bob Taylor", "role": "Principal AI Safety Researcher", "team": "Alignment"}
    }

    def resolve_user_record(variables: Dict[str, str]) -> Dict[str, Any]:
        user_id = variables.get("user_id")
        if user_id in mock_db:
            return {
                "mimeType": "application/json",
                "text": json.dumps(mock_db[user_id], indent=2)
            }
        return {"mimeType": "application/json", "text": json.dumps({"error": "User not found"})}

    server.register_template(
        uri_template="db://users/{user_id}/profile",
        name="User Database Record",
        mime_type="application/json",
        description="Dynamic user profile fetched from DB",
        resolver=resolve_user_record
    )

    client = ResourceClient("client-alpha-01", server)

    print("\n[1] Testing Resource Listing & Pagination...")
    all_resources = client.list_all_resources()
    assert len(all_resources) == 4
    uris = [r["uri"] for r in all_resources]
    assert "file:///workspace/config.json" in uris
    assert "file:///assets/logo.png" in uris
    print(f"  ✓ Successfully listed {len(all_resources)} resources across pagination pages")

    print("\n[2] Testing Resource Reading (Static & Binary Blob)...")
    config_read = client.read_resource("file:///workspace/config.json")
    assert "production" in config_read["contents"][0]["text"]
    assert config_read["contents"][0]["mimeType"] == "application/json"
    print("  ✓ Static text resource read successfully")

    logo_read = client.read_resource("file:///assets/logo.png")
    assert "blob" in logo_read["contents"][0]
    decoded_bytes = base64.b64decode(logo_read["contents"][0]["blob"])
    assert decoded_bytes.startswith(b"\x89PNG")
    print("  ✓ Binary blob resource read and base64 verified")

    print("\n[3] Testing RFC 6570 Resource Templates...")
    template_list = server.handle_resources_templates_list()
    assert len(template_list["resourceTemplates"]) == 1
    assert template_list["resourceTemplates"][0]["uriTemplate"] == "db://users/{user_id}/profile"
    print("  ✓ Resource template registered and discovered")

    user_101 = client.read_resource("db://users/101/profile")
    user_101_data = json.loads(user_101["contents"][0]["text"])
    assert user_101_data["name"] == "Alice Chen"
    print(f"  ✓ Dynamic URI 'db://users/101/profile' matched template and retrieved: {user_101_data['name']}")

    user_102 = client.read_resource("db://users/102/profile")
    user_102_data = json.loads(user_102["contents"][0]["text"])
    assert user_102_data["role"] == "Principal AI Safety Researcher"
    print(f"  ✓ Dynamic URI 'db://users/102/profile' matched template and retrieved: {user_102_data['role']}")

    print("\n[4] Testing Dynamic Subscriptions & Push Notifications...")
    target_uri = "file:///workspace/config.json"
    client.subscribe(target_uri)
    assert "client-alpha-01" in server.subscriptions[target_uri]
    print(f"  ✓ Subscribed to {target_uri}")

    # Simulate resource update on server
    server.update_resource(target_uri, json.dumps({"env": "staging", "workers": 16}))
    assert len(client.notifications_received) == 1
    assert client.notifications_received[0]["method"] == "notifications/resources/updated"
    assert client.notifications_received[0]["params"]["uri"] == target_uri
    print(f"  ✓ Real-time push notification received for {target_uri}")

    # Verify updated content is read
    updated_content = client.read_resource(target_uri)
    assert "staging" in updated_content["contents"][0]["text"]
    print("  ✓ Updated content confirmed on subsequent read")

    print("\n[5] Testing Context Window Injection Formatting...")
    prompt_messages = client.build_prompt_context(
        user_query="Who is user 101 and what environment are we running?",
        attached_uris=["file:///workspace/config.json", "db://users/101/profile"]
    )
    assert len(prompt_messages) == 2
    system_text = prompt_messages[0]["content"]
    assert "BEGIN RESOURCE: file:///workspace/config.json" in system_text
    assert "BEGIN RESOURCE: db://users/101/profile" in system_text
    assert "Alice Chen" in system_text
    print("  ✓ Prompt context successfully assembled with bounded resource headers:")
    print("    -- Sample System Content Snippet --")
    print("\n".join("    " + line for line in system_text.splitlines()[:12]))
    print("    ...")

    print("\n" + "=" * 70)
    print("ALL MCP RESOURCES & DYNAMIC CONTEXT TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
