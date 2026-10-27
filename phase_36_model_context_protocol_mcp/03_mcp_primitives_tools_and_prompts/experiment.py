#!/usr/bin/env python3
"""
Phase 36 - Topic 03: MCP Primitives - Tools & Prompts
Pure Python 3 Standard Library Implementation

Demonstrates:
- JSON Schema parameter validation (types, required fields, enums)
- Tool discovery (tools/list) and execution (tools/call)
- Error propagation and side-effect safety controls (isError flag)
- Prompt template discovery (prompts/list) and argument injection (prompts/get)
- Multi-turn workflow templating
"""

from typing import Any, Callable, Dict, List, Optional, Union


class SimpleJSONSchemaValidator:
    """Pure Python JSON Schema validator for MCP tool input schemas."""

    TYPE_MAP = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    @classmethod
    def validate(cls, schema: Dict[str, Any], data: Any, path: str = "params") -> List[str]:
        errors: List[str] = []
        if not isinstance(schema, dict):
            return errors

        # Type validation
        expected_type = schema.get("type")
        if expected_type in cls.TYPE_MAP:
            python_type = cls.TYPE_MAP[expected_type]
            # bool is a subclass of int in Python, so special check
            if expected_type in ("integer", "number") and isinstance(data, bool):
                errors.append(f"Field '{path}' expected {expected_type}, got boolean")
            elif not isinstance(data, python_type):
                errors.append(f"Field '{path}' expected {expected_type}, got {type(data).__name__}")
                return errors  # Cannot validate properties if base type is wrong

        # Enum validation
        if "enum" in schema:
            if data not in schema["enum"]:
                errors.append(f"Field '{path}' value '{data}' not in allowed enum: {schema['enum']}")

        # Object validation
        if expected_type == "object" and isinstance(data, dict):
            # Check required fields
            required_fields = schema.get("required", [])
            for req in required_fields:
                if req not in data:
                    errors.append(f"Missing required parameter '{path}.{req}'")

            # Check properties
            properties = schema.get("properties", {})
            for key, val in data.items():
                if key in properties:
                    child_schema = properties[key]
                    child_errors = cls.validate(child_schema, val, path=f"{path}.{key}")
                    errors.extend(child_errors)

        # Array validation
        if expected_type == "array" and isinstance(data, list):
            items_schema = schema.get("items")
            if items_schema:
                for idx, item in enumerate(data):
                    item_errors = cls.validate(items_schema, item, path=f"{path}[{idx}]")
                    errors.extend(item_errors)

        return errors


class ToolRegistry:
    """Manages MCP Tools with discovery, validation, and safe sandboxed execution."""

    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.handlers: Dict[str, Callable[[Dict[str, Any]], str]] = {}

    def register(self, name: str, description: str, input_schema: Dict[str, Any],
                 handler: Callable[[Dict[str, Any]], str]) -> None:
        self.tools[name] = {
            "name": name,
            "description": description,
            "inputSchema": input_schema
        }
        self.handlers[name] = handler

    # Protocol RPC: tools/list
    def handle_tools_list(self) -> Dict[str, Any]:
        return {"tools": list(self.tools.values())}

    # Protocol RPC: tools/call
    def handle_tools_call(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if name not in self.tools:
            return {
                "content": [{"type": "text", "text": f"Tool '{name}' not found."}],
                "isError": True
            }

        tool_def = self.tools[name]
        schema = tool_def["inputSchema"]

        # Validate arguments against inputSchema
        validation_errors = SimpleJSONSchemaValidator.validate(schema, arguments, path="arguments")
        if validation_errors:
            return {
                "content": [{"type": "text", "text": f"Validation Error: {'; '.join(validation_errors)}"}],
                "isError": True
            }

        # Execute handler in sandbox boundary
        handler = self.handlers[name]
        try:
            output_text = handler(arguments)
            return {
                "content": [{"type": "text", "text": str(output_text)}],
                "isError": False
            }
        except Exception as exc:
            # MCP Tool execution failures are returned with isError: True (not JSON-RPC crash)
            return {
                "content": [{"type": "text", "text": f"Runtime execution failed: {type(exc).__name__}: {str(exc)}"}],
                "isError": True
            }


class PromptRegistry:
    """Manages MCP Prompts: user-controlled workflows and parameterized prompt templates."""

    def __init__(self):
        self.prompts: Dict[str, Dict[str, Any]] = {}
        self.generators: Dict[str, Callable[[Dict[str, str]], List[Dict[str, Any]]]] = {}

    def register(self, name: str, description: str,
                 arguments: List[Dict[str, Any]],
                 generator: Callable[[Dict[str, str]], List[Dict[str, Any]]]) -> None:
        self.prompts[name] = {
            "name": name,
            "description": description,
            "arguments": arguments
        }
        self.generators[name] = generator

    # Protocol RPC: prompts/list
    def handle_prompts_list(self) -> Dict[str, Any]:
        return {"prompts": list(self.prompts.values())}

    # Protocol RPC: prompts/get
    def handle_prompts_get(self, name: str, arguments: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        if name not in self.prompts:
            raise KeyError(f"Prompt '{name}' not found.")

        prompt_def = self.prompts[name]
        provided_args = arguments or {}

        # Validate required arguments
        for arg in prompt_def.get("arguments", []):
            if arg.get("required", False) and arg["name"] not in provided_args:
                raise ValueError(f"Missing required prompt argument: '{arg['name']}'")

        generator = self.generators[name]
        messages = generator(provided_args)
        return {
            "description": prompt_def["description"],
            "messages": messages
        }


# ---------------- Verification & Demonstration ----------------
def run_tests():
    print("=" * 70)
    print("Phase 36 Topic 03: MCP Tools & Prompts Verification")
    print("=" * 70)

    # 1. Setup Tool Registry
    tools = ToolRegistry()

    # Tool A: Safe Math Calculator
    calc_schema = {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["add", "multiply", "divide"]},
            "a": {"type": "number"},
            "b": {"type": "number"}
        },
        "required": ["operation", "a", "b"]
    }

    def run_calculator(args: Dict[str, Any]) -> str:
        op = args["operation"]
        a = args["a"]
        b = args["b"]
        if op == "add":
            return str(a + b)
        elif op == "multiply":
            return str(a * b)
        elif op == "divide":
            if b == 0:
                raise ZeroDivisionError("Cannot divide by zero")
            return str(a / b)
        raise ValueError(f"Unknown operation {op}")

    tools.register("calculator", "Performs mathematical arithmetic", calc_schema, run_calculator)

    # Tool B: Code Linting Tool
    lint_schema = {
        "type": "object",
        "properties": {
            "code": {"type": "string"},
            "strict_mode": {"type": "boolean"}
        },
        "required": ["code"]
    }

    def run_linter(args: Dict[str, Any]) -> str:
        code = args["code"]
        strict = args.get("strict_mode", False)
        issues = []
        if "eval(" in code:
            issues.append("Security Risk: Use of eval() detected")
        if strict and len(code) > 100:
            issues.append("Strict warning: Code chunk exceeds 100 characters")
        if not issues:
            return "SUCCESS: 0 lint issues found."
        return "ISSUES FOUND: " + "; ".join(issues)

    tools.register("linter", "Scans code for syntax and security flaws", lint_schema, run_linter)

    print("\n[1] Testing Tool Discovery (tools/list)...")
    catalog = tools.handle_tools_list()
    assert len(catalog["tools"]) == 2
    tool_names = [t["name"] for t in catalog["tools"]]
    assert "calculator" in tool_names and "linter" in tool_names
    print(f"  ✓ Discovered {len(tool_names)} registered tools: {tool_names}")

    print("\n[2] Testing Successful Tool Execution (tools/call)...")
    call_res1 = tools.handle_tools_call("calculator", {"operation": "multiply", "a": 7, "b": 6})
    assert call_res1["isError"] is False
    assert call_res1["content"][0]["text"] == "42"
    print("  ✓ Calculator execution succeeded: 7 * 6 = 42")

    call_res2 = tools.handle_tools_call("linter", {"code": "def hello():\n    eval('1+1')", "strict_mode": True})
    assert call_res2["isError"] is False
    assert "Security Risk: Use of eval()" in call_res2["content"][0]["text"]
    print("  ✓ Linter executed and reported expected security violation")

    print("\n[3] Testing Tool Schema Validation Failures...")
    # Missing required argument 'b'
    res_bad1 = tools.handle_tools_call("calculator", {"operation": "add", "a": 10})
    assert res_bad1["isError"] is True
    assert "Missing required parameter 'arguments.b'" in res_bad1["content"][0]["text"]
    print("  ✓ Correctly rejected missing required parameter 'b'")

    # Invalid enum value
    res_bad2 = tools.handle_tools_call("calculator", {"operation": "exponentiate", "a": 2, "b": 3})
    assert res_bad2["isError"] is True
    assert "not in allowed enum" in res_bad2["content"][0]["text"]
    print("  ✓ Correctly rejected disallowed enum value 'exponentiate'")

    # Type mismatch (string passed for number)
    res_bad3 = tools.handle_tools_call("calculator", {"operation": "add", "a": "ten", "b": 5})
    assert res_bad3["isError"] is True
    assert "expected number, got str" in res_bad3["content"][0]["text"]
    print("  ✓ Correctly rejected type mismatch (string for number)")

    print("\n[4] Testing Graceful Runtime Error Handling (isError: True)...")
    res_div0 = tools.handle_tools_call("calculator", {"operation": "divide", "a": 10, "b": 0})
    assert res_div0["isError"] is True
    assert "ZeroDivisionError" in res_div0["content"][0]["text"]
    print("  ✓ Handled ZeroDivisionError gracefully inside tool result without server crash")

    # 2. Setup Prompt Registry
    prompts = PromptRegistry()

    def generate_code_review_prompt(args: Dict[str, str]) -> List[Dict[str, Any]]:
        language = args.get("language", "Python")
        code = args.get("code", "")
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": (
                        f"Please perform a rigorous security and performance code review for the following {language} snippet:\n\n"
                        f"```{language.lower()}\n{code}\n```\n\n"
                        "Check for: 1. OWASP Top 10 flaws, 2. Time complexity bottlenecks, 3. Memory leaks."
                    )
                }
            }
        ]

    prompts.register(
        name="code_review",
        description="Initiates an expert security and performance review workflow",
        arguments=[
            {"name": "code", "description": "Source code text to review", "required": True},
            {"name": "language", "description": "Programming language name", "required": False}
        ],
        generator=generate_code_review_prompt
    )

    print("\n[5] Testing Prompt Discovery (prompts/list)...")
    prompt_catalog = prompts.handle_prompts_list()
    assert len(prompt_catalog["prompts"]) == 1
    assert prompt_catalog["prompts"][0]["name"] == "code_review"
    print(f"  ✓ Discovered prompt template: {prompt_catalog['prompts'][0]['name']}")

    print("\n[6] Testing Prompt Rendering (prompts/get)...")
    rendered = prompts.handle_prompts_get("code_review", {"code": "SELECT * FROM users WHERE id = " + "1", "language": "SQL"})
    assert "description" in rendered
    assert len(rendered["messages"]) == 1
    msg_text = rendered["messages"][0]["content"]["text"]
    assert "security and performance code review for the following SQL" in msg_text
    assert "SELECT * FROM users" in msg_text
    print("  ✓ Rendered multi-turn prompt message with parameter injection:")
    print("    -- Rendered Prompt Message Preview --")
    print("    " + msg_text.replace("\n", "\n    "))

    # Missing required argument test
    try:
        prompts.handle_prompts_get("code_review", {"language": "Rust"})
        assert False, "Should have raised ValueError for missing 'code'"
    except ValueError as ve:
        assert "Missing required prompt argument: 'code'" in str(ve)
        print("  ✓ Properly raised error for missing required prompt argument")

    print("\n" + "=" * 70)
    print("ALL MCP TOOLS & PROMPTS TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
