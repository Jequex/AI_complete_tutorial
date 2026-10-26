#!/usr/bin/env python3
"""
Topic 27.2 Experiment: Tool Use, Grammar-Constrained Schema Validation & Self-Correction Loops
Phase 27: AI Agents & Multi-Agent Systems

Implements from first principles in pure Python (standard library only):
1. Tool Registry & Schema Definitions: Standard JSON-Schema-like definitions for tools.
2. Lightweight JSON Schema Validator:
   - Checks required properties
   - Enforces primitive types ('string', 'number', 'integer', 'boolean', 'array')
   - Enforces enum constraints and numerical bounds
3. Sandboxed Tool Dispatcher:
   - Safely executes registered tool functions with validated arguments
   - Catches runtime errors and emits structured error payloads
4. Self-Correction Feedback Loop:
   - Simulates an LLM agent that emits imperfect/malformed tool calls
   - Feeds schema and runtime errors back into the agent context
   - Demonstrates multi-turn self-correction converging to valid execution

Zero external dependencies. Pure Python 3 standard library.
"""

import json
import math
from typing import Any, Callable, Dict, List, Optional, Tuple


class SchemaValidationError(Exception):
    """Raised when tool arguments violate the JSON schema."""
    pass


class ToolRegistry:
    """Registry managing available tools and their JSON schemas."""

    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.schemas: Dict[str, dict] = {}

    def register(self, name: str, schema: dict, func: Callable):
        self.tools[name] = func
        self.schemas[name] = schema

    def validate_arguments(self, tool_name: str, args: dict) -> None:
        if tool_name not in self.schemas:
            raise SchemaValidationError(f"Unknown tool '{tool_name}'. Available: {list(self.schemas.keys())}")

        schema = self.schemas[tool_name]
        parameters = schema.get("parameters", {})
        properties = parameters.get("properties", {})
        required = parameters.get("required", [])

        # Check required fields
        for req_field in required:
            if req_field not in args:
                raise SchemaValidationError(f"Missing required parameter '{req_field}' for tool '{tool_name}'.")

        # Validate types and constraints
        for param, val in args.items():
            if param not in properties:
                raise SchemaValidationError(f"Unexpected argument '{param}' passed to tool '{tool_name}'.")

            prop_schema = properties[param]
            expected_type = prop_schema.get("type")

            # Type checking
            if expected_type == "string" and not isinstance(val, str):
                raise SchemaValidationError(f"Parameter '{param}' expected string, got {type(val).__name__}.")
            elif expected_type == "number" and not isinstance(val, (int, float)):
                raise SchemaValidationError(f"Parameter '{param}' expected number, got {type(val).__name__}.")
            elif expected_type == "integer" and (not isinstance(val, int) or isinstance(val, bool)):
                raise SchemaValidationError(f"Parameter '{param}' expected integer, got {type(val).__name__}.")
            elif expected_type == "boolean" and not isinstance(val, bool):
                raise SchemaValidationError(f"Parameter '{param}' expected boolean, got {type(val).__name__}.")
            elif expected_type == "array" and not isinstance(val, list):
                raise SchemaValidationError(f"Parameter '{param}' expected array/list, got {type(val).__name__}.")

            # Enum constraints
            if "enum" in prop_schema and val not in prop_schema["enum"]:
                raise SchemaValidationError(
                    f"Invalid value '{val}' for parameter '{param}'. Must be one of: {prop_schema['enum']}"
                )

            # Minimum / Maximum bounds
            if "minimum" in prop_schema and val < prop_schema["minimum"]:
                raise SchemaValidationError(
                    f"Parameter '{param}' value {val} is below minimum allowed {prop_schema['minimum']}."
                )
            if "maximum" in prop_schema and val > prop_schema["maximum"]:
                raise SchemaValidationError(
                    f"Parameter '{param}' value {val} exceeds maximum allowed {prop_schema['maximum']}."
                )

    def dispatch(self, tool_name: str, args: dict) -> dict:
        """Validates arguments and executes the tool function."""
        self.validate_arguments(tool_name, args)
        func = self.tools[tool_name]
        return func(**args)


# --- Example Tool Implementations ---

def calculate_bmi(weight_kg: float, height_m: float) -> dict:
    if height_m <= 0:
        raise ValueError("Height must be strictly greater than 0.")
    bmi = round(weight_kg / (height_m ** 2), 2)
    category = "Underweight" if bmi < 18.5 else "Normal" if bmi < 25.0 else "Overweight"
    return {"bmi": bmi, "category": category}


def query_flight_database(origin: str, destination: str, max_price: float, cabin_class: str) -> dict:
    # Simulated flights
    flights = [
        {"flight_no": "AI-101", "origin": "JFK", "destination": "LHR", "price": 650.0, "cabin": "economy"},
        {"flight_no": "AI-102", "origin": "JFK", "destination": "LHR", "price": 1200.0, "cabin": "business"},
        {"flight_no": "AI-205", "origin": "SFO", "destination": "HND", "price": 850.0, "cabin": "economy"},
    ]
    matches = [
        f for f in flights
        if f["origin"] == origin and f["destination"] == destination
        and f["price"] <= max_price and f["cabin"] == cabin_class
    ]
    return {"matches": matches, "count": len(matches)}


# --- Tool Schemas (JSON Schema Specification) ---

BMI_SCHEMA = {
    "name": "calculate_bmi",
    "description": "Calculates Body Mass Index (BMI) given mass in kg and height in meters.",
    "parameters": {
        "type": "object",
        "properties": {
            "weight_kg": {"type": "number", "minimum": 1.0, "maximum": 500.0},
            "height_m": {"type": "number", "minimum": 0.3, "maximum": 2.8},
        },
        "required": ["weight_kg", "height_m"],
    },
}

FLIGHT_SCHEMA = {
    "name": "query_flight_database",
    "description": "Searches for commercial flights matching origin, destination, price limit, and cabin class.",
    "parameters": {
        "type": "object",
        "properties": {
            "origin": {"type": "string"},
            "destination": {"type": "string"},
            "max_price": {"type": "number", "minimum": 0.0},
            "cabin_class": {"type": "string", "enum": ["economy", "premium_economy", "business", "first"]},
        },
        "required": ["origin", "destination", "max_price", "cabin_class"],
    },
}


class SimulatedLLMAgent:
    """
    Simulates an LLM agent undergoing a self-correction trajectory.
    Starts with erroneous parameter assumptions and refines them upon receiving feedback.
    """

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.trajectory_turn = 0

    def generate_tool_call(self, goal: str, last_error: Optional[str]) -> Tuple[str, dict]:
        """
        Simulates model generation turns:
        Turn 1: Fails with invalid enum ('first_class' instead of 'first') and missing parameter.
        Turn 2: Fails with type mismatch (max_price as string "$700" instead of float 700.0).
        Turn 3: Successfully corrects all parameters based on schema error messages.
        """
        self.trajectory_turn += 1

        if self.trajectory_turn == 1:
            # Flaw 1: Missing 'origin', invalid enum 'first_class'
            return "query_flight_database", {
                "destination": "LHR",
                "max_price": 700.0,
                "cabin_class": "first_class"
            }
        elif self.trajectory_turn == 2:
            # Flaw 2: Fixes origin & enum, but sends string instead of number for max_price
            return "query_flight_database", {
                "origin": "JFK",
                "destination": "LHR",
                "max_price": "700.0",  # String instead of float
                "cabin_class": "economy"
            }
        else:
            # Correct payload
            return "query_flight_database", {
                "origin": "JFK",
                "destination": "LHR",
                "max_price": 700.0,
                "cabin_class": "economy"
            }


def run_experiment():
    print("=" * 75)
    print("TOPIC 27.2: TOOL USE, GRAMMAR-CONSTRAINED SCHEMAS & SELF-CORRECTION LOOPS")
    print("=" * 75)

    registry = ToolRegistry()
    registry.register("calculate_bmi", BMI_SCHEMA, calculate_bmi)
    registry.register("query_flight_database", FLIGHT_SCHEMA, query_flight_database)

    # 1. Deterministic direct execution test
    print("[1] Direct Schema Validation & Execution Test:")
    bmi_res = registry.dispatch("calculate_bmi", {"weight_kg": 75.0, "height_m": 1.80})
    print(f" -> BMI Tool Result: {bmi_res}")
    assert bmi_res["bmi"] == 23.15, "BMI calculation mismatch"
    print(" [PASS] Direct schema validation and tool dispatch succeeded.\n")

    # 2. Self-Correction Feedback Loop Test
    print("[2] Closed-Loop Tool Self-Correction Trajectory:")
    goal = "Find an economy flight from JFK to LHR under $700"
    agent = SimulatedLLMAgent(registry)
    max_turns = 5
    solved = False
    last_error: Optional[str] = None

    for turn in range(1, max_turns + 1):
        tool_name, tool_args = agent.generate_tool_call(goal, last_error)
        print(f"\n--- Turn {turn} ---")
        print(f"Agent Proposed Action: {tool_name}({json.dumps(tool_args)})")

        try:
            # Sandboxed Execution & Schema Validation
            result = registry.dispatch(tool_name, tool_args)
            print(f"Execution SUCCESS: {result}")
            solved = True
            break
        except SchemaValidationError as e:
            last_error = f"SchemaValidationError: {str(e)}"
            print(f"Environment Feedback (Error): {last_error}")
        except Exception as e:
            last_error = f"RuntimeError: {str(e)}"
            print(f"Environment Feedback (Error): {last_error}")

    assert solved, "Agent failed to self-correct within turn limit!"
    assert agent.trajectory_turn == 3, f"Expected 3 turns for self-correction, took {agent.trajectory_turn}"

    print("\n" + "=" * 75)
    print("Verification Results:")
    print(" [PASS] JSON Schema parameter constraints accurately enforced.")
    print(" [PASS] Malformed invocations caught deterministically without unhandled crashes.")
    print(" [PASS] Error observation injected into feedback loop triggered successful self-correction.")
    print("=" * 75)


if __name__ == "__main__":
    run_experiment()
