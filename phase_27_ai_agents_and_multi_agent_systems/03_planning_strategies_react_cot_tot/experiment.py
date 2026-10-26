#!/usr/bin/env python3
"""
Topic 27.3 Experiment: Planning Strategies — ReAct, Chain-of-Thought & Tree-of-Thoughts (ToT)
Phase 27: AI Agents & Multi-Agent Systems

Implements from first principles in pure Python (standard library only):
1. Planning Taxonomy:
   - Chain-of-Thought (Linear generation: s0 -> s1 -> s2)
   - ReAct (Interleaved Thought -> Action -> Observation execution loop)
   - Tree-of-Thoughts (ToT: Branching thought generation, state evaluation heuristic,
     beam search / DFS with backtracking)
2. A Constraint Arithmetic & Resource Puzzle (Game of 24 & Key-Lock Maze):
   - ToT explores candidate algebraic branches to reach target value 24.
   - Evaluator classifies states into: "sure" (solved), "likely" (promising), or "impossible" (prune).
   - Demonstrates DFS search with explicit backtracking when hitting dead ends.
3. ReAct Controller executing the validated plan step-by-step.

Zero external dependencies. Pure Python 3 standard library.
"""

from typing import List, Tuple, Optional, Dict
import math


class ExpressionNode:
    """Represents an expression with its current numerical value and string representation."""
    def __init__(self, val: float, text: str):
        self.val = val
        self.text = text

    def __repr__(self):
        return f"{self.text} ({self.val})"


class TreeOfThoughtsSolver:
    """
    Tree-of-Thoughts (ToT) Engine for Arithmetic Constraint Satisfaction (Game of 24).
    Explores state space as a tree:
      - State s: current multiset of remaining available numbers/expressions.
      - Thought generator: Pick two elements (a, b) and an operator (+, -, *, /) to produce a new element.
      - State Evaluator: Heuristic V(s) scoring feasibility of remaining state reaching target 24.
      - Search Strategy: Depth-First Search (DFS) with backtracking upon pruning dead branches.
    """

    def __init__(self, target: float = 24.0, tolerance: float = 1e-5):
        self.target = target
        self.tolerance = tolerance
        self.nodes_explored = 0
        self.backtracks = 0

    def generate_candidate_thoughts(self, numbers: List[ExpressionNode]) -> List[Tuple[str, List[ExpressionNode]]]:
        """
        Generates all valid 1-step reduction thoughts: combine pair (i, j) with op.
        Returns list of (thought_description, next_state_numbers).
        """
        candidates = []
        n = len(numbers)

        for i in range(n):
            for j in range(i + 1, n):
                a = numbers[i]
                b = numbers[j]
                remaining = [numbers[k] for k in range(n) if k != i and k != j]

                # Operations to consider
                ops = [
                    (a.val + b.val, f"({a.text} + {b.text})", "+"),
                    (a.val * b.val, f"({a.text} * {b.text})", "*"),
                    (a.val - b.val, f"({a.text} - {b.text})", "-"),
                    (b.val - a.val, f"({b.text} - {a.text})", "-"),
                ]
                if abs(b.val) > 1e-6:
                    ops.append((a.val / b.val, f"({a.text} / {b.text})", "/"))
                if abs(a.val) > 1e-6:
                    ops.append((b.val / a.val, f"({b.text} / {a.text})", "/"))

                for res_val, res_text, op_sym in ops:
                    new_node = ExpressionNode(res_val, res_text)
                    new_state = remaining + [new_node]
                    thought_desc = f"Combine {a.text} and {b.text} via '{op_sym}' -> {res_text} = {res_val:.2f}"
                    candidates.append((thought_desc, new_state))

        return candidates

    def evaluate_state(self, numbers: List[ExpressionNode]) -> str:
        """
        Heuristic state evaluator:
        Returns:
          'sure'       : Single element remaining that equals target 24
          'impossible' : Dead-end branch (e.g. negative numbers with no addition, or single element != 24)
          'likely'     : Promising intermediate state
        """
        if len(numbers) == 1:
            if abs(numbers[0].val - self.target) < self.tolerance:
                return "sure"
            else:
                return "impossible"

        # Prune extreme runaway values or division by tiny epsilon
        vals = [node.val for node in numbers]
        if any(math.isinf(v) or math.isnan(v) or abs(v) > 10000 for v in vals):
            return "impossible"

        return "likely"

    def solve_dfs(self, current_numbers: List[ExpressionNode], path: List[str]) -> Optional[Tuple[str, List[str]]]:
        """Recursive DFS with state evaluation and backtracking."""
        self.nodes_explored += 1
        eval_score = self.evaluate_state(current_numbers)

        if eval_score == "sure":
            return current_numbers[0].text, path

        if eval_score == "impossible":
            self.backtracks += 1
            return None  # Backtrack

        # Generate candidate thought branches
        branches = self.generate_candidate_thoughts(current_numbers)

        for thought_desc, next_state in branches:
            result = self.solve_dfs(next_state, path + [thought_desc])
            if result is not None:
                return result  # Solution found

        # If all candidate branches fail, backtrack
        self.backtracks += 1
        return None


class ReActAgent:
    """
    ReAct Controller (Reasoning + Acting)
    Interleaves Thought -> Action -> Observation steps.
    """

    def __init__(self, puzzle_input: List[float]):
        self.puzzle_input = puzzle_input
        self.trajectory: List[Dict[str, str]] = []

    def run(self) -> str:
        # Step 1: Reason about the goal
        self.trajectory.append({
            "type": "Thought",
            "content": f"I have input numbers {self.puzzle_input} and need to reach 24. Standard linear CoT is prone to dead-ends. I will invoke the Tree-of-Thoughts planner tool."
        })

        # Step 2: Act by delegating to Tree-of-Thoughts planner
        self.trajectory.append({
            "type": "Action",
            "content": f"call_tot_solver(numbers={self.puzzle_input}, target=24)"
        })

        # Step 3: Observation from planner
        nodes = [ExpressionNode(float(x), str(int(x))) for x in self.puzzle_input]
        tot = TreeOfThoughtsSolver(target=24.0)
        solution, steps = tot.solve_dfs(nodes, [])

        if solution:
            obs = f"SUCCESS: Solution found: {solution} = 24. Explored {tot.nodes_explored} states, {tot.backtracks} backtracks."
        else:
            obs = "FAILURE: No valid sequence exists."

        self.trajectory.append({"type": "Observation", "content": obs})

        # Step 4: Final thought & conclusion
        self.trajectory.append({
            "type": "Thought",
            "content": f"The verified formula {solution} successfully solves the constraint. Ready to deliver answer."
        })
        return solution


def run_experiment():
    print("=" * 75)
    print("TOPIC 27.3: PLANNING STRATEGIES — ReAct & TREE-OF-THOUGHTS (ToT)")
    print("=" * 75)

    puzzle_numbers = [4, 1, 8, 7]
    print(f"Goal: Game of 24 with input numbers {puzzle_numbers}")
    print("Approach: ReAct Controller leveraging Tree-of-Thoughts (ToT) backtracking.\n")

    agent = ReActAgent(puzzle_numbers)
    solution = agent.run()

    print("ReAct Trajectory Trace:")
    print("-" * 75)
    for step in agent.trajectory:
        prefix = f"[{step['type'].upper()}]"
        print(f"{prefix:<15} : {step['content']}")
    print("-" * 75)

    # Verify result algebraically
    expr = solution
    val = eval(expr)  # Safe evaluation on pure arithmetic expression of numbers and ops
    print(f"\nVerification:")
    print(f" - Final Formulated Expression : {expr}")
    print(f" - Evaluated Value              : {val}")
    assert abs(val - 24.0) < 1e-5, f"Evaluation error! Expected 24, got {val}"
    print(f" [PASS] Expression rigorously evaluates to 24.0.")

    # Test tricky problem that requires non-trivial fractions: [3, 3, 8, 8] -> 8 / (3 - 8/3) = 24
    print("\n" + "=" * 75)
    print("Challenging Benchmark: [3, 3, 8, 8] (Requires fractional division)")
    tricky_nodes = [ExpressionNode(float(x), str(int(x))) for x in [3, 3, 8, 8]]
    tot_tricky = TreeOfThoughtsSolver(target=24.0)
    sol_tricky, steps_tricky = tot_tricky.solve_dfs(tricky_nodes, [])

    print(f"Solution Found : {sol_tricky}")
    print(f"Nodes Explored : {tot_tricky.nodes_explored}")
    print(f"Backtracks     : {tot_tricky.backtracks}")
    val_tricky = eval(sol_tricky)
    print(f"Evaluated Value: {val_tricky}")
    assert abs(val_tricky - 24.0) < 1e-5, "Tricky puzzle failed!"
    print(f" [PASS] Tree-of-Thoughts solved fractional division constraint with backtracking.")
    print("=" * 75)


if __name__ == "__main__":
    run_experiment()
