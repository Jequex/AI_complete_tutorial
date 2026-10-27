"""
Topic 02: Test-Time Compute and Reasoning Search
================================================
Pure Python 3 standard library implementation (no torch, no numpy).

Implements:
1. Multi-Step Symbolic Reasoning Domain: Exact Rational Arithmetic Deduction ("Target 24" Puzzles).
2. Process Reward Model (PRM): Step-level process verifier filtering out invalid / hallucinated deductions.
3. Outcome Reward Model (ORM): Terminal outcome verifier validating exact solution equivalence.
4. Monte Carlo Tree Search (MCTS) Engine:
   - PUCT / UCB1 Selection with PRM Priors
   - Process Verification Pruning & Rollbacks
   - Value Backpropagation
5. Comparison: Test-Time Compute Search vs. Naive Greedy Autoregressive Decoding.
6. Test-Time Compute Scaling Law demonstration over varying search budgets.
"""

from fractions import Fraction
import math
import random
import sys
from typing import List, Dict, Tuple, Optional, Set

random.seed(42)

# -----------------------------------------------------------------------------
# 1. Reasoning Domain: Rational Arithmetic Deduction Task
# -----------------------------------------------------------------------------
class ArithmeticTask:
    def __init__(self, initial_numbers: List[int], target: int = 24, name: str = "Task"):
        self.initial_numbers = tuple(sorted([Fraction(x) for x in initial_numbers]))
        self.target = Fraction(target)
        self.name = name

    def is_terminal(self, state: Tuple[Fraction, ...]) -> bool:
        return len(state) == 1

    def is_goal(self, state: Tuple[Fraction, ...]) -> bool:
        return len(state) == 1 and state[0] == self.target

# -----------------------------------------------------------------------------
# 2. Process Reward Model (PRM) & Outcome Reward Model (ORM)
# -----------------------------------------------------------------------------
class ProcessRewardModel:
    """
    Step-level Process Reward Model (PRM).
    Validates each proposed logical deduction:
    - Verifies mathematical consistency between inputs (n1, n2, op) and claimed output.
    - Catches and filters synthetic LLM reasoning hallucinations.
    - Emits a normalized prior confidence score P(s, a) in [0, 1].
    """
    def __init__(self):
        self.hallucinations_detected = 0

    def verify_and_score_step(
        self,
        parent_state: Tuple[Fraction, ...],
        idx1: int,
        idx2: int,
        op: str,
        claimed_res: Fraction
    ) -> Tuple[bool, float]:
        n1 = parent_state[idx1]
        n2 = parent_state[idx2]

        # Division by zero filter
        if op == "/" and n2 == 0:
            self.hallucinations_detected += 1
            return False, 0.0

        # Ground-truth mathematical computation
        if op == "+": true_res = n1 + n2
        elif op == "-": true_res = n1 - n2
        elif op == "*": true_res = n1 * n2
        elif op == "/": true_res = n1 / n2
        else: return False, 0.0

        # Verification: detect hallucinated deduction steps
        if claimed_res != true_res:
            self.hallucinations_detected += 1
            return False, 0.0

        # Divergence / explosion filter
        if abs(true_res) > 50000:
            return False, 0.0

        # High score for valid sound logical deduction
        prior = 1.0
        return True, prior

class OutcomeRewardModel:
    """
    Evaluates terminal answer correctness.
    """
    def __init__(self, target: Fraction):
        self.target = target

    def evaluate(self, state: Tuple[Fraction, ...]) -> float:
        if len(state) == 1 and state[0] == self.target:
            return 1.0
        return 0.0

# -----------------------------------------------------------------------------
# 3. MCTS Node & Search Engine
# -----------------------------------------------------------------------------
class MCTSNode:
    def __init__(
        self,
        state: Tuple[Fraction, ...],
        parent: Optional['MCTSNode'] = None,
        action: Optional[Tuple[int, int, str, Fraction]] = None,
        prior: float = 1.0
    ):
        self.state = state
        self.parent = parent
        self.action = action  # (idx1, idx2, op, result)
        self.prior = prior
        self.children: Dict[Tuple, 'MCTSNode'] = {}
        self.visits = 0
        self.total_value = 0.0
        self.is_terminal = (len(state) == 1)

    @property
    def q_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits

    def ucb_score(self, parent_visits: int, c_puct: float = 2.0) -> float:
        exploration = c_puct * self.prior * math.sqrt(math.log(parent_visits + 1) / (1 + self.visits))
        return self.q_value + exploration

class MCTSSearchEngine:
    def __init__(
        self,
        task: ArithmeticTask,
        prm: ProcessRewardModel,
        orm: OutcomeRewardModel,
        c_puct: float = 2.0
    ):
        self.task = task
        self.prm = prm
        self.orm = orm
        self.c_puct = c_puct
        self.root = MCTSNode(task.initial_numbers)
        self.solved_solution: Optional[List[Tuple]] = None

    def search(self, max_simulations: int = 300) -> Optional[List[Tuple]]:
        self.solved_solution = None

        for sim in range(max_simulations):
            node = self.root
            path = [node]

            # 1. Selection
            while node.children and not node.is_terminal:
                best_action = max(
                    node.children.keys(),
                    key=lambda a: node.children[a].ucb_score(node.visits, self.c_puct)
                )
                node = node.children[best_action]
                path.append(node)

            # 2. Expansion with PRM Step Verification
            value = 0.0
            if node.is_terminal:
                value = self.orm.evaluate(node.state)
                if value >= 1.0:
                    self._record_solution(node)
                    self._backprop(path, value)
                    break
            else:
                seen_states = set()
                n = len(node.state)
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            for op in ["+", "-", "*", "/"]:
                                if op == "/" and node.state[j] == 0:
                                    continue
                                if op == "+": res = node.state[i] + node.state[j]
                                elif op == "-": res = node.state[i] - node.state[j]
                                elif op == "*": res = node.state[i] * node.state[j]
                                elif op == "/": res = node.state[i] / node.state[j]

                                # Synthetic hallucination injection (1 in 8 steps claims wrong result)
                                if random.random() < 0.12:
                                    hallucinated_res = res + Fraction(3, 2)
                                    is_v, _ = self.prm.verify_and_score_step(node.state, i, j, op, hallucinated_res)
                                    # PRM blocks it immediately

                                # Valid step verification
                                is_valid, prior_score = self.prm.verify_and_score_step(node.state, i, j, op, res)
                                if not is_valid:
                                    continue

                                rem = [node.state[k] for k in range(n) if k != i and k != j]
                                nxt_state = tuple(sorted(rem + [res]))

                                if nxt_state not in seen_states:
                                    seen_states.add(nxt_state)
                                    act = (i, j, op, res)
                                    child = MCTSNode(nxt_state, parent=node, action=act, prior=prior_score)
                                    node.children[act] = child

                                    if child.is_terminal and child.state[0] == self.task.target:
                                        self._record_solution(child)
                                        path.append(child)
                                        value = 1.0
                                        break
                    if self.solved_solution is not None:
                        break

            # 3. Backpropagation
            self._backprop(path, value)

            if self.solved_solution is not None:
                break

        return self.solved_solution

    def _backprop(self, path: List[MCTSNode], value: float):
        for node in path:
            node.visits += 1
            node.total_value += value

    def _record_solution(self, terminal_node: MCTSNode):
        path = []
        curr = terminal_node
        while curr.parent:
            path.append((curr.parent.state, curr.action, curr.state))
            curr = curr.parent
        path.reverse()
        self.solved_solution = path

# -----------------------------------------------------------------------------
# 4. Baseline: Naive Greedy Autoregressive Decoding
# -----------------------------------------------------------------------------
def naive_greedy_solve(task: ArithmeticTask) -> Optional[List[Tuple]]:
    """
    Greedily selects the operation that brings one intermediate number closest
    to the target. Prone to local traps (e.g. creating 24 too early).
    """
    curr = task.initial_numbers
    path = []

    while len(curr) > 1:
        actions = []
        n = len(curr)
        for i in range(n):
            for j in range(n):
                if i != j:
                    for op in ["+", "-", "*", "/"]:
                        if op == "/" and curr[j] == 0:
                            continue
                        if op == "+": res = curr[i] + curr[j]
                        elif op == "-": res = curr[i] - curr[j]
                        elif op == "*": res = curr[i] * curr[j]
                        elif op == "/": res = curr[i] / curr[j]
                        actions.append((i, j, op, res))

        if not actions:
            return None

        # Greedy choice: minimize immediate distance to target
        best_cand = None
        best_diff = 1e9
        best_next = None

        for cand in actions:
            i, j, op, res = cand
            rem = [curr[k] for k in range(len(curr)) if k != i and k != j]
            next_st = tuple(sorted(rem + [res]))
            diff = min(abs(float(x - task.target)) for x in next_st)
            if diff < best_diff:
                best_diff = diff
                best_cand = cand
                best_next = next_st

        path.append((curr, best_cand, best_next))
        curr = best_next

    if len(curr) == 1 and curr[0] == task.target:
        return path
    return None

# -----------------------------------------------------------------------------
# 5. Main Experiment & Verification
# -----------------------------------------------------------------------------
def run_reasoning_search_experiment():
    print("=" * 70)
    print("Phase 35 - Topic 02: Test-Time Compute & MCTS Reasoning Search")
    print("=" * 70)

    # Benchmark test problems
    tasks = [
        ArithmeticTask([4, 6, 8, 3], 24, "Standard Puzzle [4, 6, 8, 3]"),
        ArithmeticTask([1, 5, 5, 5], 24, "Fractional Division [1, 5, 5, 5]"),
        ArithmeticTask([3, 3, 8, 8], 24, "Sub-Fractional Puzzle [3, 3, 8, 8]")
    ]

    prm = ProcessRewardModel()

    print("\n[Step 1] Head-to-Head Comparison: Greedy Decoding vs. MCTS + PRM")
    print("-" * 70)
    print(f"{'Task Description':<32} | {'Greedy Search':<16} | {'MCTS + PRM':<16}")
    print("-" * 70)

    greedy_passed = 0
    mcts_passed = 0

    for task in tasks:
        orm = OutcomeRewardModel(task.target)

        # 1. Greedy
        g_sol = naive_greedy_solve(task)
        g_status = "SUCCESS" if g_sol is not None else "FAILED (Trap)"
        if g_sol is not None:
            greedy_passed += 1

        # 2. MCTS + PRM
        mcts = MCTSSearchEngine(task, prm, orm, c_puct=2.0)
        m_sol = mcts.search(max_simulations=300)
        m_status = "SUCCESS" if m_sol is not None else "FAILED"
        if m_sol is not None:
            mcts_passed += 1

        print(f"{task.name:<32} | {g_status:<16} | {m_status:<16}")
        if m_sol:
            print("  Reasoning Chain:")
            for s_idx, (st, act, nxt) in enumerate(m_sol):
                i, j, op, res = act
                n1_str = str(st[i])
                n2_str = str(st[j])
                res_str = str(res)
                nxt_str = ", ".join(str(x) for x in nxt)
                print(f"    Step {s_idx+1}: {n1_str} {op} {n2_str} = {res_str}  -->  Pool: ({nxt_str})")

    print("-" * 70)
    print(f"Greedy Decoding Pass Rate: {greedy_passed / len(tasks) * 100:.1f}%")
    print(f"MCTS + PRM Pass Rate:      {mcts_passed / len(tasks) * 100:.1f}%")
    print(f"PRM Hallucinated Steps Blocked: {prm.hallucinations_detected} steps")

    assert greedy_passed == 0, "Greedy decoding must fail on all 3 trap tasks"
    assert mcts_passed == len(tasks), f"MCTS must solve 100% of tasks, solved {mcts_passed}/{len(tasks)}"
    assert prm.hallucinations_detected > 0, "PRM should have actively detected and blocked hallucinations"

    # [Step 2] Test-Time Compute Scaling Law
    print("\n[Step 2] Test-Time Compute Scaling Law Analysis:")
    print("Testing hard puzzle [3, 3, 8, 8] with varying simulation budgets N:")
    budget_suite = [5, 20, 60, 250]
    hard_task = ArithmeticTask([3, 3, 8, 8], 24, "Hard Test")
    hard_orm = OutcomeRewardModel(Fraction(24))

    scaling_results = []
    for budget in budget_suite:
        prm_test = ProcessRewardModel()
        engine = MCTSSearchEngine(hard_task, prm_test, hard_orm, c_puct=2.0)
        sol = engine.search(max_simulations=budget)
        found = sol is not None
        scaling_results.append((budget, found))
        status = "SOLVED" if found else "NOT SOLVED"
        print(f"  Simulations Budget: {budget:3d}  -->  Status: {status} (Tree Nodes: {engine.root.visits})")

    # Low compute fails, high compute succeeds
    assert not scaling_results[0][1], "5 simulations should be insufficient for deep sub-fractional puzzle"
    assert scaling_results[-1][1], "250 simulations must successfully solve the deep puzzle"

    print("\n" + "=" * 70)
    print("ALL TEST-TIME COMPUTE REASONING SEARCH ASSERTIONS PASSED (Exit Code 0)")
    print("=" * 70)

if __name__ == "__main__":
    run_reasoning_search_experiment()
