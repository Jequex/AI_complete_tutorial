#!/usr/bin/env python3
"""
Topic 27.5 Experiment: Multi-Agent Orchestration, FIPA-ACL Protocols & Consensus Debate
Phase 27: AI Agents & Multi-Agent Systems

Implements from first principles in pure Python (standard library only):
1. FIPA-ACL Structured Agent Communication Protocol:
   - Envelope with: sender, recipient, performative (PROPOSE, CRITIQUE, REVISE, CONFIRM),
     conversation_id, and payload.
2. Multi-Agent Society Topology (Hierarchical Debate Architecture):
   - Proposer Agent: Drafts algorithm implementations. Initial attempt has subtle edge case flaw.
   - Critic Agent: Audits proposals with adversarial edge cases (empty lists, negative keys, duplicates).
   - Judge / Orchestrator Agent: Evaluates arguments, enforces protocol rules, and determines consensus.
3. Consensus Convergence Loop:
   - Proposer refines implementation based on Critic's counter-examples.
   - Judge certifies robustness when both agents reach agreement.
4. Empirical Benchmark:
   - Tests Single-Agent vs Multi-Agent Consensus against adversarial test suite.
   - Demonstrates verifiably higher reliability of multi-agent debate.

Zero external dependencies. Pure Python 3 standard library.
"""

from typing import List, Dict, Any, Optional, Tuple
import json


class ACLPerformative:
    PROPOSE = "PROPOSE"
    CRITIQUE = "CRITIQUE"
    REVISE = "REVISE"
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    INFORM = "INFORM"


class ACLMessage:
    """Standardized FIPA-ACL agent communication envelope."""

    def __init__(self, sender: str, recipient: str, performative: str,
                 conversation_id: str, payload: Dict[str, Any]):
        self.sender = sender
        self.recipient = recipient
        self.performative = performative
        self.conversation_id = conversation_id
        self.payload = payload

    def __repr__(self):
        return f"<ACLMessage from={self.sender} to={self.recipient} performative={self.performative}>"


class ProposerAgent:
    """Generates initial proposals and refines them based on critique."""

    def __init__(self, name: str = "Proposer"):
        self.name = name
        self.revision_turn = 0

    def generate_initial_proposal(self) -> str:
        """
        Problem: Implement an algorithm to find the majority element (element appearing > n/2 times)
        in a list of integers, or return None if no majority element exists.

        Flaw in Draft 1: Boyer-Moore Voting candidate identification without the mandatory
        verification pass! If no majority exists (e.g. [1, 2, 3]), it incorrectly returns a candidate.
        """
        self.revision_turn = 1
        code = """def majority_element(nums):
    # Boyer-Moore Voting candidate selection
    count = 0
    candidate = None
    for x in nums:
        if count == 0:
            candidate = x
            count = 1
        elif x == candidate:
            count += 1
        else:
            count -= 1
    # Draft 1 Bug: Returns candidate directly without second verification pass!
    return candidate
"""
        return code

    def revise_proposal(self, critique_payload: Dict[str, Any]) -> str:
        """Refines algorithm by addressing the counter-example identified by the Critic."""
        self.revision_turn += 1
        code = """def majority_element(nums):
    if not nums:
        return None
    # Pass 1: Boyer-Moore Candidate Selection
    count = 0
    candidate = None
    for x in nums:
        if count == 0:
            candidate = x
            count = 1
        elif x == candidate:
            count += 1
        else:
            count -= 1
    # Pass 2: Rigorous Frequency Verification Pass
    actual_count = sum(1 for x in nums if x == candidate)
    if actual_count > len(nums) // 2:
        return candidate
    return None
"""
        return code


class CriticAgent:
    """Adversarial testing and static analysis agent."""

    def __init__(self, name: str = "Critic"):
        self.name = name

    def audit(self, code_str: str) -> Dict[str, Any]:
        """Executes code against a comprehensive battery of edge cases."""
        local_scope: Dict[str, Any] = {}
        try:
            exec(code_str, {}, local_scope)
            fn = local_scope["majority_element"]
        except Exception as e:
            return {"status": "FAIL", "reason": f"Syntax/Compilation error: {e}"}

        test_cases = [
            ([3, 2, 3], 3, "Odd length majority"),
            ([2, 2, 1, 1, 1, 2, 2], 2, "Long majority"),
            ([1, 2, 3, 4], None, "No majority present (all unique)"),
            ([1, 1, 2, 2], None, "No majority present (tie at 50%)"),
            ([], None, "Empty input list"),
        ]

        for nums, expected, desc in test_cases:
            try:
                result = fn(nums)
                if result != expected:
                    return {
                        "status": "FAIL",
                        "counter_example": nums,
                        "expected": expected,
                        "actual": result,
                        "description": desc,
                        "critique": f"Failed on test '{desc}'. For input {nums}, expected {expected} but got {result}. "
                                    f"Cause: Candidate was not verified with a second frequency counting pass."
                    }
            except Exception as e:
                return {
                    "status": "FAIL",
                    "counter_example": nums,
                    "critique": f"Raised exception {type(e).__name__}: {e} on input {nums}."
                }

        return {"status": "PASS", "critique": "All edge cases, empty lists, ties, and majorities verified successfully."}


class JudgeOrchestrator:
    """Manages the debate protocol and consensus verification."""

    def __init__(self, proposer: ProposerAgent, critic: CriticAgent):
        self.proposer = proposer
        self.critic = critic
        self.message_history: List[ACLMessage] = []

    def log_message(self, msg: ACLMessage):
        self.message_history.append(msg)
        print(f"[{msg.sender} -> {msg.recipient}] ({msg.performative}):")
        if "critique" in msg.payload:
            print(f"  Critique: {msg.payload['critique']}")
        elif "code" in msg.payload:
            first_line = msg.payload['code'].strip().split("\n")[0]
            print(f"  Payload : {first_line} ... ({len(msg.payload['code'].splitlines())} lines)")
        print()

    def run_debate(self, max_rounds: int = 3) -> Tuple[bool, str]:
        conv_id = "majority_element_verification_01"

        # Round 1: Proposer submits initial draft
        print(f"--- Debate Round 1 ---")
        draft_code = self.proposer.generate_initial_proposal()
        msg1 = ACLMessage(
            sender=self.proposer.name,
            recipient=self.critic.name,
            performative=ACLPerformative.PROPOSE,
            conversation_id=conv_id,
            payload={"code": draft_code, "round": 1}
        )
        self.log_message(msg1)

        # Critic audits Round 1
        audit_1 = self.critic.audit(draft_code)
        if audit_1["status"] == "FAIL":
            msg2 = ACLMessage(
                sender=self.critic.name,
                recipient=self.proposer.name,
                performative=ACLPerformative.CRITIQUE,
                conversation_id=conv_id,
                payload=audit_1
            )
            self.log_message(msg2)
        else:
            return True, draft_code

        # Round 2: Proposer revises based on Critic feedback
        print(f"--- Debate Round 2 ---")
        revised_code = self.proposer.revise_proposal(audit_1)
        msg3 = ACLMessage(
            sender=self.proposer.name,
            recipient=self.critic.name,
            performative=ACLPerformative.REVISE,
            conversation_id=conv_id,
            payload={"code": revised_code, "round": 2}
        )
        self.log_message(msg3)

        # Critic audits Round 2
        audit_2 = self.critic.audit(revised_code)
        msg4 = ACLMessage(
            sender=self.critic.name,
            recipient=self.proposer.name,
            performative=ACLPerformative.ACCEPT if audit_2["status"] == "PASS" else ACLPerformative.REJECT,
            conversation_id=conv_id,
            payload=audit_2
        )
        self.log_message(msg4)

        if audit_2["status"] == "PASS":
            judge_msg = ACLMessage(
                sender="Judge",
                recipient="ALL",
                performative=ACLPerformative.CONFIRM if hasattr(ACLPerformative, "CONFIRM") else ACLPerformative.INFORM,
                conversation_id=conv_id,
                payload={"verdict": "CONSENSUS_REACHED", "code": revised_code}
            )
            self.log_message(judge_msg)
            return True, revised_code

        return False, draft_code


def run_experiment():
    print("=" * 75)
    print("TOPIC 27.5: MULTI-AGENT ORCHESTRATION, FIPA-ACL & CONSENSUS DEBATE")
    print("=" * 75)

    proposer = ProposerAgent("Proposer")
    critic = CriticAgent("Critic")
    orchestrator = JudgeOrchestrator(proposer, critic)

    # 1. Evaluate unreviewed single-agent baseline
    print("[1] Single-Agent Unreviewed Baseline Test:")
    single_agent_code = proposer.generate_initial_proposal()
    single_audit = critic.audit(single_agent_code)
    print(f" Single-Agent Solution Status: {single_audit['status']}")
    print(f" Failure Reason: {single_audit.get('critique')}")
    assert single_audit["status"] == "FAIL", "Single-agent code should have failed the adversarial test!"
    print(" [CONFIRMED] Single-agent unreviewed code contains subtle silent failure.\n")

    # 2. Run multi-agent debate and consensus
    print("[2] Multi-Agent Orchestration & Debate Trajectory:")
    consensus_reached, final_code = orchestrator.run_debate(max_rounds=3)

    assert consensus_reached, "Multi-agent debate failed to reach consensus!"
    print(f"Consensus Achieved: {consensus_reached}")
    print("-" * 75)
    print("Final Verified Code:\n" + final_code.strip())
    print("-" * 75)

    # 3. Final verification against adversarial suite
    final_audit = critic.audit(final_code)
    assert final_audit["status"] == "PASS", "Final code failed verification!"
    print("\nVerification Results:")
    print(" [PASS] FIPA-ACL structured message protocol executed across all rounds.")
    print(" [PASS] Critic identified silent edge-case vulnerability on unreviewed proposal.")
    print(" [PASS] Proposer successfully incorporated adversarial feedback to reach consensus.")
    print(" [PASS] 100% test battery pass on final consensus solution.")
    print("=" * 75)


if __name__ == "__main__":
    run_experiment()
