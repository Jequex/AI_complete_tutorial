#!/usr/bin/env python3
"""
Phase 22 - Topic 05: Context Construction and RAG Evaluation
Experiment: Pure Python RAGAS-Style Evaluation Harness & Lost-in-the-Middle Reordering

This experiment demonstrates:
1. "Lost in the Middle" context reordering (placing high-ranked chunks at extremes: start & end).
2. Pure Python RAG Evaluation Harness calculating:
   - Faithfulness (Groundedness): Verified factual claims / Total generated claims.
   - Answer Relevance: Semantic alignment between generated response and query.
   - Context Precision: Mean Average Precision of relevant retrieved chunks.
   - Context Recall: Ground-truth reference statements supported by retrieved context.
3. Comparative discrimination benchmark:
   - Evaluating a High-Quality Grounded RAG response.
   - Evaluating a Hallucinated / Contaminated response.
   - Evaluating an Evasive / Off-topic response.
4. Verification of metric discrimination thresholds and assertions.

Zero external dependencies: pure standard Python 3.
"""

import math
import re
import sys
from typing import List, Dict, Tuple, Set, Any, Optional


# ============================================================================
# 1. Lost-in-the-Middle Context Reordering
# ============================================================================
def reorder_context_for_primacy_recency(chunks: List[str]) -> List[str]:
    """
    Mitigates Liu et al. (2023) "Lost in the Middle" effect.
    Alternates placement so highest-ranked chunks occupy the start and end of the context window.
    Input ranks: [1, 2, 3, 4, 5] -> Output: [1, 3, 5, 4, 2]
    """
    if len(chunks) <= 2:
        return list(chunks)

    reordered = []
    front = True
    left_side = []
    right_side = []

    for chunk in chunks:
        if front:
            left_side.append(chunk)
        else:
            right_side.insert(0, chunk)
        front = not front

    return left_side + right_side


# ============================================================================
# 2. Pure Python RAG Evaluation Framework (RAGAS Formulation)
# ============================================================================
class RAGEvaluator:
    """
    Computes core RAG evaluation metrics without external LLMs or APIs.
    """
    def __init__(self):
        self.stopwords = {
            "a", "an", "the", "is", "are", "was", "were", "in", "on", "at",
            "to", "for", "of", "with", "by", "this", "that", "it", "its"
        }

    def _tokenize(self, text: str) -> List[str]:
        words = re.findall(r"\b\w+\b", text.lower())
        return [w for w in words if w not in self.stopwords]

    def _extract_claims(self, answer: str) -> List[str]:
        # Split into individual sentence-level atomic claims
        raw = re.split(r"(?<=[.!?])\s+", answer.strip())
        return [s.strip() for s in raw if len(s.strip()) > 5]

    def calculate_faithfulness(self, answer: str, context: List[str]) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Faithfulness (Groundedness) = (Number of claims verified in context) / (Total claims in answer).
        Verifies if each factual assertion in the answer is entailed by the context.
        """
        claims = self._extract_claims(answer)
        if not claims:
            return 0.0, []

        context_text = " ".join(context).lower()
        context_tokens = set(self._tokenize(context_text))

        claim_evaluations = []
        grounded_count = 0

        for claim in claims:
            claim_tokens = self._tokenize(claim)
            if not claim_tokens:
                continue

            # Check overlap fraction of key substantive words
            overlap = [t for t in claim_tokens if t in context_tokens]
            overlap_ratio = len(overlap) / len(claim_tokens)

            # A claim is verified if >= 75% of its informative content is supported by context
            is_grounded = overlap_ratio >= 0.75
            if is_grounded:
                grounded_count += 1

            claim_evaluations.append({
                "claim": claim,
                "overlap_ratio": overlap_ratio,
                "grounded": is_grounded
            })

        faithfulness = grounded_count / len(claim_evaluations) if claim_evaluations else 0.0
        return faithfulness, claim_evaluations

    def calculate_answer_relevance(self, query: str, answer: str) -> float:
        """
        Answer Relevance = Semantic cosine similarity & question intent coverage.
        Penalizes answers that evade the question or discuss unrelated topics.
        """
        q_tokens = self._tokenize(query)
        a_tokens = self._tokenize(answer)
        if not q_tokens or not a_tokens:
            return 0.0

        # Term frequency vectors
        vocab = sorted(list(set(q_tokens + a_tokens)))
        v_q = [q_tokens.count(w) for w in vocab]
        v_a = [a_tokens.count(w) for w in vocab]

        # Cosine similarity
        dot = sum(q * a for q, a in zip(v_q, v_a))
        norm_q = math.sqrt(sum(q * q for q in v_q))
        norm_a = math.sqrt(sum(a * a for a in v_a))

        if norm_q == 0.0 or norm_a == 0.0:
            return 0.0

        cos_sim = dot / (norm_q * norm_a)
        
        # Coverage multiplier: fraction of query keywords addressed
        q_matched = sum(1 for w in set(q_tokens) if w in set(a_tokens))
        coverage = q_matched / len(set(q_tokens))

        return (cos_sim * 0.5) + (coverage * 0.5)

    def calculate_context_precision(self, ranked_chunk_relevance: List[bool]) -> float:
        """
        Context Precision = Mean Average Precision (MAP) of retrieved chunks.
        Evaluates whether relevant chunks were placed at top ranks:
        Precision@k = (Relevant chunks up to k) / k
        MAP = sum(Precision@k * rel_k) / (Total relevant chunks)
        """
        total_relevant = sum(1 for r in ranked_chunk_relevance if r)
        if total_relevant == 0:
            return 0.0

        precisions = []
        relevant_so_far = 0

        for k, is_rel in enumerate(ranked_chunk_relevance, start=1):
            if is_rel:
                relevant_so_far += 1
                precision_at_k = relevant_so_far / k
                precisions.append(precision_at_k)

        return sum(precisions) / total_relevant

    def calculate_context_recall(self, ground_truth_statements: List[str], retrieved_context: List[str]) -> float:
        """
        Context Recall = (Ground truth statements found in context) / (Total ground truth statements).
        """
        if not ground_truth_statements:
            return 1.0

        context_blob = " ".join(retrieved_context).lower()
        context_words = set(self._tokenize(context_blob))

        covered = 0
        for statement in ground_truth_statements:
            st_words = self._tokenize(statement)
            match_count = sum(1 for w in st_words if w in context_words)
            if len(st_words) > 0 and (match_count / len(st_words)) >= 0.70:
                covered += 1

        return covered / len(ground_truth_statements)


def run_experiment():
    print("=" * 80)
    print("PHASE 22 - TOPIC 05: CONTEXT CONSTRUCTION & RAG EVALUATION EXPERIMENT")
    print("=" * 80)

    # 1. Test "Lost in the Middle" Reordering
    print("\n[Step 1] Testing 'Lost in the Middle' Primacy-Recency Reordering...")
    ranked_chunks = [
        "[Rank 1: Most Relevant]",
        "[Rank 2: Second Relevant]",
        "[Rank 3: Third Relevant]",
        "[Rank 4: Fourth Relevant]",
        "[Rank 5: Fifth Relevant]"
    ]
    reordered = reorder_context_for_primacy_recency(ranked_chunks)
    print("  Original Ranked Order: ", ranked_chunks)
    print("  Reordered Context:      ", reordered)

    # The top-1 chunk should be at index 0, and top-2 chunk should be at the very end
    assert reordered[0] == "[Rank 1: Most Relevant]", "Top chunk must be at the very beginning (Primacy bias)."
    assert reordered[-1] == "[Rank 2: Second Relevant]", "Second chunk must be at the very end (Recency bias)."
    print("✓ Lost-in-the-Middle reordering verified: Top facts placed at primacy & recency positions.")

    # 2. Context & Query Setup for Evaluation Harness
    evaluator = RAGEvaluator()

    query = "What causes the high-temperature alert in reactor turbine three?"
    retrieved_context = [
        "Turbine three operates with a nominal lubricating oil pressure of 45 PSI.",
        "A high-temperature alert in reactor turbine three is caused by coolant valve failure restricting circulation.",
        "Routine maintenance was conducted by technician Alpha on September 12."
    ]
    ground_truth = [
        "A high-temperature alert in reactor turbine three is caused by coolant valve failure restricting circulation."
    ]

    # Candidate Responses to Evaluate
    good_response = (
        "The high-temperature alert in reactor turbine three is triggered by coolant valve failure that restricts fluid circulation."
    )
    
    hallucinated_response = (
        "The high-temperature alert in reactor turbine three is caused by a massive cyber attack by rogue hackers. "
        "They injected malicious ransomware into the industrial control logic and overclocked the magnetic rotors."
    )
    
    evasive_response = (
        "Nuclear reactors produce clean thermal energy by splitting heavy uranium atoms inside graphite moderator assemblies."
    )

    # 3. Evaluate High-Quality Grounded RAG Response
    print("\n[Step 2] Evaluating Grounded RAG Response...")
    faith_good, claims_good = evaluator.calculate_faithfulness(good_response, retrieved_context)
    rel_good = evaluator.calculate_answer_relevance(query, good_response)
    print(f"  Response: \"{good_response}\"")
    print(f"  -> Faithfulness (Groundedness): {faith_good * 100:.1f}%")
    print(f"  -> Answer Relevance:           {rel_good * 100:.1f}%")

    assert faith_good == 1.0, f"Expected 100% faithfulness for grounded response, got {faith_good}"
    assert rel_good > 0.60, f"Expected high relevance for grounded response, got {rel_good}"
    print("✓ Grounded response achieves high faithfulness and relevance.")

    # 4. Evaluate Hallucinated Response
    print("\n[Step 3] Evaluating Hallucinated Response (Contaminated)...")
    faith_hallucinated, claims_hallucinated = evaluator.calculate_faithfulness(hallucinated_response, retrieved_context)
    rel_hallucinated = evaluator.calculate_answer_relevance(query, hallucinated_response)
    print(f"  Response: \"{hallucinated_response[:75]}...\"")
    print(f"  -> Faithfulness (Groundedness): {faith_hallucinated * 100:.1f}%")
    print(f"  -> Answer Relevance:           {rel_hallucinated * 100:.1f}%")

    assert faith_hallucinated == 0.0, f"Hallucinated claims must yield 0% faithfulness, got {faith_hallucinated}"
    print("✓ Hallucination correctly caught: Faithfulness collapses to 0.0.")

    # 5. Evaluate Evasive / Off-Topic Response
    print("\n[Step 4] Evaluating Off-Topic Evasive Response...")
    rel_evasive = evaluator.calculate_answer_relevance(query, evasive_response)
    print(f"  Response: \"{evasive_response}\"")
    print(f"  -> Answer Relevance: {rel_evasive * 100:.1f}%")
    assert rel_evasive < rel_good * 0.5, "Off-topic response must score significantly lower in relevance."
    print("✓ Off-topic evasion penalized: Relevance score drops.")

    # 6. Evaluate Context Precision (MAP) & Context Recall
    print("\n[Step 5] Evaluating Retrieval Metrics (Context Precision & Recall)...")
    # Scenario A: Relevant chunk ranked #1, followed by 2 irrelevant chunks [True, False, False]
    precision_optimal = evaluator.calculate_context_precision([True, False, False])
    # Scenario B: Relevant chunk buried at #3 [False, False, True]
    precision_suboptimal = evaluator.calculate_context_precision([False, False, True])
    
    print(f"  Optimal Retrieval Precision (Rank 1):  {precision_optimal * 100:.1f}%")
    print(f"  Suboptimal Precision (Buried Rank 3): {precision_suboptimal * 100:.1f}%")
    assert precision_optimal > precision_suboptimal, "Placing relevant chunks at rank 1 must yield higher Context Precision."

    recall = evaluator.calculate_context_recall(ground_truth, retrieved_context)
    print(f"  Context Recall (Ground Truth Coverage): {recall * 100:.1f}%")
    assert recall == 1.0, "Retrieved context must contain 100% of ground truth statements."
    print("✓ Context Precision and Recall metrics properly discriminate retrieval quality.")

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED: Topic 05 Context & RAG Evaluation fully verified (Exit Code 0).")
    print("=" * 80)


if __name__ == "__main__":
    run_experiment()
    sys.exit(0)
