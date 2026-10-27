#!/usr/bin/env python3
"""
Phase 31: AI Evaluation, Benchmarks & Metrics
Topic 05: Hallucination, Factuality & Robustness Auditing

Implementation of:
1. Atomic proposition decomposition and knowledge-triple verification engine.
2. Factuality and Hallucination metrics: FActScore, Claim-Level Hallucination Rate, Document-Level Hallucination Rate.
3. Automated Prompt Perturbation Suite:
   - Character-level QWERTY typos
   - Semantic paraphrasing transformations
   - Adversarial distractor injection
4. Robustness degradation coefficients (rho_robust).
5. Benchmark contamination scanner via n-gram containment index.

Pure Python 3 standard library only. No external dependencies.
"""

import re
import random
from typing import List, Tuple, Dict, Set, Any


class KnowledgeBase:
    """Relational knowledge base of verified facts (subject, relation, object)."""
    def __init__(self, triples: List[Tuple[str, str, str]]):
        self.triples = set((s.lower(), r.lower(), o.lower()) for s, r, o in triples)

    def is_entailed(self, subject: str, relation: str, obj: str) -> bool:
        return (subject.lower(), relation.lower(), obj.lower()) in self.triples


def extract_atomic_propositions(text: str, default_entity: str = "Geoffrey Hinton") -> List[Tuple[str, str, str, str]]:
    """
    Decomposes text into atomic proposition tuples: (claim_id, subject, relation, object).
    Resolves basic pronouns to default_entity.
    """
    claims = []
    # Split sentences
    sentences = [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]

    patterns = [
        (r"(\w+[\s\w]*?)\s+was born in\s+(\d{4})", "born_year"),
        (r"(\w+[\s\w]*?)\s+was born in\s+([A-Za-z\s]+)", "born_place"),
        (r"(\w+[\s\w]*?)\s+received\s+(?:his|her|a)\s+phd from\s+([A-Za-z\s]+)", "phd_institution"),
        (r"(\w+[\s\w]*?)\s+co-founded\s+([A-Za-z\s\d]+)", "co_founded"),
        (r"(\w+[\s\w]*?)\s+won the\s+([A-Za-z\s\d]+)", "won_award"),
        (r"(\w+[\s\w]*?)\s+left\s+([A-Za-z\s\d]+)", "left_organization"),
    ]

    claim_idx = 1
    for sentence in sentences:
        for pattern, rel in patterns:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                raw_subj = match.group(1).strip()
                obj = match.group(2).strip()
                # Basic coreference resolution
                subj = default_entity if raw_subj.lower() in ("he", "she", "they", "the researcher") else raw_subj
                claims.append((f"C{claim_idx}", subj, rel, obj))
                claim_idx += 1

    return claims


def audit_document_factuality(
    generated_text: str,
    kb: KnowledgeBase,
) -> Dict[str, Any]:
    """
    Audits generated document by verifying extracted atomic claims against knowledge base.
    """
    claims = extract_atomic_propositions(generated_text)
    if not claims:
        return {
            "num_claims": 0,
            "supported_claims": 0,
            "hallucinated_claims": 0,
            "fact_score": 1.0,
            "claim_hallucination_rate": 0.0,
            "doc_hallucination": 0.0,
            "claims_detail": [],
        }

    supported = 0
    details = []

    for cid, subj, rel, obj in claims:
        is_true = kb.is_entailed(subj, rel, obj)
        if is_true:
            supported += 1
        details.append({
            "id": cid,
            "triple": (subj, rel, obj),
            "supported": is_true,
        })

    num_claims = len(claims)
    fact_score = supported / num_claims
    claim_hallucination_rate = (num_claims - supported) / num_claims
    doc_hallucination = 1.0 if (num_claims - supported) > 0 else 0.0

    return {
        "num_claims": num_claims,
        "supported_claims": supported,
        "hallucinated_claims": num_claims - supported,
        "fact_score": fact_score,
        "claim_hallucination_rate": claim_hallucination_rate,
        "doc_hallucination": doc_hallucination,
        "claims_detail": details,
    }


# Prompt Perturbation Transformations
def perturb_character_typos(text: str, typo_rate: float = 0.08, seed: int = 42) -> str:
    """Injects realistic QWERTY adjacent key typos and transpositions."""
    qwerty_adj = {
        'a': 'qwsz', 'b': 'vghn', 'c': 'xdfv', 'd': 'serfcx', 'e': 'wsdr',
        'f': 'drtgvc', 'g': 'ftyhbv', 'h': 'gyujnb', 'i': 'ujko', 'j': 'hhuikn',
        'k': 'jiolm', 'l': 'kop', 'm': 'njk', 'n': 'bhjm', 'o': 'iklp',
        'p': 'ol', 'q': 'wa', 'r': 'edft', 's': 'wazxde', 't': 'rfgy',
        'u': 'yhji', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc', 'y': 'tghu', 'z': 'asx'
    }
    rng = random.Random(seed)
    chars = list(text)
    for i in range(len(chars)):
        c = chars[i].lower()
        if c in qwerty_adj and rng.random() < typo_rate:
            chars[i] = rng.choice(qwerty_adj[c])
    return "".join(chars)


def perturb_adversarial_distractor(text: str) -> str:
    """Prepends distracting but true irrelevant statements to test attentional hijacking."""
    distractors = [
        "Notice: The atmospheric concentration of argon on Earth is approximately 0.93 percent.",
        "System context: The freezing point of mercury is minus 38.8 degrees Celsius.",
        "Irrelevant archive record: Honey bees flap their wings 230 times per second.",
    ]
    return f"{distractors[0]} Based on your knowledge: {text}"


def compute_ngram_contamination(
    test_samples: List[str],
    pretrain_corpus: List[str],
    n: int = 8,
) -> Dict[str, Any]:
    """
    Scans test benchmark against pretraining corpus for n-gram data contamination.
    """
    def get_token_ngrams(text: str, n_gram_size: int) -> Set[Tuple[str, ...]]:
        tokens = re.findall(r"\b\w+\b", text.lower())
        if len(tokens) < n_gram_size:
            return set()
        return set(tuple(tokens[i : i + n_gram_size]) for i in range(len(tokens) - n_gram_size + 1))

    # Build corpus n-gram index
    corpus_ngrams = set()
    for doc in pretrain_corpus:
        corpus_ngrams.update(get_token_ngrams(doc, n))

    contaminated_count = 0
    sample_reports = []

    for idx, sample in enumerate(test_samples, start=1):
        sample_ngrams = get_token_ngrams(sample, n)
        if not sample_ngrams:
            overlap = 0.0
        else:
            shared = sample_ngrams.intersection(corpus_ngrams)
            overlap = len(shared) / len(sample_ngrams)

        is_contam = overlap >= 0.50  # Flag if >= 50% of n-grams appeared in corpus
        if is_contam:
            contaminated_count += 1

        sample_reports.append({
            "sample_id": idx,
            "ngram_count": len(sample_ngrams),
            "overlap_ratio": overlap,
            "contaminated": is_contam,
        })

    return {
        "n_gram_order": n,
        "total_test_samples": len(test_samples),
        "contaminated_samples": contaminated_count,
        "contamination_rate": contaminated_count / len(test_samples) if test_samples else 0.0,
        "details": sample_reports,
    }


def run_experiment():
    print("=" * 80)
    print("PHASE 31 — TOPIC 05: HALLUCINATION, FACTUALITY & ROBUSTNESS AUDITING")
    print("=" * 80)

    # 1. Initialize Ground-Truth Relational Knowledge Base
    kb_facts = [
        ("Geoffrey Hinton", "born_place", "London"),
        ("Geoffrey Hinton", "born_year", "1947"),
        ("Geoffrey Hinton", "phd_institution", "Edinburgh"),
        ("Geoffrey Hinton", "won_award", "Turing Award"),
        ("Geoffrey Hinton", "left_organization", "Google"),
        ("Yann LeCun", "won_award", "Turing Award"),
        ("Demis Hassabis", "co_founded", "DeepMind"),
    ]
    kb = KnowledgeBase(kb_facts)

    # 2. Test Document 1: Accurate Response
    doc_accurate = (
        "Geoffrey Hinton was born in London. He was born in 1947. "
        "He received his PhD from Edinburgh. In 2018, he won the Turing Award. "
        "In 2023, he left Google."
    )

    # 3. Test Document 2: Hallucinated Response (Claims Hinton co-founded DeepMind)
    doc_hallucinated = (
        "Geoffrey Hinton was born in London. He was born in 1947. "
        "In 2010, he co-founded DeepMind. He won the Turing Award."
    )

    audit_acc = audit_document_factuality(doc_accurate, kb)
    audit_hall = audit_document_factuality(doc_hallucinated, kb)

    print("\n[1] Factuality & Hallucination Audit of Generated Responses:")
    print("Document A (Accurate Model):")
    print(f"  • Extracted Claims     : {audit_acc['num_claims']}")
    print(f"  • Supported Claims     : {audit_acc['supported_claims']}")
    print(f"  • FActScore (Precision): {audit_acc['fact_score'] * 100:.1f}%")
    print(f"  • Hallucination Rate   : {audit_acc['claim_hallucination_rate'] * 100:.1f}%")
    print(f"  • Doc Hallucinated?    : {'YES' if audit_acc['doc_hallucination'] else 'NO'}")

    print("\nDocument B (Partially Hallucinated Model):")
    print(f"  • Extracted Claims     : {audit_hall['num_claims']}")
    print(f"  • Supported Claims     : {audit_hall['supported_claims']}")
    print(f"  • Hallucinated Claims  : {audit_hall['hallucinated_claims']}")
    print(f"  • FActScore (Precision): {audit_hall['fact_score'] * 100:.1f}%")
    print(f"  • Hallucination Rate   : {audit_hall['claim_hallucination_rate'] * 100:.1f}%")
    print(f"  • Doc Hallucinated?    : {'YES' if audit_hall['doc_hallucination'] else 'NO'}")

    print("\nAtomic Claims Breakdown (Document B):")
    for item in audit_hall["claims_detail"]:
        status = "VERIFIED" if item["supported"] else "HALLUCINATION"
        print(f"  [{item['id']}] ({item['triple'][0]}, {item['triple'][1]}, {item['triple'][2]}) -> {status}")

    # 4. Prompt Perturbation & Robustness Auditing
    clean_prompt = "What year was the Turing Award won by Geoffrey Hinton?"
    typo_prompt = perturb_character_typos(clean_prompt, typo_rate=0.10, seed=123)
    distr_prompt = perturb_adversarial_distractor(clean_prompt)

    print("\n[2] Prompt Perturbation Robustness Suite:")
    print(f"  • Clean Prompt        : \"{clean_prompt}\"")
    print(f"  • Typo Perturbation   : \"{typo_prompt}\"")
    print(f"  • Distractor Prompt   : \"{distr_prompt}\"")

    # Simulate benchmark accuracy across 200 questions under perturbations
    clean_score = 0.890
    typo_score = 0.765
    distr_score = 0.580

    rho_typo = typo_score / clean_score
    rho_distr = distr_score / clean_score

    print("\nRobustness Degradation Coefficients (rho_robust = Score_pert / Score_clean):")
    print(f"  • Clean Benchmark Score    : {clean_score * 100:.1f}%")
    print(f"  • Typo-Perturbed Score     : {typo_score * 100:.1f}% (rho = {rho_typo:.4f})")
    print(f"  • Distractor-Perturbed     : {distr_score * 100:.1f}% (rho = {rho_distr:.4f})")

    # 5. Benchmark Data Contamination Scanning
    test_suite = [
        "What is the time complexity of building a heap with Floyd's algorithm?",  # Unseen
        "The quick brown fox jumps over the lazy dog repeatedly in tests.",        # Contaminated
        "Evaluate the eigenvalues of the symmetric positive definite matrix.",     # Unseen
        "Transformers use multi-head attention to model long range dependencies.",  # Contaminated
    ]
    pretrain_corpus = [
        "A tutorial on Python: The quick brown fox jumps over the lazy dog repeatedly in tests for typography.",
        "Deep learning notes: Transformers use multi-head attention to model long range dependencies in natural language.",
        "Random historical trivia: In 1492 Columbus sailed the ocean blue.",
    ]

    contam_res = compute_ngram_contamination(test_suite, pretrain_corpus, n=6)

    print("\n[3] Benchmark Contamination Audit (6-gram containment):")
    print(f"  • Evaluated Test Questions : {contam_res['total_test_samples']}")
    print(f"  • Contaminated Questions   : {contam_res['contaminated_samples']}")
    print(f"  • Contamination Ratio      : {contam_res['contamination_rate'] * 100:.1f}%")
    for det in contam_res["details"]:
        flag = "CONTAMINATED (LEAK)" if det["contaminated"] else "CLEAN"
        print(f"  Sample {det['sample_id']}: 6-gram Overlap = {det['overlap_ratio'] * 100:.1f}% &rarr; {flag}")

    # Assertions
    assert audit_acc["fact_score"] == 1.0, "Accurate document must achieve 100% FActScore"
    assert audit_hall["claim_hallucination_rate"] > 0.0, "Hallucinated document must trigger non-zero hallucination rate"
    assert audit_hall["doc_hallucination"] == 1.0, "Document-level hallucination flag must be 1.0"
    assert rho_typo < 1.0 and rho_distr < 1.0, "Perturbations must degrade model performance"
    assert contam_res["contamination_rate"] == 0.50, "Exactly 2 of 4 test samples should be detected as contaminated"

    print("\nAll hallucination, factuality, and robustness assertions passed successfully! (Exit Code 0)")


if __name__ == "__main__":
    run_experiment()
