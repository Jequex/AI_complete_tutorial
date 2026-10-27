#!/usr/bin/env python3
"""
Phase 31: AI Evaluation, Benchmarks & Metrics
Topic 03: NLP & Language Model Metrics

Implementation of:
1. Language Model Perplexity (PPL) derived from cross-entropy token log probabilities.
2. BLEU-4 with clipped n-gram precision and brevity penalty (BP).
3. ROUGE family: ROUGE-1, ROUGE-2, and ROUGE-L (via 2D Dynamic Programming LCS).
4. SQuAD-style Exact Match (EM) and token-level F1 with text normalization and multi-reference matching.

Pure Python 3 standard library only. No external dependencies.
"""

import math
import re
import collections
from typing import List, Tuple, Dict, Set, Any


def compute_perplexity(log_probs: List[float]) -> Dict[str, float]:
    """
    Computes cross-entropy loss and perplexity from list of conditional token log-probabilities ln(P(x_i | x_<i)).
    """
    n = len(log_probs)
    assert n > 0, "Token list cannot be empty"
    # Cross entropy in natural nats
    cross_entropy_nats = -sum(log_probs) / n
    # Cross entropy in base-2 bits
    cross_entropy_bits = cross_entropy_nats / math.log(2.0)
    # Perplexity
    ppl = math.exp(cross_entropy_nats)

    return {
        "num_tokens": float(n),
        "cross_entropy_nats": cross_entropy_nats,
        "cross_entropy_bits": cross_entropy_bits,
        "perplexity": ppl,
    }


def tokenize(text: str) -> List[str]:
    """Simple regex word tokenizer."""
    return re.findall(r"\b\w+\b", text.lower())


def get_ngrams(tokens: List[str], n: int) -> collections.Counter:
    """Extracts n-grams as tuples of tokens."""
    if len(tokens) < n:
        return collections.Counter()
    return collections.Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def compute_bleu(
    candidate: str,
    references: List[str],
    max_n: int = 4,
    smoothing: bool = True,
) -> Dict[str, Any]:
    """
    Computes BLEU-N score with modified clipped precision and brevity penalty.
    """
    cand_tokens = tokenize(candidate)
    ref_token_lists = [tokenize(ref) for ref in references]

    c = len(cand_tokens)
    # Effective reference length r: closest reference length to c
    r = min(ref_token_lists, key=lambda refs: (abs(len(refs) - c), len(refs)))
    r_len = len(r)

    # Brevity penalty
    if c == 0:
        return {"bleu": 0.0, "brevity_penalty": 0.0, "precisions": [0.0] * max_n}

    bp = 1.0 if c > r_len else math.exp(1.0 - float(r_len) / float(c))

    precisions = []
    for n in range(1, max_n + 1):
        cand_ngrams = get_ngrams(cand_tokens, n)
        cand_total = sum(cand_ngrams.values())

        if cand_total == 0:
            p_n = 0.0
        else:
            # Maximum reference count for each ngram
            ref_max_counts = collections.defaultdict(int)
            for ref_tokens in ref_token_lists:
                ref_ngrams = get_ngrams(ref_tokens, n)
                for gram, count in ref_ngrams.items():
                    if count > ref_max_counts[gram]:
                        ref_max_counts[gram] = count

            clipped_matches = sum(
                min(count, ref_max_counts[gram]) for gram, count in cand_ngrams.items()
            )

            if smoothing and clipped_matches == 0:
                # Chen & Cherry smoothing method 1: replace 0 with small epsilon
                p_n = 0.1 / cand_total
            else:
                p_n = clipped_matches / cand_total

        precisions.append(p_n)

    # Geometric mean
    log_sum = 0.0
    for p in precisions:
        if p <= 0.0:
            return {"bleu": 0.0, "brevity_penalty": bp, "precisions": precisions}
        log_sum += (1.0 / max_n) * math.log(p)

    bleu = bp * math.exp(log_sum)
    return {
        "bleu": bleu,
        "brevity_penalty": bp,
        "precisions": precisions,
        "cand_length": c,
        "ref_length": r_len,
    }


def lcs_length(seq_a: List[str], seq_b: List[str]) -> int:
    """
    Computes length of Longest Common Subsequence (LCS) using 2D Dynamic Programming.
    Time: O(N * M), Space: O(min(N, M)).
    """
    n = len(seq_a)
    m = len(seq_b)
    # Memory-efficient DP with 2 rows
    prev = [0] * (m + 1)
    curr = [0] * (m + 1)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq_a[i - 1] == seq_b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = list(curr)

    return curr[m]


def compute_rouge(candidate: str, reference: str, beta: float = 1.0) -> Dict[str, Dict[str, float]]:
    """
    Computes ROUGE-1, ROUGE-2, and ROUGE-L (Precision, Recall, F1).
    """
    cand_tokens = tokenize(candidate)
    ref_tokens = tokenize(reference)

    results = {}

    # ROUGE-1 & ROUGE-2
    for n in (1, 2):
        cand_ngrams = get_ngrams(cand_tokens, n)
        ref_ngrams = get_ngrams(ref_tokens, n)

        ref_total = sum(ref_ngrams.values())
        cand_total = sum(cand_ngrams.values())

        match_count = sum(min(count, ref_ngrams[gram]) for gram, count in cand_ngrams.items())

        rec = match_count / ref_total if ref_total > 0 else 0.0
        prec = match_count / cand_total if cand_total > 0 else 0.0
        f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        results[f"rouge_{n}"] = {"recall": rec, "precision": prec, "f1": f1}

    # ROUGE-L (LCS)
    lcs_val = lcs_length(cand_tokens, ref_tokens)
    m = len(ref_tokens)
    n = len(cand_tokens)

    r_lcs = lcs_val / m if m > 0 else 0.0
    p_lcs = lcs_val / n if n > 0 else 0.0
    denom = r_lcs + (beta * beta) * p_lcs
    f_lcs = (1.0 + beta * beta) * r_lcs * p_lcs / denom if denom > 0 else 0.0

    results["rouge_l"] = {"recall": r_lcs, "precision": p_lcs, "f1": f_lcs, "lcs": float(lcs_val)}
    return results


def squad_normalize_answer(s: str) -> str:
    """
    Standard SQuAD evaluation string normalizer:
    1. Lowercase
    2. Strip punctuation
    3. Remove articles ('a', 'an', 'the')
    4. Collapse contiguous whitespace
    """
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_squad_metrics(prediction: str, references: List[str]) -> Dict[str, float]:
    """
    Computes Exact Match (EM) and Token-level F1 against a list of acceptable reference answers.
    """
    norm_pred = squad_normalize_answer(prediction)
    pred_tokens = norm_pred.split()

    exact_match = 0.0
    max_f1 = 0.0

    for ref in references:
        norm_ref = squad_normalize_answer(ref)
        if norm_pred == norm_ref:
            exact_match = 1.0

        ref_tokens = norm_ref.split()
        common = collections.Counter(pred_tokens) & collections.Counter(ref_tokens)
        num_same = sum(common.values())

        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            f1 = 1.0 if pred_tokens == ref_tokens else 0.0
        elif num_same == 0:
            f1 = 0.0
        else:
            prec = num_same / len(pred_tokens)
            rec = num_same / len(ref_tokens)
            f1 = (2.0 * prec * rec) / (prec + rec)

        if f1 > max_f1:
            max_f1 = f1

    return {
        "exact_match": exact_match,
        "token_f1": max_f1,
    }


def run_experiment():
    print("=" * 80)
    print("PHASE 31 — TOPIC 03: NLP & LANGUAGE MODEL METRICS")
    print("=" * 80)

    # 1. Perplexity computation
    # Simulated token probabilities for a 6-token sentence
    token_probs = [0.25, 0.12, 0.50, 0.35, 0.08, 0.45]
    log_probs = [math.log(p) for p in token_probs]
    ppl_res = compute_perplexity(log_probs)

    print("\n[1] Language Model Perplexity Evaluation:")
    print(f"  • Token Count          : {int(ppl_res['num_tokens'])}")
    print(f"  • Cross-Entropy (Nats) : {ppl_res['cross_entropy_nats']:.4f} nats/token")
    print(f"  • Cross-Entropy (Bits) : {ppl_res['cross_entropy_bits']:.4f} bits/token")
    print(f"  • Perplexity (PPL)     : {ppl_res['perplexity']:.4f} (Branching factor)")

    # 2. BLEU-4 Evaluation
    candidate_text = "The deep neural network achieved state of the art accuracy on the benchmark."
    reference_texts = [
        "The deep neural network achieved state of the art performance on the test benchmark.",
        "A deep neural net attained state-of-the-art accuracy on the benchmark dataset.",
    ]
    bleu_res = compute_bleu(candidate_text, reference_texts, max_n=4)

    print("\n[2] BLEU-4 Machine Translation & Generation Evaluation:")
    print(f"  • Candidate Text       : \"{candidate_text}\"")
    print(f"  • Candidate Length (c) : {bleu_res['cand_length']} words")
    print(f"  • Reference Length (r) : {bleu_res['ref_length']} words")
    print(f"  • Brevity Penalty (BP) : {bleu_res['brevity_penalty']:.4f}")
    for n, p_n in enumerate(bleu_res["precisions"], start=1):
        print(f"  • Modified p_{n} (n-gram): {p_n:.4f}")
    print(f"  • BLEU-4 Score         : {bleu_res['bleu'] * 100:.2f}%")

    # 3. ROUGE Evaluation (ROUGE-1, ROUGE-2, ROUGE-L)
    cand_summary = "Transformers use multi-head attention mechanisms to model long-range sequence dependencies."
    ref_summary = "The Transformer architecture relies on self-attention mechanisms to capture long-range token dependencies."
    rouge_res = compute_rouge(cand_summary, ref_summary)

    print("\n[3] ROUGE Summarization Evaluation:")
    print(f"  • Candidate Summary: \"{cand_summary}\"")
    print(f"  • Reference Summary: \"{ref_summary}\"")
    print(f"  • ROUGE-1 : Precision={rouge_res['rouge_1']['precision']:.4f}, Recall={rouge_res['rouge_1']['recall']:.4f}, F1={rouge_res['rouge_1']['f1']:.4f}")
    print(f"  • ROUGE-2 : Precision={rouge_res['rouge_2']['precision']:.4f}, Recall={rouge_res['rouge_2']['recall']:.4f}, F1={rouge_res['rouge_2']['f1']:.4f}")
    print(f"  • ROUGE-L : LCS={int(rouge_res['rouge_l']['lcs'])}, Precision={rouge_res['rouge_l']['precision']:.4f}, Recall={rouge_res['rouge_l']['recall']:.4f}, F1={rouge_res['rouge_l']['f1']:.4f}")

    # 4. Question Answering Evaluation (SQuAD Exact Match & Token F1)
    qa_cases = [
        {
            "pred": "The Apollo 11 mission",
            "refs": ["Apollo 11", "Apollo eleven"],
            "expected_em": 0.0,
        },
        {
            "pred": "Albert Einstein",
            "refs": ["Albert Einstein", "Dr. Albert Einstein"],
            "expected_em": 1.0,
        },
        {
            "pred": "in 1969",
            "refs": ["1969", "July 1969"],
            "expected_em": 0.0,
        },
    ]

    print("\n[4] SQuAD Question Answering Evaluation:")
    for idx, case in enumerate(qa_cases, start=1):
        qa_metric = compute_squad_metrics(case["pred"], case["refs"])
        print(f"  Case {idx}: Pred=\"{case['pred']}\", Refs={case['refs']}")
        print(f"    &rarr; Exact Match (EM) = {qa_metric['exact_match']}, Token F1 = {qa_metric['token_f1']:.4f}")
        assert qa_metric["exact_match"] == case["expected_em"]

    # Assertions
    assert ppl_res["perplexity"] > 1.0, "Perplexity must be strictly greater than 1"
    assert 0.0 <= bleu_res["bleu"] <= 1.0, "BLEU must lie in [0, 1]"
    assert 0.0 <= rouge_res["rouge_l"]["f1"] <= 1.0, "ROUGE-L must lie in [0, 1]"
    print("\nAll NLP & Language Model metric assertions passed successfully! (Exit Code 0)")


if __name__ == "__main__":
    run_experiment()
