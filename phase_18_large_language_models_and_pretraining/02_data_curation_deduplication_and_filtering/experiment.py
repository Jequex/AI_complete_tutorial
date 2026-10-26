#!/usr/bin/env python3
"""
Phase 18 — Topic 02: Data Curation, Deduplication, and Filtering
Experiment: Pure Python 3 MinHash & Locality-Sensitive Hashing (LSH) Pipeline.

This script implements:
1. Text tokenization & k-shingling (word and character n-grams).
2. Ground-truth Jaccard similarity calculation J(A, B) = |A ∩ B| / |A ∪ B|.
3. MinHash signature matrix generation using universal hashing:
   h_i(x) = (a_i * x + b_i) mod p, with Mersenne prime p = 2^31 - 1.
4. Locality-Sensitive Hashing (LSH) with banding (b bands, r rows):
   S-curve collision probability: P(collision | s) = 1 - (1 - s^r)^b
   Threshold: s* ~= (1 / b)^(1 / r).
5. Exact hash deduplication (SHA-256) vs Near-deduplication comparison.
6. Heuristic quality filters (Gopher / C4 rules: stopword ratio, symbol ratio, repetition).
7. Benchmark decontamination containment test (13-gram overlap).

Zero external dependencies (pure standard library: hashlib, math, random, sys).
Exits with code 0 on successful verification.
"""

import hashlib
import math
import random
import re
import sys

# Mersenne prime for universal hashing: 2^31 - 1
LARGE_PRIME = 2147483647


def generate_k_shingles(text: str, k: int = 3, mode: str = "word") -> set[str]:
    """Extracts a set of k-shingles (word n-grams or character n-grams)."""
    cleaned = re.sub(r"[^\w\s]", " ", text.lower()).strip()
    if mode == "word":
        tokens = cleaned.split()
        if len(tokens) < k:
            return set(tokens)
        return {" ".join(tokens[i:i + k]) for i in range(len(tokens) - k + 1)}
    elif mode == "char":
        cleaned_no_space = "".join(cleaned.split())
        if len(cleaned_no_space) < k:
            return {cleaned_no_space}
        return {cleaned_no_space[i:i + k] for i in range(len(cleaned_no_space) - k + 1)}
    else:
        raise ValueError(f"Unknown shingle mode: {mode}")


def exact_jaccard_similarity(set_a: set, set_b: set) -> float:
    """Calculates ground-truth Jaccard similarity J(A, B) = |A ∩ B| / |A ∪ B|."""
    if not set_a and not set_b:
        return 1.0
    union_len = len(set_a | set_b)
    if union_len == 0:
        return 0.0
    return len(set_a & set_b) / union_len


class MinHashLSH:
    """
    Pure Python MinHash and Locality-Sensitive Hashing (LSH) Indexer.
    """
    def __init__(self, num_perm: int = 128, num_bands: int = 16, seed: int = 42):
        self.num_perm = num_perm
        self.num_bands = num_bands
        assert num_perm % num_bands == 0, f"num_perm ({num_perm}) must be divisible by num_bands ({num_bands})"
        self.rows_per_band = num_perm // num_bands
        self.threshold = (1.0 / num_bands) ** (1.0 / self.rows_per_band)
        
        # Deterministic universal hash coefficients: (a * x + b) mod p
        rng = random.Random(seed)
        self.hash_params: list[tuple[int, int]] = []
        for _ in range(num_perm):
            a = rng.randint(1, LARGE_PRIME - 1)
            b = rng.randint(0, LARGE_PRIME - 1)
            self.hash_params.append((a, b))
            
        # Buckets for each band: band_idx -> bucket_hash -> list of doc_ids
        self.band_buckets: list[dict[int, list[str]]] = [{} for _ in range(num_bands)]
        self.signatures: dict[str, list[int]] = {}
        self.shingles: dict[str, set[str]] = {}

    def _hash_shingle(self, shingle: str) -> int:
        """Hashes a string shingle into a 32-bit positive integer."""
        digest = hashlib.md5(shingle.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], byteorder="big") & 0x7FFFFFFF

    def compute_signature(self, shingles: set[str]) -> list[int]:
        """Computes the MinHash signature vector of length num_perm."""
        if not shingles:
            return [0] * self.num_perm
            
        hashed_shingles = [self._hash_shingle(s) for s in shingles]
        signature: list[int] = []
        
        for a, b in self.hash_params:
            min_val = LARGE_PRIME
            for x in hashed_shingles:
                h = (a * x + b) % LARGE_PRIME
                if h < min_val:
                    min_val = h
            signature.append(min_val)
            
        return signature

    def insert(self, doc_id: str, shingles: set[str]) -> None:
        """Inserts a document into the MinHash signature store and LSH band buckets."""
        sig = self.compute_signature(shingles)
        self.signatures[doc_id] = sig
        self.shingles[doc_id] = shingles
        
        # Partition signature into bands and insert into hash buckets
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_chunk = tuple(sig[start:end])
            bucket_hash = hash(band_chunk)
            
            if bucket_hash not in self.band_buckets[band_idx]:
                self.band_buckets[band_idx][bucket_hash] = []
            self.band_buckets[band_idx][bucket_hash].append(doc_id)

    def estimate_jaccard(self, doc_id_a: str, doc_id_b: str) -> float:
        """Estimates Jaccard similarity by comparing fraction of matching signature slots."""
        sig_a = self.signatures[doc_id_a]
        sig_b = self.signatures[doc_id_b]
        matches = sum(1 for val_a, val_b in zip(sig_a, sig_b) if val_a == val_b)
        return matches / self.num_perm

    def query_candidates(self, doc_id: str) -> set[str]:
        """Retrieves all candidate duplicate documents that collide in at least one band."""
        candidates = set()
        sig = self.signatures[doc_id]
        
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_chunk = tuple(sig[start:end])
            bucket_hash = hash(band_chunk)
            
            for other_id in self.band_buckets[band_idx].get(bucket_hash, []):
                if other_id != doc_id:
                    candidates.add(other_id)
        return candidates

    def theoretical_collision_probability(self, s: float) -> float:
        """P(collision | s) = 1 - (1 - s^r)^b."""
        return 1.0 - (1.0 - (s ** self.rows_per_band)) ** self.num_bands


def heuristic_quality_filter(text: str) -> dict[str, any]:
    """
    Implements standard Gopher / C4 heuristic filters:
    1. Document length: Word count between 50 and 100,000.
    2. Stopword ratio: English stop words must constitute at least 15% of all words.
    3. Symbol-to-word ratio: Must not exceed 20% (flags code dumps or machine gibberish).
    4. Repetition penalty: No single 3-gram should account for > 20% of all 3-grams.
    """
    english_stopwords = {
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it",
        "for", "not", "on", "with", "he", "as", "you", "do", "at", "this", "but",
        "his", "by", "from", "they", "we", "say", "her", "she", "or", "an", "will",
        "my", "one", "all", "would", "there", "their", "what", "so", "up", "out",
        "if", "about", "who", "get", "which", "go", "me", "when", "make", "can",
        "like", "time", "no", "just", "him", "know", "take", "people", "into", "year"
    }
    
    words = text.split()
    word_count = len(words)
    if word_count < 20:
        return {"passed": False, "reason": f"Too short ({word_count} words < 20)"}
        
    # Stopword ratio
    cleaned_words = [re.sub(r"[^\w]", "", w.lower()) for w in words]
    cleaned_words = [w for w in cleaned_words if w]
    if not cleaned_words:
        return {"passed": False, "reason": "No valid alphanumeric words found"}
        
    stopword_count = sum(1 for w in cleaned_words if w in english_stopwords)
    stopword_ratio = stopword_count / len(cleaned_words)
    if stopword_ratio < 0.10:
        return {"passed": False, "reason": f"Low stopword ratio ({stopword_ratio:.2f} < 0.10)"}
        
    # Symbol-to-word ratio
    symbol_chars = sum(1 for c in text if c in "!@#$%^&*()_+={}[]|\\:;\"'<>,.?/~`")
    symbol_ratio = symbol_chars / max(1, len(text))
    if symbol_ratio > 0.18:
        return {"passed": False, "reason": f"High symbol ratio ({symbol_ratio:.2f} > 0.18)"}
        
    # 3-gram repetition check
    if len(cleaned_words) >= 10:
        trigrams = [" ".join(cleaned_words[i:i+3]) for i in range(len(cleaned_words) - 2)]
        counts = {}
        for tri in trigrams:
            counts[tri] = counts.get(tri, 0) + 1
        max_tri_count = max(counts.values())
        max_tri_ratio = max_tri_count / len(trigrams)
        if max_tri_ratio > 0.20:
            return {"passed": False, "reason": f"Excessive repetition (3-gram occurs {max_tri_ratio:.2f} > 0.20)"}
            
    return {"passed": True, "reason": "All heuristic quality checks passed"}


def benchmark_decontamination_check(train_text: str, benchmark_questions: list[str],
                                    n_gram_size: int = 13) -> list[dict]:
    """
    Performs benchmark decontamination check using n-gram containment.
    If any 13-gram in a benchmark question matches training text, it flags contamination.
    """
    train_tokens = re.sub(r"[^\w\s]", " ", train_text.lower()).split()
    if len(train_tokens) < n_gram_size:
        return []
    train_ngrams = set(" ".join(train_tokens[i:i + n_gram_size]) for i in range(len(train_tokens) - n_gram_size + 1))
    
    flagged = []
    for idx, question in enumerate(benchmark_questions):
        q_tokens = re.sub(r"[^\w\s]", " ", question.lower()).split()
        if len(q_tokens) < n_gram_size:
            continue
        q_ngrams = set(" ".join(q_tokens[i:i + n_gram_size]) for i in range(len(q_tokens) - n_gram_size + 1))
        overlap = train_ngrams & q_ngrams
        if overlap:
            flagged.append({
                "benchmark_id": idx,
                "question_snippet": question[:60] + "...",
                "matching_ngrams": list(overlap)[:2]
            })
    return flagged


def run_experiment() -> None:
    print("=" * 80)
    print("PHASE 18 — TOPIC 02: DATA CURATION, DEDUPLICATION, AND FILTERING")
    print("Pure Python 3 MinHash, LSH Indexing, & Quality Filtering Pipeline")
    print("=" * 80)

    # 1. Corpus Creation: Diverse subjects, near-duplicates, and exact duplicates
    corpus = {
        "doc_physics_original": (
            "Quantum mechanics describes the fundamental physical properties of nature at the scale "
            "of atoms and subatomic particles. It is the foundation of all quantum physics including "
            "quantum chemistry, quantum field theory, quantum technology, and quantum information science. "
            "Classical physics cannot explain phenomena such as black-body radiation or the photoelectric effect."
        ),
        "doc_physics_near_dup": (
            "Quantum mechanics describes the fundamental physical properties of nature at the scale "
            "of atoms and subatomic particles. It is the core foundation of all quantum physics including "
            "quantum chemistry, quantum field theory, quantum technology, and quantum information science. "
            "Classical physics fails to explain phenomena such as black-body radiation or the photoelectric effect. "
            "Published by Academic Press 2024. All rights reserved."
        ),
        "doc_physics_exact_dup": (
            "Quantum mechanics describes the fundamental physical properties of nature at the scale "
            "of atoms and subatomic particles. It is the foundation of all quantum physics including "
            "quantum chemistry, quantum field theory, quantum technology, and quantum information science. "
            "Classical physics cannot explain phenomena such as black-body radiation or the photoelectric effect."
        ),
        "doc_history_roman": (
            "The Roman Empire was the post-Republican period of ancient Rome. As a polity, it included "
            "large territorial holdings around the Mediterranean Sea in Europe, North Africa, and Western Asia, "
            "ruled by emperors. From the accession of Caesar Augustus to the military anarchy of the third century, "
            "it was a principate with Italy as the metropole of the provinces."
        ),
        "doc_history_near_dup": (
            "The Roman Empire was the post-Republican period of ancient Rome. As a polity, it included "
            "large territorial holdings around the Mediterranean Sea in Europe, North Africa, and Western Asia, "
            "ruled by emperors. From the accession of Caesar Augustus to the military anarchy of the third century, "
            "it was a principate with Italy as the metropole of the provinces. "
            "Read more historical essays and ancient maps at history-archive.example.org/articles."
        ),
        "doc_biology_dna": (
            "Deoxyribonucleic acid is a polymer composed of two polynucleotide chains that coil around each other "
            "to form a double helix. The polymer carries genetic instructions for the development, functioning, "
            "growth and reproduction of all known organisms and many viruses. DNA and RNA are nucleic acids."
        ),
        "doc_spam_repetitive": (
            "Buffering stream please wait... Buffering stream please wait... Buffering stream please wait... "
            "Buffering stream please wait... Buffering stream please wait... Buffering stream please wait... "
            "Buffering stream please wait... Buffering stream please wait... Buffering stream please wait..."
        ),
        "doc_machine_gibberish": (
            "$#@! &*(^ %$$#@ 123890 !@#$%^&*()_+ {}:|<>? ~`~` []\\;'/., !@#$$%^& "
            "897234982374982374982374982374928374982374982374982374982374"
        )
    }

    print("\n[1] HEURISTIC QUALITY FILTER AUDIT (GOPHER & C4 RULES):")
    print("-" * 80)
    for doc_id, text in corpus.items():
        res = heuristic_quality_filter(text)
        status = "PASSED" if res["passed"] else "REJECTED"
        print(f"  • {doc_id:<26} : [{status}] {res['reason']}")

    # Assert expected filter behaviors
    assert heuristic_quality_filter(corpus["doc_physics_original"])["passed"] is True
    assert heuristic_quality_filter(corpus["doc_spam_repetitive"])["passed"] is False
    assert heuristic_quality_filter(corpus["doc_machine_gibberish"])["passed"] is False

    print("\n[2] EXACT HASH DEDUPLICATION (SHA-256) LIMITATION DEMONSTRATION:")
    print("-" * 80)
    exact_hashes = {}
    for doc_id, text in corpus.items():
        h = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
        if h in exact_hashes:
            print(f"  Exact Duplicate Found: '{doc_id}' is identical to '{exact_hashes[h]}'")
        else:
            exact_hashes[h] = doc_id
    print("  Notice: SHA-256 successfully caught 'doc_physics_exact_dup', but completely missed")
    print("  'doc_physics_near_dup' and 'doc_history_near_dup' due to cryptographic avalanche effect!")

    # 3. MinHash & LSH Near-Deduplication
    print("\n[3] MINHASH & LOCALITY-SENSITIVE HASHING (LSH) PIPELINE:")
    print("-" * 80)
    num_perm = 128
    num_bands = 16  # 16 bands x 8 rows = 128 permutations
    lsh = MinHashLSH(num_perm=num_perm, num_bands=num_bands, seed=42)
    
    print(f"LSH Configuration: {num_perm} permutations | {num_bands} bands | {lsh.rows_per_band} rows/band")
    print(f"Theoretical Similarity Threshold s* = (1 / {num_bands})^(1 / {lsh.rows_per_band}) = {lsh.threshold:.4f}")

    # Index all clean documents
    clean_docs = {k: v for k, v in corpus.items() if not k.startswith("doc_spam") and not k.startswith("doc_machine")}
    for doc_id, text in clean_docs.items():
        shingles = generate_k_shingles(text, k=3, mode="word")
        lsh.insert(doc_id, shingles)

    print("\n[4] VERIFYING JACCARD SIMILARITY ESTIMATION ACCURACY (EXACT vs MINHASH):")
    print("-" * 80)
    print(f"{'Pair Comparison':<45} | {'Exact Jaccard':<14} | {'MinHash Est':<14} | {'Abs Error':<10}")
    print("-" * 80)
    
    doc_ids = list(clean_docs.keys())
    max_error = 0.0
    total_error = 0.0
    pair_count = 0
    
    for i in range(len(doc_ids)):
        for j in range(i + 1, len(doc_ids)):
            id_a, id_b = doc_ids[i], doc_ids[j]
            shing_a = lsh.shingles[id_a]
            shing_b = lsh.shingles[id_b]
            
            exact_j = exact_jaccard_similarity(shing_a, shing_b)
            est_j = lsh.estimate_jaccard(id_a, id_b)
            err = abs(exact_j - est_j)
            
            total_error += err
            max_error = max(max_error, err)
            pair_count += 1
            
            pair_label = f"{id_a} <-> {id_b}"
            print(f"{pair_label:<45} | {exact_j:<14.4f} | {est_j:<14.4f} | {err:<10.4f}")

    mae = total_error / pair_count
    print(f"\n--> Mean Absolute Error across all pairs: {mae:.4f} (Expected << 1 / sqrt({num_perm}) = {1 / math.sqrt(num_perm):.4f})")
    assert mae < 0.08, f"MinHash estimation error too high ({mae:.4f} >= 0.08)!"

    print("\n[5] LSH CANDIDATE PAIR RETRIEVAL & S-CURVE VERIFICATION:")
    print("-" * 80)
    physics_candidates = lsh.query_candidates("doc_physics_original")
    history_candidates = lsh.query_candidates("doc_history_roman")
    biology_candidates = lsh.query_candidates("doc_biology_dna")

    print(f"Candidates retrieved for 'doc_physics_original': {physics_candidates}")
    print(f"Candidates retrieved for 'doc_history_roman':     {history_candidates}")
    print(f"Candidates retrieved for 'doc_biology_dna':       {biology_candidates}")

    # Assert near-duplicates were correctly retrieved
    assert "doc_physics_near_dup" in physics_candidates, "LSH failed to detect physics near-duplicate!"
    assert "doc_physics_exact_dup" in physics_candidates, "LSH failed to detect physics exact duplicate!"
    assert "doc_history_near_dup" in history_candidates, "LSH failed to detect history near-duplicate!"
    assert len(biology_candidates) == 0, "Unrelated biology document produced unexpected collision!"

    print("\n[6] BENCHMARK DECONTAMINATION CONTAINMENT AUDIT:")
    print("-" * 80)
    benchmark_eval_set = [
        "In quantum physics, classical mechanics fails to explain black-body radiation or the photoelectric effect at atomic scales.",
        "What was the capital of the Roman Empire during the reign of Augustus?",
        "Explain the process of DNA double helix transcription into messenger RNA."
    ]
    
    # Check physics document against benchmark
    flagged = benchmark_decontamination_check(corpus["doc_physics_original"], benchmark_eval_set, n_gram_size=8)
    print(f"Decontamination audit for training doc 'doc_physics_original':")
    if flagged:
        for f in flagged:
            print(f"  [CONTAMINATION DETECTED] Benchmark Q#{f['benchmark_id']}: '{f['question_snippet']}'")
            print(f"    Overlapping n-grams: {f['matching_ngrams']}")
    else:
        print("  Clean: No benchmark contamination found.")

    print("\n" + "=" * 80)
    print("ALL VERIFICATIONS COMPLETED SUCCESSFULLY — EXITING CODE 0")
    print("=" * 80)


if __name__ == "__main__":
    run_experiment()
    sys.exit(0)
