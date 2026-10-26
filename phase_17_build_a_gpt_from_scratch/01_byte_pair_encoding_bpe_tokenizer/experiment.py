#!/usr/bin/env python3
"""
Phase 17 - Topic 01: Byte-Pair Encoding (BPE) Tokenizer
Complete pure Python 3 implementation with ZERO external dependencies.

Implements:
1. Byte-level BPE Tokenizer from scratch (as popularized by Sennrich et al. 2016 & GPT-2 Radford et al. 2019).
2. Base byte vocabulary (256 distinct byte values), guaranteeing 0 Out-Of-Vocabulary (OOV) tokens.
3. Special tokens handling (<|endoftext|>).
4. Iterative training: counting pair frequencies, greedy pair merging, merge rule recording.
5. Linear/priority-based encoding with learned merge rules.
6. Lossless decoding back to exact raw UTF-8 strings.
7. Verification suite: asserts exact round-trip reconstruction, byte fallback on unseen unicode/emojis.
"""

from typing import Dict, List, Tuple, Optional
import sys


class BPETokenizer:
    """
    Byte-level Byte Pair Encoding (BPE) Tokenizer.
    
    Base vocabulary contains:
    - 256 raw byte tokens (0 to 255)
    - Optional special tokens (e.g., '<|endoftext|>')
    - Subsequent merged tokens learned during training.
    """
    def __init__(self):
        # Merges: maps tuple of token IDs (id1, id2) -> merged_token_id
        self.merges: Dict[Tuple[int, int], int] = {}
        # Vocabulary: token_id -> bytes
        self.vocab: Dict[int, bytes] = {}
        # Inverse vocabulary: bytes -> token_id
        self.inv_vocab: Dict[bytes, int] = {}
        
        # Special tokens: string -> token_id
        self.special_tokens: Dict[str, int] = {}
        self.inv_special_tokens: Dict[int, str] = {}
        
        # Initialize base 256 byte vocabulary
        self._init_base_vocab()
        
    def _init_base_vocab(self):
        """Initialize the 256 fundamental byte tokens."""
        for b in range(256):
            byte_val = bytes([b])
            self.vocab[b] = byte_val
            self.inv_vocab[byte_val] = b
            
    def add_special_token(self, token_str: str) -> int:
        """Register a special token like '<|endoftext|>' with a unique ID."""
        if token_str in self.special_tokens:
            return self.special_tokens[token_str]
        
        new_id = len(self.vocab)
        self.special_tokens[token_str] = new_id
        self.inv_special_tokens[new_id] = token_str
        token_bytes = token_str.encode("utf-8")
        self.vocab[new_id] = token_bytes
        self.inv_vocab[token_bytes] = new_id
        return new_id

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @staticmethod
    def get_stats(token_ids: List[int]) -> Dict[Tuple[int, int], int]:
        """Count frequencies of adjacent token pairs."""
        counts: Dict[Tuple[int, int], int] = {}
        for i in range(len(token_ids) - 1):
            pair = (token_ids[i], token_ids[i + 1])
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    @staticmethod
    def merge(token_ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        """Replace all occurrences of pair (p0, p1) with new_id."""
        new_tokens = []
        i = 0
        n = len(token_ids)
        p0, p1 = pair
        while i < n:
            if i < n - 1 and token_ids[i] == p0 and token_ids[i + 1] == p1:
                new_tokens.append(new_id)
                i += 2
            else:
                new_tokens.append(token_ids[i])
                i += 1
        return new_tokens

    def train(self, text: str, num_merges: int, verbose: bool = False):
        """
        Train BPE tokenizer on raw text by discovering `num_merges` new subword tokens.
        """
        # Convert text into raw UTF-8 bytes, then list of initial token IDs (0..255)
        raw_bytes = text.encode("utf-8")
        token_ids = list(raw_bytes)
        
        if verbose:
            print(f"[BPE Train] Initial corpus length: {len(token_ids)} byte tokens.")
            print(f"[BPE Train] Base vocabulary size: {self.vocab_size} (256 raw bytes + special tokens).")

        for merge_idx in range(num_merges):
            stats = self.get_stats(token_ids)
            if not stats:
                if verbose:
                    print(f"[BPE Train] No more pairs to merge at step {merge_idx + 1}.")
                break
                
            # Greedily pick the most frequent adjacent pair.
            # Tie-breaker: lexicographical order of pair IDs for deterministic behavior
            best_pair = max(stats.keys(), key=lambda p: (stats[p], -p[0], -p[1]))
            best_count = stats[best_pair]
            
            if best_count < 2:
                # If no pair occurs at least twice, further merges offer diminishing returns
                if verbose:
                    print(f"[BPE Train] Stopping early: most frequent pair occurs only {best_count} times.")
                break
                
            new_id = len(self.vocab)
            token_ids = self.merge(token_ids, best_pair, new_id)
            
            # Record merge rule
            self.merges[best_pair] = new_id
            
            # Form concatenated byte representation
            merged_bytes = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.vocab[new_id] = merged_bytes
            self.inv_vocab[merged_bytes] = new_id
            
            if verbose and ((merge_idx + 1) % 10 == 0 or merge_idx < 5 or merge_idx == num_merges - 1):
                preview = merged_bytes.decode("utf-8", errors="replace")
                print(f"  Merge {merge_idx + 1:3d}/{num_merges}: pair {best_pair} -> ID {new_id} ({best_count}x) -> '{preview}'")

        if verbose:
            compression_ratio = len(raw_bytes) / len(token_ids) if len(token_ids) > 0 else 1.0
            print(f"[BPE Train] Finished {len(self.merges)} merges.")
            print(f"[BPE Train] Final vocab size: {self.vocab_size}")
            print(f"[BPE Train] Compression: {len(raw_bytes)} bytes -> {len(token_ids)} tokens ({compression_ratio:.2f}x)")

    def encode_chunk(self, chunk_bytes: bytes) -> List[int]:
        """Encode raw bytes using the learned merge rules."""
        token_ids = list(chunk_bytes)
        if len(token_ids) < 2:
            return token_ids
            
        while len(token_ids) >= 2:
            # Find the pair in token_ids that has the earliest merge rule priority
            # Merges dict insertion order is preservation of training merge order!
            stats = self.get_stats(token_ids)
            # Find candidate pairs present in our merge table
            candidate_pairs = [p for p in stats if p in self.merges]
            if not candidate_pairs:
                break
            
            # Pick pair that was learned first (lowest new_id assigned)
            pair_to_merge = min(candidate_pairs, key=lambda p: self.merges[p])
            new_id = self.merges[pair_to_merge]
            token_ids = self.merge(token_ids, pair_to_merge, new_id)
            
        return token_ids

    def encode(self, text: str) -> List[int]:
        """
        Encode string into token IDs, properly preserving registered special tokens.
        """
        if not self.special_tokens:
            return self.encode_chunk(text.encode("utf-8"))
            
        # Segment text by special tokens if present
        # Sort special tokens by length descending to match longest first
        specials = sorted(self.special_tokens.keys(), key=len, reverse=True)
        
        tokens: List[int] = []
        i = 0
        n = len(text)
        
        while i < n:
            matched_special = None
            for s in specials:
                if text.startswith(s, i):
                    matched_special = s
                    break
                    
            if matched_special:
                tokens.append(self.special_tokens[matched_special])
                i += len(matched_special)
            else:
                # Find index of next special token
                next_special_idx = n
                for s in specials:
                    idx = text.find(s, i)
                    if idx != -1 and idx < next_special_idx:
                        next_special_idx = idx
                        
                plain_chunk = text[i:next_special_idx]
                tokens.extend(self.encode_chunk(plain_chunk.encode("utf-8")))
                i = next_special_idx
                
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode a sequence of token IDs back into a Unicode string.
        Guaranteed lossless for valid UTF-8 sequences.
        """
        byte_chunks: List[bytes] = []
        for tid in token_ids:
            if tid in self.inv_special_tokens:
                byte_chunks.append(self.inv_special_tokens[tid].encode("utf-8"))
            elif tid in self.vocab:
                byte_chunks.append(self.vocab[tid])
            else:
                raise ValueError(f"Unrecognized token ID {tid}")
                
        joined_bytes = b"".join(byte_chunks)
        # Use replace or strict depending on need; strict verifies lossless UTF-8 round-trip
        return joined_bytes.decode("utf-8", errors="replace")


def run_experiment():
    print("=" * 70)
    print("Phase 17 - Topic 01: Byte-Pair Encoding (BPE) Tokenizer Lab")
    print("=" * 70)

    # 1. Initialize Tokenizer & Special Tokens
    tokenizer = BPETokenizer()
    eot_id = tokenizer.add_special_token("<|endoftext|>")
    pad_id = tokenizer.add_special_token("<|pad|>")
    print(f"[*] Initialized tokenizer with {tokenizer.vocab_size} base tokens (256 bytes + 2 special).")
    print(f"    Special '<|endoftext|>' ID: {eot_id}")
    print(f"    Special '<|pad|>' ID: {pad_id}")

    # 2. Prepare Representative Training Corpus
    training_text = (
        "The quick brown fox jumps over the lazy dog.\n"
        "Transformers rely on self-attention mechanisms to model long-range context.\n"
        "Byte-Pair Encoding (BPE) constructs subword vocabularies by iteratively merging\n"
        "the most frequent adjacent pairs of tokens.\n"
        "In modern language models such as GPT-2, GPT-3, and GPT-4, byte-level BPE\n"
        "guarantees that every possible character, punctuation mark, emoji, and foreign script\n"
        "can be encoded without an out-of-vocabulary (<UNK>) token!\n"
        "Attention is all you need. Deep learning models transform raw sequences of tokens.\n"
        "Low perplexity, autoregressive language modeling, and causal masking.\n"
    ) * 8  # Repeat corpus to establish strong pair statistics

    print(f"\n[*] Training corpus size: {len(training_text.encode('utf-8'))} bytes.")
    
    # 3. Train BPE Tokenizer
    target_merges = 60
    tokenizer.train(training_text, num_merges=target_merges, verbose=True)

    print(f"\n[*] Training Complete.")
    print(f"    Total merges learned: {len(tokenizer.merges)}")
    print(f"    Final vocabulary size: {tokenizer.vocab_size}")

    # 4. Tokenization Round-Trip Tests
    test_cases = [
        "Attention is all you need.",
        "The quick brown fox jumps over the lazy dog.",
        "Byte-level BPE solves OOV completely! 🚀 🔥 🤖",
        "Mathematics: ∀ x ∈ ℝ, e^{iπ} + 1 = 0.",
        "<|endoftext|>A prompt starting with special token.<|endoftext|>",
        "Hieroglyphs & rare scripts: 𓀀 𓀁 𓀂 𓀃 𓀄 and accents: résumé, café, naïve.",
        "1234567890 + 987654321 = 1111111111.",
        "Whitespace test:   \t\n\r  multiple spaces and tabs."
    ]

    print("\n" + "=" * 70)
    print("[*] Testing Lossless Round-Trip Encoding & Decoding:")
    print("=" * 70)

    for idx, test_str in enumerate(test_cases, 1):
        encoded = tokenizer.encode(test_str)
        decoded = tokenizer.decode(encoded)
        
        # Verify exact round-trip match
        assert decoded == test_str, f"Mismatch on test case {idx}!\nOrig: {test_str}\nDecoded: {decoded}"
        
        # Compute compression metric
        raw_byte_len = len(test_str.encode("utf-8"))
        token_len = len(encoded)
        ratio = raw_byte_len / token_len if token_len > 0 else 1.0
        
        print(f"Case {idx}:")
        print(f"  Input text   : {repr(test_str)}")
        print(f"  Token IDs    : {encoded[:12]}{'...' if len(encoded) > 12 else ''} (Total: {token_len} tokens)")
        print(f"  Compression  : {raw_byte_len} bytes -> {token_len} tokens ({ratio:.2f}x)")
        print(f"  Lossless     : ✅ PASSED\n")

    # 5. Inspection of Specific Token Merges
    print("=" * 70)
    print("[*] Top 10 Learned Token Merges Inspection:")
    print("=" * 70)
    for i, (pair, merged_id) in enumerate(list(tokenizer.merges.items())[:10], 1):
        p0_repr = tokenizer.vocab[pair[0]].decode("utf-8", errors="replace")
        p1_repr = tokenizer.vocab[pair[1]].decode("utf-8", errors="replace")
        m_repr = tokenizer.vocab[merged_id].decode("utf-8", errors="replace")
        print(f"  Rule #{i:02d}: ID {pair[0]:3d} ({repr(p0_repr)}) + ID {pair[1]:3d} ({repr(p1_repr)}) -> ID {merged_id:3d} ({repr(m_repr)})")

    # 6. Verification of Out-Of-Vocabulary Immunity
    print("\n" + "=" * 70)
    print("[*] Testing Out-Of-Vocabulary (OOV) Resilience:")
    print("=" * 70)
    alien_text = "Здравствуй, мир! こんにちは世界 🌟 🪐 🛸"
    alien_encoded = tokenizer.encode(alien_text)
    alien_decoded = tokenizer.decode(alien_encoded)
    assert alien_decoded == alien_text, "Failed on multilingual/emoji stress test!"
    print(f"  Alien input : {repr(alien_text)}")
    print(f"  Token IDs   : {alien_encoded}")
    print(f"  Decoded     : {repr(alien_decoded)}")
    print(f"  Result      : ✅ Zero <UNK> tokens, 100% exact lossless recovery!")

    print("\n" + "=" * 70)
    print("✅ All Byte-Pair Encoding (BPE) tests passed successfully!")
    print("=" * 70)
    sys.exit(0)


if __name__ == "__main__":
    run_experiment()
