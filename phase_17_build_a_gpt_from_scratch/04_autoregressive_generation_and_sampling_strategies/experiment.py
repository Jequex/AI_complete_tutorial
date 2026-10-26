#!/usr/bin/env python3
"""
Phase 17 - Topic 04: Autoregressive Generation, Sampling Strategies & KV-Cache
Complete pure Python 3 implementation with ZERO external dependencies.

Implements:
1. Autoregressive token-by-token generation loop.
2. Complete decoding & sampling strategy suite:
   - Greedy Search (Argmax)
   - Temperature Scaling (Sharpness vs Entropy)
   - Top-k Filtering
   - Top-p (Nucleus) Sampling (Holtzman et al. 2020)
3. Key-Value (KV) Cache mechanism for O(T) per-token generation vs O(T^2) naive recomputation.
4. Exact numerical verification that KV-cached generation produces identical outputs to naive generation.
5. Benchmark demonstrating generation efficiency with KV-caching.
"""

import math
import random
import time
import sys
from typing import List, Tuple, Optional, Dict

random.seed(42)


# =====================================================================
# 1. Pure Python Math & Activation Primitives
# =====================================================================

def gelu(x: float) -> float:
    return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x ** 3))))


def softmax_1d(vec: List[float]) -> List[float]:
    max_val = max(vec)
    exp_vec = [math.exp(v - max_val) for v in vec]
    sum_exp = sum(exp_vec)
    if sum_exp == 0.0:
        return [1.0 / len(vec)] * len(vec)
    return [v / sum_exp for v in exp_vec]


def matmul_2d(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    M = len(A)
    K = len(A[0])
    N = len(B[0])
    B_T = [[B[r][c] for r in range(K)] for c in range(N)]
    out = [[0.0] * N for _ in range(M)]
    for i in range(M):
        row_A = A[i]
        for j in range(N):
            row_B_T = B_T[j]
            s = 0.0
            for k in range(K):
                s += row_A[k] * row_B_T[k]
            out[i][j] = s
    return out


def transpose_2d(A: List[List[float]]) -> List[List[float]]:
    return [[A[r][c] for r in range(len(A))] for c in range(len(A[0]))]


# =====================================================================
# 2. Sampling Strategies
# =====================================================================

def sample_token(
    logits: List[float],
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    greedy: bool = False
) -> int:
    """
    Sample next token ID from unnormalized logits using specified strategy:
    - greedy: deterministic argmax
    - temperature: controls sharpness (higher = more uniform, lower = more deterministic)
    - top_k: keeps top k highest probability tokens
    - top_p: nucleus sampling, keeps cumulative probability mass up to p
    """
    # 1. Greedy Search (Argmax)
    if greedy or temperature <= 1e-6:
        best_id = 0
        best_logit = logits[0]
        for idx in range(1, len(logits)):
            if logits[idx] > best_logit:
                best_logit = logits[idx]
                best_id = idx
        return best_id

    # 2. Temperature Scaling
    scaled_logits = [l / temperature for l in logits]

    # 3. Top-k Filtering
    if top_k is not None and 0 < top_k < len(scaled_logits):
        # Find k-th largest value
        sorted_indices = sorted(range(len(scaled_logits)), key=lambda i: scaled_logits[i], reverse=True)
        cutoff_value = scaled_logits[sorted_indices[top_k - 1]]
        for i in range(len(scaled_logits)):
            if scaled_logits[i] < cutoff_value:
                scaled_logits[i] = -1e9

    # 4. Softmax to Probabilities
    probs = softmax_1d(scaled_logits)

    # 5. Top-p (Nucleus) Sampling
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_pairs = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
        cumulative_mass = 0.0
        allowed_indices = set()
        for idx, prob in sorted_pairs:
            allowed_indices.add(idx)
            cumulative_mass += prob
            if cumulative_mass >= top_p:
                break
        
        # Zero out unallowed and renormalize
        filtered_probs = [probs[i] if i in allowed_indices else 0.0 for i in range(len(probs))]
        sum_filt = sum(filtered_probs)
        if sum_filt > 0:
            probs = [p / sum_filt for p in filtered_probs]

    # 6. Categorical Sampling from distribution
    r = random.random()
    cum = 0.0
    for idx, prob in enumerate(probs):
        cum += prob
        if r <= cum:
            return idx
    return len(probs) - 1


# =====================================================================
# 3. Transformer Sub-Layers with KV-Cache Support
# =====================================================================

class LayerNorm:
    def __init__(self, d_model: int, eps: float = 1e-5):
        self.d_model = d_model
        self.eps = eps
        self.gamma = [1.0] * d_model
        self.beta = [0.0] * d_model

    def forward(self, X: List[List[float]]) -> List[List[float]]:
        """X has shape [T, d_model]."""
        T = len(X)
        D = self.d_model
        out = [[0.0] * D for _ in range(T)]
        for t in range(T):
            vec = X[t]
            mean = sum(vec) / D
            var = sum((v - mean) ** 2 for v in vec) / D
            std = math.sqrt(var + self.eps)
            for d in range(D):
                out[t][d] = ((vec[d] - mean) / std) * self.gamma[d] + self.beta[d]
        return out


class CausalSelfAttentionWithKVCache:
    """
    Causal Self-Attention supporting both:
    1. Full sequence forward pass (training / prompt pre-fill).
    2. Step-by-step token generation with KV-Cache.
    """
    def __init__(self, d_model: int, n_heads: int):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_k)
        
        init_scale = 1.0 / math.sqrt(d_model)
        self.W_q = [[random.gauss(0.0, init_scale) for _ in range(d_model)] for _ in range(d_model)]
        self.W_k = [[random.gauss(0.0, init_scale) for _ in range(d_model)] for _ in range(d_model)]
        self.W_v = [[random.gauss(0.0, init_scale) for _ in range(d_model)] for _ in range(d_model)]
        self.W_o = [[random.gauss(0.0, init_scale) for _ in range(d_model)] for _ in range(d_model)]

    def forward(
        self,
        X: List[List[float]],
        kv_cache: Optional[Tuple[List[List[float]], List[List[float]]]] = None
    ) -> Tuple[List[List[float]], Tuple[List[List[float]], List[List[float]]]]:
        """
        X: [T, d_model]
        kv_cache: Optional tuple of (cached_K, cached_V) where each is [T_past, d_model].
        Returns:
            out: [T, d_model]
            new_kv_cache: (K_full, V_full)
        """
        T = len(X)
        D = self.d_model
        H = self.n_heads
        d_k = self.d_k

        Q = matmul_2d(X, self.W_q)
        K_new = matmul_2d(X, self.W_k)
        V_new = matmul_2d(X, self.W_v)

        if kv_cache is not None:
            K_past, V_past = kv_cache
            # Append newly computed K and V to historical cache
            K = K_past + K_new
            V = V_past + V_new
        else:
            K = K_new
            V = V_new

        T_total = len(K)
        T_start = T_total - T  # offset where new queries begin

        head_outputs = [[[0.0] * d_k for _ in range(T)] for _ in range(H)]

        for h in range(H):
            offset = h * d_k
            Q_h = [[Q[t][offset + i] for i in range(d_k)] for t in range(T)]
            K_h = [[K[t][offset + i] for i in range(d_k)] for t in range(T_total)]
            V_h = [[V[t][offset + i] for i in range(d_k)] for t in range(T_total)]

            # S = Q_h * K_h^T -> shape [T, T_total]
            scores = matmul_2d(Q_h, transpose_2d(K_h))

            # Apply causal mask: query at position (T_start + i) cannot attend to key j > (T_start + i)
            attn_weights = [[0.0] * T_total for _ in range(T)]
            for i in range(T):
                query_pos = T_start + i
                row = [0.0] * T_total
                for j in range(T_total):
                    if j <= query_pos:
                        row[j] = scores[i][j] * self.scale
                    else:
                        row[j] = -1e9
                attn_weights[i] = softmax_1d(row)

            # context = attn_weights * V_h -> [T, d_k]
            context_h = matmul_2d(attn_weights, V_h)
            head_outputs[h] = context_h

        # Concatenate heads [T, D]
        concat_heads = [[0.0] * D for _ in range(T)]
        for t in range(T):
            for h in range(H):
                offset = h * d_k
                for i in range(d_k):
                    concat_heads[t][offset + i] = head_outputs[h][t][i]

        out = matmul_2d(concat_heads, self.W_o)
        return out, (K, V)


class MLP:
    def __init__(self, d_model: int):
        self.d_model = d_model
        d_ff = 4 * d_model
        init_1 = 1.0 / math.sqrt(d_model)
        init_2 = 1.0 / math.sqrt(d_ff)
        self.W_1 = [[random.gauss(0.0, init_1) for _ in range(d_ff)] for _ in range(d_model)]
        self.b_1 = [0.0] * d_ff
        self.W_2 = [[random.gauss(0.0, init_2) for _ in range(d_model)] for _ in range(d_ff)]
        self.b_2 = [0.0] * d_model

    def forward(self, X: List[List[float]]) -> List[List[float]]:
        T = len(X)
        D = self.d_model
        d_ff = 4 * D
        h1 = matmul_2d(X, self.W_1)
        for t in range(T):
            for j in range(d_ff):
                h1[t][j] = gelu(h1[t][j] + self.b_1[j])
        h2 = matmul_2d(h1, self.W_2)
        out = [[0.0] * D for _ in range(T)]
        for t in range(T):
            for j in range(D):
                out[t][j] = h2[t][j] + self.b_2[j]
        return out


class TransformerBlock:
    def __init__(self, d_model: int, n_heads: int):
        self.ln_1 = LayerNorm(d_model)
        self.attn = CausalSelfAttentionWithKVCache(d_model, n_heads)
        self.ln_2 = LayerNorm(d_model)
        self.mlp = MLP(d_model)

    def forward(
        self,
        X: List[List[float]],
        kv_cache: Optional[Tuple[List[List[float]], List[List[float]]]] = None
    ) -> Tuple[List[List[float]], Tuple[List[List[float]], List[List[float]]]]:
        T = len(X)
        D = len(X[0])
        normed_1 = self.ln_1.forward(X)
        attn_out, new_cache = self.attn.forward(normed_1, kv_cache=kv_cache)
        x_res1 = [[X[t][d] + attn_out[t][d] for d in range(D)] for t in range(T)]
        normed_2 = self.ln_2.forward(x_res1)
        mlp_out = self.mlp.forward(normed_2)
        out = [[x_res1[t][d] + mlp_out[t][d] for d in range(D)] for t in range(T)]
        return out, new_cache


# =====================================================================
# 4. GPT Model with Dual Forward (Full / KV-Cached)
# =====================================================================

class GPT:
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_layer: int,
        n_head: int,
        d_model: int
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_model = d_model

        init_scale = 0.02
        self.wte = [[random.gauss(0.0, init_scale) for _ in range(d_model)] for _ in range(vocab_size)]
        self.wpe = [[random.gauss(0.0, init_scale) for _ in range(d_model)] for _ in range(block_size)]
        self.blocks = [TransformerBlock(d_model, n_head) for _ in range(n_layer)]
        self.ln_f = LayerNorm(d_model)

    def forward_naive(self, token_ids: List[int]) -> List[float]:
        """Full forward pass without caching. Returns next-token logits for final position."""
        T = len(token_ids)
        assert T <= self.block_size
        D = self.d_model

        # Embeddings
        X = [[self.wte[token_ids[t]][d] + self.wpe[t][d] for d in range(D)] for t in range(T)]

        # Blocks
        for block in self.blocks:
            X, _ = block.forward(X, kv_cache=None)

        # Final LN
        X = self.ln_f.forward(X)

        # Weight-tied LM Head projection for position T-1
        last_h = X[-1]
        logits = [0.0] * self.vocab_size
        for v in range(self.vocab_size):
            dot = 0.0
            w_v = self.wte[v]
            for d in range(D):
                dot += last_h[d] * w_v[d]
            logits[v] = dot
        return logits

    def forward_kv_cached(
        self,
        new_token_id: int,
        pos: int,
        layer_caches: List[Optional[Tuple[List[List[float]], List[List[float]]]]]
    ) -> Tuple[List[float], List[Tuple[List[List[float]], List[List[float]]]]]:
        """
        Step-by-step forward pass feeding ONLY single newest token.
        Reuses cached keys and values for all past tokens!
        """
        D = self.d_model
        # Slice single token embedding
        X = [[self.wte[new_token_id][d] + self.wpe[pos][d] for d in range(D)]]

        new_caches = []
        for l, block in enumerate(self.blocks):
            X, new_cache = block.forward(X, kv_cache=layer_caches[l])
            new_caches.append(new_cache)

        X = self.ln_f.forward(X)

        # LM Head on single output vector
        h = X[0]
        logits = [0.0] * self.vocab_size
        for v in range(self.vocab_size):
            dot = 0.0
            w_v = self.wte[v]
            for d in range(D):
                dot += h[d] * w_v[d]
            logits[v] = dot

        return logits, new_caches

    def generate(
        self,
        prompt_ids: List[int],
        max_new_tokens: int,
        use_cache: bool = True,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        greedy: bool = False,
        eos_id: Optional[int] = None
    ) -> List[int]:
        """Autoregressive generation loop."""
        tokens = list(prompt_ids)

        if not use_cache:
            # Naive generation: recomputes all tokens from scratch at every step!
            for _ in range(max_new_tokens):
                if len(tokens) >= self.block_size:
                    break
                logits = self.forward_naive(tokens)
                next_tok = sample_token(logits, temperature=temperature, top_k=top_k, top_p=top_p, greedy=greedy)
                tokens.append(next_tok)
                if eos_id is not None and next_tok == eos_id:
                    break
            return tokens

        # KV-Cached generation:
        # Pre-fill cache with prompt tokens first
        layer_caches = [None] * self.n_layer
        for pos, tok in enumerate(tokens):
            logits, layer_caches = self.forward_kv_cached(tok, pos, layer_caches)

        # Generate new tokens one-by-one
        for _ in range(max_new_tokens):
            if len(tokens) >= self.block_size:
                break
            next_tok = sample_token(logits, temperature=temperature, top_k=top_k, top_p=top_p, greedy=greedy)
            tokens.append(next_tok)
            if eos_id is not None and next_tok == eos_id:
                break
            # Feed ONLY the single newly generated token
            logits, layer_caches = self.forward_kv_cached(next_tok, len(tokens) - 1, layer_caches)

        return tokens


# =====================================================================
# 5. Verification, Sampling Benchmarks & KV-Cache Proof
# =====================================================================

def run_experiment():
    print("=" * 70)
    print("Phase 17 - Topic 04: Autoregressive Generation & Sampling Strategies")
    print("=" * 70)

    # 1. Setup Model
    vocab_map = {c: i for i, c in enumerate(" abcdefghijklmnopqrstuvwxyz.<>")}
    inv_vocab = {i: c for c, i in vocab_map.items()}
    V = len(vocab_map)
    T = 32
    D = 16
    H = 2
    L = 2

    model = GPT(vocab_size=V, block_size=T, n_layer=L, n_head=H, d_model=D)
    print(f"[*] Initialized Generator GPT (V={V}, T_max={T}, d_model={D}, layers={L})")

    # Helper function to encode/decode simple strings
    def encode(s: str) -> List[int]:
        return [vocab_map.get(c, 0) for c in s.lower()]

    def decode(ids: List[int]) -> str:
        return "".join(inv_vocab.get(i, "?") for i in ids)

    prompt = "ai is"
    prompt_ids = encode(prompt)
    print(f"[*] Prompt: '{prompt}' (IDs: {prompt_ids})")

    # 2. Test Sampling Strategies
    print("\n" + "=" * 70)
    print("[*] Comparing Decoding & Sampling Strategies (Generating 15 tokens):")
    print("=" * 70)

    strategies = [
        ("Greedy Search", {"greedy": True}),
        ("Temperature = 0.5 (Low Entropy)", {"temperature": 0.5}),
        ("Temperature = 1.0 (Standard)", {"temperature": 1.0}),
        ("Temperature = 2.0 (High Entropy)", {"temperature": 2.0}),
        ("Top-k (k=3)", {"top_k": 3, "temperature": 0.8}),
        ("Top-p Nucleus (p=0.85)", {"top_p": 0.85, "temperature": 0.8}),
    ]

    for name, kwargs in strategies:
        random.seed(123)  # reset seed for comparison
        gen_ids = model.generate(prompt_ids, max_new_tokens=15, use_cache=True, **kwargs)
        gen_text = decode(gen_ids)
        print(f"  {name:<30} -> '{gen_text}'")

    # 3. Exact Mathematical Equivalence: Naive vs KV-Cache Generation
    print("\n" + "=" * 70)
    print("[*] Verifying KV-Cache Exact Numerical Equivalence:")
    print("=" * 70)
    # Under deterministic Greedy decoding, naive recomputation and KV-cache generation
    # MUST yield 100% IDENTICAL tokens and logits!
    gen_naive = model.generate(prompt_ids, max_new_tokens=12, use_cache=False, greedy=True)
    gen_cached = model.generate(prompt_ids, max_new_tokens=12, use_cache=True, greedy=True)

    print(f"  Naive recomputation output : '{decode(gen_naive)}'")
    print(f"  KV-Cached generation output: '{decode(gen_cached)}'")
    assert gen_naive == gen_cached, "KV-Cache produced different token sequence from naive forward pass!"
    print("  Identical token-for-token output confirmed: ✅ PASSED")

    # 4. Performance & Complexity Benchmark
    print("\n" + "=" * 70)
    print("[*] Benchmarking Generation Complexity: Naive O(T^2) vs Cached O(T):")
    print("=" * 70)

    n_tokens = 20
    # Measure Naive
    start_naive = time.perf_counter()
    _ = model.generate(prompt_ids, max_new_tokens=n_tokens, use_cache=False, greedy=True)
    time_naive = time.perf_counter() - start_naive

    # Measure Cached
    start_cached = time.perf_counter()
    _ = model.generate(prompt_ids, max_new_tokens=n_tokens, use_cache=True, greedy=True)
    time_cached = time.perf_counter() - start_cached

    speedup = time_naive / time_cached if time_cached > 0 else 1.0
    print(f"  Naive generation time  (re-evaluating full history) : {time_naive * 1000:6.2f} ms")
    print(f"  KV-Cached generation   (evaluating 1 token + cache) : {time_cached * 1000:6.2f} ms")
    print(f"  Speedup Factor                                     : {speedup:6.2f}x faster")
    assert speedup >= 1.0, "KV-cache should be faster or equal to naive recomputation"
    print("  KV-Cache complexity reduction demonstrated: ✅ PASSED")

    print("\n" + "=" * 70)
    print("✅ All Autoregressive Generation & Sampling tests passed successfully!")
    print("=" * 70)
    sys.exit(0)


if __name__ == "__main__":
    run_experiment()
