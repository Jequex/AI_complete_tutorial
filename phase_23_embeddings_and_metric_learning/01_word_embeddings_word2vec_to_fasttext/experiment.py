#!/usr/bin/env python3
"""
Phase 23 - Topic 01: Word Embeddings (Word2Vec to FastText)
Experiment: Skip-Gram with Negative Sampling (SGNS) & Subword Embedding Engine

This experiment implements from first principles in pure Python 3:
1. Tokenization and Vocabulary compilation with 3/4-exponent unigram noise distribution.
2. Skip-Gram with Negative Sampling (SGNS) model with dual embedding matrices (input W and context W').
3. Analytical gradient updates for positive and negative samples avoiding the full vocabulary softmax.
4. Model training on a structured semantic corpus (Animals, Fruits, Royalty).
5. Cosine similarity evaluation and vector space geometry validation.
6. FastText character n-gram subword synthesizer demonstrating out-of-vocabulary (OOV)
   and typo-resilient embedding synthesis.

Zero external dependencies: pure standard Python 3.
"""

import math
import random
import sys
from typing import List, Dict, Tuple, Set


def sigmoid(x: float) -> float:
    """Numerically stable logistic sigmoid function."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def dot(u: List[float], v: List[float]) -> float:
    """Computes dot product of two vectors."""
    return sum(a * b for a, b in zip(u, v))


def norm(u: List[float]) -> float:
    """Computes Euclidean L2 norm of a vector."""
    return math.sqrt(sum(a * a for a in u))


def cosine_similarity(u: List[float], v: List[float]) -> float:
    """Computes cosine similarity between two vectors."""
    nu = norm(u)
    nv = norm(v)
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return dot(u, v) / (nu * nv)


class SGNS:
    """
    Skip-Gram with Negative Sampling (SGNS) Word Embedding Model.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 16, seed: int = 42):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        random.seed(seed)
        
        # Initialize input embeddings W and context embeddings W'
        scale = 0.1 / embed_dim
        self.W = [
            [(random.random() - 0.5) * scale for _ in range(embed_dim)]
            for _ in range(vocab_size)
        ]
        self.W_prime = [
            [(random.random() - 0.5) * scale for _ in range(embed_dim)]
            for _ in range(vocab_size)
        ]

    def train_step(self, center_id: int, context_id: int, neg_ids: List[int], lr: float) -> float:
        """
        Executes a single SGNS update step on a center word, positive context word,
        and k negative noise words using exact analytical gradients.
        
        Returns the negative log-likelihood loss for this step.
        """
        v_I = self.W[center_id]
        v_O = self.W_prime[context_id]
        
        # 1. Positive pair
        z_O = dot(v_O, v_I)
        sig_O = sigmoid(z_O)
        loss = -math.log(max(sig_O, 1e-12))
        
        # Gradient coefficient for positive context: (sigma - 1)
        grad_pos_coeff = sig_O - 1.0
        
        # Accumulate gradient for center input vector
        grad_v_I = [grad_pos_coeff * v_O[j] for j in range(self.embed_dim)]
        
        # Update positive context vector
        for j in range(self.embed_dim):
            v_O[j] -= lr * grad_pos_coeff * v_I[j]
            
        # 2. Negative pairs
        for neg_id in neg_ids:
            v_neg = self.W_prime[neg_id]
            z_neg = dot(v_neg, v_I)
            sig_neg = sigmoid(z_neg)
            
            loss += -math.log(max(1.0 - sig_neg, 1e-12))
            
            grad_neg_coeff = sig_neg
            for j in range(self.embed_dim):
                grad_v_I[j] += grad_neg_coeff * v_neg[j]
                v_neg[j] -= lr * grad_neg_coeff * v_I[j]
                
        # 3. Update center word input vector
        for j in range(self.embed_dim):
            v_I[j] -= lr * grad_v_I[j]
            
        return loss

    def get_embedding(self, word_id: int) -> List[float]:
        """Returns the input embedding vector for a given word index."""
        return self.W[word_id]


class FastTextSubwordSynthesizer:
    """
    Extracts character n-grams and synthesizes word representations via subword aggregation.
    """
    def __init__(self, n_min: int = 3, n_max: int = 4, embed_dim: int = 16, seed: int = 42):
        self.n_min = n_min
        self.n_max = n_max
        self.embed_dim = embed_dim
        self.ngram_to_id: Dict[str, int] = {}
        self.subword_vectors: List[List[float]] = []
        random.seed(seed)

    def _get_or_create_subword_id(self, ngram: str) -> int:
        if ngram not in self.ngram_to_id:
            idx = len(self.subword_vectors)
            self.ngram_to_id[ngram] = idx
            scale = 0.5 / self.embed_dim
            self.subword_vectors.append([(random.random() - 0.5) * scale for _ in range(self.embed_dim)])
        return self.ngram_to_id[ngram]

    def extract_ngrams(self, word: str) -> List[str]:
        """Extracts character n-grams with boundary delimiters < and >."""
        tagged = f"<{word}>"
        ngrams = []
        L = len(tagged)
        for n in range(self.n_min, self.n_max + 1):
            for i in range(L - n + 1):
                ngrams.append(tagged[i:i + n])
        ngrams.append(tagged)  # special whole word token
        return ngrams

    def register_word(self, word: str, word_vector: List[float]):
        """Associates character n-grams of a word with its learned vector."""
        ngrams = self.extract_ngrams(word)
        m = len(ngrams)
        for ng in ngrams:
            idx = self._get_or_create_subword_id(ng)
            for j in range(self.embed_dim):
                self.subword_vectors[idx][j] = word_vector[j] / m

    def synthesize_oov_vector(self, oov_word: str) -> List[float]:
        """Synthesizes an embedding vector for an unseen word or typo via its constituent n-grams."""
        ngrams = self.extract_ngrams(oov_word)
        vec = [0.0] * self.embed_dim
        count = 0
        for ng in ngrams:
            if ng in self.ngram_to_id:
                idx = self.ngram_to_id[ng]
                sub_vec = self.subword_vectors[idx]
                for j in range(self.embed_dim):
                    vec[j] += sub_vec[j]
                count += 1
        if count == 0:
            return vec
        return [x / count for x in vec]


def create_corpus() -> List[List[str]]:
    """Generates a structured text corpus with three distinct topical clusters."""
    sentences = [
        # Animal Cluster
        ["dog", "puppy", "pet", "fur", "bark", "tail", "play"],
        ["cat", "kitten", "pet", "fur", "purr", "paws", "play"],
        ["dog", "puppy", "chase", "ball", "friendly", "bark"],
        ["cat", "kitten", "catch", "mouse", "quiet", "purr"],
        ["pet", "animal", "dog", "cat", "puppy", "kitten", "fur"],
        ["puppy", "dog", "faithful", "guard", "bark", "tail"],
        ["kitten", "cat", "gentle", "claws", "purr", "sleep"],
        ["dog", "cat", "pet", "companion", "fur", "tail"],

        # Fruit Cluster
        ["apple", "banana", "sweet", "fruit", "ripe", "orchard"],
        ["orange", "citrus", "juice", "fruit", "sweet", "peel"],
        ["banana", "yellow", "sweet", "fruit", "ripe", "peel"],
        ["apple", "red", "crisp", "fruit", "sweet", "orchard"],
        ["orange", "apple", "banana", "fruit", "salad", "delicious"],
        ["fresh", "fruit", "harvest", "apple", "banana", "orange"],
        ["sweet", "delicious", "fruit", "juice", "ripe", "fresh"],
        ["tasty", "apple", "orange", "banana", "fruit", "ripe"],

        # Royalty Cluster
        ["king", "queen", "royal", "crown", "throne", "palace"],
        ["prince", "princess", "royal", "palace", "crown", "monarch"],
        ["king", "ruler", "kingdom", "throne", "crown", "royal"],
        ["queen", "ruler", "kingdom", "palace", "crown", "royal"],
        ["prince", "son", "king", "queen", "royal", "palace"],
        ["princess", "daughter", "king", "queen", "royal", "court"],
        ["monarch", "reign", "royal", "crown", "throne", "palace"],
        ["noble", "king", "queen", "reign", "royal", "throne"]
    ]
    return sentences


def build_vocab(corpus: List[List[str]]) -> Tuple[Dict[str, int], Dict[int, str], List[int]]:
    """Compiles vocabulary and unigram frequency table."""
    counts: Dict[str, int] = {}
    for sent in corpus:
        for w in sent:
            counts[w] = counts.get(w, 0) + 1
            
    words = sorted(counts.keys())
    word_to_id = {w: i for i, w in enumerate(words)}
    id_to_word = {i: w for i, w in enumerate(words)}
    freqs = [counts[w] for w in words]
        
    return word_to_id, id_to_word, freqs


def build_negative_sampler(freqs: List[int], table_size: int = 100000) -> List[int]:
    """Constructs negative sampling lookup table using 3/4-exponent unigram smoothing."""
    smoothed_powers = [count ** 0.75 for count in freqs]
    total_power = sum(smoothed_powers)
    table = []
    for word_id, power in enumerate(smoothed_powers):
        slots = int(round((power / total_power) * table_size))
        table.extend([word_id] * slots)
    while len(table) < table_size:
        table.append(0)
    return table


def main():
    print("=" * 80)
    print("PHASE 23 - TOPIC 01: WORD EMBEDDINGS (WORD2VEC SGNS & FASTTEXT)")
    print("=" * 80)
    
    # 1. Corpus compilation & Vocabulary build
    corpus = create_corpus()
    word_to_id, id_to_word, freqs = build_vocab(corpus)
    vocab_size = len(word_to_id)
    print(f"[+] Loaded structured corpus: {len(corpus)} sentences, {sum(freqs)} tokens.")
    print(f"[+] Compiled vocabulary: {vocab_size} unique words.")

    # 2. Setup Negative Sampling Table
    neg_table = build_negative_sampler(freqs)
    print(f"[+] Constructed 3/4 unigram negative sampling table: {len(neg_table)} entries.")

    # 3. Model Initialization
    embed_dim = 16
    k_neg = 5
    window_size = 2
    base_lr = 0.05
    epochs = 200
    model = SGNS(vocab_size=vocab_size, embed_dim=embed_dim, seed=42)
    print(f"[+] Initialized SGNS model: embed_dim={embed_dim}, k_neg={k_neg}, window={window_size}")

    # 4. Extract training pairs
    pairs = []
    for sent in corpus:
        sent_ids = [word_to_id[w] for w in sent]
        L = len(sent_ids)
        for i in range(L):
            center = sent_ids[i]
            for j in range(max(0, i - window_size), min(L, i + window_size + 1)):
                if i != j:
                    pairs.append((center, sent_ids[j]))
    print(f"[+] Extracted {len(pairs)} co-occurrence training pairs.")

    # 5. Training Loop with Linear Learning Rate Decay
    print("\n[+] Training Skip-Gram with Negative Sampling...")
    random.seed(42)
    for epoch in range(1, epochs + 1):
        cur_lr = base_lr * (1.0 - (epoch - 1) / epochs)
        total_loss = 0.0
        random.shuffle(pairs)
        for center_id, context_id in pairs:
            # Sample k negatives from table
            neg_ids = []
            for _ in range(k_neg):
                neg_id = random.choice(neg_table)
                while neg_id == center_id or neg_id == context_id:
                    neg_id = random.choice(neg_table)
                neg_ids.append(neg_id)
                
            loss = model.train_step(center_id, context_id, neg_ids, cur_lr)
            total_loss += loss
            
        if epoch % 50 == 0 or epoch == epochs:
            avg_loss = total_loss / len(pairs)
            print(f"    Epoch {epoch:3d}/{epochs} (lr={cur_lr:.4f}) | Avg Loss: {avg_loss:.4f}")

    # 6. Evaluate Cosine Similarities within and across clusters
    print("\n" + "=" * 80)
    print("SEMANTIC CLUSTER EVALUATION: COSINE SIMILARITY")
    print("=" * 80)

    def sim(w1: str, w2: str) -> float:
        v1 = model.get_embedding(word_to_id[w1])
        v2 = model.get_embedding(word_to_id[w2])
        return cosine_similarity(v1, v2)

    # In-cluster pairs (Expected HIGH similarity)
    sim_cat_kitten = sim("cat", "kitten")
    sim_dog_puppy = sim("dog", "puppy")
    sim_apple_banana = sim("apple", "banana")
    sim_king_queen = sim("king", "queen")

    # Cross-cluster pairs (Expected LOW similarity)
    sim_cat_apple = sim("cat", "apple")
    sim_dog_banana = sim("dog", "banana")
    sim_king_apple = sim("king", "apple")
    sim_queen_dog = sim("queen", "dog")

    print(f"[*] In-Cluster Animal : sim(cat, kitten)    = {sim_cat_kitten:+.4f}")
    print(f"[*] In-Cluster Animal : sim(dog, puppy)     = {sim_dog_puppy:+.4f}")
    print(f"[*] In-Cluster Fruit  : sim(apple, banana)  = {sim_apple_banana:+.4f}")
    print(f"[*] In-Cluster Royalty: sim(king, queen)    = {sim_king_queen:+.4f}")
    print("-" * 50)
    print(f"[*] Cross-Cluster     : sim(cat, apple)     = {sim_cat_apple:+.4f}")
    print(f"[*] Cross-Cluster     : sim(dog, banana)    = {sim_dog_banana:+.4f}")
    print(f"[*] Cross-Cluster     : sim(king, apple)    = {sim_king_apple:+.4f}")
    print(f"[*] Cross-Cluster     : sim(queen, dog)     = {sim_queen_dog:+.4f}")

    # Assertions for semantic clustering
    assert sim_cat_kitten > sim_cat_apple, "Failed: cat should be closer to kitten than to apple"
    assert sim_dog_puppy > sim_dog_banana, "Failed: dog should be closer to puppy than to banana"
    assert sim_apple_banana > sim_cat_apple, "Failed: apple-banana should exceed cross-cluster"
    assert sim_king_queen > sim_king_apple, "Failed: king should be closer to queen than to apple"
    print("\n[✓] All Word2Vec Cosine Similarity cluster separation assertions PASSED!")

    # 7. FastText Subword OOV Synthesizer Demonstration
    print("\n" + "=" * 80)
    print("FASTTEXT SUBWORD SYNTHESIZER: OOV & TYPO RESILIENCE")
    print("=" * 80)
    
    synthesizer = FastTextSubwordSynthesizer(n_min=3, n_max=4, embed_dim=embed_dim)
    for word, wid in word_to_id.items():
        synthesizer.register_word(word, model.get_embedding(wid))

    # Test typos and out-of-vocabulary words
    typos = [
        ("kiten", "kitten"),
        ("oraneg", "orange"),
        ("bananna", "banana"),
        ("puppies", "puppy")
    ]

    for typo, target in typos:
        oov_vec = synthesizer.synthesize_oov_vector(typo)
        target_vec = model.get_embedding(word_to_id[target])
        unrelated_vec = model.get_embedding(word_to_id["king" if target != "king" else "apple"])
        
        sim_typo_target = cosine_similarity(oov_vec, target_vec)
        sim_typo_unrelated = cosine_similarity(oov_vec, unrelated_vec)
        
        print(f"[*] Typo: '{typo:7s}' -> Target: '{target:7s}' | Cosine: {sim_typo_target:+.4f} vs Unrelated: {sim_typo_unrelated:+.4f}")
        assert sim_typo_target > sim_typo_unrelated, f"Failed: Typo '{typo}' should be closer to '{target}'"

    print("\n[✓] FastText Subword OOV & Typo Synthesis assertions PASSED!")
    print("\n[★] EXPERIMENT COMPLETE: ALL TESTS PASSED SUCCESSFULLY (Exit Code 0).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
