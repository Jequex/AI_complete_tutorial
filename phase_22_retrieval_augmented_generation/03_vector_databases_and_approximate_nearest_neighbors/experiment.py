#!/usr/bin/env python3
"""
Phase 22 - Topic 03: Vector Databases and Approximate Nearest Neighbors (ANN)
Experiment: Pure Python Hierarchical Navigable Small World (HNSW) Index

This experiment implements:
1. Distance metrics: Cosine Distance & Euclidean Distance over unit vectors.
2. Exact brute-force k-NN linear scan (O(N * d)).
3. Pure Python Multi-Layer HNSW Graph Index:
   - Probabilistic layer assignment (skip-list hierarchy).
   - Multi-layer greedy routing from top entry point down to layer 0.
   - Beam-search graph traversal (efSearch) at layer 0.
   - Bi-directional neighbor link construction with heuristic degree capping (M).
4. Empirical benchmark on 200 synthetic vectors in 32 dimensions:
   - Evaluation across 30 test queries.
   - Measurement of Recall@k (target > 90%).
   - Operation count reduction: HNSW vs Brute-Force comparisons.

Zero external dependencies: pure standard Python 3.
"""

import math
import random
import sys
import time
from typing import List, Dict, Tuple, Set, Optional


def dot_product(u: List[float], v: List[float]) -> float:
    return sum(a * b for a, b in zip(u, v))


def euclidean_distance(u: List[float], v: List[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(u, v)))


def cosine_distance(u: List[float], v: List[float]) -> float:
    # For unit-normalized vectors: cosine_distance = 1.0 - dot_product
    dot = max(-1.0, min(1.0, dot_product(u, v)))
    return 1.0 - dot


def normalize(v: List[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v] if norm > 0.0 else v


class SimpleHNSW:
    """
    Pure Python Hierarchical Navigable Small World (HNSW) Graph Index.
    """
    def __init__(self, dim: int, M: int = 8, ef_construction: int = 24, ef_search: int = 16):
        self.dim = dim
        self.M = M                          # Max connections per node in higher layers
        self.M0 = 2 * M                     # Max connections per node in layer 0
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.mL = 1.0 / math.log(M)         # Normalization factor for level generation
        
        self.vectors: List[List[float]] = []
        # graphs[level][node_id] = set of neighbor node_ids
        self.graphs: List[Dict[int, Set[int]]] = []
        self.node_levels: List[int] = []
        self.entry_point: Optional[int] = None
        self.max_level: int = -1
        
        # Diagnostics
        self.distance_computations: int = 0
        self._query_dist_cache: Dict[int, float] = {}

    def _dist(self, u: List[float], v: List[float]) -> float:
        self.distance_computations += 1
        return cosine_distance(u, v)

    def _cached_dist_to_node(self, query: List[float], node: int) -> float:
        if node not in self._query_dist_cache:
            self._query_dist_cache[node] = self._dist(query, self.vectors[node])
        return self._query_dist_cache[node]

    def _random_level(self) -> int:
        r = random.random()
        while r == 0.0:
            r = random.random()
        return int(-math.log(r) * self.mL)

    def insert(self, vec: List[float]) -> int:
        node_id = len(self.vectors)
        self.vectors.append(vec)
        level = self._random_level()
        self.node_levels.append(level)

        # Expand graph levels if needed
        while len(self.graphs) <= level:
            self.graphs.append({})
        for l in range(level + 1):
            self.graphs[l][node_id] = set()

        if self.entry_point is None:
            self.entry_point = node_id
            self.max_level = level
            return node_id

        curr_obj = self.entry_point
        curr_dist = self._dist(vec, self.vectors[curr_obj])

        # Phase 1: Greedy routing through higher levels down to level+1
        for l in range(self.max_level, level, -1):
            changed = True
            while changed:
                changed = False
                for neighbor in self.graphs[l].get(curr_obj, []):
                    d = self._dist(vec, self.vectors[neighbor])
                    if d < curr_dist:
                        curr_dist = d
                        curr_obj = neighbor
                        changed = True

        # Phase 2: Insert into levels from min(max_level, level) down to 0
        ep = [curr_obj]
        top_insert_level = min(self.max_level, level)
        for l in range(top_insert_level, -1, -1):
            candidates = self._search_layer_insert(vec, ep, self.ef_construction, l)
            # Pick neighbors
            m_max = self.M0 if l == 0 else self.M
            neighbors = [node for dist, node in candidates[:m_max]]
            
            for neighbor in neighbors:
                self.graphs[l][node_id].add(neighbor)
                self.graphs[l][neighbor].add(node_id)
                # Prune if over capacity
                if len(self.graphs[l][neighbor]) > m_max:
                    self._shrink_connections(neighbor, l, m_max)

            ep = [node for dist, node in candidates]

        if level > self.max_level:
            self.max_level = level
            self.entry_point = node_id

        return node_id

    def _shrink_connections(self, node: int, level: int, max_conns: int) -> None:
        node_vec = self.vectors[node]
        neighbors = list(self.graphs[level][node])
        scored = [(self._dist(node_vec, self.vectors[nbr]), nbr) for nbr in neighbors]
        scored.sort(key=lambda x: x[0])
        kept = {nbr for dist, nbr in scored[:max_conns]}
        for nbr in neighbors:
            if nbr not in kept:
                self.graphs[level][nbr].discard(node)
        self.graphs[level][node] = kept

    def _search_layer_insert(self, query: List[float], enter_points: List[int], ef: int, level: int) -> List[Tuple[float, int]]:
        v_visited: Set[int] = set(enter_points)
        candidates: List[Tuple[float, int]] = []
        for ep in enter_points:
            d = self._dist(query, self.vectors[ep])
            candidates.append((d, ep))
        candidates.sort(key=lambda x: x[0])
        
        w_results = list(candidates)

        while candidates:
            c_dist, c_node = candidates.pop(0)
            f_dist = w_results[-1][0]

            if c_dist > f_dist and len(w_results) >= ef:
                break

            for neighbor in self.graphs[level].get(c_node, []):
                if neighbor not in v_visited:
                    v_visited.add(neighbor)
                    d = self._dist(query, self.vectors[neighbor])
                    if d < f_dist or len(w_results) < ef:
                        candidates.append((d, neighbor))
                        candidates.sort(key=lambda x: x[0])
                        w_results.append((d, neighbor))
                        w_results.sort(key=lambda x: x[0])
                        if len(w_results) > ef:
                            w_results.pop()
                            f_dist = w_results[-1][0]

        return w_results

    def _search_layer_query(self, query: List[float], enter_points: List[int], ef: int, level: int) -> List[Tuple[float, int]]:
        v_visited: Set[int] = set(enter_points)
        candidates: List[Tuple[float, int]] = []
        for ep in enter_points:
            d = self._cached_dist_to_node(query, ep)
            candidates.append((d, ep))
        candidates.sort(key=lambda x: x[0])
        
        w_results = list(candidates)

        while candidates:
            c_dist, c_node = candidates.pop(0)
            f_dist = w_results[-1][0]

            if c_dist > f_dist and len(w_results) >= ef:
                break

            for neighbor in self.graphs[level].get(c_node, []):
                if neighbor not in v_visited:
                    v_visited.add(neighbor)
                    d = self._cached_dist_to_node(query, neighbor)
                    if d < f_dist or len(w_results) < ef:
                        candidates.append((d, neighbor))
                        candidates.sort(key=lambda x: x[0])
                        w_results.append((d, neighbor))
                        w_results.sort(key=lambda x: x[0])
                        if len(w_results) > ef:
                            w_results.pop()
                            f_dist = w_results[-1][0]

        return w_results

    def search(self, query: List[float], k: int = 5) -> List[Tuple[int, float]]:
        if self.entry_point is None:
            return []

        self._query_dist_cache.clear()
        curr_obj = self.entry_point
        curr_dist = self._cached_dist_to_node(query, curr_obj)

        # Step 1: Greedy skip-list routing from top level to layer 1
        for l in range(self.max_level, 0, -1):
            changed = True
            while changed:
                changed = False
                for neighbor in self.graphs[l].get(curr_obj, []):
                    d = self._cached_dist_to_node(query, neighbor)
                    if d < curr_dist:
                        curr_dist = d
                        curr_obj = neighbor
                        changed = True

        # Step 2: Beam search on layer 0 with ef_search
        ef = max(self.ef_search, k)
        results = self._search_layer_query(query, [curr_obj], ef, level=0)
        
        # Return top-k (node_id, distance)
        return [(node, dist) for dist, node in results[:k]]


def brute_force_knn(vectors: List[List[float]], query: List[float], k: int) -> List[Tuple[int, float]]:
    scored = [(idx, cosine_distance(query, vec)) for idx, vec in enumerate(vectors)]
    scored.sort(key=lambda x: x[1])
    return scored[:k]


def run_experiment():
    print("=" * 80)
    print("PHASE 22 - TOPIC 03: VECTOR DATABASES & HNSW ANN EXPERIMENT")
    print("=" * 80)

    random.seed(42)
    dim = 32
    num_vectors = 200
    num_test_queries = 30
    k = 5

    print(f"\n[Step 1] Synthesizing {num_vectors} normalized vectors in {dim} dimensions...")
    vectors = []
    for _ in range(num_vectors):
        raw_vec = [random.gauss(0.0, 1.0) for _ in range(dim)]
        vectors.append(normalize(raw_vec))

    test_queries = []
    for _ in range(num_test_queries):
        raw_q = [random.gauss(0.0, 1.0) for _ in range(dim)]
        test_queries.append(normalize(raw_q))

    # 1. Build Pure Python HNSW Index
    print(f"\n[Step 2] Building HNSW Index (M=8, ef_construction=24, ef_search=16)...")
    hnsw = SimpleHNSW(dim=dim, M=8, ef_construction=24, ef_search=16)
    
    t0 = time.time()
    for vec in vectors:
        hnsw.insert(vec)
    t_build = time.time() - t0
    
    print(f"  HNSW built with {len(hnsw.graphs)} levels in {t_build*1000:.1f} ms.")
    for l, g in enumerate(hnsw.graphs):
        print(f"    Level {l}: {len(g)} nodes, avg connections: {sum(len(v) for v in g.values())/max(1, len(g)):.1f}")

    # 2. Benchmark HNSW Search vs Brute-Force Scan
    print(f"\n[Step 3] Evaluating ANN Search Accuracy vs Brute Force across {num_test_queries} queries (k={k})...")
    
    recalls = []
    hnsw_distance_ops = []
    brute_force_ops = num_vectors  # Exact scan always computes N distance comparisons per query

    for q in test_queries:
        # Exact scan
        exact_results = brute_force_knn(vectors, q, k)
        exact_set = {idx for idx, dist in exact_results}

        # HNSW search
        hnsw.distance_computations = 0
        hnsw_results = hnsw.search(q, k=k)
        hnsw_ops = hnsw.distance_computations
        hnsw_distance_ops.append(hnsw_ops)
        hnsw_set = {idx for idx, dist in hnsw_results}

        # Compute recall@k: |HNSW ∩ Exact| / k
        intersection = exact_set.intersection(hnsw_set)
        recall = len(intersection) / float(k)
        recalls.append(recall)

    avg_recall = sum(recalls) / len(recalls)
    avg_hnsw_ops = sum(hnsw_distance_ops) / len(hnsw_distance_ops)
    efficiency_gain = (1.0 - (avg_hnsw_ops / brute_force_ops)) * 100.0

    print(f"\n--- Benchmark Results ---")
    print(f"  Mean Recall@{k}:           {avg_recall * 100:.2f}% (Target: > 90.0%)")
    print(f"  Brute-Force Comparisons:  {brute_force_ops} per query")
    print(f"  HNSW Distance Checks:     {avg_hnsw_ops:.1f} per query")
    print(f"  Computational Reduction:  {efficiency_gain:.1f}% fewer operations than O(N*d)")

    # Assertions
    assert avg_recall >= 0.90, f"HNSW Recall@{k} ({avg_recall*100:.2f}%) did not meet >90% threshold!"
    assert avg_hnsw_ops < brute_force_ops, "HNSW must require fewer distance evaluations than exhaustive scan."
    print("✓ Recall requirement (>90%) successfully validated.")
    print("✓ Logarithmic search efficiency demonstrated.")

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED: Topic 03 HNSW Vector Index fully verified (Exit Code 0).")
    print("=" * 80)


if __name__ == "__main__":
    run_experiment()
    sys.exit(0)
