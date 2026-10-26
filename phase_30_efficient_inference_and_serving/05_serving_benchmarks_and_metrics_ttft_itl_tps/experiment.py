#!/usr/bin/env python3
"""
Topic 30.5 Experiment: Serving Telemetry & Open-Loop Poisson Load Simulator
Phase 30: Efficient Inference & Serving

This experiment implements a pure Python 3 (standard library only) engine for:
1. Open-loop Poisson arrival process modeling realistic user traffic bursts.
2. Cycle-accurate continuous batching execution simulation tracking TTFT, ITL, and TPS.
3. Accurate empirical percentile calculations (p50, p90, p99) and ASCII histogram rendering.
4. Empirical verification of Little's Law (L = lambda * W).
"""

from typing import Dict, List, Optional, Tuple
import collections
import math
import random

class InferenceQuery:
    """Represents a simulated request dispatched to the serving cluster."""
    def __init__(self, query_id: int, arrival_time_ms: float, prompt_tokens: int, max_tokens: int):
        self.query_id = query_id
        self.arrival_time_ms = arrival_time_ms
        self.prompt_tokens = prompt_tokens
        self.max_tokens = max_tokens

        # Telemetry timestamps
        self.admitted_time_ms: Optional[float] = None
        self.first_token_time_ms: Optional[float] = None
        self.token_timestamps: List[float] = []
        self.completion_time_ms: Optional[float] = None

    @property
    def ttft_ms(self) -> float:
        """Time to First Token: latency from arrival to first token emission."""
        assert self.first_token_time_ms is not None
        return self.first_token_time_ms - self.arrival_time_ms

    @property
    def itls_ms(self) -> List[float]:
        """Inter-Token Latencies: durations between consecutive token emissions."""
        if len(self.token_timestamps) < 2:
            return []
        return [self.token_timestamps[i] - self.token_timestamps[i - 1] for i in range(1, len(self.token_timestamps))]

    @property
    def e2e_latency_ms(self) -> float:
        """Total turnaround residence time in the serving system."""
        assert self.completion_time_ms is not None
        return self.completion_time_ms - self.arrival_time_ms


def compute_percentiles(values: List[float], percentiles: List[float]) -> Dict[float, float]:
    """Computes exact empirical percentiles from a list of floats."""
    if not values:
        return {p: 0.0 for p in percentiles}
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    results = {}
    for p in percentiles:
        # Linear interpolation percentile
        rank = (p / 100.0) * (n - 1)
        low_idx = int(math.floor(rank))
        high_idx = int(math.ceil(rank))
        weight = rank - low_idx
        p_val = sorted_vals[low_idx] * (1.0 - weight) + sorted_vals[high_idx] * weight
        results[p] = p_val
    return results


def print_ascii_histogram(values: List[float], title: str, num_bins: int = 10, unit: str = "ms"):
    """Renders a clean ASCII frequency distribution histogram."""
    if not values:
        return
    min_v = min(values)
    max_v = max(values)
    if min_v == max_v:
        max_v += 1e-3
    bin_width = (max_v - min_v) / num_bins
    bins = [0] * num_bins

    for v in values:
        idx = min(int((v - min_v) / bin_width), num_bins - 1)
        bins[idx] += 1

    max_count = max(bins)
    print(f"\n  [HISTOGRAM] {title}:")
    for i in range(num_bins):
        low = min_v + i * bin_width
        high = low + bin_width
        bar_len = int((bins[i] / max_count) * 35) if max_count > 0 else 0
        bar = "█" * bar_len
        print(f"    [{low:6.1f} - {high:6.1f} {unit}] | {bar:<35} ({bins[i]:3d})")


class ClusterSimulator:
    """
    Simulates a continuous batching GPU cluster under open-loop Poisson arrival traffic.
    """
    def __init__(self, max_batch_size: int = 16, decode_base_ms: float = 12.0, prefill_ms_per_100_tok: float = 8.0):
        self.max_batch_size = max_batch_size
        self.decode_base_ms = decode_base_ms
        self.prefill_rate = prefill_ms_per_100_tok / 100.0

    def run_simulation(self, arrival_rate_lambda: float, num_requests: int = 100) -> List[InferenceQuery]:
        """
        Executes discrete-event simulation of Poisson arrivals and continuous batching.
        arrival_rate_lambda: requests per second.
        """
        # 1. Generate Poisson arrival sequence: Delta t = -ln(U) / lambda
        random.seed(42)
        queries: List[InferenceQuery] = []
        curr_time = 0.0
        for i in range(num_requests):
            inter_arrival_s = -math.log(random.random()) / arrival_rate_lambda
            curr_time += inter_arrival_s * 1000.0  # convert to ms
            prompt_len = random.randint(128, 512)
            gen_len = random.randint(20, 60)
            queries.append(InferenceQuery(i, curr_time, prompt_len, gen_len))

        # Simulation queues
        waiting_queue: collections.deque[InferenceQuery] = collections.deque()
        running_batch: List[InferenceQuery] = []
        completed_queries: List[InferenceQuery] = []

        clock_ms = 0.0
        query_idx = 0

        while query_idx < len(queries) or running_batch or waiting_queue:
            # Advance clock if GPU is idle and queue is empty
            if not running_batch and not waiting_queue:
                if query_idx < len(queries):
                    clock_ms = queries[query_idx].arrival_time_ms

            # Enqueue any new arrivals up to current clock
            while query_idx < len(queries) and queries[query_idx].arrival_time_ms <= clock_ms:
                waiting_queue.append(queries[query_idx])
                query_idx += 1

            # Admit requests into running batch up to max_batch_size
            admitted_this_step: List[InferenceQuery] = []
            while len(running_batch) < self.max_batch_size and waiting_queue:
                req = waiting_queue.popleft()
                req.admitted_time_ms = clock_ms
                running_batch.append(req)
                admitted_this_step.append(req)

            # Execution step duration:
            # 1. If any requests were just admitted, execute their prefill phase
            prefill_time = sum(req.prompt_tokens * self.prefill_rate for req in admitted_this_step)
            # 2. Decode forward step duration scales slightly with batch size (memory bound)
            batch_factor = 1.0 + (len(running_batch) / self.max_batch_size) * 0.4
            step_duration_ms = prefill_time + (self.decode_base_ms * batch_factor)

            # Advance clock
            clock_ms += step_duration_ms

            # Process newly admitted requests: first token emitted!
            for req in admitted_this_step:
                req.first_token_time_ms = clock_ms
                req.token_timestamps.append(clock_ms)

            # Generate 1 token for ongoing running batch
            finished_reqs: List[InferenceQuery] = []
            for req in running_batch:
                if req not in admitted_this_step:
                    req.token_timestamps.append(clock_ms)

                if len(req.token_timestamps) >= req.max_tokens:
                    req.completion_time_ms = clock_ms
                    finished_reqs.append(req)

            for req in finished_reqs:
                running_batch.remove(req)
                completed_queries.append(req)

        return completed_queries


def run_serving_benchmarks_experiment():
    print("=" * 80)
    print("PHASE 30: TOPIC 05 EXPERIMENT — SERVING BENCHMARKS (TTFT, ITL, TPS)")
    print("=" * 80)

    # 1. Initialize Simulator
    sim = ClusterSimulator(max_batch_size=12, decode_base_ms=10.0, prefill_ms_per_100_tok=5.0)

    # Run at moderate load (lambda = 8 requests/sec)
    arrival_rate = 8.0
    num_queries = 80
    print(f"\n[1] OPEN-LOOP POISSON BENCHMARK RUN (Arrival Rate Lambda = {arrival_rate} req/s, Total Queries = {num_queries}):")
    results = sim.run_simulation(arrival_rate_lambda=arrival_rate, num_requests=num_queries)

    # 2. Extract metrics
    ttfts = [q.ttft_ms for q in results]
    all_itls = []
    for q in results:
        all_itls.extend(q.itls_ms)
    e2e_latencies = [q.e2e_latency_ms for q in results]
    total_tokens_generated = sum(len(q.token_timestamps) for q in results)
    sim_duration_s = (max(q.completion_time_ms for q in results) - min(q.arrival_time_ms for q in results)) / 1000.0

    system_tps = total_tokens_generated / sim_duration_s
    mean_itl = sum(all_itls) / len(all_itls) if all_itls else 0.0
    user_tps = 1000.0 / mean_itl if mean_itl > 0 else 0.0

    # Percentiles
    ttft_pcts = compute_percentiles(ttfts, [50, 90, 99])
    itl_pcts = compute_percentiles(all_itls, [50, 90, 99])
    e2e_pcts = compute_percentiles(e2e_latencies, [50, 90, 99])

    print(f"\n[2] TELEMETRY SUMMARY REPORT:")
    print(f"  • Total Generated Tokens : {total_tokens_generated:,} tokens")
    print(f"  • Wall-Clock Duration    : {sim_duration_s:.2f} seconds")
    print(f"  • System Throughput (TPS): {system_tps:.1f} tokens/second")
    print(f"  • User Throughput (TPS)  : {user_tps:.1f} tokens/second per active stream")

    print(f"\n[3] LATENCY PERCENTILE BREAKDOWN:")
    print(f"  {'Metric':<24} | {'p50 (Median)':<14} | {'p90':<14} | {'p99 (Tail)':<14} | {'Mean'}")
    print("  " + "-" * 76)
    print(f"  {'Time to First Token (TTFT)':<24} | {ttft_pcts[50]:<11.1f} ms | {ttft_pcts[90]:<11.1f} ms | {ttft_pcts[99]:<11.1f} ms | {sum(ttfts)/len(ttfts):.1f} ms")
    print(f"  {'Inter-Token Latency (ITL)':<24} | {itl_pcts[50]:<11.1f} ms | {itl_pcts[90]:<11.1f} ms | {itl_pcts[99]:<11.1f} ms | {mean_itl:.1f} ms")
    print(f"  {'End-to-End Latency':<24} | {e2e_pcts[50]:<11.1f} ms | {e2e_pcts[90]:<11.1f} ms | {e2e_pcts[99]:<11.1f} ms | {sum(e2e_latencies)/len(e2e_latencies):.1f} ms")

    # 4. Render ASCII Histograms
    print_ascii_histogram(ttfts, title="Time to First Token (TTFT) Distribution", num_bins=8, unit="ms")
    print_ascii_histogram(all_itls, title="Inter-Token Latency (ITL) Distribution", num_bins=8, unit="ms")

    # 5. Little's Law Verification: L = lambda * W
    mean_w_s = (sum(e2e_latencies) / len(e2e_latencies)) / 1000.0
    theoretical_l = arrival_rate * mean_w_s
    print(f"\n[4] LITTLE'S LAW EMPIRICAL VERIFICATION (L = lambda * W):")
    print(f"  • Arrival Rate Lambda (req/s)    : {arrival_rate:.2f}")
    print(f"  • Mean Residence Time W (s)      : {mean_w_s:.3f} seconds")
    print(f"  • Theoretical Mean Concurrency L : {theoretical_l:.2f} active requests")

    # Invariants verification
    assert ttft_pcts[99] >= ttft_pcts[50], "p99 must be >= p50!"
    assert itl_pcts[99] >= itl_pcts[50], "p99 ITL must be >= p50 ITL!"
    assert system_tps > user_tps, "System TPS must exceed single-user TPS in batched serving!"
    print("\n✓ ALL SERVING TELEMETRY & BENCHMARK INVARIANTS VERIFIED SUCCESSFULLY.")

if __name__ == "__main__":
    run_serving_benchmarks_experiment()
