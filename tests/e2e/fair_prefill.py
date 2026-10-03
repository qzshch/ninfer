"""Opt-in real-model fairness regression, including dormant sparse lane reuse."""
from concurrent.futures import ThreadPoolExecutor
import threading


def run_fair_prefill(suite):
    """Grow all cold owners, reduce membership, then reuse every physical slot.

    This is a correctness regression. Timing is retained as evidence, without an
    absolute latency pass threshold that would turn a slower GPU into a failure.
    The existing parallel-lane oracle checks old-history retrieval, no cross-lane
    markers, complete transport and same-input isolated greedy output.
    """
    rounds = []
    for repeat in range(suite.args.repeats):
        rounds.append(suite.parallel_lanes(repeat * 2))
        # A one-row graph activation between full batches exercises a dormant
        # lane's address tables and graph ingress without clearing the engine.
        suite.health()
        rounds.append(suite.parallel_lanes(repeat * 2 + 1))
    # One long prefill followed by a short request must both complete. This
    # covers all-cold scheduling and the transition to a live decode owner.
    barrier = threading.Barrier(2)
    def request(long):
        barrier.wait(timeout=10)
        messages = (suite.long_messages(suite.args.window * 64 + 1024) if long else
                    [{'role': 'user', 'content': 'Reply with exactly READY.'}])
        return suite.request(messages, tokens=64, stream=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        mixed = list(pool.map(request, (True, False)))
    suite.liveness()
    return {'isolated_and_batched_rounds': rounds, 'mixed_long_and_short': mixed,
            'token_budget': suite.args.prefill_token_budget,
            'configured_lanes': suite.args.concurrency}
