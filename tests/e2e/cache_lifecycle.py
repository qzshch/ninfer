"""Seeded, public conversation histories that cross sparse cache boundaries."""
from __future__ import annotations

import copy
import hashlib
import json
import random


TOOLS = [{"type": "function", "function": {
    "name": "read_status", "description": "Read a test status record.",
    "parameters": {"type": "object", "properties": {"id": {"type": "string"}}}}}]


def conversation(seed: int, window_tokens: int):
    rng = random.Random(seed)
    words = ["alpha", "beta", "gamma", "delta"]
    rng.shuffle(words)
    # A stable system prefix exceeds the sparse Device window. New turns must
    # keep that exact prefix while extending the checkpoint publication frontier.
    system = (f"Test conversation {seed}. Follow the final instruction exactly. "
              + (" ".join(words) + " ") * (window_tokens // 4 + 128))
    return [{"role": "system", "content": system},
            {"role": "user", "content": "Read the test status."},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": f"status_{seed}", "type": "function", "function": {
                    "name": "read_status", "arguments": '{"id":"test"}'}}]},
            {"role": "tool", "tool_call_id": f"status_{seed}",
             "content": "Status is online. " * 128},
            {"role": "user", "content": "Reply with exactly READY."}]


def extend(messages, seed: int, turn: int):
    history = copy.deepcopy(messages)
    history += [{"role": "assistant", "content":
                 f"Public result {seed}/{turn}. " + "alpha beta gamma delta " * (96 + turn * 16)},
                {"role": "user", "content": "Reply with exactly READY."}]
    return history


def run_lifecycle(suite, seed: int, *, control=False):
    window = suite.args.window * 64
    original = conversation(seed, window)
    journal = []

    def execute(action, messages, require_hit=False):
        result = suite.request(messages, tokens=32, stream=True, tools=TOOLS)
        # A generated health prompt itself publishes a competing checkpoint. It
        # must not evict the very prefix whose immediate warm reuse is under test.
        suite.liveness()
        usage = result["usage"]
        cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        if usage["prompt_tokens"] <= window:
            raise AssertionError(f"{action}: history did not exceed its sparse window")
        if require_hit and not control and cached <= window:
            raise AssertionError(f"{action}: no long retained prefix was exercised ({cached} cached)")
        if control and cached != 0:
            raise AssertionError(f"{action}: cold reference unexpectedly reused {cached} tokens")
        journal.append({"action": action, "seed": seed, "messages": len(messages),
                        "prompt_tokens": usage["prompt_tokens"], "cached_tokens": cached,
                        "completion_tokens": usage["completion_tokens"], "seconds": result["seconds"],
                        "text": result["text"], "request_messages_sha256": hashlib.sha256(
                            json.dumps(messages, sort_keys=True, separators=(',', ':')).encode()).hexdigest()})
        if hasattr(suite.args, 'output'):
            (suite.args.output / f'cache-transitions-{seed}.json').write_text(
                json.dumps(journal, indent=2), encoding='utf-8')
        if not control:
            references = getattr(suite.args, 'reference_outputs', {}).get(str(seed), {})
            if action not in references:
                raise AssertionError(f"{action}: no matching cold reference was collected")
            if result['text'] != references[action]:
                raise AssertionError(f"{action}: cached output differs from the same-input cold control: "
                                     f"cached={result['text']!r}, cold={references[action]!r}")

    execute("cold", original)
    execute("warm", original, require_hit=True)
    current = original
    for turn in range(3):
        current = extend(current, seed, turn)
        execute(f"grow-{turn}", current, require_hit=True)
        execute(f"republish-{turn}", current, require_hit=True)
    # Return to an earlier frontier and fork it; the later snapshot must not
    # overwrite immutable shared history or leak a previous active owner's state.
    execute("return-to-earlier-frontier", original)
    execute("branch-from-earlier-frontier", extend(original, seed + 1, 0))
    rewritten = copy.deepcopy(original)
    rewritten[0]["content"] = "REWRITTEN. " + rewritten[0]["content"]
    execute("rewrite-root", rewritten)
    # Distinct roots exceed Host/Device state slots and force eviction. Revisiting
    # the original history is valid even when it has become a cold activation.
    for offset in range(1, 7):
        execute(f"cache-pressure-{offset}", conversation(seed + 100 + offset, window))
    execute("revisit-after-eviction", original)
    return {"seed": seed, "transitions": journal}
