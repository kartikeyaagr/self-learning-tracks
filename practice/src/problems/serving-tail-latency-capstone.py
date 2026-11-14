import math

META = dict(title='Capstone: simulate a serving scheduler and fix its tail latency', tags=['Inference'], diff='Hard', runtime='python',
            related=['simulate-continuous-batching', 'ttft-tpot-itl-and-token-throughput', 'pagedattention-block-allocation', 'simulate-chunked-prefill-scheduling'])

STATEMENT = r"""
Write `simulate(requests, budget, kv_capacity, base, per_token, chunked)`. Each request is `[arrival, prompt_len, output_len]`, already sorted by arrival. Time is continuous; a scheduler **step** costs `base + per_token × tokens_in_step`.

State: a FCFS `waiting` queue, and admitted requests that are either *prefilling* (prompt tokens left) or *decoding*. Loop until every request is done:

1. If nothing is admitted and nobody has arrived by the current time, jump to the next arrival.
2. **Admit** arrived waiting requests in order while `prompt_len + output_len` fits in the free KV capacity (stop at the first that does not fit).
3. **Build the step.**
   - `chunked = False` (prefill-first): if any admitted request has prompt tokens left, the step is a prefill-only step: take whole remaining prompts in admission order while their total is ≤ `budget` (always take at least one). Otherwise every decoding request decodes one token.
   - `chunked = True`: every decoding request decodes one token; then fill the rest of `budget` with prompt chunks in admission order.
4. Advance time by the step cost. Then, for every request in the step:
   - a request whose prompt is now complete emits its **first token** at this time (TTFT = time − arrival);
   - a decoding request emits one token; its inter-token gap is the time since its previous token.
   A request whose emitted tokens reach `output_len` is done and frees its KV reservation.

Return `{"ttft_p50", "ttft_p99", "itl_p99", "throughput", "steps"}`. Percentiles use the nearest-rank method ($\lceil pn/100 \rceil$-th smallest, 0.0 if empty); throughput = total output tokens / time of the last token.
"""

STARTER = '''def simulate(requests, budget, kv_capacity, base, per_token, chunked):
    """Discrete-event simulation of an LLM serving scheduler."""
    return {"ttft_p50": 0.0, "ttft_p99": 0.0, "itl_p99": 0.0, "throughput": 0.0, "steps": 0}
'''

SOLUTION = '''import math

def _pct(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    return float(xs[max(math.ceil(p * len(xs) / 100), 1) - 1])

def simulate(requests, budget, kv_capacity, base, per_token, chunked):
    R = [{"arr": a, "left": p, "out": o, "need": p + o, "emitted": 0, "last": None} for a, p, o in requests]
    waiting, active = list(range(len(R))), []
    t, free, steps, ttft, itl, end = 0.0, kv_capacity, 0, [], [], 0.0
    while waiting or active:
        if not active and R[waiting[0]]["arr"] > t:
            t = R[waiting[0]]["arr"]
        while waiting and R[waiting[0]]["arr"] <= t and R[waiting[0]]["need"] <= free:
            i = waiting.pop(0); free -= R[i]["need"]; active.append(i)
        if not active:
            t = R[waiting[0]]["arr"]
            continue
        pre = [i for i in active if R[i]["left"] > 0]
        dec = [i for i in active if R[i]["left"] == 0]
        work = {}
        if not chunked:
            if pre:
                used = 0
                for i in pre:
                    if work and used + R[i]["left"] > budget:
                        break
                    work[i] = R[i]["left"]; used += R[i]["left"]
                dec = []
        else:
            room = budget - len(dec)
            for i in pre:
                if room <= 0:
                    break
                take = min(R[i]["left"], room); work[i] = take; room -= take
        tokens = sum(work.values()) + len(dec)
        t += base + per_token * tokens
        steps += 1
        emit = []
        for i, n in work.items():
            R[i]["left"] -= n
            if R[i]["left"] == 0:
                ttft.append(t - R[i]["arr"]); emit.append(i)
        for i in dec:
            itl.append(t - R[i]["last"]); emit.append(i)
        for i in emit:
            R[i]["emitted"] += 1; R[i]["last"] = t; end = t
            if R[i]["emitted"] >= R[i]["out"]:
                active.remove(i); free += R[i]["need"]
    total = sum(r["out"] for r in R)
    return {"ttft_p50": _pct(ttft, 50), "ttft_p99": _pct(ttft, 99), "itl_p99": _pct(itl, 99),
            "throughput": total / end if end > 0 else 0.0, "steps": steps}
'''

HINTS = [
    'Keep each request as a small record: tokens of prompt left, tokens emitted, time of its last token, and its KV reservation.',
    'Prefill-first scheduling makes every decoding request wait during a long prefill; that wait shows up as a large inter-token gap.',
    'With chunking, decodes go first in every step, so their gap is bounded by one step of at most `budget` tokens.',
]

_burst = [[0.0, 50, 20], [0.0, 60, 20], [1.0, 2000, 10], [1.5, 30, 20], [2.0, 1500, 5]]
_steady = [[0.2 * i, 200 + 37 * (i % 5), 30 + 7 * (i % 3)] for i in range(20)]
TESTS = [
    (_burst, 512, 10000, 0.01, 0.0005, False),
    (_burst, 512, 10000, 0.01, 0.0005, True),
    (_steady, 256, 3000, 0.01, 0.0005, False),
    (_steady, 256, 3000, 0.01, 0.0005, True),
    ([[0.0, 10, 1]], 64, 100, 0.01, 0.001, True),
    ([[0.0, 10, 3], [5.0, 10, 3]], 64, 100, 0.01, 0.001, False),
]

EXPLANATION = r"""
## The tail problem
With prefill-first scheduling a 2,000-token prompt runs as one long step, and every request that is already decoding waits for it. Users see that as a stall in the middle of a streamed answer: a large p99 inter-token latency even though the average looks fine.

## Chunked prefill
Splitting prompts into chunks that share each step with decodes caps every step at `budget` tokens, so decode gaps stay bounded. The cost is a slightly later first token for the long prompt, and a few more steps. Comparing the two settings on the same trace is the standard way to choose the budget.

## The other lever: KV capacity
Requests are admitted only when their full KV reservation fits. A small capacity queues requests and inflates TTFT; paged allocation (reserve blocks as tokens are produced) admits more requests for the same memory.

## Interview follow-ups
- How would disaggregating prefill and decode onto separate GPUs change these numbers?
"""
