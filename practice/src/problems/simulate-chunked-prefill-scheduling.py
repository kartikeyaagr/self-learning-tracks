META = dict(title='Chunked prefill under a per-step token budget', tags=['Inference'], diff='Hard', runtime='python',
            lessons=['ieng:10.02', 'ieng:07.04'],
            related=['simulate-continuous-batching', 'ttft-tpot-itl-and-token-throughput', 'prefix-cache-matching-and-reuse'])

STATEMENT = r"""
A 30 000-token prompt prefilled in one step freezes every other request's decoding for that step. **Chunked prefill** caps the tokens processed per step (the token budget), serves decodes first, and splits long prompts across several steps with whatever budget is left. Simulate it.

Write `chunked_prefill(requests, token_budget, max_seqs)`.

## Inputs
- `requests`: list of `(arrival_step, prompt_len, output_len)` ints with `prompt_len >= 1`, `output_len >= 1`. Ids are list indices; requests are admitted in `(arrival_step, id)` order.
- `token_budget`: tokens processed per step, with `token_budget >= max_seqs` (so decodes always fit).
- `max_seqs`: the most admitted (prefilling or decoding) requests at once. There is no KV limit here.

## Each step $t = 0, 1, 2, \dots$
1. **Admit** arrived requests (`arrival_step <= t`) in order while fewer than `max_seqs` are admitted.
2. **Decode first.** Every request already in the decode phase uses 1 token of the budget and produces one output token.
3. **Prefill with what is left.** Go through prefilling requests in admission order; each takes `min(remaining prompt tokens, remaining budget)` tokens until the budget is used up.
4. A request whose **last** prompt chunk ran in this step also produces its **first** output token in this step (it costs no extra budget) and decodes from step $t + 1$ on.
5. A request that has produced `output_len` tokens finishes at step $t$ and leaves after the step.

## Output
A tuple `(schedule, used)`:
- `schedule`: list of `[first_token_step, finish_step]` per request id.
- `used`: list of tokens processed (decode + prefill) at each step from 0 to the last finish step.
"""

STARTER = '''def chunked_prefill(requests, token_budget, max_seqs):
    """Return (schedule: [[first_token_step, finish_step], ...], used: [tokens per step])."""
    raise NotImplementedError
'''

SOLUTION = '''def chunked_prefill(requests, token_budget, max_seqs):
    order = sorted(range(len(requests)), key=lambda i: (requests[i][0], i))
    admitted = []                                   # ids in admission order
    left = {}                                       # id -> prompt tokens still to prefill
    made = {}                                       # id -> output tokens produced
    sched = [[None, None] for _ in requests]
    used, t, done = [], 0, 0
    while done < len(requests):
        while order and requests[order[0]][0] <= t and len(admitted) < max_seqs:
            i = order.pop(0)
            admitted.append(i)
            left[i], made[i] = requests[i][1], 0

        budget = token_budget
        decoding = [i for i in admitted if left[i] == 0]    # decided before this step's prefills finish
        for i in decoding:                                  # decodes first: they keep TPOT steady
            budget -= 1
            made[i] += 1
        for i in admitted:
            if left[i] == 0 or budget == 0:
                continue
            chunk = min(left[i], budget)
            left[i] -= chunk
            budget -= chunk
            if left[i] == 0:                                # last chunk ran: the first token comes with it
                made[i] = 1
                sched[i][0] = t
        used.append(token_budget - budget)

        for i in list(admitted):
            if made[i] == requests[i][2]:
                sched[i][1] = t
                admitted.remove(i)
                done += 1
        t += 1
    return sched, used
'''

HINTS = [
    'Track, per admitted request, how many prompt tokens are still to prefill and how many output tokens it has produced. A request is decoding when its remaining prompt is 0.',
    'Decide who decodes **before** running this step\'s prefill chunks: a request that finishes prefilling now gets its first token from the prefill, not an extra decode.',
    'Spend the budget greedily in admission order. Retire finished requests only after the step is complete, then increase `t`.',
]

TESTS = [
    ([(0, 5, 2)], 3, 2),
    ([(0, 2, 3), (0, 6, 1)], 4, 4),
    ([(0, 1, 6), (1, 20, 2)], 8, 2),
    ([(0, 4, 1), (0, 4, 1), (0, 4, 1)], 4, 1),
    ([(2, 3, 3), (0, 3, 3)], 2, 2),
    ([(0, 1, 1)], 1, 1),
    ([((i * 2) % 5, 1 + (i * 7) % 13, 1 + (i * 3) % 5) for i in range(12)], 6, 3),
    ([(0, 30, 4)] + [(i, 2, 6) for i in range(1, 6)], 8, 4),
]

EXPLANATION = r"""
## Why it works
Every step processes at most `token_budget` tokens, so step time (and therefore every running request's inter-token latency) is bounded no matter how long an incoming prompt is. Serving decodes first protects TPOT; leftover budget goes to prefill so the GPU still does compute-heavy work in every step. A long prompt's TTFT grows a little, since its prefill is spread over $\lceil \text{prompt} / \text{budget} \rceil$ steps, but nobody else stalls.

## Choosing the budget
A larger budget means fewer steps per prompt (better TTFT and throughput, because big prefill chunks use the tensor cores well), a smaller budget means faster, steadier decode steps. Engines expose it as "max batched tokens"; values of a few thousand tokens are typical on large GPUs.

## Common bugs
- Letting a request that just finished prefill also decode in the same step, so it gets two tokens in one iteration.
- Prefilling in arrival order across **all** queued requests instead of admitted ones, which bypasses `max_seqs`.
- Forgetting that a prompt shorter than the leftover budget leaves budget for the next prefilling request in the same step (test 8 fills each step completely).

## Interview follow-ups
- How does chunked prefill interact with prefix caching? (Cached blocks are skipped, so only uncached tokens consume budget; chunk boundaries should align with KV blocks.)
- Why is mixing prefill and decode in one batch good for utilisation? (Decode is memory-bound and prefill compute-bound: one fused step uses both the bandwidth and the FLOPs.)
"""
