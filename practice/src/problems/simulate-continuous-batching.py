META = dict(title='Continuous batching scheduler with a KV budget', tags=['Inference'], diff='Hard', runtime='python',
            lessons=['ieng:07.03', 'ieng:07.06'],
            related=['dynamic-request-batching', 'simulate-chunked-prefill-scheduling', 'pagedattention-block-allocation'])

STATEMENT = r"""
LLM engines schedule at the granularity of one **decode step** (iteration): after every step, finished requests leave the batch and waiting ones join, so short answers never wait for long ones. Simulate that loop.

Write `continuous_batching(requests, max_seqs, kv_capacity)`.

## Inputs
- `requests`: list of `(arrival_step, prompt_len, output_len)` ints, `output_len` $\ge 1$. Request ids are list indices. The waiting queue is ordered by `(arrival_step, id)`.
- `max_seqs`: the most requests that may run at once. `kv_capacity`: KV slots (tokens) in the cache.
- Every request fits alone: `prompt_len + output_len <= kv_capacity`.

## Each step $t = 0, 1, 2, \dots$
1. **Admit.** While the head of the waiting queue has arrived (`arrival_step <= t`), fewer than `max_seqs` requests are running, and its **reservation** `prompt_len + output_len` fits in the free KV slots, admit it and reserve those slots. Stop at the first request that can't be admitted (strict FCFS, no skipping ahead).
2. **Run.** Every running request produces one token. A request admitted in this step does its prefill and produces its **first** token in this same step.
3. **Retire.** A request that has now produced `output_len` tokens finishes at step $t$; its slots are freed after the step (usable from step $t + 1$).

## Output
A tuple `(schedule, avg_batch)`:
- `schedule`: list of `[first_token_step, finish_step]` per request id.
- `avg_batch`: the total number of tokens produced divided by the number of steps from 0 to the last finish step inclusive (idle steps count), as a float.
"""

STARTER = '''def continuous_batching(requests, max_seqs, kv_capacity):
    """Return (schedule: [[first_token_step, finish_step], ...], avg_batch: float)."""
    # static batching: requests run one after another
    t, out = 0, []
    for arr, p, o in sorted(requests):
        t = max(t, arr)
        out.append([t, t + o - 1])
        t += o
    return out, 1.0
'''

SOLUTION = '''def continuous_batching(requests, max_seqs, kv_capacity):
    waiting = sorted(range(len(requests)), key=lambda i: (requests[i][0], i))
    running = {}                                      # id -> tokens produced so far
    sched = [[None, None] for _ in requests]
    free, t, done, tokens = kv_capacity, 0, 0, 0
    while done < len(requests):
        # 1. admit in strict FCFS order: the first request that can't start blocks the rest
        while waiting:
            arr, p, o = requests[waiting[0]]
            if arr > t or len(running) >= max_seqs or p + o > free:
                break
            i = waiting.pop(0)
            free -= p + o                             # reserve the worst case, so no preemption is needed
            running[i] = 0
            sched[i][0] = t                           # prefill + first token in the admission step

        # 2. one token for every running request, 3. retire finished ones after the step
        for i in list(running):
            running[i] += 1
            tokens += 1
            if running[i] == requests[i][2]:
                sched[i][1] = t
                free += requests[i][1] + requests[i][2]
                del running[i]
                done += 1
        t += 1
    return sched, tokens / t
'''

HINTS = [
    'Keep a FCFS list of waiting ids, a dict of running ids with their token counts, and an integer of free KV slots. Advance one step at a time.',
    'Admission checks three things for the head of the queue only: it has arrived, there is a free sequence slot, and `prompt + output` slots are free. Break on the first failure.',
    'Free a finished request\'s slots only after every running request has produced its token for the step. When the loop ends, `t` equals the last finish step plus one.',
]

TESTS = [
    ([(0, 2, 3), (0, 2, 1)], 4, 100),
    ([(0, 4, 2), (0, 4, 2), (1, 4, 2)], 2, 100),
    ([(0, 5, 5), (0, 5, 5), (0, 1, 1)], 8, 12),
    ([(0, 1, 10), (0, 9, 1), (0, 1, 1)], 8, 11),
    ([(3, 2, 2), (0, 2, 2)], 1, 10),
    ([(0, 3, 1)], 1, 4),
    ([((i * 3) % 7, 2 + (i * 5) % 9, 1 + (i * 7) % 6) for i in range(15)], 4, 30),
    ([(i // 2, 8, 3 + (i % 5)) for i in range(12)], 3, 40),
]

EXPLANATION = r"""
## Why it works
Scheduling per iteration means a slot freed by a finished request is reused at the very next step, instead of idling until the longest request in a static batch finishes. Reserving `prompt + output` slots up front is the simple admission policy: a running request can never run out of cache, so no preemption is needed. Its cost is wasted reservation, because real outputs often stop early; paged KV caches allocate block by block instead and preempt when memory runs out.

## Cost
Each step is $O(\text{running})$ plus queue work, so the simulation is $O(\text{steps} \times \text{max\_seqs})$. `avg_batch` is the average number of sequences decoded per step: a direct proxy for throughput, since a memory-bound decode step costs about the same for 1 or 32 sequences.

## Common bugs
- Admitting past a blocked head of the queue. In test 3 the tiny third request would fit at step 0 but must wait behind the second one; skipping ahead improves utilisation but can starve large requests, so engines choose the policy deliberately.
- Freeing slots in the middle of the step, so a newly freed slot is reused in the same step it was released.
- Giving a newly admitted request its first token one step late: prefill and the first sampled token happen in the same iteration.

## Interview follow-ups
- What does this ignore that real engines handle? (Prefill costs much more than a decode token, so admitting a long prompt slows that step for everyone; chunked prefill bounds the cost.)
- How would you add preemption? (Allocate KV lazily; when the cache is full, evict the newest request's blocks and either swap them to CPU or recompute them later.)
"""
