META = dict(title='Bill a conversation with prompt-cache writes, reads and TTL expiry', tags=['Inference', 'Language Models'], diff='Medium',
            runtime='python', lessons=['aieng:12.04', 'aieng:04.04'],
            related=['prefix-cache-matching-and-reuse', 'model-cascade-routing', 'optimize-throughput-and-cost-under-latency-slo'])

STATEMENT = r"""
Hosted LLM APIs can cache the processed prefix of a prompt. A request marks a **breakpoint**: everything before it is
eligible for caching. Tokens read from the cache are billed at a discount, tokens written to it at a premium, and
an entry expires if nobody uses it for `ttl` seconds.

Write `bill(requests, prices, ttl, min_cache)` that bills a sequence of requests (sorted by `t`) twice: with
caching and without.

Each request is a dict:
- `t`: start time in seconds,
- `version`: an id for the cacheable prefix content (system prompt, tools). Requests only share cache entries
  with the same `version`; editing the system prompt changes it,
- `bp`: tokens up to and including the breakpoint (the cacheable prefix),
- `prompt`: total input tokens (`prompt >= bp`),
- `output`: output tokens.

`prices` has `input` and `output` (dollars per million tokens), `write_mult` and `read_mult`.

**With caching**, for each request:
1. The cache holds entries `(version, length, last_used)`. An entry is alive at time `t` if `t < last_used + ttl`.
2. If `bp < min_cache`, nothing is read or written: all `prompt` tokens are billed at `input`.
3. Otherwise, the hit is the longest alive entry with the same `version` and `length <= bp`
   (a conversation only grows, so any shorter entry of the same version is a prefix). Let `L` be its length,
   or 0 if there is none. Bill `L` tokens at `input * read_mult`, `bp - L` at `input * write_mult` and
   `prompt - bp` at `input`. Set the hit entry's `last_used` to `t` (reads refresh it for free), then store
   `(version, bp, t)` (if an entry of that version and length exists, just set its `last_used` to `t`).
4. Output tokens are billed at `output`.

**Without caching**, every input token is billed at `input`.

Return `{"cached": dollars, "uncached": dollars, "hits": requests with L > 0, "read": total tokens read,
"written": total tokens written}`.
"""

STARTER = '''def bill(requests, prices, ttl, min_cache):
    """Total cost with and without prompt caching."""
    cached = uncached = 0.0
    for r in requests:
        uncached += (r["prompt"] * prices["input"] + r["output"] * prices["output"]) / 1e6
        cached += (r["bp"] * prices["input"] * prices["read_mult"]
                   + (r["prompt"] - r["bp"]) * prices["input"] + r["output"] * prices["output"]) / 1e6
    return {"cached": cached, "uncached": uncached, "hits": len(requests), "read": 0, "written": 0}
'''

SOLUTION = '''def bill(requests, prices, ttl, min_cache):
    p_in, p_out = prices["input"] / 1e6, prices["output"] / 1e6
    cache = {}                                   # (version, length) -> last_used
    cached = uncached = 0.0
    hits = read = written = 0
    for r in requests:
        t, v, bp, prompt, out = r["t"], r["version"], r["bp"], r["prompt"], r["output"]
        uncached += prompt * p_in + out * p_out
        if bp < min_cache:                       # below the minimum: silently not cached
            cached += prompt * p_in + out * p_out
            continue
        alive = [n for (ver, n), last in cache.items() if ver == v and n <= bp and t < last + ttl]
        L = max(alive, default=0)
        if L:
            hits += 1
            cache[(v, L)] = t                    # a read refreshes the entry
        read += L
        written += bp - L
        cached += (L * prices["read_mult"] + (bp - L) * prices["write_mult"] + (prompt - bp)) * p_in + out * p_out
        cache[(v, bp)] = t
    return {"cached": cached, "uncached": uncached, "hits": hits, "read": read, "written": written}
'''

HINTS = [
    'Keep the cache as a dict from `(version, length)` to the time it was last used. Expired entries can stay in the dict; just ignore them when looking for a hit.',
    'The hit is `max` over alive lengths `<= bp` of the same version. Everything between the hit and the breakpoint is a write, everything after the breakpoint is plain input.',
    'Do the `min_cache` check first: a short prefix is neither read nor written, even if an entry exists. Refresh the hit entry and store the new one after billing.',
]

_P = {"input": 3.0, "output": 15.0, "write_mult": 1.25, "read_mult": 0.1}
_P1H = {"input": 3.0, "output": 15.0, "write_mult": 2.0, "read_mult": 0.1}


def _chat(times, sys_tokens=4000, turn=600, out=300, version="v1"):
    """A chat whose history grows by `turn` input tokens and `out` output tokens per turn."""
    reqs, hist = [], sys_tokens
    for t in times:
        reqs.append({"t": t, "version": version, "bp": hist + turn, "prompt": hist + turn, "output": out})
        hist += turn + out
    return reqs


_SUPPORT = [{"t": 30 * k, "version": "v1", "bp": 6000, "prompt": 6400, "output": 250} for k in range(8)]
_EDITED = _SUPPORT[:4] + [dict(r, version="v2") for r in _SUPPORT[4:]]
TESTS = [
    (_chat([0, 40, 100, 160, 230]), _P, 300, 1024),
    (_SUPPORT, _P, 300, 1024),
    (_chat([0, 200, 600, 650, 1000]), _P, 300, 1024),
    (_chat([0, 200, 600, 650, 1000]), _P1H, 3600, 1024),
    (_EDITED, _P, 300, 1024),
    ([{"t": 0, "version": "a", "bp": 900, "prompt": 1200, "output": 100},
      {"t": 10, "version": "a", "bp": 900, "prompt": 1300, "output": 100}], _P, 300, 1024),
    ([{"t": 0, "version": "a", "bp": 2000, "prompt": 2000, "output": 10},
      {"t": 300, "version": "a", "bp": 2000, "prompt": 2000, "output": 10},
      {"t": 599, "version": "a", "bp": 2000, "prompt": 2000, "output": 10}], _P, 300, 1024),
    (_chat(list(range(0, 1200, 60)), sys_tokens=12000, turn=400, out=500), _P, 300, 1024),
]

EXPLANATION = r"""
## What the bill is made of
With caching, one request's input splits into three parts priced differently:
$$\text{cost} = \frac{L \cdot r + (b - L)\cdot w + (P - b)}{10^6}\, p_{\text{in}} + \frac{O}{10^6}\, p_{\text{out}}$$
where $L$ is the cached length found, $b$ the breakpoint, $P$ the prompt, $r$ and $w$ the read and write multipliers.
With Anthropic's published multipliers ($w = 1.25$ for a 5-minute entry, $w = 2$ for a 1-hour entry, $r = 0.1$),
one write followed by one read costs $1.35\times$ the uncached price of that prefix, so caching pays from the
second use onward. A prefix written and never read costs 25% more than not caching at all.

## Why the tests look like this
- **Growing chat (test 1):** each turn reads the previous turn's entry and writes only the new tokens.
- **Support bot (test 2):** the same 6,000-token prefix every 30 s: one write, then seven reads.
- **Gaps longer than the TTL (tests 3 and 4):** the turns arrive after gaps of 200, 400, 50 and 350 s. With a
  5-minute TTL the 400 s and 350 s gaps expire the entry and the whole history is written again (2 hits). With a
  1-hour TTL every write costs $2\times$ instead of $1.25\times$, but all 4 follow-ups hit and the bill drops by a quarter.
- **Edited system prompt (test 5):** the new `version` cannot reuse anything; one full write, then reads again.
- **Below the minimum (test 6):** no error, no caching. Check the usage fields in real responses for this.
- **TTL boundary (test 7):** the request at $t = 300$ misses (the entry from $t = 0$ died at 300). The one at 599
  hits the entry written at 300.

## What real caches add
- Providers match on blocks and look back a limited number of positions from each breakpoint (20 blocks on
  Anthropic's API), so a breakpoint far past the last write can miss even though the prefix is identical.
- A cache entry becomes readable only once the first response starts. Ten parallel requests with a cold prefix
  all pay the write.
- Hits are per machine or per region: routing matters (see the prefix-cache-aware routing lesson).

## Common bugs
- Billing all of `bp` as a read on a hit instead of only the cached length $L$.
- Forgetting that reads refresh the entry, so a busy prefix never expires.
- Using `<=` for liveness: an entry used at time 0 with a 300 s TTL is gone at exactly 300.

## Interview follow-ups
- When does a 1-hour TTL beat a 5-minute one? (Each 5-minute expiry turns a $0.1\,b$ read into a $1.25\,b$
  rewrite, $1.15\,b$ extra; the 1-hour entry costs $0.75\,b$ extra per write. Avoiding one expiry per write pays for it.)
- How do you raise the hit rate in a multi-tenant app? (Stable content first, per-tenant data after the
  breakpoint, no timestamps in the system prompt, a routing key per prefix.)
"""
