META = dict(title='Critical path and cost of an LLM request trace', tags=['MLOps', 'Inference'], diff='Hard', runtime='python',
            lessons=['aieng:12.07', 'ieng:24.03'],
            related=['latency-budget-allocation', 'prompt-cache-cost', 'retry-backoff-schedule'])

STATEMENT = r"""
A trace of one agent request is a tree of spans. Each span is a dict with `id`, `parent` (`None` for the root),
`name`, `start` and `end` (ms) and `attrs`, a dict of OpenTelemetry GenAI attributes. Model-call spans have
`gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens` and, when the provider served
some input from its prompt cache, `gen_ai.usage.cache_read.input_tokens`. As in the OpenTelemetry convention,
`input_tokens` **includes** the cache-read tokens.

Write `analyse(spans, prices)` returning `{"e2e": ..., "path": [...], "cost": ...}`.

**`e2e`** is the root's `end - start`.

**`path`** is the critical path: the chain of spans that determined the end time. Build it recursively:
for a span, set `cursor` to its `end`; repeatedly pick, among its children with `end <= cursor`, the one with the
latest `end` (ties: earliest `start`, then first in input order), then set `cursor` to that child's `start`.
Stop when no child qualifies. The span's path is its own `id` followed by the paths of the picked children in
order of increasing start time. Return the root's path.

**`cost`**: for every span with `gen_ai.request.model`, with `prices[model] = {"input", "cache_read", "output"}`
in dollars per million tokens:
$$\frac{(\text{in} - \text{cached})\,p_{\text{in}} + \text{cached}\,p_{\text{cache}} + \text{out}\,p_{\text{out}}}{10^6}$$
summed over the trace.
"""

STARTER = '''def analyse(spans, prices):
    """End-to-end latency, critical path and model cost of one trace."""
    root = next(s for s in spans if s["parent"] is None)
    path = [s["id"] for s in sorted(spans, key=lambda s: s["start"])]
    cost = 0.0
    for s in spans:
        a = s["attrs"]
        if "gen_ai.request.model" in a:
            p = prices[a["gen_ai.request.model"]]
            cost += (a["gen_ai.usage.input_tokens"] * p["input"] + a["gen_ai.usage.output_tokens"] * p["output"]) / 1e6
    return {"e2e": root["end"] - root["start"], "path": path, "cost": cost}
'''

SOLUTION = '''def analyse(spans, prices):
    kids = {}
    for k, s in enumerate(spans):
        kids.setdefault(s["parent"], []).append((k, s))

    def crit(s):
        picked, cursor = [], s["end"]
        cands = kids.get(s["id"], [])
        while True:
            ok = [(k, c) for k, c in cands if c["end"] <= cursor and all(c is not p for p in picked)]
            if not ok:
                break
            _, c = max(ok, key=lambda kc: (kc[1]["end"], -kc[1]["start"], -kc[0]))
            picked.append(c)
            cursor = c["start"]
        path = [s["id"]]
        for c in reversed(picked):               # picked from last to first; emit in start order
            path += crit(c)
        return path

    root = kids[None][0][1]
    cost = 0.0
    for s in spans:
        a = s["attrs"]
        if "gen_ai.request.model" in a:
            p = prices[a["gen_ai.request.model"]]
            cached = a.get("gen_ai.usage.cache_read.input_tokens", 0)
            fresh = a["gen_ai.usage.input_tokens"] - cached   # input_tokens already includes cache reads
            cost += (fresh * p["input"] + cached * p["cache_read"] + a["gen_ai.usage.output_tokens"] * p["output"]) / 1e6
    return {"e2e": root["end"] - root["start"], "path": crit(root), "cost": cost}
'''

HINTS = [
    'Index children by parent id once. The critical path walks backwards from a span\'s end: the last child to finish is the one the parent was waiting for.',
    'After picking a child, only children that finished before it started can be on the path. Children that overlap it were not what the parent waited for.',
    'For cost, subtract the cache-read tokens from `input_tokens` before pricing the rest at the full input rate. Spans without a model attribute cost nothing here.',
]


def _s(i, parent, name, start, end):
    return {"id": i, "parent": parent, "name": name, "start": start, "end": end, "attrs": {}}


def _llm(i, parent, start, end, model, tin, tout, cached=0):
    a = {"gen_ai.operation.name": "chat", "gen_ai.request.model": model,
         "gen_ai.usage.input_tokens": tin, "gen_ai.usage.output_tokens": tout}
    if cached:
        a["gen_ai.usage.cache_read.input_tokens"] = cached
    return {"id": i, "parent": parent, "name": f"chat {model}", "start": start, "end": end, "attrs": a}


_PR = {"big": {"input": 3.0, "cache_read": 0.3, "output": 15.0},
       "small": {"input": 1.0, "cache_read": 0.1, "output": 5.0}}

# A RAG agent turn: guardrail and retrieval in parallel, one planning call, two tool calls in parallel, final answer.
_AGENT = [
    _s("root", None, "invoke_agent support", 0, 4200),
    _llm("guard", "root", 5, 180, "small", 900, 5),
    _s("ret", "root", "retrieve kb", 5, 420),
    _llm("plan", "root", 430, 1300, "big", 9000, 120, cached=8000),
    _s("t1", "root", "execute_tool get_order", 1310, 1650),
    _s("t2", "root", "execute_tool get_policy", 1310, 2100),
    _llm("answer", "root", 2110, 4150, "big", 10500, 380, cached=9000),
    _s("log", "root", "write audit log", 4150, 4400),
]
# Same trace but the retrieval span has an embedded call and a nested rerank.
_NESTED = _AGENT[:2] + [_s("ret", "root", "retrieve kb", 5, 420),
                         _s("emb", "ret", "embeddings", 10, 90),
                         _s("search", "ret", "vector search", 95, 210),
                         _llm("rerank", "ret", 215, 415, "small", 4000, 20)] + _AGENT[3:]
TESTS = [
    (_AGENT, _PR),
    (_NESTED, _PR),
    ([_s("r", None, "root", 0, 100), _s("a", "r", "a", 0, 60), _s("b", "r", "b", 0, 60), _s("c", "r", "c", 60, 100)], _PR),
    ([_s("r", None, "root", 0, 100), _s("a", "r", "a", 10, 60), _s("b", "r", "b", 0, 60), _s("c", "r", "c", 70, 90)], _PR),
    ([_llm("only", None, 100, 1600, "small", 2000, 300)], _PR),
    ([_s("r", None, "root", 0, 50), _s("x", "r", "x", 0, 20), _s("y", "r", "y", 30, 45), _s("z", "y", "z", 31, 44),
      _s("late", "r", "fire and forget", 40, 90)], _PR),
]

EXPLANATION = r"""
## Why the critical path, not the slowest span
The root waits for whatever finishes last; anything overlapping that span was hidden behind it. In the agent
trace, `get_order` (340 ms) runs beside `get_policy` (790 ms), so speeding up `get_order` saves nothing. The
guardrail runs beside retrieval and is off the path too. The path here is retrieval, planning call,
`get_policy`, answer call, covering all but a few milliseconds of the 4.2 s; the two model calls take 2.9 s of it. That is where to
spend effort (shorter outputs, a smaller planning model, streaming the answer).

## Spans that outlive their parent
The audit-log span in the agent trace ends after the root (fire-and-forget work). It cannot be on the path because the
parent did not wait for it. Its cost still counts.

## Token accounting
OpenTelemetry's GenAI conventions say `gen_ai.usage.input_tokens` SHOULD include cached tokens, with
`gen_ai.usage.cache_read.input_tokens` reported separately. Some provider APIs report input tokens *excluding*
cached and cache-write tokens. Mixing the two conventions double-counts or drops the cached part; normalise
at ingestion and test it with a trace like this one.

## Complexity
Each span's children are scanned once per pick, so $O(c^2)$ per span with $c$ children; fine for traces of a
few hundred spans. Sorting children by end time makes each pick a binary search.

## Common bugs
- Taking the span with the longest duration instead of walking back from the parent's end.
- Emitting children in the order they were picked (latest first) instead of start order.
- Pricing all input tokens at the full rate, or subtracting cache reads twice.

## Interview follow-ups
- How do you aggregate critical paths across thousands of traces? (Sum time per span name on the path, per
  percentile bucket: what is on the path for p99 requests is often different from p50.)
- What should a trace never contain by default? (Prompt and completion text, tool arguments with personal data;
  record ids, token counts and hashes, and keep content in an access-controlled store if you need it.)
"""
