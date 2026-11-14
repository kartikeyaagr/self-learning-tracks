META = dict(title='Capstone: an inference request handler with stops and usage', tags=['Inference', 'Language Models'], diff='Medium', runtime='python',
            related=['decode-attention-with-kv-cache', 'dynamic-request-batching', 'ttft-tpot-itl-and-token-throughput'])

STATEMENT = r"""
Write `handle(requests, next_token, max_context, eos)` that serves a list of generation requests with a toy deterministic model: `next_token` maps the **string** of the last token id to the next id (missing keys mean `eos`).

Each request is `{"id", "prompt": [ids], "max_tokens": int, "stop": [[ids], ...]}`. For each, in order:

1. Validate: a prompt that is empty or longer than `max_context` → `{"id", "error": "prompt length must be 1..<max_context>"}`; `max_tokens < 1` → `{"id", "error": "max_tokens must be positive"}`.
2. Generate one token at a time from the last token of prompt + output so far. Stop, checking in this order after each new token:
   - the token is `eos`: drop it, `finish_reason "stop"`;
   - the output ends with any stop sequence: remove that sequence from the output, `"stop"`;
   - output length reached `max_tokens`: `"length"`;
   - prompt + output reached `max_context`: `"length"`.
3. Stream the output in chunks of at most 2 tokens, in order (`chunks` is a list of lists).

Return a list of `{"id", "tokens", "chunks", "finish_reason", "usage": {"prompt_tokens", "completion_tokens", "total_tokens"}}` or error dicts. `completion_tokens` counts every generated token including a dropped eos or stop sequence (they cost compute).
"""

STARTER = '''def handle(requests, next_token, max_context, eos):
    """Serve generation requests with a deterministic toy model."""
    out = []
    for r in requests:
        toks = [next_token.get(str(r["prompt"][-1]), eos)]
        out.append({"id": r["id"], "tokens": toks, "chunks": [toks], "finish_reason": "length",
                    "usage": {"prompt_tokens": len(r["prompt"]), "completion_tokens": 1, "total_tokens": len(r["prompt"]) + 1}})
    return out
'''

SOLUTION = '''def handle(requests, next_token, max_context, eos):
    out = []
    for r in requests:
        p = list(r["prompt"])
        if not 1 <= len(p) <= max_context:
            out.append({"id": r["id"], "error": f"prompt length must be 1..{max_context}"}); continue
        if r["max_tokens"] < 1:
            out.append({"id": r["id"], "error": "max_tokens must be positive"}); continue
        gen, produced, reason = [], 0, None
        while reason is None:
            last = (p + gen)[-1]
            t = next_token.get(str(last), eos)
            produced += 1
            if t == eos:
                reason = "stop"; break
            gen.append(t)
            hit = next((s for s in r.get("stop", []) if s and gen[-len(s):] == list(s)), None)
            if hit is not None:
                gen = gen[:-len(hit)]; reason = "stop"
            elif len(gen) >= r["max_tokens"] or len(p) + len(gen) >= max_context:
                reason = "length"
        out.append({"id": r["id"], "tokens": gen, "chunks": [gen[i:i + 2] for i in range(0, len(gen), 2)],
                    "finish_reason": reason,
                    "usage": {"prompt_tokens": len(p), "completion_tokens": produced, "total_tokens": len(p) + produced}})
    return out
'''

HINTS = [
    'Check eos before appending, and stop sequences after appending, since a stop sequence spans several tokens.',
    'Keep a separate counter of produced tokens for usage; the returned text may be shorter.',
    'The context limit counts prompt plus output, so long prompts leave little room to generate.',
]

_M = {"1": 2, "2": 3, "3": 4, "4": 5, "5": 0, "7": 8, "8": 7, "9": 9}
TESTS = [
    ([{"id": "a", "prompt": [1], "max_tokens": 10, "stop": []}], _M, 16, 0),
    ([{"id": "b", "prompt": [1], "max_tokens": 2, "stop": []}], _M, 16, 0),
    ([{"id": "c", "prompt": [1], "max_tokens": 10, "stop": [[3, 4]]}], _M, 16, 0),
    ([{"id": "d", "prompt": [7], "max_tokens": 50, "stop": []}], _M, 6, 0),
    ([{"id": "e", "prompt": [], "max_tokens": 5, "stop": []}, {"id": "f", "prompt": [2], "max_tokens": 0, "stop": []},
      {"id": "g", "prompt": [6], "max_tokens": 5, "stop": []}], _M, 8, 0),
    ([{"id": "h", "prompt": [9, 9], "max_tokens": 3, "stop": [[9, 9, 9, 9]]}], _M, 10, 0),
]

EXPLANATION = r"""
## What an inference server does around the model
Most of a serving stack is not the forward pass: request validation, a decode loop with several stopping rules, streaming partial output, and accurate accounting. The model here is a lookup table so the control flow can be tested exactly; in a real server `next_token` is a batched forward pass with a KV cache and a sampler.

## Stopping rules
End-of-sequence, user stop sequences, the token budget and the context window are separate conditions, and the finish reason tells the client which one fired. Stop sequences must be removed from the returned text but still counted, because the model computed them.

## Streaming
Clients see tokens as they are produced, which is what makes time-to-first-token the latency users feel. Chunking trades a little latency for fewer network messages.

## Interview follow-ups
- How do stop sequences interact with tokenisation when a stop string can be split across different token boundaries?
"""
