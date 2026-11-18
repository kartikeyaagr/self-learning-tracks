META = dict(title='Block tool calls tainted by untrusted content', tags=['Language Models'], diff='Medium', runtime='python',
            lessons=['aieng:12.03'], related=['validate-tool-arguments', 'mcp-dispatch-tool-call', 'run-react-loop'])

STATEMENT = r"""
Prompt injection turns into damage when text an attacker controls decides the arguments of a dangerous tool call: the recipient of an email, the URL of a fetch, the repository a pull request goes to. A policy layer outside the model can stop that by tracking where every value came from (its *taint*) and refusing guarded arguments that depend on untrusted content.

Write `check_plan(tools, steps, mode, approved)` that decides, step by step, which tool calls may run.

## Inputs
- `tools` maps a tool name to `{"output": "trusted" | "untrusted", "guarded": [argument names]}`. An `"untrusted"` tool returns content an outsider can write (web pages, inbound email, public issues). A guarded argument must not depend on untrusted content.
- `steps` is the agent's sequence of calls, each `{"id": str, "tool": str, "args": {name: value}}`. A value that is a dict with the single key `"from"` (a list of earlier step ids) means "computed from those steps' outputs"; any other value is a literal.
- `mode` says where literals come from:
  - `"dataflow"`: a planner wrote the literals from the user's request before reading any tool output (the CaMeL / plan-then-execute setting). A literal is never tainted. A `"from"` value is tainted if any step it lists has tainted output.
  - `"context"`: a plain agent loop, where the model writes every argument after reading everything in its context. Once any executed step has produced tainted output, **every** argument of every later step is tainted, literals included. Before that, the dataflow rule applies.
- `approved`: a list of step ids a human confirmed after seeing the exact arguments.

## Rules, in step order
1. If a `"from"` value lists a step that did not run (blocked or skipped), the decision is `"skip"`; the step does not run.
2. Otherwise, if any guarded argument of the step's tool is tainted: `"approved"` if the step id is in `approved` (it runs), else `"block"` (it does not run).
3. Otherwise `"allow"` (it runs).

A step that runs has tainted output if its tool's output is `"untrusted"` or any of its arguments is tainted. Steps that do not run produce no output and do not taint the context.

Return `{"decisions": [(id, decision), ...] in step order, "tainted": sorted ids of steps that ran with tainted output}`.
"""

STARTER = '''def check_plan(tools, steps, mode, approved):
    """Decide allow / block / approved / skip for each step."""
    untrusted, decisions = set(), []
    for s in steps:
        tool = tools[s["tool"]]
        bad = any(isinstance(v, dict) and set(v["from"]) & untrusted
                  for a, v in s["args"].items() if a in tool["guarded"])
        decisions.append((s["id"], "block" if bad else "allow"))
        if tool["output"] == "untrusted":
            untrusted.add(s["id"])
    return {"decisions": decisions, "tainted": sorted(untrusted)}
'''

SOLUTION = '''def check_plan(tools, steps, mode, approved):
    ran, tainted, decisions = set(), set(), []
    context_dirty = False
    for s in steps:
        tool, args = tools[s["tool"]], s["args"]
        refs = {a: v["from"] for a, v in args.items() if isinstance(v, dict) and set(v) == {"from"}}
        if any(r not in ran for ids in refs.values() for r in ids):
            decisions.append((s["id"], "skip"))
            continue
        def is_tainted(a):
            if mode == "context" and context_dirty:
                return True
            return a in refs and any(r in tainted for r in refs[a])
        arg_taint = {a: is_tainted(a) for a in args}
        if any(arg_taint.get(a, False) for a in tool["guarded"]):
            if s["id"] not in approved:
                decisions.append((s["id"], "block"))
                continue
            decisions.append((s["id"], "approved"))
        else:
            decisions.append((s["id"], "allow"))
        ran.add(s["id"])
        if tool["output"] == "untrusted" or any(arg_taint.values()):
            tainted.add(s["id"])
            context_dirty = True
    return {"decisions": decisions, "tainted": sorted(tainted)}
'''

HINTS = [
    'Keep two sets as you go: steps that ran, and steps whose output is tainted. Everything else follows from them.',
    'Taint is transitive: `summarize(read_inbox())` is tainted even though `summarize` itself is a trusted tool.',
    'In `"context"` mode a single tainted output flips a flag that taints every later argument, even a literal the model typed. That is why plain agent loops need so many confirmations.',
    'A step that is blocked or skipped never ran, so it neither taints the context nor can be referenced later; anything that depends on it is skipped.',
]

_TOOLS = {
    "read_inbox": {"output": "untrusted", "guarded": []},
    "get_contact": {"output": "trusted", "guarded": []},
    "read_calendar": {"output": "trusted", "guarded": []},
    "summarize": {"output": "trusted", "guarded": []},
    "send_email": {"output": "trusted", "guarded": ["to"]},
    "fetch_url": {"output": "untrusted", "guarded": ["url"]},
    "create_pr": {"output": "trusted", "guarded": ["repo", "body"]},
}

_EMAIL = [
    {"id": "s1", "tool": "read_inbox", "args": {"folder": "inbox", "limit": 20}},
    {"id": "s2", "tool": "get_contact", "args": {"name": "Bob"}},
    {"id": "s3", "tool": "summarize", "args": {"text": {"from": ["s1"]}}},
    {"id": "s4", "tool": "send_email", "args": {"to": {"from": ["s2"]}, "body": {"from": ["s3"]}}},
    {"id": "s5", "tool": "send_email", "args": {"to": {"from": ["s1"]}, "body": {"from": ["s2"]}}},
    {"id": "s6", "tool": "fetch_url", "args": {"url": {"from": ["s5"]}}},
]

_TRUSTED_FIRST = [
    {"id": "c1", "tool": "read_calendar", "args": {"day": "today"}},
    {"id": "c2", "tool": "get_contact", "args": {"name": "Ana"}},
    {"id": "c3", "tool": "send_email", "args": {"to": {"from": ["c2"]}, "body": {"from": ["c1"]}}},
    {"id": "c4", "tool": "fetch_url", "args": {"url": "https://status.example.com"}},
    {"id": "c5", "tool": "send_email", "args": {"to": "ana@example.com", "body": "status checked"}},
    {"id": "c6", "tool": "fetch_url", "args": {"url": "https://docs.example.com"}},
]

_REPO = [
    {"id": "r1", "tool": "fetch_url", "args": {"url": "https://github.com/acme/public/issues/7"}},
    {"id": "r2", "tool": "summarize", "args": {"text": {"from": ["r1"]}, "style": "short"}},
    {"id": "r3", "tool": "create_pr", "args": {"repo": "acme/private", "body": {"from": ["r2"]}}},
    {"id": "r4", "tool": "create_pr", "args": {"repo": "acme/private", "body": "fix typo"}},
    {"id": "r5", "tool": "summarize", "args": {"text": {"from": ["r3", "r4"]}}},
    {"id": "r6", "tool": "send_email", "args": {"to": "me@example.com", "body": {"from": ["r5"]}}},
]

TESTS = [
    (_TOOLS, _EMAIL, "dataflow", []),
    (_TOOLS, _EMAIL, "context", []),
    (_TOOLS, _EMAIL, "context", ["s4", "s5"]),
    (_TOOLS, _TRUSTED_FIRST, "context", []),
    (_TOOLS, _TRUSTED_FIRST, "dataflow", []),
    (_TOOLS, _REPO, "dataflow", []),
    (_TOOLS, _REPO, "context", ["r4"]),
    (_TOOLS, _REPO, "dataflow", ["r3"]),
]

EXPLANATION = r"""
## What this models
This is the core of capability-style defences such as CaMeL: every value carries its provenance, and a policy checked **outside the model** decides whether a tool call may use it. The model can be fully fooled by an injected instruction and the exfiltrating call still does not run, because the recipient or URL it needs came from untrusted content.

## Dataflow versus context taint
In a plain agent loop the model writes each argument after reading everything in its context, so once one web page or email has been read, nothing it writes afterwards can be trusted, literals included. That is the `"context"` mode: safe, but it blocks nearly every later guarded call, which in practice means a confirmation prompt per action. Separating a planner that never reads untrusted text from the code that executes the plan (the `"dataflow"` mode) keeps literals clean, so only values that really came from untrusted content are blocked.

## Complexity
One pass over the steps; each step checks its references against sets, so the total is linear in the number of arguments and references.

## Interview follow-ups
- Which arguments of `send_email` would you guard, and why is the body less dangerous than the recipient when the recipient is trusted?
- How do you stop the planner from needing untrusted data to decide *which* tool to call next?
- How would you keep approval prompts rare enough that people still read them?
"""
