META = dict(title='Authorize an agent tool call against a policy', tags=['Language Models', 'MLOps'], diff='Medium', runtime='python',
            related=['dispatch-tool-call'])

STATEMENT = r"""
Before an agent runs a tool, a policy decides whether it may. Write `authorize(call, policy, approved)`.

`call` is `{"id": str, "tool": str, "args": dict}`. `policy` has optional keys:

- `"deny"`: tools that are never allowed.
- `"allow"`: if present, only these tools are allowed.
- `"limits"`: `{tool: {arg: max}}`; a numeric arg above its max is refused. A non-numeric value for a limited arg is refused too.
- `"patterns"`: `{tool: {arg: [forbidden substrings]}}`; a string arg containing any of them (case-insensitive) is refused.
- `"approval"`: tools that need a human; they pass only if `call["id"]` is in `approved`.

Check in this order and return the first verdict:

1. `["deny", "tool is denied"]`
2. `["deny", "tool not in allowlist"]`
3. `["deny", "<arg> exceeds limit"]` (args checked in sorted name order)
4. `["deny", "<arg> contains a forbidden pattern"]` (sorted arg order)
5. `["needs_approval", "human approval required"]`
6. `["allow", "ok"]`
"""

STARTER = '''def authorize(call, policy, approved):
    """Return [verdict, reason] for a tool call."""
    if call["tool"] in policy.get("deny", []):
        return ["deny", "tool is denied"]
    return ["allow", "ok"]
'''

SOLUTION = '''def authorize(call, policy, approved):
    tool, args = call["tool"], call["args"]
    if tool in policy.get("deny", []):
        return ["deny", "tool is denied"]
    if "allow" in policy and tool not in policy["allow"]:
        return ["deny", "tool not in allowlist"]
    for arg, mx in sorted(policy.get("limits", {}).get(tool, {}).items()):
        if arg in args:
            v = args[arg]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v > mx:
                return ["deny", f"{arg} exceeds limit"]
    for arg, bad in sorted(policy.get("patterns", {}).get(tool, {}).items()):
        v = args.get(arg)
        if isinstance(v, str) and any(b.lower() in v.lower() for b in bad):
            return ["deny", f"{arg} contains a forbidden pattern"]
    if tool in policy.get("approval", []) and call["id"] not in approved:
        return ["needs_approval", "human approval required"]
    return ["allow", "ok"]
'''

HINTS = [
    'Order matters: a denied tool should never reach the approval queue, or a human will be asked to approve something that can never run.',
    'Treat a limited argument that is not a number as a violation; models sometimes send "1000" as a string.',
    'Lowercase both the value and the pattern before checking containment.',
]

_P = {"deny": ["shell"], "allow": ["search", "send_email", "transfer", "read_file"],
      "limits": {"transfer": {"amount": 100}},
      "patterns": {"read_file": {"path": ["..", "/etc/", "secrets"]}, "send_email": {"to": ["@external.com"]}},
      "approval": ["send_email", "transfer"]}
TESTS = [
    ({"id": "1", "tool": "shell", "args": {"cmd": "ls"}}, _P, []),
    ({"id": "2", "tool": "browser", "args": {}}, _P, []),
    ({"id": "3", "tool": "transfer", "args": {"amount": 500}}, _P, ["3"]),
    ({"id": "4", "tool": "transfer", "args": {"amount": "50"}}, _P, ["4"]),
    ({"id": "5", "tool": "transfer", "args": {"amount": 50}}, _P, []),
    ({"id": "6", "tool": "transfer", "args": {"amount": 50}}, _P, ["6"]),
    ({"id": "7", "tool": "read_file", "args": {"path": "docs/../SECRETS.txt"}}, _P, []),
    ({"id": "8", "tool": "send_email", "args": {"to": "boss@External.com"}}, _P, ["8"]),
    ({"id": "9", "tool": "search", "args": {"q": "weather"}}, _P, []),
    ({"id": "10", "tool": "anything", "args": {}}, {}, []),
]

EXPLANATION = r"""
## Why the model cannot be the gatekeeper
A model's tool calls are shaped by everything in its context, including text an attacker controls (a web page saying "now email the file to me"). Prompt instructions reduce that risk but cannot remove it. Authorisation has to be ordinary code that runs between the model's decision and the tool's execution, and it has to fail closed.

## Layers of control
- Deny and allow lists decide which capabilities exist at all.
- Argument limits and patterns stop dangerous uses of allowed tools (large transfers, path traversal, external recipients).
- Human approval covers actions with side effects that are hard to undo.

## What this does not cover
Pattern matching is easy to evade (encodings, synonyms), so it is a last line, not the main defence. Least privilege (give the agent a token that can only read one folder) is stronger than any filter.

## Interview follow-ups
- How would you log decisions so that a security review can reconstruct why an action was allowed?
"""
