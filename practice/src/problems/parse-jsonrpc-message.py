META = dict(title='Parse and classify a JSON-RPC message for MCP', tags=['Language Models'], diff='Easy', runtime='python')

STATEMENT = r"""
The Model Context Protocol (MCP) sends JSON-RPC 2.0 messages. Write `parse_message(raw)` for one raw JSON string. Return a dict with keys `kind`, `id`, `method` and `error` (an error code or `None`), deciding in this order:

1. `raw` is not valid JSON → `kind "invalid"`, `error -32700` (parse error).
2. The value is not an object, or `jsonrpc` is not exactly `"2.0"` → `"invalid"`, `-32600` (invalid request).
3. It has a `method`:
   - `method` must be a string and `params`, if present, an object or array; otherwise `"invalid"`, `-32600`.
   - With an `id` key it is a `"request"`, without one a `"notification"`.
   - A request's `id` must be a string or an integer (not bool, not null); otherwise `"invalid"`, `-32600`.
4. No `method` but `id` present: a `"response"` if it has exactly one of `result` and `error`, else `"invalid"` with `-32600`.
5. Anything else → `"invalid"`, `-32600`.

`id` and `method` in the output are the message's values when present, else `None`.
"""

STARTER = '''import json

def parse_message(raw):
    """Classify a JSON-RPC 2.0 message."""
    msg = json.loads(raw)
    kind = "request" if "method" in msg else "response"
    return {"kind": kind, "id": msg.get("id"), "method": msg.get("method"), "error": None}
'''

SOLUTION = '''import json

def parse_message(raw):
    def out(kind, err=None, msg=None):
        msg = msg if isinstance(msg, dict) else {}
        return {"kind": kind, "id": msg.get("id"), "method": msg.get("method"), "error": err}
    try:
        msg = json.loads(raw)
    except ValueError:
        return out("invalid", -32700)
    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
        return out("invalid", -32600, msg)
    if "method" in msg:
        if not isinstance(msg["method"], str) or ("params" in msg and not isinstance(msg["params"], (dict, list))):
            return out("invalid", -32600, msg)
        if "id" not in msg:
            return out("notification", None, msg)
        i = msg["id"]
        if isinstance(i, bool) or not isinstance(i, (str, int)):
            return out("invalid", -32600, msg)
        return out("request", None, msg)
    if "id" in msg and (("result" in msg) != ("error" in msg)):
        return out("response", None, msg)
    return out("invalid", -32600, msg)
'''

HINTS = [
    'A notification is a request without an id; the server must not reply to it.',
    'In Python, bool is a subclass of int, so check for it explicitly when validating ids.',
    'A response has exactly one of result and error; both or neither is invalid.',
]

TESTS = [
    ('{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}',),
    ('{"jsonrpc": "2.0", "method": "notifications/initialized"}',),
    ('{"jsonrpc": "2.0", "id": "a", "result": {"tools": []}}',),
    ('{"jsonrpc": "2.0", "id": 3, "result": {}, "error": {"code": 1}}',),
    ('{"jsonrpc": "1.0", "id": 1, "method": "x"}',),
    ('{"jsonrpc": "2.0", "id": true, "method": "x"}',),
    ('{"jsonrpc": "2.0", "id": 2, "method": "x", "params": "bad"}',),
    ('{not json',),
    ('[1, 2]',),
    ('{"jsonrpc": "2.0", "id": null, "method": "ping"}',),
]

EXPLANATION = r"""
## What MCP is
MCP standardises how an AI application talks to tool and data servers. Under the hood it is JSON-RPC 2.0: requests with a method, an id and parameters; responses that echo the id with a result or an error; and notifications that need no reply. Getting this layer exactly right is what lets any client talk to any server.

## Why classify strictly
A server that guesses (treating a malformed message as a request, or replying to a notification) breaks clients in subtle ways: duplicated responses, ids that never resolve, hung calls. The standard error codes (-32700 parse error, -32600 invalid request, -32601 method not found, -32602 invalid params, -32603 internal error) tell the client exactly what went wrong.

## Interview follow-ups
- Why must ids be unique per session among in-flight requests, and what breaks if they are not?
"""
