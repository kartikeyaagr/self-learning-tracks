META = dict(title='Capstone: a minimal MCP server over a message stream', tags=['Language Models'], diff='Hard', runtime='python',
            related=['parse-jsonrpc-message', 'mcp-tool-descriptor', 'validate-tool-arguments', 'mcp-dispatch-tool-call', 'mcp-read-resource'])

STATEMENT = r"""
Write `mcp_server(messages, tools, descriptors, resources, templates, versions)` that processes a list of raw JSON strings in order and returns the list of responses it sends.

- Classify each message as in the parsing step. An invalid message gets an error response: code `-32700` with message `"Parse error"`, or `-32600` with `"Invalid Request"`. Its `id` is the message's id when that is a string or a non-bool integer, else `None`.
- Notifications and responses from the client get no reply.
- Requests:
  - `initialize`: reply with `{"protocolVersion": v, "capabilities": {"tools": {}, "resources": {}}, "serverInfo": {"name": "tiny-mcp", "version": "0.1.0"}}`, where `v` is the client's `params.protocolVersion` if it is in `versions`, else `versions[0]` (the server's preferred version). The server is now initialised.
  - `ping`: reply with an empty result, at any time.
  - Any other request before `initialize` (a rule for this exercise): error `-32600`, `"Server not initialized"`.
  - `tools/list`: `{"tools": descriptors sorted by name}`.
  - `tools/call`: as in the tool-call step, using `tools`.
  - `resources/list` and `resources/read`, and any other `resources/...` method: as in the resources step.
  - Anything else: `-32601`, `"Method not found"`.
"""

STARTER = '''def mcp_server(messages, tools, descriptors, resources, templates, versions):
    """Process a stream of JSON-RPC messages and return the responses."""
    return []
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

PY = {"string": str, "boolean": bool, "array": list, "object": dict}

def _type_ok(v, t):
    if t == "integer":
        return isinstance(v, int) and not isinstance(v, bool)
    if t == "number":
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    return isinstance(v, PY[t])

def validate_args(schema, args):
    if not isinstance(args, dict):
        return ["arguments must be an object"]
    props, errs = schema.get("properties", {}), []
    errs += [f"missing required '{r}'" for r in schema.get("required", []) if r not in args]
    if schema.get("additionalProperties") is False:
        errs += [f"unexpected '{k}'" for k in args if k not in props]
    for k, v in args.items():
        if k not in props:
            continue
        p = props[k]
        if "type" in p and not _type_ok(v, p["type"]):
            errs.append(f"'{k}' must be {p['type']}")
            continue
        if "enum" in p and v not in p["enum"]:
            errs.append(f"'{k}' must be one of {p['enum']}")
        if "minimum" in p and v < p["minimum"]:
            errs.append(f"'{k}' must be >= {p['minimum']}")
        if "maximum" in p and v > p["maximum"]:
            errs.append(f"'{k}' must be <= {p['maximum']}")
    return sorted(errs)

def handle_tools_call(request, tools):
    rid = request.get("id")
    def err(code, msg):
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": msg}}
    def res(text, is_error=False):
        return {"jsonrpc": "2.0", "id": rid, "result": {"content": [{"type": "text", "text": text}], "isError": is_error}}
    p = request.get("params")
    if not isinstance(p, dict) or not isinstance(p.get("name"), str):
        return err(-32602, "Invalid params")
    if p["name"] not in tools:
        return err(-32602, f"Unknown tool: {p['name']}")
    tool, args = tools[p["name"]], p.get("arguments", {})
    errors = validate_args(tool["schema"], args)
    if errors:
        return res("Invalid arguments: " + "; ".join(errors), True)
    impl = tool["impl"]
    if impl["kind"] == "echo":
        return res(args["text"])
    if impl["kind"] == "sum":
        return res(format(sum(args["numbers"]), "g"))
    v = args[impl["key"]]
    return res(str(impl["table"][v])) if v in impl["table"] else res(f"Not found: {v}", True)

import re

def handle_resources(request, resources, templates):
    rid = request.get("id")
    ok = lambda r: {"jsonrpc": "2.0", "id": rid, "result": r}
    err = lambda e: {"jsonrpc": "2.0", "id": rid, "error": e}
    m = request.get("method")
    if m == "resources/list":
        return ok({"resources": [{"uri": r["uri"], "name": r["name"], "mimeType": r["mimeType"]}
                                 for r in sorted(resources, key=lambda r: r["uri"])]})
    if m != "resources/read":
        return err({"code": -32601, "message": "Method not found"})
    p = request.get("params")
    uri = p.get("uri") if isinstance(p, dict) else None
    if not isinstance(uri, str):
        return err({"code": -32602, "message": "Invalid params"})
    for r in resources:
        if r["uri"] == uri:
            return ok({"contents": [{"uri": uri, "mimeType": r["mimeType"], "text": r["text"]}]})
    for t in templates:
        pre, rest = t["uriTemplate"].split("{", 1)
        post = rest.split("}", 1)[1]
        mt = re.fullmatch(re.escape(pre) + r"([^/]+)" + re.escape(post), uri)
        if mt and mt.group(1) in t["data"]:
            return ok({"contents": [{"uri": uri, "mimeType": t["mimeType"], "text": t["data"][mt.group(1)]}]})
    return err({"code": -32002, "message": "Resource not found", "data": {"uri": uri}})

def mcp_server(messages, tools, descriptors, resources, templates, versions):
    out, ready = [], False
    for raw in messages:
        info = parse_message(raw)
        if info["kind"] == "invalid":
            i = info["id"] if isinstance(info["id"], (str, int)) and not isinstance(info["id"], bool) else None
            msg = "Parse error" if info["error"] == -32700 else "Invalid Request"
            out.append({"jsonrpc": "2.0", "id": i, "error": {"code": info["error"], "message": msg}})
            continue
        if info["kind"] != "request":
            continue
        req = json.loads(raw)
        rid, m = req["id"], req["method"]
        ok = lambda r: {"jsonrpc": "2.0", "id": rid, "result": r}
        if m == "initialize":
            want = (req.get("params") or {}).get("protocolVersion")
            ready = True
            out.append(ok({"protocolVersion": want if want in versions else versions[0],
                           "capabilities": {"tools": {}, "resources": {}},
                           "serverInfo": {"name": "tiny-mcp", "version": "0.1.0"}}))
        elif m == "ping":
            out.append(ok({}))
        elif not ready:
            out.append({"jsonrpc": "2.0", "id": rid, "error": {"code": -32600, "message": "Server not initialized"}})
        elif m == "tools/list":
            out.append(ok({"tools": sorted(descriptors, key=lambda d: d["name"])}))
        elif m == "tools/call":
            out.append(handle_tools_call(req, tools))
        elif m.startswith("resources/"):
            out.append(handle_resources(req, resources, templates))
        else:
            out.append({"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "Method not found"}})
    return out
'''

HINTS = [
    'Parse and classify first; only requests go on to method routing.',
    'Keep one piece of session state: whether initialize has happened.',
    'Route resources/* to the resource handler so it can return its own method-not-found for unsupported resource methods.',
]

_obj = lambda props, req: {"type": "object", "properties": props, "required": req, "additionalProperties": False}
_TOOLS = {"echo": {"schema": _obj({"text": {"type": "string"}}, ["text"]), "impl": {"kind": "echo"}},
          "capital": {"schema": _obj({"country": {"type": "string"}}, ["country"]),
                      "impl": {"kind": "lookup", "key": "country", "table": {"France": "Paris"}}}}
_DESC = [{"name": "echo", "description": "Echo text.", "inputSchema": _TOOLS["echo"]["schema"]},
         {"name": "capital", "description": "Capital of a country.", "inputSchema": _TOOLS["capital"]["schema"]}]
_RES = [{"uri": "config://app", "name": "App config", "mimeType": "application/json", "text": "{}"}]
_TPL = [{"uriTemplate": "notes://{id}", "mimeType": "text/plain", "data": {"1": "Buy milk"}}]
_V = ["2025-06-18", "2025-03-26"]
_session = ['{"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}}',
            '{"jsonrpc": "2.0", "method": "notifications/initialized"}',
            '{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}',
            '{"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "capital", "arguments": {"country": "France"}}}',
            '{"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "capital", "arguments": {}}}',
            '{"jsonrpc": "2.0", "id": 4, "method": "resources/read", "params": {"uri": "notes://1"}}',
            '{"jsonrpc": "2.0", "id": 5, "method": "prompts/list"}']
_early = ['{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}', '{"jsonrpc": "2.0", "id": 2, "method": "ping"}',
          '{"jsonrpc": "2.0", "id": 3, "method": "initialize", "params": {"protocolVersion": "1999-01-01"}}',
          '{"jsonrpc": "2.0", "id": 4, "method": "resources/list"}']
_bad = ['{oops', '{"jsonrpc": "2.0", "id": 9, "method": 5}', '{"jsonrpc": "2.0", "id": true, "method": "ping"}',
        '{"jsonrpc": "2.0", "id": 1, "result": {}}']
TESTS = [
    (_session, _TOOLS, _DESC, _RES, _TPL, _V),
    (_early, _TOOLS, _DESC, _RES, _TPL, _V),
    (_bad, _TOOLS, _DESC, _RES, _TPL, _V),
    (_session[:1] + ['{"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {"name": "echo", "arguments": {"text": "hi"}}}'], _TOOLS, _DESC, _RES, _TPL, _V),
    ([], _TOOLS, _DESC, _RES, _TPL, _V),
]

EXPLANATION = r"""
## The whole server
An MCP server is a message loop: classify, route, respond, and remember a little session state. Real servers add a transport (stdio or streamable HTTP), concurrency, cancellation and progress notifications, but the core is what you built here, and the SDKs mostly hide the JSON-RPC layer behind decorators that register tools and resources.

## Lifecycle
The client starts with `initialize`, the two sides agree on a protocol version and capabilities, then the client sends an `initialized` notification. Version negotiation matters: if the server does not support the client's version it answers with one it does support, and the client decides whether to continue.

## What to test
Malformed input, calls before initialisation, unknown methods, protocol errors versus tool errors, and that notifications never get a reply. These are where interoperability bugs come from.

## Interview follow-ups
- How would you add authentication when the server runs over HTTP instead of stdio?
"""
