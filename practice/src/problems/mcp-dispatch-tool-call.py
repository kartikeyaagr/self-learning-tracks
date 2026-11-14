META = dict(title='Handle an MCP tools/call request', tags=['Language Models'], diff='Medium', runtime='python',
            related=['validate-tool-arguments', 'mcp-tool-descriptor'])

STATEMENT = r"""
Write `handle_tools_call(request, tools)`. `request` is a parsed JSON-RPC request with `params = {"name": ..., "arguments": {...}}`. `tools` maps a tool name to `{"schema": <inputSchema>, "impl": <spec>}`.

MCP separates **protocol errors** (a JSON-RPC `error`) from **tool errors** (a normal `result` with `isError: true`, so the model can read and react to them):

1. `params` missing, not an object, or without a string `name` → error `-32602`, message `"Invalid params"`.
2. Unknown tool → error `-32602`, message `"Unknown tool: <name>"`.
3. `arguments` (default `{}`) fails validation with the same rules as the validation step → tool error with text `"Invalid arguments: "` + the sorted errors joined by `"; "`.
4. Run the impl:
   - `{"kind": "echo"}` → text is `arguments["text"]`.
   - `{"kind": "sum"}` → text is `format(sum(arguments["numbers"]), "g")`.
   - `{"kind": "lookup", "key": k, "table": {...}}` → text is `str(table[v])` for `v = arguments[k]`, or a tool error with text `"Not found: <v>"`.

A result is `{"content": [{"type": "text", "text": ...}], "isError": bool}`. Return `{"jsonrpc": "2.0", "id": ..., "result": ...}` or `{"jsonrpc": "2.0", "id": ..., "error": {"code": ..., "message": ...}}`.
"""

STARTER = '''def handle_tools_call(request, tools):
    """Return the JSON-RPC response for a tools/call request."""
    name = request["params"]["name"]
    text = str(request["params"].get("arguments"))
    return {"jsonrpc": "2.0", "id": request["id"], "result": {"content": [{"type": "text", "text": text}], "isError": False}}
'''

SOLUTION = '''PY = {"string": str, "boolean": bool, "array": list, "object": dict}

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
'''

HINTS = [
    'Decide first whether the problem is with the protocol (bad params, unknown tool) or with the tool call itself.',
    'Reuse your argument validator from the previous step unchanged.',
    'A lookup miss is a tool error, not a protocol error: the request was well formed.',
]

_obj = lambda props, req: {"type": "object", "properties": props, "required": req, "additionalProperties": False}
_TOOLS = {"echo": {"schema": _obj({"text": {"type": "string"}}, ["text"]), "impl": {"kind": "echo"}},
          "sum": {"schema": _obj({"numbers": {"type": "array"}}, ["numbers"]), "impl": {"kind": "sum"}},
          "capital": {"schema": _obj({"country": {"type": "string"}}, ["country"]),
                      "impl": {"kind": "lookup", "key": "country", "table": {"France": "Paris", "Japan": "Tokyo"}}}}
_rq = lambda i, params: {"jsonrpc": "2.0", "id": i, "method": "tools/call", "params": params}
TESTS = [
    (_rq(1, {"name": "echo", "arguments": {"text": "hi"}}), _TOOLS),
    (_rq(2, {"name": "sum", "arguments": {"numbers": [1, 2.5, 3]}}), _TOOLS),
    (_rq(3, {"name": "capital", "arguments": {"country": "Peru"}}), _TOOLS),
    (_rq(4, {"name": "capital", "arguments": {"country": 7, "extra": True}}), _TOOLS),
    (_rq(5, {"name": "weather", "arguments": {}}), _TOOLS),
    (_rq("x", {"arguments": {}}), _TOOLS),
    ({"jsonrpc": "2.0", "id": 7, "method": "tools/call"}, _TOOLS),
    (_rq(8, {"name": "echo"}), _TOOLS),
]

EXPLANATION = r"""
## Two kinds of failure
A protocol error means the client sent something the server cannot process; the client application handles it, and the model may never see it. A tool error means the call was understood but did not succeed; it is returned as content with `isError: true`, which the client passes to the model so it can try again. Mixing the two hides fixable mistakes from the model or crashes clients on routine failures.

## Content blocks
Tool results are a list of content blocks rather than a bare string, so a tool can return text, images or embedded resources. Text blocks are by far the most common.

## Interview follow-ups
- How should a long-running tool report progress without blocking other requests?
"""
