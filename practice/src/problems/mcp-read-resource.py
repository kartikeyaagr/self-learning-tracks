import re

META = dict(title='List and read MCP resources, including URI templates', tags=['Language Models'], diff='Medium', runtime='python')

STATEMENT = r"""
Write `handle_resources(request, resources, templates)` for the `resources/list` and `resources/read` methods.

- `resources` is a list of `{"uri", "name", "mimeType", "text"}`.
- `templates` is a list of `{"uriTemplate", "mimeType", "data"}`, where the template contains exactly one `{var}` placeholder (for example `notes://{id}`) and `data` maps a variable value to text.

Methods:

- `resources/list` → result `{"resources": [{"uri", "name", "mimeType"}, ...]}` sorted by uri (templates are not listed).
- `resources/read` with `params.uri` (a string, else error `-32602` `"Invalid params"`):
  1. An exact match in `resources` wins.
  2. Otherwise try templates in order: the placeholder matches one or more characters other than `/`; the rest of the template must match literally. If a template matches and its `data` has the value, use it.
  3. Found → result `{"contents": [{"uri": uri, "mimeType": ..., "text": ...}]}`.
  4. Not found → error `{"code": -32002, "message": "Resource not found", "data": {"uri": uri}}`.
- Any other method → error `-32601` `"Method not found"`.

Return the full JSON-RPC response dict with the request's id.
"""

STARTER = '''def handle_resources(request, resources, templates):
    """Handle resources/list and resources/read."""
    uri = request["params"]["uri"]
    r = next(r for r in resources if r["uri"] == uri)
    return {"jsonrpc": "2.0", "id": request["id"], "result": {"contents": [{"uri": uri, "text": r["text"]}]}}
'''

SOLUTION = '''import re

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
'''

HINTS = [
    'Turn the template into a regex: escape the literal parts and replace the placeholder with ([^/]+).',
    'Use re.fullmatch so a template does not match a longer URI by accident.',
    'Check exact resources before templates, so a static resource can override a template.',
]

_RES = [{"uri": "file:///readme.md", "name": "README", "mimeType": "text/markdown", "text": "# Project"},
        {"uri": "config://app", "name": "App config", "mimeType": "application/json", "text": '{"debug": false}'}]
_TPL = [{"uriTemplate": "notes://{id}", "mimeType": "text/plain", "data": {"1": "Buy milk", "2": "Call Sam"}},
        {"uriTemplate": "users://{name}/profile", "mimeType": "application/json", "data": {"ada": '{"role": "admin"}'}}]
_rq = lambda i, m, p=None: {"jsonrpc": "2.0", "id": i, "method": m, **({"params": p} if p is not None else {})}
TESTS = [
    (_rq(1, "resources/list"), _RES, _TPL),
    (_rq(2, "resources/read", {"uri": "config://app"}), _RES, _TPL),
    (_rq(3, "resources/read", {"uri": "notes://2"}), _RES, _TPL),
    (_rq(4, "resources/read", {"uri": "notes://9"}), _RES, _TPL),
    (_rq(5, "resources/read", {"uri": "users://ada/profile"}), _RES, _TPL),
    (_rq(6, "resources/read", {"uri": "users://ada/x/profile"}), _RES, _TPL),
    (_rq(7, "resources/read", {}), _RES, _TPL),
    (_rq(8, "resources/subscribe", {"uri": "notes://1"}), _RES, _TPL),
]

EXPLANATION = r"""
## Resources versus tools
Tools are actions the model chooses to call. Resources are data the application can read and put into context: files, records, configuration. The split lets a client show users what data a server exposes and decide what to attach, instead of letting the model fetch anything.

## Templates
A server cannot list every row of a database. URI templates describe families of resources (`notes://{id}`) so clients can construct URIs for items they learn about elsewhere. Matching must be strict, or `users://ada/x/profile` would resolve to the wrong record.

## Errors
Resource-not-found uses its own error code with the URI in `data`, so clients can tell a missing item from a malformed request.

## Interview follow-ups
- How would you enforce per-user access to resources when one server serves many users?
"""
