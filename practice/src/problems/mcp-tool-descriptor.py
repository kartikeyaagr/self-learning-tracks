import re

META = dict(title='Build an MCP tool descriptor with a JSON Schema', tags=['Language Models'], diff='Easy', runtime='python')

STATEMENT = r"""
An MCP server advertises each tool with a name, a description and an `inputSchema` (JSON Schema). Write `tool_descriptor(name, description, params)` where `params` is a list of dicts with keys `name`, `type`, `description`, `required` (bool) and optionally `enum`, `minimum`, `maximum`.

- If `name` does not match `^[A-Za-z0-9_-]{1,64}$`, return `{"error": "invalid tool name"}`.
- If two params share a name, return `{"error": "duplicate parameter '<name>'"}` for the first repeated one.
- If a type is not one of `string`, `integer`, `number`, `boolean`, `array`, `object`, return `{"error": "unsupported type '<type>'"}`.
- Otherwise return:

```
{"name": name, "description": description,
 "inputSchema": {"type": "object", "properties": {...}, "required": [...], "additionalProperties": false}}
```

Each property is `{"type": ..., "description": ...}` plus `enum`, `minimum` and `maximum` when given. `required` lists required param names in input order.
"""

STARTER = '''def tool_descriptor(name, description, params):
    """Return an MCP tool descriptor or {"error": ...}."""
    props = {p["name"]: {"type": p["type"]} for p in params}
    return {"name": name, "description": description, "inputSchema": {"type": "object", "properties": props}}
'''

SOLUTION = '''import re

TYPES = {"string", "integer", "number", "boolean", "array", "object"}

def tool_descriptor(name, description, params):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
        return {"error": "invalid tool name"}
    props, req, seen = {}, [], set()
    for p in params:
        if p["name"] in seen:
            return {"error": f"duplicate parameter '{p['name']}'"}
        seen.add(p["name"])
        if p["type"] not in TYPES:
            return {"error": f"unsupported type '{p['type']}'"}
        prop = {"type": p["type"], "description": p["description"]}
        for k in ("enum", "minimum", "maximum"):
            if k in p:
                prop[k] = p[k]
        props[p["name"]] = prop
        if p.get("required"):
            req.append(p["name"])
    return {"name": name, "description": description,
            "inputSchema": {"type": "object", "properties": props, "required": req, "additionalProperties": False}}
'''

HINTS = [
    'Use re.fullmatch so the whole name must match, not a prefix.',
    'Setting additionalProperties to false lets the server reject arguments the model invented.',
]

_city = {"name": "city", "type": "string", "description": "City name", "required": True}
_units = {"name": "units", "type": "string", "description": "Units", "required": False, "enum": ["metric", "imperial"]}
_days = {"name": "days", "type": "integer", "description": "Days ahead", "required": False, "minimum": 0, "maximum": 7}
TESTS = [
    ("get_forecast", "Weather forecast for a city.", [_city, _units, _days]),
    ("get forecast", "Bad name.", [_city]),
    ("x" * 65, "Too long.", []),
    ("dup", "Duplicate.", [_city, _days, _city]),
    ("bad_type", "Type.", [{"name": "when", "type": "date", "description": "d", "required": True}]),
    ("ping", "No arguments.", []),
]

EXPLANATION = r"""
## The descriptor is the model's documentation
The client passes tool descriptors to the model, so the name, description and schema are effectively prompt text. A precise schema (types, enums, ranges, required fields) prevents a large share of bad calls before they happen, and a clear description decides whether the model picks the tool at all.

## Validate at registration
Catching a duplicate parameter or a typo in a type when the server starts is far cheaper than debugging a model that keeps sending malformed calls in production.

## Interview follow-ups
- How would you version a tool's schema without breaking clients that cached the old descriptor?
"""
