META = dict(title='Validate tool arguments against a JSON Schema subset', tags=['Language Models', 'Data Processing'], diff='Medium', runtime='python',
            related=['mcp-tool-descriptor'])

STATEMENT = r"""
Write `validate_args(schema, args)` for the object schemas produced by a tool descriptor. Return a **sorted** list of error strings (empty if valid).

- If `args` is not a dict: return `["arguments must be an object"]`.
- For each name in `required` missing from `args`: `"missing required '<name>'"`.
- If `additionalProperties` is false, for each key not in `properties`: `"unexpected '<name>'"`.
- For each present key that is in `properties`, check its `type` first:
  - `string` → `str`; `boolean` → `bool`; `integer` → `int` but not `bool`; `number` → `int` or `float` but not `bool`; `array` → `list`; `object` → `dict`.
  - A wrong type gives `"'<name>' must be <type>"` and no further checks for that key.
- Then, if present: `enum` → `"'<name>' must be one of <enum>"` using `str(list)`; `minimum` / `maximum` → `"'<name>' must be >= <min>"` / `"'<name>' must be <= <max>"`.
"""

STARTER = '''def validate_args(schema, args):
    """Return a sorted list of validation errors."""
    return sorted(f"missing required '{r}'" for r in schema.get("required", []) if r not in args)
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
'''

HINTS = [
    'Stop checking a key after a type error; comparing a string with a numeric minimum would raise.',
    'JSON has one number type, but schemas distinguish integer from number; 3.0 is not an integer here.',
    'Sorting the errors makes the output deterministic regardless of dict order.',
]

_S = {"type": "object", "properties": {"city": {"type": "string"}, "units": {"type": "string", "enum": ["metric", "imperial"]},
      "days": {"type": "integer", "minimum": 0, "maximum": 7}, "verbose": {"type": "boolean"}, "lat": {"type": "number", "minimum": -90, "maximum": 90}},
      "required": ["city"], "additionalProperties": False}
TESTS = [
    (_S, {"city": "Oslo", "days": 3}),
    (_S, {"days": 9, "units": "kelvin", "extra": 1}),
    (_S, {"city": 5, "days": "3", "verbose": 1}),
    (_S, {"city": "Oslo", "days": True, "lat": 91.5}),
    (_S, {"city": "Oslo", "lat": 45, "days": 2.0}),
    (_S, ["Oslo"]),
    ({"type": "object", "properties": {}}, {"anything": 1}),
]

EXPLANATION = r"""
## Why validate on the server
The schema is shown to the model, but the model does not have to follow it. Strings where numbers belong, invented parameters and out-of-range values are all common. The server is the only place where the schema is enforced.

## Error messages are for the model
Specific, sorted messages ("'days' must be <= 7") can be returned as a tool error so the model fixes the call on its next attempt. A full JSON Schema validator supports much more (nested objects, patterns, formats); this subset covers what most tool schemas use.

## Interview follow-ups
- Should the server coerce "3" to 3, or reject it? What are the risks of each?
"""
