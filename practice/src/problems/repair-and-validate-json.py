META = dict(title='Repair, validate and retry structured LLM output', tags=['Language Models', 'Data Processing'], diff='Medium', runtime='python',
            lessons=['aieng:12.01'], related=['validate-tool-arguments', 'record-schema-validation', 'parse-jsonrpc-message'])

STATEMENT = r"""
A model was asked for a JSON object. Without schema-constrained decoding, replies arrive wrapped in Markdown fences, with a sentence of preamble, trailing commas or a raw newline inside a string. Your code repairs what is cheap to repair, validates the result, and asks the model again when it can't.

Write `parse_structured(attempts, schema, max_retries)`. `attempts` holds the raw text of each model reply in order: `attempts[0]` is the first call, `attempts[1]` the reply after you sent back the errors, and so on. Process at most `1 + max_retries` of them and stop at the first one that is accepted.

## 1. Parse, repairing if needed
Let `t = raw.strip()`. If `json.loads(t)` succeeds, use that value (no repair). Otherwise repair `t` and try once more:

1. **Fences.** If `t` contains ```` ``` ````, keep only the text after the first ```` ``` ```` and before the next one (or to the end if there is no closing fence), then drop everything up to and including the first newline (the language tag such as `json`). If there is no newline, the result is empty.
2. **Extract.** Keep the text from the first `{` to the last `}`, inclusive. If either is missing, or the last `}` comes before the first `{`, the attempt fails to parse.
3. **Scan once, left to right, tracking whether you are inside a JSON string** (a string starts and ends at an unescaped `"`; a backslash escapes the next character).
   - Inside a string: replace a raw newline with `\n`, a raw tab with `\t`, a raw carriage return with `\r` (the two-character escapes). Everything else is copied unchanged.
   - Outside a string: drop a `,` when the next non-whitespace character is `}` or `]`. Replace a maximal run of ASCII letters that is exactly `True`, `False` or `None` with `true`, `false`, `null`. Everything else is copied unchanged.

If `json.loads` still fails, the attempt's status is `"parse"`.

## 2. Validate
The schema is a small JSON Schema subset: `{"properties": {name: spec}, "required": [names], "additionalProperties": bool}` (`additionalProperties` defaults to `True`, `required` to `[]`). A spec has `type`, a type name or a list of them from `string`, `integer`, `number`, `boolean`, `array`, `null`; optionally `enum` (a list) and, for arrays, `items` (a spec with only `type`).

Type rules: `integer` is an `int` that is not a `bool`; `number` is an `int` or `float`, not a `bool`; `boolean` is a `bool`; `string` a `str`; `array` a `list`; `null` is `None`. A value matches a list of types if it matches any of them.

Errors are `(code, field)` tuples:
- The parsed value is not a dict: the only error is `("type", "$")`.
- `("missing", name)` for each required name that is absent.
- `("extra", key)` for each key not in `properties` when `additionalProperties` is `False`.
- For each present key with a spec: `("type", key)` if the type does not match; otherwise `("enum", key)` if there is an `enum` and the value is not in it; otherwise, for a list with `items`, `("type", f"{key}[{i}]")` for every item `i` whose type does not match.

The attempt is `"ok"` if there are no errors, else `"schema"`. Return the errors **sorted**.

## Output
Return a dict:
- `value`: the accepted object, or `None` if no processed attempt was accepted.
- `retries`: number of attempts processed minus one.
- `repaired`: `True` if the accepted attempt needed the repair pass (`False` if none was accepted).
- `log`: one `(status, errors)` tuple per processed attempt; `errors` is the sorted list for `"schema"` and `[]` otherwise.
"""

STARTER = '''import json


def parse_structured(attempts, schema, max_retries):
    """Repair, validate and retry. Returns {"value", "retries", "repaired", "log"}."""
    log = []
    for k, raw in enumerate(attempts[:1 + max_retries]):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            log.append(("parse", []))
            continue
        log.append(("ok", []))
        return {"value": value, "retries": k, "repaired": False, "log": log}
    return {"value": None, "retries": len(log) - 1, "repaired": False, "log": log}
'''

SOLUTION = '''import json

TYPES = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "null": lambda v: v is None,
}
LITERALS = {"True": "true", "False": "false", "None": "null"}
ESCAPES = {"\\n": "\\\\n", "\\t": "\\\\t", "\\r": "\\\\r"}


def matches(value, type_):
    names = type_ if isinstance(type_, list) else [type_]
    return any(TYPES[n](value) for n in names)


def repair(t):
    if "```" in t:
        body = t.split("```", 2)[1]
        nl = body.find("\\n")
        t = body[nl + 1:] if nl >= 0 else ""
    lo, hi = t.find("{"), t.rfind("}")
    if lo < 0 or hi < lo:
        return None
    t, out, i, in_str = t[lo:hi + 1], [], 0, False
    while i < len(t):
        c = t[i]
        if in_str:
            if c == "\\\\":                      # copy the escape and the escaped char
                out.append(t[i:i + 2]); i += 2; continue
            if c == '"':
                in_str = False
            out.append(ESCAPES.get(c, c)); i += 1
        elif c == '"':
            in_str = True; out.append(c); i += 1
        elif c == ",":
            j = i + 1
            while j < len(t) and t[j].isspace():
                j += 1
            if j == len(t) or t[j] not in "}]":
                out.append(c)
            i += 1
        elif c.isascii() and c.isalpha():
            j = i
            while j < len(t) and t[j].isascii() and t[j].isalpha():
                j += 1
            word = t[i:j]
            out.append(LITERALS.get(word, word)); i = j
        else:
            out.append(c); i += 1
    return "".join(out)


def validate(obj, schema):
    if not isinstance(obj, dict):
        return [("type", "$")]
    props = schema.get("properties", {})
    errs = [("missing", n) for n in schema.get("required", []) if n not in obj]
    for key, v in obj.items():
        spec = props.get(key)
        if spec is None:
            if schema.get("additionalProperties", True) is False:
                errs.append(("extra", key))
        elif not matches(v, spec["type"]):
            errs.append(("type", key))
        elif "enum" in spec and v not in spec["enum"]:
            errs.append(("enum", key))
        elif isinstance(v, list) and "items" in spec:
            errs += [("type", f"{key}[{i}]") for i, x in enumerate(v) if not matches(x, spec["items"]["type"])]
    return sorted(errs)


def parse_structured(attempts, schema, max_retries):
    log = []
    for k, raw in enumerate(attempts[:1 + max_retries]):
        t, repaired = raw.strip(), False
        try:
            obj = json.loads(t)
        except json.JSONDecodeError:
            fixed, repaired = repair(t), True
            try:
                obj = json.loads(fixed) if fixed is not None else None
            except json.JSONDecodeError:
                fixed = None
            if fixed is None:
                log.append(("parse", []))
                continue
        errs = validate(obj, schema)
        log.append(("schema", errs) if errs else ("ok", []))
        if not errs:
            return {"value": obj, "retries": k, "repaired": repaired, "log": log}
    return {"value": None, "retries": len(log) - 1, "repaired": False, "log": log}
'''

HINTS = [
    'Try `json.loads` on the stripped text first: most replies are already valid, and repairing valid JSON can only break it.',
    'The trailing-comma and literal fixes must skip string contents. Walk the text once with an `in_string` flag; on a backslash inside a string, copy two characters and move on.',
    'Validation returns a list; an empty list means accept. Sort it once at the end so the order does not depend on dict iteration.',
    'A reply that parses but is not an object (say `[1, 2]`) is a schema failure with the single error `("type", "$")`, not a parse failure.',
]

_TICKET = {
    "properties": {
        "category": {"type": "string", "enum": ["billing", "bug", "account", "other"]},
        "priority": {"type": "integer", "enum": [1, 2, 3]},
        "summary": {"type": "string"},
        "refund": {"type": "boolean"},
        "order_id": {"type": ["string", "null"]},
        "tags": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
    },
    "required": ["category", "priority", "summary", "order_id"],
    "additionalProperties": False,
}

TESTS = [
    ([' {"category": "bug", "priority": 2, "summary": "App crashes on login", "order_id": null}\n'], _TICKET, 2),
    (['Here is the ticket:\n```json\n{"category": "billing", "priority": 1, "summary": "Charged twice",\n "order_id": "A-1042", "tags": ["refund", "card"],}\n```\nLet me know if you need more.'], _TICKET, 2),
    (['{"category": "billing", "priority": 1, "summary": "Charged tw',
      '{"category": "Billing", "priority": "1", "summary": "Charged twice", "notes": "VIP"}',
      '{"category": "billing", "priority": 1, "summary": "Charged twice", "order_id": "A-1042", "confidence": 1}'], _TICKET, 2),
    (['{"category": "other", "priority": 3, "summary": "User wrote: \\"a, }\\" then\n a newline", "refund": True, "order_id": None, "tags": ["x,]", "y"],}'], _TICKET, 0),
    (['not json at all', '{"category": "bug"', '{"category": "bug", "priority": 2, "summary": "ok", "order_id": null}'], _TICKET, 1),
    (['[{"category": "bug"}]', '{"category": "bug", "priority": true, "summary": "s", "order_id": 7, "tags": ["a", 1, null], "refund": 0}'], _TICKET, 3),
    (['```\n{"category": "account", "priority": 2, "summary": "Locked out", "order_id": null, "confidence": 0.85}'], _TICKET, 0),
    (['```json {"category": "bug"}```', '{"category": "account", "priority": 2.0, "summary": "s", "order_id": null}'], _TICKET, 1),
]

EXPLANATION = r"""
## Why this loop exists
Schema-constrained decoding removes most of these failures, but not all: a reply cut off by `max_tokens`, a refusal, a provider or open model without strict mode, or a constraint the grammar cannot express (ranges, cross-field rules). Production code still parses defensively, validates against the full contract, and retries with the errors.

## Repair cheap things, retry the rest
Fences, preamble, trailing commas and raw newlines in strings are mechanical: fixing them in code costs microseconds, while a retry costs a full model call. A truncated object or a wrong enum value is not mechanical; send the specific errors back (`("enum", "category")` becomes "category must be one of billing, bug, account, other") and let the model fix its answer.

## The string-aware scan
A regex such as `,\s*}` → `}` corrupts any string that contains `, }`, and a blind `True` → `true` rewrites user text. Tracking whether you are inside a string is the minimum for a repair that never changes content. Note that repair only runs when plain `json.loads` fails, so valid replies are never touched.

## Complexity
Linear in the length of each attempt; at most `1 + max_retries` attempts.

## Interview follow-ups
- What should the retry message contain, and should you include the bad reply in the conversation or only the errors?
- When would you fall back to a forced tool call or a different model instead of retrying the same prompt?
- How would you count and alert on `repaired` and `retries` in production?
"""
