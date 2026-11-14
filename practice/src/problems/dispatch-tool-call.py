META = dict(title='Dispatch a tool call and format the observation', tags=['Language Models'], diff='Easy', runtime='python',
            related=['parse-react-step'])

STATEMENT = r"""
Write `dispatch_tool(tool, args, tools)` that runs one tool call and returns the **observation string** shown to the model.

`tools` maps a name to a spec. Three kinds exist:

- `{"kind": "calculator"}`: args `a`, `op`, `b`; `op` is one of `+ - * /`. Result formatted with `format(x, "g")`.
- `{"kind": "lookup", "key": <arg name>, "table": {...}}`: returns `str(table[value])` for the arg named by `key`, or `"No result for '<value>'."`.
- `{"kind": "search", "docs": {id: text}}`: arg `query`; return the ids (sorted) of docs containing every lowercase word of the query (split on whitespace) as a substring of the lowercase text, joined by `", "`, or `"No matches."`.

Errors are observations too, checked in this order:

1. Unknown tool: `"Error: unknown tool '<name>'."`
2. A required arg is missing (calculator: a, op, b in that order; lookup: its key; search: query): `"Error: missing argument '<arg>'."`
3. Calculator: `a` or `b` not a number (bool does not count) → `"Error: a and b must be numbers."`; unknown op → `"Error: unsupported operator '<op>'."`; division by zero → `"Error: division by zero."`

Successful results are prefixed with `"Result: "`.
"""

STARTER = '''def dispatch_tool(tool, args, tools):
    """Run the tool and return the observation string."""
    spec = tools[tool]
    if spec["kind"] == "calculator":
        return "Result: " + str(eval(f"{args['a']}{args['op']}{args['b']}"))
    return "Result: "
'''

SOLUTION = '''def dispatch_tool(tool, args, tools):
    if tool not in tools:
        return f"Error: unknown tool '{tool}'."
    spec = tools[tool]
    kind = spec["kind"]
    need = {"calculator": ["a", "op", "b"], "lookup": [spec.get("key")], "search": ["query"]}[kind]
    for name in need:
        if name not in args:
            return f"Error: missing argument '{name}'."
    if kind == "calculator":
        a, op, b = args["a"], args["op"], args["b"]
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in (a, b)):
            return "Error: a and b must be numbers."
        if op not in ("+", "-", "*", "/"):
            return f"Error: unsupported operator '{op}'."
        if op == "/" and b == 0:
            return "Error: division by zero."
        x = {"+": a + b, "-": a - b, "*": a * b, "/": a / b if op == "/" else 0}[op]
        return "Result: " + format(x, "g")
    if kind == "lookup":
        v = args[spec["key"]]
        return f"Result: {spec['table'][v]}" if v in spec["table"] else f"No result for '{v}'."
    words = str(args["query"]).lower().split()
    hits = sorted(i for i, t in spec["docs"].items() if all(w in t.lower() for w in words))
    return "Result: " + ", ".join(hits) if hits else "No matches."
'''

HINTS = [
    'Never eval model-provided strings; map each operator to a Python operation explicitly.',
    'isinstance(True, int) is True in Python, so check for bool first.',
    'Return a string in every branch: the observation is what the model reads next.',
]

_T = {"calc": {"kind": "calculator"},
      "capital": {"kind": "lookup", "key": "country", "table": {"France": "Paris", "Japan": "Tokyo"}},
      "search": {"kind": "search", "docs": {"d1": "Paris is the capital of France.", "d2": "Tokyo hosts the 2020 Olympics.", "d3": "The Paris Olympics were in 2024."}}}
TESTS = [
    ("calc", {"a": 6, "op": "*", "b": 7}, _T),
    ("calc", {"a": 1, "op": "/", "b": 0}, _T),
    ("calc", {"a": "1", "op": "+", "b": 2}, _T),
    ("calc", {"a": 2, "op": "**", "b": 3}, _T),
    ("calc", {"a": 1, "b": 2}, _T),
    ("capital", {"country": "Japan"}, _T),
    ("capital", {"country": "Peru"}, _T),
    ("search", {"query": "Paris olympics"}, _T),
    ("search", {"query": "Berlin"}, _T),
    ("weather", {}, _T),
    ("calc", {"a": True, "op": "+", "b": 1}, _T),
    ("calc", {"a": 1.5, "op": "/", "b": 4}, _T),
]

EXPLANATION = r"""
## The dispatcher is a trust boundary
Tool arguments come from a model, which means they come from whatever text the model read, including web pages and documents an attacker can write. The dispatcher must validate everything and must never pass model text to `eval`, a shell or SQL without strict checks.

## Errors as observations
Telling the model exactly what went wrong ("missing argument 'country'") lets it repair the call on the next step. Generic failures ("tool error") waste steps.

## Deterministic formatting
Formatting numbers the same way every time keeps traces reproducible and lets you write evaluation checks against exact observations.

## Interview follow-ups
- Which tools would you run in a sandbox, and what limits (time, memory, network) would you set?
"""
