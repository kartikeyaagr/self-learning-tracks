META = dict(title='Run the ReAct loop with a scripted model', tags=['Language Models'], diff='Medium', runtime='python',
            related=['parse-react-step', 'dispatch-tool-call', 'advance-agent-state'])

STATEMENT = r"""
Put the parser, dispatcher and state update together. Write `run_react(outputs, tools, max_steps)`, where `outputs` is a scripted list of model replies: the reply for step `i` is `outputs[i]` (a real agent would call an LLM with the scratchpad here).

Use the same rules as the three previous steps:

- Parse each reply into an action, a final answer or an error (ReAct text format, same checks and messages).
- Actions are dispatched to `tools` (calculator, lookup and search kinds, same observation strings). Parse errors become the observation `"Error: could not parse step: <message>"`.
- Update the state (scratchpad lines, step count, `max_steps` limit) as before.
- If the loop needs reply `i` but `outputs` has fewer than `i + 1` entries, stop with `stop_reason = "no_output"` without counting a step.

Return `{"answer", "steps", "stop_reason", "calls", "scratchpad"}` where `calls` lists `[tool, args, observation]` for every dispatched action, in order.
"""

STARTER = '''def run_react(outputs, tools, max_steps):
    """Run the agent loop over scripted model outputs."""
    return {"answer": None, "steps": 0, "stop_reason": "no_output", "calls": [], "scratchpad": []}
'''

SOLUTION = '''import json

FIELDS = ("Thought:", "Action Input:", "Action:", "Final Answer:")

def parse_react_step(text):
    f = {}
    for line in text.splitlines():
        line = line.strip()
        for key in FIELDS:
            if line.startswith(key):
                f.setdefault(key[:-1], line[len(key):].strip())
                break
    thought = f.get("Thought", "")
    if "Action" in f and "Final Answer" in f:
        return {"type": "error", "message": "both action and final answer"}
    if "Final Answer" in f:
        return {"type": "final", "thought": thought, "answer": f["Final Answer"]}
    if not f.get("Action"):
        return {"type": "error", "message": "missing action"}
    if "Action Input" not in f:
        return {"type": "error", "message": "missing action input"}
    try:
        args = json.loads(f["Action Input"])
    except ValueError:
        args = None
    if not isinstance(args, dict):
        return {"type": "error", "message": "action input is not a JSON object"}
    return {"type": "action", "thought": thought, "tool": f["Action"], "args": args}

def dispatch_tool(tool, args, tools):
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


def advance_state(state, step, observation, max_steps):
    s = {**state, "scratchpad": list(state["scratchpad"])}
    if s["done"]:
        return s
    s["steps"] += 1
    pad = s["scratchpad"]
    if step["type"] in ("action", "final") and step.get("thought"):
        pad.append(f"Thought: {step['thought']}")
    if step["type"] == "action":
        pad += [f"Action: {step['tool']}", f"Action Input: {json.dumps(step['args'], sort_keys=True)}", f"Observation: {observation}"]
    elif step["type"] == "final":
        pad.append(f"Final Answer: {step['answer']}")
        s.update(done=True, answer=step["answer"], stop_reason="final")
    else:
        pad.append(f"Observation: {observation}")
    if not s["done"] and s["steps"] >= max_steps:
        s.update(done=True, stop_reason="max_steps")
    return s

def run_react(outputs, tools, max_steps):
    state = {"scratchpad": [], "steps": 0, "done": False, "answer": None, "stop_reason": None}
    calls = []
    while not state["done"]:
        if state["steps"] >= len(outputs):
            state = {**state, "done": True, "stop_reason": "no_output"}
            break
        step = parse_react_step(outputs[state["steps"]])
        obs = ""
        if step["type"] == "action":
            obs = dispatch_tool(step["tool"], step["args"], tools)
            calls.append([step["tool"], step["args"], obs])
        elif step["type"] == "error":
            obs = "Error: could not parse step: " + step["message"]
        state = advance_state(state, step, obs, max_steps)
    return {"answer": state["answer"], "steps": state["steps"], "stop_reason": state["stop_reason"],
            "calls": calls, "scratchpad": state["scratchpad"]}
'''

HINTS = [
    'Loop until the state says done; the state update owns the step limit.',
    'Check for a missing reply before parsing, and do not count it as a step.',
    'Only actions are dispatched. A final answer ends the loop without calling a tool.',
]

_T = {"calc": {"kind": "calculator"},
      "capital": {"kind": "lookup", "key": "country", "table": {"France": "Paris", "Japan": "Tokyo"}},
      "population": {"kind": "lookup", "key": "city", "table": {"Paris": 2100000, "Tokyo": 14000000}}}

_OK = ['Thought: Find the capital.\nAction: capital\nAction Input: {"country": "France"}',
       'Thought: Now its population.\nAction: population\nAction Input: {"city": "Paris"}',
       'Thought: Done.\nFinal Answer: Paris has about 2,100,000 people.']
_LOOP = ['Action: capital\nAction Input: {"country": "Peru"}'] * 10
_FIX = ['Action: calc\nAction Input: {"a": 2, "op": "x", "b": 3}', 'Action: calc\nAction Input: not json',
        'Action: calc\nAction Input: {"a": 2, "op": "*", "b": 3}', 'Final Answer: 6']
TESTS = [
    (_OK, _T, 5),
    (_OK, _T, 2),
    (_LOOP, _T, 4),
    (_FIX, _T, 6),
    (_OK[:1], _T, 5),
    ([], _T, 3),
]

EXPLANATION = r"""
## The loop
Model proposes, code executes, observation goes back, repeat until the model answers or a limit is hit. Everything interesting about agents (tool design, error recovery, cost, safety) lives in the pieces around this small loop.

## Testing with scripted outputs
Replacing the LLM with a fixed list of replies makes the loop deterministic, so you can test that parse errors are recoverable, that limits stop runaway loops, and that a missing reply ends the run cleanly. The same harness later replays real model transcripts for regression tests.

## Failure modes to expect
- Repeating the same failed call until the step limit (the `_LOOP` case).
- Recovering from bad input after an error observation (the `_FIX` case).
- Answering early without using a tool at all, which is only visible in evaluation.

## Interview follow-ups
- How would you detect and break a loop of identical calls before the step limit?
"""
