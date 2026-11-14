META = dict(title='Capstone: a guarded ReAct agent with evaluation', tags=['Language Models'], diff='Hard', runtime='python',
            related=['parse-react-step', 'dispatch-tool-call', 'advance-agent-state', 'run-react-loop', 'authorize-tool-call', 'evaluate-agent-trajectory'])

STATEMENT = r"""
Build the full agent: `react_agent(outputs, tools, policy, approved, max_steps, reference)`.

Run the ReAct loop over the scripted replies in `outputs` exactly as in the loop step (same parser, dispatcher, state update, `no_output` rule), with one addition: every parsed action is first checked by the policy (same rules and verdicts as the authorisation step), using the call id `str(step_index)` where `step_index` is the number of steps taken before this one.

- `allow`: dispatch as usual.
- `deny`: do not run the tool; the observation is `"Error: not authorized (<reason>)."`.
- `needs_approval`: do not run the tool; the observation is `"Error: waiting for approval (<reason>)."`.

Every action, including refused ones, is recorded in `calls` as `[tool, args, observation]`.

Return `{"run": {"answer", "steps", "stop_reason", "calls"}, "eval": <trajectory evaluation of run against reference>}` using the evaluation step's metrics.
"""

STARTER = '''def react_agent(outputs, tools, policy, approved, max_steps, reference):
    """Run a guarded ReAct agent and evaluate the trajectory."""
    raise NotImplementedError
'''

SOLUTION = '''import json, re

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

def authorize(call, policy, approved):
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

import re

def _norm(s):
    return " ".join(re.sub(r"[^a-z0-9 ]", "", s.lower()).split())

def evaluate_trajectory(run, reference):
    ans = run.get("answer")
    success = ans is not None and _norm(reference["answer"]) in _norm(ans)
    used = {c[0] for c in run["calls"]}
    req = set(reference["required_tools"])
    recall = len(req & used) / len(req) if req else 1.0
    invalid = sum(str(c[2]).startswith("Error") for c in run["calls"])
    seen, repeated = set(), 0
    for tool, args, _ in run["calls"]:
        key = (tool, json.dumps(args, sort_keys=True))
        repeated += key in seen
        seen.add(key)
    eff = min(1.0, reference["max_steps"] / run["steps"]) if success and run["steps"] > 0 else 0.0
    verdict = "pass" if success and recall == 1.0 else ("partial" if success or recall == 1.0 else "fail")
    return {"success": success, "tool_recall": recall, "invalid_calls": invalid,
            "repeated_calls": repeated, "efficiency": eff, "verdict": verdict}

def react_agent(outputs, tools, policy, approved, max_steps, reference):
    state = {"scratchpad": [], "steps": 0, "done": False, "answer": None, "stop_reason": None}
    calls = []
    while not state["done"]:
        if state["steps"] >= len(outputs):
            state = {**state, "done": True, "stop_reason": "no_output"}
            break
        step = parse_react_step(outputs[state["steps"]])
        obs = ""
        if step["type"] == "action":
            verdict, reason = authorize({"id": str(state["steps"]), "tool": step["tool"], "args": step["args"]}, policy, approved)
            if verdict == "allow":
                obs = dispatch_tool(step["tool"], step["args"], tools)
            elif verdict == "deny":
                obs = f"Error: not authorized ({reason})."
            else:
                obs = f"Error: waiting for approval ({reason})."
            calls.append([step["tool"], step["args"], obs])
        elif step["type"] == "error":
            obs = "Error: could not parse step: " + step["message"]
        state = advance_state(state, step, obs, max_steps)
    run = {"answer": state["answer"], "steps": state["steps"], "stop_reason": state["stop_reason"], "calls": calls}
    return {"run": run, "eval": evaluate_trajectory(run, reference)}
'''

HINTS = [
    'Reuse your five earlier functions unchanged; the capstone only adds the policy check between parsing and dispatch.',
    'Refused calls still count as steps and still appear in calls, so the evaluation sees them as invalid.',
    'The call id is the step index as a string, so approvals can be granted per step.',
]

_T = {"calc": {"kind": "calculator"},
      "capital": {"kind": "lookup", "key": "country", "table": {"France": "Paris", "Japan": "Tokyo"}},
      "population": {"kind": "lookup", "key": "city", "table": {"Paris": 2100000, "Tokyo": 14000000}},
      "transfer": {"kind": "calculator"}}
_P = {"deny": ["shell"], "approval": ["transfer"], "limits": {"transfer": {"b": 100}}}
_ref = {"answer": "14000000", "required_tools": ["capital", "population"], "max_steps": 3}
_good = ['Thought: Capital first.\nAction: capital\nAction Input: {"country": "Japan"}',
         'Action: population\nAction Input: {"city": "Tokyo"}',
         'Final Answer: Tokyo has 14000000 people.']
_memory = ['Thought: I remember this.\nFinal Answer: About 14000000.']
_shell = ['Action: shell\nAction Input: {"cmd": "cat /etc/passwd"}', 'Action: capital\nAction Input: {"country": "Japan"}',
          'Action: population\nAction Input: {"city": "Tokyo"}', 'Final Answer: 14000000']
_xfer = ['Action: transfer\nAction Input: {"a": 5, "op": "+", "b": 50}', 'Action: transfer\nAction Input: {"a": 5, "op": "+", "b": 500}', 'Final Answer: done']
TESTS = [
    (_good, _T, _P, [], 5, _ref),
    (_memory, _T, _P, [], 5, _ref),
    (_shell, _T, _P, [], 5, _ref),
    (_shell, _T, _P, [], 3, _ref),
    (_xfer, _T, _P, [], 5, {"answer": "done", "required_tools": ["transfer"], "max_steps": 2}),
    (_xfer, _T, _P, ["0", "1"], 5, {"answer": "done", "required_tools": ["transfer"], "max_steps": 2}),
]

EXPLANATION = r"""
## What the pieces add up to
Parser, dispatcher, state, loop, policy and evaluation are the whole of a minimal agent framework. Frameworks such as LangGraph or the OpenAI Agents SDK add persistence, streaming, parallel tool calls and tracing, but each of those maps onto one of these functions.

## What the tests show
- The memory run answers correctly without tools: success but a partial verdict.
- The shell run wastes a step on a refused call, then recovers; with a tighter step budget the same run fails.
- The transfer run shows approval gating: without approval nothing executes; with it, the over-limit call is still refused because limits are checked before approval.

## The loop to keep in mind
Build the agent, evaluate it on a fixed task suite, inspect failing trajectories, change one thing (tool description, prompt, policy), and evaluate again. Without trajectory-level evaluation, agent changes are guesswork.

## Interview follow-ups
- Where would you put retries, and how would you stop them from multiplying cost?
"""
