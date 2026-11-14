META = dict(title='Evaluate an agent trajectory against a reference', tags=['Language Models', 'Metrics & Evaluation'], diff='Medium', runtime='python',
            related=['run-react-loop'])

STATEMENT = r"""
Write `evaluate_trajectory(run, reference)`.

`run` is the output of an agent loop: `{"answer": str or None, "steps": int, "stop_reason": str, "calls": [[tool, args, observation], ...]}`. `reference` has `"answer"` (expected text), `"required_tools"` (list) and `"max_steps"` (a budget).

Normalise answers by lowercasing, removing every character that is not a letter, digit or space, and collapsing whitespace. Return:

- `success`: the normalised run answer equals, or contains, the normalised reference answer (False if the run has no answer).
- `tool_recall`: fraction of distinct required tools that were called at least once (1.0 if none are required).
- `invalid_calls`: number of calls whose observation starts with `"Error"`.
- `repeated_calls`: number of calls whose `(tool, args)` pair already appeared earlier in the run (compare args with `json.dumps(args, sort_keys=True)`).
- `efficiency`: `min(1.0, reference max_steps / steps)` if the run succeeded and took at least one step, else 0.0.
- `verdict`: `"pass"` if success and tool_recall is 1.0, `"partial"` if exactly one of them holds, else `"fail"`.
"""

STARTER = '''def evaluate_trajectory(run, reference):
    """Score an agent run against a reference."""
    ok = run["answer"] == reference["answer"]
    return {"success": ok, "tool_recall": 1.0, "invalid_calls": 0, "repeated_calls": 0,
            "efficiency": 1.0 if ok else 0.0, "verdict": "pass" if ok else "fail"}
'''

SOLUTION = '''import json, re

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
'''

HINTS = [
    'Normalise before comparing, so "Paris." and "paris" match, and use containment because agents add context around the answer.',
    'Serialise args with sorted keys so {"a": 1, "b": 2} and {"b": 2, "a": 1} count as the same call.',
    'Success without the required tools usually means the model answered from memory: a "partial", not a pass.',
]

_ref = {"answer": "Paris", "required_tools": ["capital"], "max_steps": 2}
TESTS = [
    ({"answer": "The capital is Paris.", "steps": 2, "stop_reason": "final", "calls": [["capital", {"country": "France"}, "Result: Paris"]]}, _ref),
    ({"answer": "Paris", "steps": 1, "stop_reason": "final", "calls": []}, _ref),
    ({"answer": None, "steps": 4, "stop_reason": "max_steps", "calls": [["capital", {"country": "Peru"}, "No result for 'Peru'."]] * 4}, _ref),
    ({"answer": "Lyon", "steps": 3, "stop_reason": "final",
      "calls": [["calc", {"a": 1, "op": "x", "b": 2}, "Error: unsupported operator 'x'."], ["capital", {"b": 1, "a": 2}, "Result: ?"], ["capital", {"a": 2, "b": 1}, "Result: ?"]]}, _ref),
    ({"answer": "PARIS!!", "steps": 5, "stop_reason": "final", "calls": [["capital", {"country": "France"}, "Result: Paris"]]}, _ref),
    ({"answer": "6", "steps": 0, "stop_reason": "final", "calls": []}, {"answer": "6", "required_tools": [], "max_steps": 1}),
]

EXPLANATION = r"""
## Outcome is not enough
An agent can reach the right answer by luck or from memory while ignoring the tools it was supposed to use, or it can succeed after ten failed calls. Trajectory metrics separate those cases: did it use the right tools, how many calls failed, how much did it repeat itself, how many steps did it take relative to a good run.

## Why rule-based checks first
Exact and normalised matches are cheap, deterministic and easy to debug. Many teams add an LLM judge for open-ended answers, but keep these structural metrics because they explain why a run failed, not only that it did.

## Using the metrics
Aggregate over a fixed suite of tasks and track them per model and prompt version. A rise in repeated calls usually means the model is not reading its observations; a rise in invalid calls usually means a tool schema or description changed.

## Interview follow-ups
- How would you evaluate an agent whose correct trajectory is not unique?
"""
