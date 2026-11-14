META = dict(title='Advance the agent state by one step', tags=['Language Models'], diff='Easy', runtime='python',
            related=['parse-react-step', 'dispatch-tool-call'])

STATEMENT = r"""
Write `advance_state(state, step, observation, max_steps)` that returns a **new** state after one agent step. Do not modify `state`.

`state` has keys `scratchpad` (list of strings), `steps` (int), `done` (bool), `answer` (str or None) and `stop_reason` (str or None). `step` is a parsed step (`"action"`, `"final"` or `"error"`, as from a ReAct parser) and `observation` is the observation string for actions and errors (ignored for finals).

1. If `state["done"]` is already true, return an unchanged copy.
2. Increment `steps`.
3. Append to the scratchpad:
   - action: `"Thought: <thought>"`, `"Action: <tool>"`, `"Action Input: <json.dumps(args, sort_keys=True)>"`, `"Observation: <observation>"`
   - final: `"Thought: <thought>"`, `"Final Answer: <answer>"`, then set `done`, `answer`, and `stop_reason = "final"`.
   - error: `"Observation: <observation>"`
   Skip a Thought line whose thought is empty.
4. If not done and `steps >= max_steps`, set `done` and `stop_reason = "max_steps"`.
"""

STARTER = '''def advance_state(state, step, observation, max_steps):
    """Return the next agent state."""
    state["steps"] += 1
    return state
'''

SOLUTION = '''import json

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
'''

HINTS = [
    'Copy the scratchpad list as well as the dict; a shallow dict copy still shares the list.',
    'A final answer on the last allowed step should end with stop_reason "final", not "max_steps".',
]

def _st(**kw):
    s = {"scratchpad": [], "steps": 0, "done": False, "answer": None, "stop_reason": None}
    s.update(kw); return s

_act = {"type": "action", "thought": "Look it up.", "tool": "capital", "args": {"country": "France"}}
TESTS = [
    (_st(), _act, "Result: Paris", 5),
    (_st(steps=4, scratchpad=["x"]), _act, "Result: Paris", 5),
    (_st(steps=4), {"type": "final", "thought": "Done.", "answer": "Paris"}, "", 5),
    (_st(), {"type": "error", "message": "missing action"}, "Error: could not parse step: missing action", 3),
    (_st(done=True, steps=2, answer="x", stop_reason="final"), _act, "ignored", 5),
    (_st(), {"type": "action", "thought": "", "tool": "calc", "args": {"b": 2, "a": 1, "op": "+"}}, "Result: 3", 1),
]

EXPLANATION = r"""
## State as data
Keeping the agent state as an immutable value (a new dict per step) makes runs replayable: you can store every intermediate state, diff two runs, or restart from step 3 with a different model. Mutating shared state makes those bugs hard to see.

## The scratchpad is the prompt
Each step's thought, action and observation are appended to the transcript that is sent back to the model. That is how the model "remembers" what it did. It is also why long agent runs get expensive: the prompt grows with every step, and old observations often need truncating or summarising.

## Always have a step limit
A model can loop forever, calling the same tool with the same arguments. A hard limit, with a distinct stop reason, turns that into a measurable failure instead of a hung process.

## Interview follow-ups
- How would you shorten the scratchpad once it no longer fits the context window?
"""
