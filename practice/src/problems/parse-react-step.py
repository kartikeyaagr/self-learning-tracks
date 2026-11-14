META = dict(title='Parse one ReAct step from model output', tags=['Language Models', 'NLP'], diff='Easy', runtime='python')

STATEMENT = r"""
A ReAct agent asks the model to reply in this format:

```
Thought: <reasoning>
Action: <tool name>
Action Input: <JSON object>
```

or, when it is done, `Final Answer: <text>` (a Thought line may come first).

Write `parse_react_step(text)`. Look at lines after stripping each one; a field is a line that starts with `Thought:`, `Action:`, `Action Input:` or `Final Answer:`, and its value is the rest of the line stripped. If a field appears twice, keep the first. Then return, in this order of checks:

1. `{"type": "error", "message": "both action and final answer"}` if both `Action` and `Final Answer` are present.
2. `{"type": "final", "thought": ..., "answer": ...}` if `Final Answer` is present.
3. `{"type": "error", "message": "missing action"}` if there is no `Action`, or its value is empty.
4. `{"type": "error", "message": "missing action input"}` if there is no `Action Input`.
5. `{"type": "error", "message": "action input is not a JSON object"}` if it does not parse with `json.loads` to a dict.
6. Otherwise `{"type": "action", "thought": ..., "tool": ..., "args": dict}`.

A missing Thought is `""`.
"""

STARTER = '''import json

def parse_react_step(text):
    """Parse model output into an action, a final answer or an error."""
    return {"type": "final", "thought": "", "answer": text}
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
'''

HINTS = [
    'Check "Action Input:" before "Action:", because every Action Input line also starts with "Action".',
    'json.loads raises ValueError (JSONDecodeError is a subclass) on bad input; a valid list or number is still not a dict.',
    'Return errors as data. The agent loop turns them into an observation so the model can retry.',
]

TESTS = [
    ('Thought: I need the population.\nAction: lookup\nAction Input: {"query": "Paris"}',),
    ("Thought: I know this.\nFinal Answer: 42",),
    ("Action: lookup\nAction Input: Paris",),
    ('Action: calculator\nAction Input: [1, 2]',),
    ("Thought: hmm",),
    ('Action: search\nAction Input: {"q": "x"}\nFinal Answer: done',),
    ('  Thought:  spaced  \n  Action:   add  \n  Action Input: {"a": 1, "b": 2}  \nAction: ignored',),
    ("Thought: no input\nAction: lookup",),
]

EXPLANATION = r"""
## The contract between model and loop
ReAct interleaves reasoning ("Thought") with tool use ("Action") and the tool's result ("Observation"). The loop around the model is ordinary code, so it needs a strict, parseable contract. Everything that can go wrong in that contract (no action, malformed JSON, the model trying to answer and act at once) must become a specific error the loop can feed back.

## Why not raise
A parse failure is normal model behaviour, not a bug. Turning it into an observation ("Error: action input is not a JSON object") gives the model a chance to correct itself on the next step, which works surprisingly often.

## Modern variants
Native function calling replaces this text format with structured tool-call objects, which removes most parse errors. The loop logic, the error-as-observation pattern and the need to validate arguments stay the same.

## Interview follow-ups
- Why should the loop stop generation at "Observation:" when using the text format?
"""
