META = dict(title='Retraining trigger policy with cooldown', tags=['MLOps', 'Time Series'], diff='Medium', runtime='python',
            related=['population-stability-index'])

STATEMENT = r"""
Write `retrain_days(psi, auc, baseline_auc, psi_th, consec, auc_drop, max_age, cooldown, age0)` that decides on which days to retrain.

Inputs are daily lists: `psi[d]` is the drift score and `auc[d]` the measured AUC, or `None` if labels have not arrived. `age0` is the model's age in days on day 0.

On each day `d`, first update state: age is `age0 + d` for the first model, or days since the last retrain otherwise; the drift streak counts consecutive days with `psi[d] > psi_th`. Then the triggers are, in priority order:

1. `"performance"`: `auc[d]` is not `None` and `auc[d] < baseline_auc - auc_drop`.
2. `"drift"`: the drift streak is at least `consec`.
3. `"age"`: age is at least `max_age`.

Retrain on day `d` only if a trigger fires and at least `cooldown` days have passed since the last retrain (always allowed if there was none). After a retrain, age becomes 0 (it is 1 on the next day) and the drift streak resets to 0.

Return a list of `[day, reason]` pairs.
"""

STARTER = '''def retrain_days(psi, auc, baseline_auc, psi_th, consec, auc_drop, max_age, cooldown, age0):
    """Return [[day, reason], ...]."""
    return [[d, "drift"] for d, p in enumerate(psi) if p > psi_th]
'''

SOLUTION = '''def retrain_days(psi, auc, baseline_auc, psi_th, consec, auc_drop, max_age, cooldown, age0):
    out, last, streak = [], None, 0
    for d in range(len(psi)):
        age = age0 + d if last is None else d - last
        streak = streak + 1 if psi[d] > psi_th else 0
        if auc[d] is not None and auc[d] < baseline_auc - auc_drop:
            reason = "performance"
        elif streak >= consec:
            reason = "drift"
        elif age >= max_age:
            reason = "age"
        else:
            reason = None
        if reason and (last is None or d - last >= cooldown):
            out.append([d, reason])
            last, streak = d, 0
    return out
'''

HINTS = [
    'Update the streak before checking triggers, so today counts toward it.',
    'A trigger blocked by the cooldown is not remembered; it simply fires again on a later day if the condition still holds.',
    'Reset the streak only when a retrain actually happens.',
]

TESTS = [
    ([0.05] * 10, [None] * 10, 0.8, 0.2, 3, 0.03, 7, 2, 0),
    ([0.05, 0.3, 0.3, 0.3, 0.3, 0.3, 0.05, 0.05], [None] * 8, 0.8, 0.2, 3, 0.03, 30, 2, 0),
    ([0.05] * 6, [None, 0.79, 0.75, 0.74, None, 0.70], 0.8, 0.2, 3, 0.03, 30, 3, 5),
    ([0.3] * 12, [None] * 12, 0.8, 0.2, 2, 0.03, 30, 4, 0),
    ([0.05] * 5, [None] * 5, 0.8, 0.2, 3, 0.03, 10, 1, 9),
    ([0.25, 0.1, 0.25, 0.25, 0.25, 0.1], [0.81, None, None, 0.6, None, None], 0.8, 0.2, 2, 0.05, 100, 0, 0),
]

EXPLANATION = r"""
## Three reasons to retrain
- Performance: the direct signal, but labels often arrive late, so it cannot be the only trigger.
- Drift: input statistics move before labels do. Requiring several consecutive days avoids reacting to one noisy day.
- Age: a schedule backstop that catches slow decay nothing else flags.

## Why a cooldown
Retraining takes time and the new model needs time to show its effect. Without a cooldown a persistent drift signal retrains every day, wasting compute and making it impossible to tell whether any retrain helped.

## Priority order
When several triggers fire, record the strongest reason. The log of reasons is what lets you later tune thresholds, for example if most retrains are age-triggered the drift detector may be too insensitive.

## Interview follow-ups
- When is retraining the wrong response to drift (for example, a broken upstream feature)?
"""
