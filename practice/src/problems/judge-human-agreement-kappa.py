import random

META = dict(title='Validate an LLM judge against human labels', tags=['Metrics & Evaluation', 'Language Models'],
            diff='Medium', runtime='python', lessons=['aieng:10.03'],
            related=['cohens-kappa', 'pairwise-judge-position-swap', 'remove-response-length-bias-from-judge-scores',
                     'normalized-confusion-matrix'])

STATEMENT = r"""
Before an LLM judge grades thousands of production traces, you check it against a few hundred outputs that a domain expert labelled pass/fail. Raw agreement is not enough: if 80% of outputs pass, a judge that always says "pass" agrees 80% of the time and catches no failures. Write `judge_agreement(human, judge, observed_pass_rate=None)` that reports what you need to decide whether to trust the judge, and corrects the judge's pass rate on unlabelled data for its known errors.

## Inputs
- `human`, `judge`: equal-length lists of ints, `1` = pass and `0` = fail, at least one element. `human` is treated as the truth.
- `observed_pass_rate`: optional float in $[0, 1]$, the fraction of **unlabelled** traces the judge passed.

## Outputs
Return a dict:

| key | value |
|---|---|
| `"confusion"` | `[[tp, fn], [fp, tn]]`: rows are human pass, human fail; columns are judge pass, judge fail |
| `"agreement"` | $(tp + tn) / n$ |
| `"kappa"` | Cohen's $\kappa = (p_o - p_e)/(1 - p_e)$, with $p_o$ the agreement and $p_e = h\,j + (1-h)(1-j)$, where $h$ and $j$ are the human and judge pass fractions. If $p_e = 1$, return `1.0` |
| `"tpr"` | $tp / (tp + fn)$, the share of human passes the judge passes; `None` if the human passed nothing |
| `"tnr"` | $tn / (tn + fp)$, the share of human failures the judge catches; `None` if the human failed nothing |
| `"corrected_pass_rate"` | see below |

## Correcting the observed pass rate
If the judge passes true passes with rate TPR and wrongly passes true failures with rate $1 - \text{TNR}$, then on data with true pass rate $\theta$ it passes $p_{obs} = \theta\,\text{TPR} + (1-\theta)(1-\text{TNR})$. Solving for $\theta$:

$$\theta = \frac{p_{obs} + \text{TNR} - 1}{\text{TPR} + \text{TNR} - 1}$$

Return $\theta$ clipped to $[0, 1]$. Return `None` if `observed_pass_rate` is `None`, if TPR or TNR is `None`, or if $\text{TPR} + \text{TNR} - 1 \le 0$ (the judge is no better than chance, so its output says nothing about $\theta$).
"""

STARTER = '''def judge_agreement(human, judge, observed_pass_rate=None):
    """Return a dict: confusion, agreement, kappa, tpr, tnr, corrected_pass_rate."""
    n = len(human)
    agree = sum(h == j for h, j in zip(human, judge)) / n
    return {"confusion": None, "agreement": agree, "kappa": agree,
            "tpr": None, "tnr": None, "corrected_pass_rate": observed_pass_rate}
'''

SOLUTION = '''def judge_agreement(human, judge, observed_pass_rate=None):
    tp = sum(1 for h, j in zip(human, judge) if h == 1 and j == 1)
    fn = sum(1 for h, j in zip(human, judge) if h == 1 and j == 0)
    fp = sum(1 for h, j in zip(human, judge) if h == 0 and j == 1)
    tn = sum(1 for h, j in zip(human, judge) if h == 0 and j == 0)
    n = tp + fn + fp + tn

    p_o = (tp + tn) / n
    h, j = (tp + fn) / n, (tp + fp) / n            # human and judge pass fractions
    p_e = h * j + (1 - h) * (1 - j)                # agreement expected by chance
    kappa = 1.0 if p_e == 1 else (p_o - p_e) / (1 - p_e)

    tpr = tp / (tp + fn) if tp + fn else None
    tnr = tn / (tn + fp) if tn + fp else None
    corrected = None
    if observed_pass_rate is not None and tpr is not None and tnr is not None and tpr + tnr - 1 > 0:
        theta = (observed_pass_rate + tnr - 1) / (tpr + tnr - 1)
        corrected = min(1.0, max(0.0, theta))
    return {"confusion": [[tp, fn], [fp, tn]], "agreement": p_o, "kappa": kappa,
            "tpr": tpr, "tnr": tnr, "corrected_pass_rate": corrected}
'''

HINTS = [
    'Count the four cells first; every output is a function of tp, fn, fp and tn.',
    'Chance agreement: if the human passes a fraction h and the judge a fraction j independently, they agree with probability h*j + (1-h)*(1-j).',
    'For the correction, write the judge\'s expected pass rate as theta*TPR + (1 - theta)*(1 - TNR) and solve the linear equation for theta. Guard the denominator before dividing.',
]


def _labelled(tp, fn, fp, tn, seed):
    """Shuffled labels with exactly the given confusion counts."""
    pairs = [(1, 1)] * tp + [(1, 0)] * fn + [(0, 1)] * fp + [(0, 0)] * tn
    random.Random(seed).shuffle(pairs)
    return [h for h, _ in pairs], [j for _, j in pairs]


def _noisy(n, base, tpr, tnr, seed):
    r = random.Random(seed)
    human = [1 if r.random() < base else 0 for _ in range(n)]
    judge = [(1 if r.random() < tpr else 0) if h else (0 if r.random() < tnr else 1) for h in human]
    return human, judge


TESTS = [
    (*_labelled(150, 10, 20, 20, 0), 0.88),           # the lesson: 85% agreement, kappa 0.48, misses half the failures
    (*_labelled(152, 8, 6, 34, 1), 0.88),             # after adding failing examples to the judge prompt
    (*_labelled(160, 0, 40, 0, 2), 0.90),             # always says pass: 80% agreement, kappa 0, no correction possible
    ([1, 1, 1, 1], [1, 1, 0, 1], 0.5),                # no human failures: tnr undefined
    ([1, 0, 1, 0], [1, 0, 1, 0]),                     # perfect judge, no observed rate
    ([1, 1, 0, 0], [0, 0, 1, 1], 0.4),                # always wrong: kappa -1, worse than chance
    ([0, 0, 0], [0, 0, 0], 0.2),                      # both always fail: p_e = 1
    (*_labelled(30, 10, 2, 8, 3), 0.98),              # correction clips to 1
    (*_noisy(300, 0.7, 0.92, 0.75, 4), 0.81),
    (*_noisy(500, 0.9, 0.97, 0.6, 5), 0.93),
]

EXPLANATION = r"""
## Why kappa and not agreement
Agreement mixes skill with base rate. In the first test the judge agrees with the expert on 85% of outputs, which sounds fine, but with 80% of outputs passing, chance alone gives $p_e = 0.8 \times 0.85 + 0.2 \times 0.15 = 0.71$. Kappa rescales the gap: $(0.85 - 0.71)/(1 - 0.71) = 0.48$, only moderate. For binary labels there is a closed form that avoids the intermediate fractions:
$$\kappa = \frac{2(tp\cdot tn - fn\cdot fp)}{(tp+fp)(fp+tn) + (tp+fn)(fn+tn)}$$

## Why TPR and TNR separately
The failures are what an eval exists to find, so TNR (the share of real failures the judge flags) usually matters most. The first judge has TPR 0.94 but TNR 0.50: it waves through half the failures. Test 2 is the same judge after failing examples and the expert's critiques were added to its prompt: TNR rises to 0.85 and kappa to 0.79.

## The correction
A judge with known error rates reports a biased pass rate. Inverting the mixing equation (the Rogan-Gladen estimator from diagnostic testing) recovers the true rate: the first judge's 88% observed becomes 86.9%. The estimate is only as good as TPR and TNR, which are themselves estimates from the labelled set. In practice, bootstrap the labelled set to put an interval on the corrected rate. When TPR + TNR is at most 1, the judge's verdicts are independent of the truth or anti-correlated with it, and no correction is meaningful.

## Common bugs
- Swapping rows and columns of the confusion matrix (rows here are the human, the reference).
- Using the judge's pass fraction for both marginals in $p_e$.
- Dividing by zero when one class is missing from the labelled set instead of returning `None`.
- Tuning the judge prompt on the same labelled cases you report agreement on. Split them into a dev set for iterating and a test set for this report.

## Interview follow-ups
- Your labelled set has 160 passes and 40 failures. How precise is the TNR estimate, and how would you get more failures to label?
- Two experts disagree with each other at kappa 0.6. What does that mean for the judge's target?
"""
