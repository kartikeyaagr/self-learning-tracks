META = dict(title='Canonicalize pairwise preference data', tags=['Data Processing', 'Reinforcement Learning'],
            diff='Medium', runtime='python', lessons=['air-fanout-v1:10.03'],
            related=['bradley-terry-reward-model-loss', 'dpo-closed-form-loss', 'perform-stable-exact-deduplication'])

STATEMENT = r"""
Preference datasets for reward models and DPO arrive as raw annotations: a prompt, two responses in whatever order the UI showed them, and a label. Before training they are canonicalised into `(prompt, chosen, rejected)` triples: ties and broken rows are dropped, and repeated annotations of the same comparison are merged by majority vote.

Write `canonicalize(records)` that returns the tuple `(pairs, stats)`.

## Input
`records` is a list of dicts with string keys `"prompt"`, `"a"`, `"b"` and `"label"`, where `label` is `"a"`, `"b"` or `"tie"`.

## Steps
1. **Normalise** the prompt and both responses: strip, and collapse whitespace runs to single spaces (`" ".join(s.split())`). Case is kept.
2. **Drop** a record, counting it in `stats`, checked in this order:
   - `"empty"`: the prompt or either response is empty after normalising;
   - `"identical"`: the two responses are equal after normalising;
   - `"ties"`: the label is `"tie"`.
3. Each surviving record is one **vote**: the winner is the response its label points at.
4. **Group** votes by the key (prompt, unordered pair of responses), so a vote for "x over y" and one for "y over x" on the same prompt belong to the same comparison.
5. For each group, the response with more votes is `chosen`. If the votes are split evenly, drop the group and count it in `"conflicts"`.

## Output
- `pairs`: list of dicts `{"prompt", "chosen", "rejected", "votes", "total"}` where `votes` is the winner's vote count and `total` the group's vote count. Ordered by the position of each group's **first** vote in the input.
- `stats`: dict with int values for `"empty"`, `"identical"`, `"ties"` (records dropped in step 2), `"conflicts"` (groups dropped in step 5) and `"merged"` (votes that were folded into an existing group, i.e. total votes minus number of groups).
"""

STARTER = '''def canonicalize(records):
    """Return (pairs, stats)."""
    raise NotImplementedError
'''

SOLUTION = '''def _norm(s):
    return " ".join(s.split())

def canonicalize(records):
    stats = {"empty": 0, "identical": 0, "ties": 0, "conflicts": 0, "merged": 0}
    groups = {}                                   # key -> {winner: count}; dicts keep first-insertion order
    for r in records:
        prompt, a, b = _norm(r["prompt"]), _norm(r["a"]), _norm(r["b"])
        if not (prompt and a and b):
            stats["empty"] += 1
            continue
        if a == b:
            stats["identical"] += 1
            continue
        if r["label"] == "tie":
            stats["ties"] += 1
            continue
        key = (prompt, frozenset((a, b)))         # order-free: (x, y) and (y, x) are the same comparison
        winner = a if r["label"] == "a" else b
        if key in groups:
            stats["merged"] += 1
        else:
            groups[key] = {a: 0, b: 0}
        groups[key][winner] += 1

    pairs = []
    for (prompt, _), counts in groups.items():
        (x, cx), (y, cy) = counts.items()
        if cx == cy:
            stats["conflicts"] += 1
            continue
        chosen, rejected = (x, y) if cx > cy else (y, x)
        pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected,
                      "votes": max(cx, cy), "total": cx + cy})
    return pairs, stats
'''

HINTS = [
    'Normalise first, then run the three drop checks in the stated order so each dropped record is counted exactly once.',
    'Key the groups by `(prompt, frozenset({a, b}))`. A frozenset ignores the order the two responses were shown in.',
    'A Python dict keeps insertion order, so a dict of groups already yields them in order of their first vote. Store a count per response and compare the two at the end.',
]

_R = lambda p, a, b, l: {"prompt": p, "a": a, "b": b, "label": l}
TESTS = [
    ([_R('Hi', 'Hello!', 'Go away.', 'a'), _R('Hi', 'Go away.', 'Hello!', 'b')],),
    ([_R('Q', 'x', 'y', 'tie'), _R('Q', 'x', ' x ', 'a'), _R('', 'x', 'y', 'a'), _R('Q', 'x', 'y', 'b')],),
    ([_R('Q', 'x', 'y', 'a'), _R('Q', 'y', 'x', 'a')],),
    ([_R('2+2?', '4', '5', 'a'), _R('Capital of France?', 'Paris', 'Lyon', 'a'), _R('2+2?', '5', '4', 'a'),
      _R('2+2?', '4', '5', 'a'), _R('Capital of France?', 'Lyon', 'Paris', 'b'), _R('2+2?', '4', '22', 'a')],),
    ([],),
    ([_R('  Write   a haiku ', 'Leaves  fall\nslowly', 'leaves fall slowly', 'b'), _R('Write a haiku', 'leaves fall slowly', 'Leaves fall slowly', 'a'),
      _R('Write a haiku', 'Leaves fall slowly', 'Leaves fall\tslowly', 'a'), _R('Write a haiku', 'A', '   ', 'a')],),
    ([_R(f'p{i % 4}', f'r{i % 3}', f'r{(i + 1) % 3}', 'ab'[i % 2] if i % 5 else 'tie') for i in range(40)],),
]

EXPLANATION = r"""
## Why canonicalise
Reward-model and DPO losses take `(chosen, rejected)` pairs; they have no slot for "tie" or for "the annotator saw these in the other order". Leaving duplicates in also silently up-weights popular comparisons, and keeping both directions of a disputed comparison feeds the model contradictory gradients that cancel out while still costing compute.

## Order-free grouping
The UI randomises which response is shown as "a" to avoid position bias, so the same comparison appears in both orders. A `frozenset` key makes the two orders collide. Test 1 is one comparison annotated twice in opposite display orders; it becomes a single pair with 2 of 2 votes. Test 3 is a genuine disagreement and is dropped as a conflict.

## Normalisation choices
Whitespace collapse catches responses that differ only in formatting (test 6), which would otherwise teach the model a preference between two identical texts. Case is preserved because it can matter ("Paris" vs "paris" in a factual answer).

## Complexity
$O(\text{total text length})$ with hashing.

## Common bugs
- Keying groups on the ordered tuple `(prompt, a, b)`, so opposite display orders never merge.
- Resolving conflicts by keeping the first or last vote.
- Counting a record in more than one drop category.

## Interview follow-ups
- Majority vote throws away the strength of agreement. How could you use `votes / total` as a soft label in the Bradley-Terry loss?
- Annotators differ in reliability. How would you weight their votes?
"""
