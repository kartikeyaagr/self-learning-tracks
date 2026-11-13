META = dict(title='Weighted edit distance with transpositions', tags=['NLP'], diff='Medium', runtime='python',
            lessons=['sdsd:09.15'], related=['spell-correction', 'evaluation-metrics-bleu-rouge'])

STATEMENT = r"""
The edit distance between two sequences is the cheapest way to turn `a` into `b` using single-element edits. It powers spell checkers, fuzzy matching, and word error rate (WER) for speech recognition when the sequences are lists of words.

Write `edit_distance(a, b, ins=1, dele=1, sub=1, swap=None)`.

## Inputs
- `a`, `b`: two strings, or two lists of strings (each element is one symbol).
- `ins`, `dele`, `sub`: non-negative costs of inserting a symbol, deleting a symbol, and replacing one symbol by a different one. Matching equal symbols costs 0.
- `swap`: `None` to disable transpositions, or the cost of exchanging two **adjacent** symbols (`ab` to `ba`).

## Transpositions
Use the *optimal string alignment* rule: a transposition can only be applied to two symbols that are not touched by any other edit. In the DP below this is the extra case
$$D[i][j] = \min\big(D[i][j],\ D[i-2][j-2] + \text{swap}\big) \quad \text{if } i, j \ge 2,\ a_{i-1} = b_{j-2},\ a_{i-2} = b_{j-1}.$$

## Output
The minimum total cost, as a number (an int when all costs are ints).
"""

STARTER = '''def edit_distance(a, b, ins=1, dele=1, sub=1, swap=None):
    """Minimum cost to turn sequence a into sequence b."""
    raise NotImplementedError
'''

SOLUTION = '''def edit_distance(a, b, ins=1, dele=1, sub=1, swap=None):
    n, m = len(a), len(b)
    # D[i][j] = cost of turning a[:i] into b[:j]
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[i][0] = D[i - 1][0] + dele
    for j in range(1, m + 1):
        D[0][j] = D[0][j - 1] + ins
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            same = a[i - 1] == b[j - 1]
            D[i][j] = min(D[i - 1][j] + dele,                      # drop a[i-1]
                          D[i][j - 1] + ins,                       # add b[j-1]
                          D[i - 1][j - 1] + (0 if same else sub))  # match or replace
            if (swap is not None and i > 1 and j > 1
                    and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]):
                D[i][j] = min(D[i][j], D[i - 2][j - 2] + swap)
    return D[n][m]
'''

HINTS = [
    'Let `D[i][j]` be the cheapest way to turn the first `i` symbols of `a` into the first `j` of `b`. The first row and column are all insertions or all deletions.',
    'Each cell takes the minimum of three moves: from the cell above (delete), from the left (insert), and from the diagonal (match for 0 or substitute).',
    'For transpositions add one more candidate from two rows and two columns back, `D[i-2][j-2] + swap`, when the last two symbols are crossed over. Use `==` on elements, so the same code works for strings and word lists.',
]

TESTS = [
    ("kitten", "sitting"),
    ("acb", "abc", 1, 1, 1, 1),
    ("", ""),
    ("", "abc"),
    ("abc", "abc"),
    ("ca", "abc", 1, 1, 1, 1),
    ("flaw", "lawn", 2, 3, 1),
    ("intention", "execution", 1, 1, 2),
    (["the", "cat", "sat", "on", "mat"], ["the", "cat", "sat", "on", "the", "mat"]),
    (["a", "big", "red", "dog"], ["a", "red", "big", "cat"], 1, 1, 1, 0.5),
]

EXPLANATION = r"""
## Why the DP is correct
Look at the last symbols of an optimal edit script. Either `a[i-1]` was deleted, `b[j-1]` was inserted, or the two were aligned (matched or substituted). Each case leaves a smaller prefix problem, so the best cost is the minimum over the three, plus the cost of that last step. With transpositions there is a fourth way to end: the last two symbols were swapped.

## Optimal string alignment vs true Damerau-Levenshtein
OSA forbids editing a substring more than once. That is why `edit_distance("ca", "abc", swap=1)` is 3 here, while unrestricted Damerau-Levenshtein gives 2 (swap to `ac`, then insert `b` in the middle). The full version needs an extra table of last positions per symbol.

## Complexity
$O(nm)$ time. Only the last two rows are ever read, so memory can drop to $O(m)$ (three rows with transpositions).

## Common bugs
- Initialising the first row/column with `i` and `j` instead of the cumulative costs, which is wrong once `ins != dele`.
- Charging `sub` when the symbols are equal.
- Checking the transposition condition with the indices off by one.

## Interview follow-ups
- How do you compute WER from this? (Word lists, unit costs, divided by the reference length; test 9 gives one insertion.)
- How would you recover the actual alignment? (Store which move won in each cell and backtrack from `D[n][m]`.)
"""
