META = dict(title='Pack a sentence pair for next-sentence prediction', tags=['Transformers', 'NLP'], diff='Easy', runtime='python',
            lessons=['aieng:05.03'], related=['padding-and-truncation', 'masked-language-modeling', 'span-extraction'])

STATEMENT = r"""
BERT's next-sentence-prediction task feeds two segments A and B as one sequence and asks whether B really followed A. The same pair format is used for question answering and sentence-pair classification:

```
[CLS] A ... [SEP] B ... [SEP] [PAD] ...
```

Write `pack_pair(a, b, max_len, cls_id, sep_id, pad_id=0)` that returns `(input_ids, token_type_ids, attention_mask)`, three lists of ints of length `max_len`.

## Truncation (longest first)
There is room for `max_len - 3` segment tokens. While `len(a) + len(b)` exceeds that, remove the **last** token of the longer segment; if they are equally long, remove from `a`. Do not modify the caller's lists.

## Layout
- `input_ids`: `cls_id`, then `a`, then `sep_id`, then `b`, then `sep_id`, then `pad_id` up to `max_len`.
- `token_type_ids`: 0 for `[CLS]`, every token of `a` and the first `[SEP]`; 1 for every token of `b` and the second `[SEP]`; 0 for padding.
- `attention_mask`: 1 for every non-padding position, 0 for padding.

`max_len` is at least 3.
"""

STARTER = '''def pack_pair(a, b, max_len, cls_id, sep_id, pad_id=0):
    """Return (input_ids, token_type_ids, attention_mask), each a list of length max_len."""
    raise NotImplementedError
'''

SOLUTION = '''def pack_pair(a, b, max_len, cls_id, sep_id, pad_id=0):
    a, b = list(a), list(b)                       # copies: truncation must not touch the caller's lists
    while len(a) + len(b) > max_len - 3:
        (a if len(a) >= len(b) else b).pop()      # longest first, ties trim a
    ids = [cls_id] + a + [sep_id] + b + [sep_id]
    types = [0] * (len(a) + 2) + [1] * (len(b) + 1)
    n_pad = max_len - len(ids)
    return ids + [pad_id] * n_pad, types + [0] * n_pad, [1] * len(ids) + [0] * n_pad
'''

HINTS = [
    'Copy both segments, then trim in a loop: each iteration pops one token from whichever segment is currently longer (`a` on a tie).',
    'Build the unpadded sequence first. Segment ids are `len(a) + 2` zeros (CLS, a, SEP) followed by `len(b) + 1` ones (b, SEP).',
    'Pad all three lists by the same amount: `pad_id` for the ids, 0 for the segment ids, 0 for the attention mask.',
]

TESTS = [
    ([7, 8], [9], 8, 101, 102),
    ([1, 2, 3, 4, 5], [6, 7, 8], 7, 101, 102),
    ([], [], 3, 101, 102),
    ([1, 2], [3, 4], 5, 101, 102),
    ([], [5, 6, 7, 8], 6, 101, 102, -1),
    (list(range(10, 30)), list(range(40, 45)), 16, 101, 102),
    ([1, 2, 3], [4, 5, 6], 12, 2, 3, 1),
]

EXPLANATION = r"""
## Why it works
Packing both segments into one sequence lets self-attention compare every token of A with every token of B. The segment (token type) ids add a learned embedding that tells the model which side a token came from, and the `[SEP]` tokens mark the boundary explicitly. For next-sentence prediction, the final hidden state of `[CLS]` feeds a binary classifier: half the training pairs are real consecutive sentences, half pair A with a random sentence.

## Longest-first truncation
Trimming the longer segment one token at a time keeps as much of both segments as possible. Cutting only `b` (or only `a`) could delete a short segment entirely while a long one keeps tokens that matter less.

## Complexity
$O(\text{len}(a) + \text{len}(b) + \text{max\_len})$.

## Common bugs
- Forgetting that the three special tokens use up room: the budget is `max_len - 3`.
- Giving the second `[SEP]` segment id 0.
- `a.pop()` on the caller's list, which mutates the dataset between epochs.

## Interview follow-ups
- RoBERTa dropped NSP and trained on contiguous full-length text instead. Why might NSP have been too easy to be useful? (Random negatives usually differ in topic, so the model can solve it without learning coherence.)
- ALBERT's sentence-order prediction uses swapped consecutive sentences as negatives. What does that force the model to learn?
"""
