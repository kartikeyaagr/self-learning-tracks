META = dict(title='Assistant-only loss mask for chat SFT', tags=['Language Models', 'NLP'], diff='Easy',
            runtime='python', lessons=['air-fanout-v1:10.03', 'air-fanout-v1:10.04'],
            related=['padding-and-truncation', 'mutation-trap', 'canonicalize-pairwise-preference-data'])

STATEMENT = r"""
Supervised fine-tuning on conversations trains the model to write the **assistant's** turns, not to imitate users or recite the system prompt. The whole conversation is fed in as context, but the loss is computed only on assistant tokens. Frameworks express this with a `labels` list in which ignored positions hold `-100`.

Write `sft_example(turns, max_len, last_only=False)` that returns a dict.

## Chat template
Token ids: `PAD = 0`, `BOS = 1`, role headers `system = 2`, `user = 3`, `assistant = 4`, end-of-turn `EOT = 5`. Content tokens are ids $\ge 10$.

The sequence is `BOS`, then for each turn `(role, content)` in order: the role's header id, the content ids, then `EOT`.

## Labels
- An assistant turn's **content tokens and its `EOT`** keep their ids as labels (the model must learn when to stop).
- Everything else gets `-100`: `BOS`, all role headers (including the assistant header), and every system or user token.
- With `last_only=True`, only the **last** assistant turn of the conversation is trained; earlier assistant turns are context like any other.

Labels are aligned with `input_ids` (the model shifts them internally); don't shift.

## Length
Truncate the sequence (and labels) to the first `max_len` tokens, then right-pad to exactly `max_len` with `PAD`, label `-100` and attention mask 0.

## Output
| key | value |
|---|---|
| `"input_ids"`, `"labels"`, `"attention_mask"` | lists of `max_len` ints |
| `"n_trained"` | int, number of labels that are not `-100` |

`turns` is a list of `(role, content)` pairs with `role` in `"system"`, `"user"`, `"assistant"` and `content` a list of ints.
"""

STARTER = '''def sft_example(turns, max_len, last_only=False):
    """Return {"input_ids", "labels", "attention_mask", "n_trained"}."""
    raise NotImplementedError
'''

SOLUTION = '''PAD, BOS, EOT, IGNORE = 0, 1, 5, -100
HEADER = {"system": 2, "user": 3, "assistant": 4}

def sft_example(turns, max_len, last_only=False):
    assistant_turns = [i for i, (role, _) in enumerate(turns) if role == "assistant"]
    trained = set(assistant_turns[-1:] if last_only else assistant_turns)

    ids, labels = [BOS], [IGNORE]
    for i, (role, content) in enumerate(turns):
        body = list(content) + [EOT]
        ids += [HEADER[role]] + body
        labels += [IGNORE] + (body if i in trained else [IGNORE] * len(body))   # header is never a target

    ids, labels = ids[:max_len], labels[:max_len]
    pad = max_len - len(ids)
    return {
        "input_ids": ids + [PAD] * pad,
        "labels": labels + [IGNORE] * pad,
        "attention_mask": [1] * len(ids) + [0] * pad,
        "n_trained": sum(l != IGNORE for l in labels),
    }
'''

HINTS = [
    'Build `input_ids` and `labels` side by side, turn by turn, so they can never drift out of alignment.',
    'For each turn, the header is always `-100`; the body (content plus `EOT`) is copied into the labels only for trained assistant turns.',
    'Decide which turns are trained before the loop (all assistant turns, or only the last one), and truncate both lists before padding.',
]

TESTS = [
    ([('user', [10, 11]), ('assistant', [12])], 10),
    ([('system', [20]), ('user', [10]), ('assistant', [12, 13]), ('user', [14]), ('assistant', [15])], 16),
    ([('system', [20]), ('user', [10]), ('assistant', [12, 13]), ('user', [14]), ('assistant', [15])], 16, True),
    ([('user', [10, 11, 12]), ('assistant', [13, 14, 15, 16])], 8),
    ([('user', [10, 11, 12]), ('assistant', [13, 14, 15, 16])], 4),
    ([('user', [10])], 5),
    ([('user', [10]), ('assistant', []), ('user', [11]), ('assistant', [12])], 12, False),
    ([('system', list(range(30, 36))), ('user', list(range(40, 45))), ('assistant', list(range(50, 58))),
      ('user', [60, 61]), ('assistant', list(range(70, 75)))], 32, True),
]

EXPLANATION = r"""
## Why mask the prompt
If user and system tokens were targets, part of every gradient would teach the model to predict prompts, which it never needs to do, and long system prompts repeated across every example would dominate the loss. Masking also stops the model from learning to "write the user's next message" at the end of its turn.

## Why train on `EOT`
The end-of-turn token is how a chat model stops generating. If it is masked, the model never sees a gradient for stopping and rambles past its answer. The assistant **header**, on the other hand, is always supplied by the chat template at inference time, so it needs no gradient.

## `last_only`
For multi-turn data where the earlier assistant turns were written by a weaker model (or by a human in a different style), training only the final turn avoids imitating them while still conditioning on them (test 3 vs test 2).

## Truncation
Truncating from the right can cut an assistant answer mid-way (test 5); then it has no `EOT` target, which is fine but wasteful. Real pipelines drop or pack such examples rather than truncate.

## Complexity
Linear in the conversation length.

## Common bugs
- Shifting labels by one yourself when the training framework shifts them again: every target becomes the token two steps ahead.
- Including the assistant header in the labels.
- Padding with label `0` (the PAD id) instead of `-100`, which trains the model to emit padding.

## Interview follow-ups
- How does sequence packing (several conversations per row) change the attention mask and the labels?
- Should the loss be averaged per token across the batch or per conversation first? What does each choice do to long answers?
"""
