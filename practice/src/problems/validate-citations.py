META = dict(title='Check that cited chunks support each answer sentence', tags=['NLP', 'Language Models', 'Metrics & Evaluation'], diff='Medium', runtime='python',
            related=['rag-context-packing'])

STATEMENT = r"""
A RAG answer cites chunks with markers such as `[c2]`. Write `validate_citations(answer, retrieved, min_support)`.

- `retrieved` maps chunk id to chunk text.
- Split `answer` into sentences with the regex `(?<=[.!?])\s+` and drop empty pieces.
- In each sentence, citation ids are the contents of `\[([^\]]+)\]`. Remove the markers, then take **content words**: lowercase matches of `[a-z0-9]+` with at least 3 characters.
- Each sentence gets a status, checked in this order:
  1. `"uncited"` if it has no markers.
  2. `"unknown"` if any cited id is not in `retrieved`.
  3. Otherwise compute support = fraction of distinct content words that appear among the content words of the cited chunks (1.0 if the sentence has no content words). Status is `"supported"` if support $\ge$ `min_support`, else `"unsupported"`.
- Report support as 0.0 for uncited and unknown sentences, rounded to 6 places otherwise.

Return `{"sentences": [[status, support], ...], "grounded": fraction of sentences that are supported}` (0.0 for an empty answer).
"""

STARTER = '''import re

def validate_citations(answer, retrieved, min_support):
    """Check each sentence of a cited answer against the retrieved chunks."""
    sents = [s for s in re.split(r"(?<=[.!?])\\s+", answer) if s]
    return {"sentences": [["supported", 1.0] for _ in sents], "grounded": 1.0 if sents else 0.0}
'''

SOLUTION = '''import re

def _words(text):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 3}

def validate_citations(answer, retrieved, min_support):
    sents = [s for s in re.split(r"(?<=[.!?])\\s+", answer.strip()) if s]
    rows = []
    for s in sents:
        ids = [x.strip() for x in re.findall(r"\\[([^\\]]+)\\]", s)]
        if not ids:
            rows.append(["uncited", 0.0]); continue
        if any(i not in retrieved for i in ids):
            rows.append(["unknown", 0.0]); continue
        words = _words(re.sub(r"\\[[^\\]]+\\]", " ", s))
        evidence = set().union(*(_words(retrieved[i]) for i in ids))
        sup = 1.0 if not words else len(words & evidence) / len(words)
        rows.append(["supported" if sup >= min_support else "unsupported", round(sup, 6)])
    grounded = sum(r[0] == "supported" for r in rows) / len(rows) if rows else 0.0
    return {"sentences": rows, "grounded": grounded}
'''

HINTS = [
    'Strip the citation markers before extracting words, or ids like "c2" leak into the overlap count.',
    'Check for unknown ids before computing support: a citation to a chunk that was never retrieved is a hallucinated source.',
    'The evidence is the union of words over every chunk the sentence cites.',
]

_R = {"c1": "The Eiffel Tower was completed in 1889 for the World's Fair in Paris.",
      "c2": "It is 330 metres tall and was the tallest structure until 1930.",
      "c3": "Gustave Eiffel's company designed and built the tower."}
TESTS = [
    ("The Eiffel Tower was completed in 1889 [c1]. It is 330 metres tall [c2].", _R, 0.8),
    ("The tower was completed in 1889 [c2]. Eiffel's company built it [c3].", _R, 0.6),
    ("It opened in Paris [c1][c3]. It was painted red [c4]. Everyone loves it.", _R, 0.5),
    ("", _R, 0.5),
    ("Built by Gustave Eiffel's company [ c3 ]! Tallest until 1930 [c2]?", _R, 0.9),
    ("It is [c1].", _R, 0.5),
]

EXPLANATION = r"""
## Why check citations
A model can attach a real chunk id to a claim that chunk never makes. Users see the citation and trust the sentence more, so a wrong citation is worse than none. A cheap lexical check catches the obvious cases: claims with no source, sources that were never retrieved, and sentences whose content words are mostly absent from the cited text.

## What lexical overlap misses
Paraphrases score low even when they are faithful, and a sentence can reuse every word of the source while reversing its meaning ("not completed in 1889"). Production systems add an entailment model or an LLM judge for the sentences that pass this filter. The lexical check is still worth running first because it is fast, deterministic and explainable.

## Using the score
Track the grounded fraction over a fixed evaluation set. A drop after a prompt or model change usually means the generator has started answering from its own memory instead of the context.

## Interview follow-ups
- How would you handle a sentence that combines facts from two chunks but cites only one?
"""
