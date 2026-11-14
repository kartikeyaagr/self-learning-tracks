---
id: "06.01"
course: "deep-learning"
module: "06"
slug: "text-preprocessing-and-tokenisation"
title: "Text preprocessing and tokenisation"
keyIdeas: ["Every NLP model starts by mapping text to integer ids: normalise, pre-tokenise, split into subwords, look up ids. Each step is a modelling decision, and the model is tied to exactly the pipeline it was trained with.", "Unicode normalisation (NFC, NFKC) makes visually identical strings byte-identical; case-folding and accent stripping are lossy choices that some tasks want and others do not.", "Sentence splitting on '.' fails on abbreviations, decimals and initials. Use rules with exception lists or a trained splitter, and test on your own domain.", "Subword tokenisers sit between words (huge vocabularies, unknown words) and characters (very long sequences). BPE learns merges by pair frequency; WordPiece encodes by greedy longest match with '##' continuations.", "Byte-level BPE starts from the 256 byte values, so any string can be encoded and there is no unknown token."]
question: "Why can a byte-level BPE tokeniser (GPT-2 style) never produce an unknown token, while BERT's WordPiece can? What does each model actually receive for an emoji that never appeared in tokeniser training?"
answer: "Byte-level BPE's base vocabulary is all 256 byte values, and merges only combine symbols that already exist, so any UTF-8 string can at worst be encoded byte by byte. An unseen emoji (4 bytes in UTF-8) arrives as up to four byte-level tokens; the model gets the information, though it costs more tokens and the model may have seen those byte sequences rarely. WordPiece's base vocabulary is a fixed set of characters seen in training. Greedy longest-match encoding fails if no vocabulary entry matches a remaining piece of the word, and then the whole word is replaced by [UNK]. So BERT receives [UNK] for the emoji (and for the whole whitespace-delimited word containing it), which erases all information about it."
sources: [{"title": "Neural Machine Translation of Rare Words with Subword Units (Sennrich et al., 2016)", "url": "https://arxiv.org/abs/1508.07909", "note": "The BPE-for-NLP paper; the merge algorithm fits in a dozen lines in section 3.2."}, {"title": "Hugging Face LLM Course, chapter 6: Tokenizers", "url": "https://huggingface.co/learn/llm-course/chapter6/1", "note": "Normalisation, pre-tokenisation, and BPE, WordPiece and Unigram side by side with code."}, {"title": "SentencePiece (Kudo and Richardson, 2018)", "url": "https://arxiv.org/abs/1808.06226", "note": "Language-independent subword tokenisation treating whitespace as a symbol."}, {"title": "Python unicodedata", "url": "https://docs.python.org/3/library/unicodedata.html", "note": "NFC, NFD, NFKC and NFKD normalisation forms and what each folds."}]
practice: ["byte-pair-encoding", "subword-tokenization", "text-normalization", "sentence-segmentation", "regex-patterns-for-nlp"]
tags: ["tokenisation", "BPE", "WordPiece"]
tools: ["Python"]
---

## Where this module sits

Module 05 built the architectures that read sequences. This module is about text specifically: how raw strings become token ids, the classical counting models that still serve as baselines and features, word embeddings, token-level tasks, pretrained encoders, and how generated text is decoded and scored. It ends where the AI Engineering and AI Research courses pick up with large language models.

~~~dg-flow Each stage is a choice the model is then locked into: the same text must go through the same stages at training and serving time.
Raw text -(NFKC, lowercase?)-> Normalised -(regex split)-> Pre-tokens -(BPE / WordPiece)-> *Subwords -(vocab lookup)-> Ids + offsets
~~~

## Normalisation

Unicode allows the same visible text to have several byte sequences. "é" can be one code point (U+00E9) or "e" followed by a combining accent (U+0065 U+0301). NFC composes these into a canonical form; NFKC additionally folds compatibility characters, so the ligature "ﬁ" becomes "fi" and full-width "Ａ" becomes "A". Without normalisation the same word gets different ids depending on which keyboard or PDF extractor produced it.

Other common steps, each with a cost:

- **Lowercasing** merges "Apple" and "apple": good for small-data classifiers, bad for named-entity recognition, where case is a strong signal. BERT ships both cased and uncased models for this reason.
- **Accent stripping** helps English search, hurts French and Vietnamese.
- **Whitespace and control characters**: collapse runs of spaces, remove zero-width characters, normalise line endings.

```python
import unicodedata, re
def normalise(s):
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", s)   # zero-width characters
    return re.sub(r"\s+", " ", s).strip()
```

## Regular expressions and pre-tokenisation

Before subword splitting, most tokenisers pre-split text into words, numbers and punctuation with a regex, so merges never cross a word boundary. GPT-2's pattern, for instance, separates contractions ("'s", "'re"), runs of letters, runs of digits and runs of other symbols, and keeps a leading space attached to the word. Regexes are also the workhorse for extracting URLs, e-mail addresses, mentions and hashtags before they are mangled by later steps.

Two failure modes to know: greedy quantifiers that swallow too much (`.+` vs `.+?`), and nested quantifiers like `(a+)+` that cause catastrophic backtracking on adversarial input, which can hang a server.

## Sentence splitting

"Splitting on full stops" breaks on "Dr. Smith", "3.14", "U.S. troops" and "e.g. this". Practical splitters combine rules (a full stop followed by whitespace and a capital letter is a candidate boundary) with exception lists of abbreviations, or learn abbreviations from unlabelled text, as the Punkt algorithm does. Domain matters: legal text, chat messages and clinical notes all break general-purpose splitters in different ways, so evaluate on a sample of your own data.

## Why subwords

A word-level vocabulary for English web text needs millions of entries to cover names, typos and inflections, and anything unseen becomes `[UNK]`. A character-level model has no unknowns but makes sequences about 4-5× longer, which hurts attention cost (quadratic in length) and makes the model learn spelling from scratch. Subword vocabularies of 30k to 100k units give frequent words a single token and split rare words into meaningful pieces. English text averages about 4 characters, or roughly 0.75 words, per token with typical LLM tokenisers.

## BPE and WordPiece

**Byte-pair encoding** (Sennrich et al., 2016) learns a merge list from a corpus:

1. Start with each word as a sequence of characters (or bytes), with word counts.
2. Count every adjacent symbol pair, weighted by word frequency.
3. Merge the most frequent pair into a new symbol, record the merge.
4. Repeat until the vocabulary reaches the target size.

Encoding a new word replays the merges in the order they were learned.

<div class="ex">
<div class="lab">Worked example</div>

**the first BPE merges**

Corpus: low ×5, lower ×2, newest ×6, widest ×3, with `_` marking word end.

Pair counts include e s = 9, s t = 9, t _ = 9, w e = 8, l o = 7, o w = 7. Three pairs tie at 9; with first-seen tie-breaking the first merge is e+s → es.

Then es+t → est (9) and est+_ → est_ (9), so "newest" and "widest" share a suffix token. Next come l+o and lo+w (7 each), giving the frequent stem "low".

</div>

```python
from collections import Counter
def bpe_step(words):                          # words: {tuple_of_symbols: count}
    pairs = Counter()
    for w, c in words.items():
        for a, b in zip(w, w[1:]): pairs[a, b] += c
    best = max(pairs, key=pairs.get)          # ties: first seen wins
    merged = {}
    for w, c in words.items():
        out, i = [], 0
        while i < len(w):
            if i + 1 < len(w) and (w[i], w[i + 1]) == best: out.append(w[i] + w[i + 1]); i += 2
            else: out.append(w[i]); i += 1
        merged[tuple(out)] = c
    return best, merged
```

**WordPiece** (BERT) differs in two ways. Training picks the pair that most increases corpus likelihood, roughly $\text{count}(ab) / (\text{count}(a)\,\text{count}(b))$, which favours pairs whose parts rarely appear apart. Encoding is greedy longest-match-first: take the longest vocabulary entry that prefixes the word, mark continuation pieces with `##`, repeat. "unaffable" becomes `un ##aff ##able`. If some remainder matches nothing, the whole word becomes `[UNK]`.

**Byte-level BPE** (GPT-2 onward) runs BPE over UTF-8 bytes rather than characters. The base vocabulary is the 256 byte values, so every string, including emoji and any script, has an encoding and there is no unknown token; rare characters simply cost several tokens. SentencePiece's unigram model is a third option, which prunes a large candidate vocabulary by likelihood and treats whitespace as an ordinary symbol, which suits languages without spaces.

Keep character offsets for every token. Token-level tasks (lesson 06.05) need them to map predictions back to the original text.

<div class="take">
<div class="lab">Key takeaway</div>

tokenisation is part of the model. Normalise deliberately, pre-split with a regex, and use a subword tokeniser so vocabulary stays bounded and unknown words decompose into known pieces; byte-level BPE removes unknowns entirely.

</div>
