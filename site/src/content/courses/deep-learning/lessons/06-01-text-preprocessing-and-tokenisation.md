---
id: "06.01"
course: "deep-learning"
module: "06"
slug: "text-preprocessing-and-tokenisation"
title: "Text preprocessing and tokenisation"
keyIdeas: ["Every NLP model starts by mapping text to integer ids: normalise, pre-tokenise, split into subwords, look up ids. Each step is a modelling decision, and the model is tied to exactly the pipeline it was trained with.", "Unicode normalisation (NFC, NFKC) makes visually identical strings byte-identical; lowercasing and accent stripping are lossy choices that help some tasks (small-data classification, search) and hurt others (named entities, accented languages).", "Sentence splitting on full stops fails on abbreviations, decimals and initials. Use rules with exception lists or a trained splitter such as Punkt, and test on your own domain.", "Words leave a long Zipf tail of unknown words, and characters make sequences several times longer. Subword vocabularies of tens of thousands of entries give frequent words one token and build rare words from known pieces.", "BPE learns an ordered list of merges by pair frequency and encodes by replaying them; WordPiece merges by a likelihood score and encodes by greedy longest match with `##` continuations, falling back to `[UNK]` for the whole word.", "Byte-level BPE starts from the 256 byte values, so any string can be encoded and there is no unknown token, but scripts the tokeniser rarely saw cost several tokens per character."]
question: "Your team trained its own WordPiece tokeniser and a classifier on clean English support tickets. In production, accuracy drops on tickets pasted from PDFs (with ligatures like 'ﬁ' and full-width characters) and on tickets containing emoji and product codes such as 'XR-7b'. Logs show many [UNK] ids, and the same visible word sometimes maps to different ids. Explain each symptom, say what you would check first, and propose fixes. Would switching to a byte-level BPE tokeniser solve everything?"
answer: "Different ids for the same visible word: the text is not Unicode-normalised. A PDF extractor produces ligatures ('ﬁ' as one code point), full-width letters or combining accents, which are different code points from the clean training text, so the tokeniser splits them differently. Fix: apply NFKC normalisation (plus zero-width character removal and whitespace collapsing) and make sure the exact same normalisation runs at training and serving time; ideally ship the tokeniser and its normaliser as one artefact with the model and add a test that known strings map to known ids. Many [UNK]s: WordPiece encodes by greedy longest match from a base vocabulary of characters seen in training. An emoji (or any unseen character) matches no vocabulary entry, so the whole whitespace-delimited word containing it becomes [UNK] and its information is erased. Product codes may also shatter or hit unseen characters. Check first: measure the [UNK] rate and fertility (tokens per word) on a sample of production tickets and eyeball the splits of the most common product codes; also diff the ids produced by the training and serving pipelines on the same strings. Fixes: normalise, add missing characters (or common emoji and code prefixes) to the vocabulary and retrain the tokeniser and model, or pre-extract product codes with a regex into a placeholder or a separate feature. Byte-level BPE would remove [UNK] entirely, because its base vocabulary is the 256 byte values and any string encodes at worst byte by byte; an emoji arrives as up to four byte tokens. But it would not fix inconsistent normalisation (two spellings still give different byte sequences), rare scripts and symbols would cost several tokens each, and changing tokeniser means retraining at least the embedding layer, since every id changes meaning."
sources: [{"title": "Neural Machine Translation of Rare Words with Subword Units (Sennrich et al., 2016)", "url": "https://arxiv.org/abs/1508.07909", "note": "The BPE-for-NLP paper; the merge algorithm fits in a dozen lines in section 3.2."}, {"title": "Hugging Face LLM Course, chapter 6: Tokenizers", "url": "https://huggingface.co/learn/llm-course/chapter6/1", "note": "Normalisation, pre-tokenisation, and BPE, WordPiece and Unigram side by side with code."}, {"title": "SentencePiece (Kudo and Richardson, 2018)", "url": "https://arxiv.org/abs/1808.06226", "note": "Language-independent subword tokenisation treating whitespace as a symbol."}, {"title": "Python unicodedata", "url": "https://docs.python.org/3/library/unicodedata.html", "note": "NFC, NFD, NFKC and NFKD normalisation forms and what each folds."}]
practice: ["byte-pair-encoding", "subword-tokenization", "text-normalization", "sentence-segmentation", "regex-patterns-for-nlp"]
tags: ["tokenisation", "BPE", "WordPiece"]
tools: ["Python"]
---

## What you will learn

- What happens to a string before a network sees it: normalise, split, cut into subwords, look up ids.
- How Unicode normalisation, lowercasing and whitespace cleaning work, and what each one throws away.
- How regex pre-tokenisation and sentence splitting work, and the classic ways they break.
- Why modern models use **subwords** rather than whole words or single characters, measured on real text.
- How to train byte-pair encoding (BPE) by hand, how WordPiece and byte-level BPE differ, and how to check a tokeniser.

*Before you start:* you should be comfortable with Python strings, lists and dictionaries, and know what an embedding layer does (it turns an integer id into a vector; module 05 used one in every sequence model). No maths beyond counting is needed until "Going deeper".

## Where this module starts

Module 05 built networks that read sequences of vectors. Those networks never see letters: they see integers, one per **token**, which an embedding table turns into vectors. This module is about text. It starts with the step that produces those integers (this lesson), then covers counting models (06.02, 06.03), word embeddings (06.04), token-level tasks (06.05), pretrained encoders (06.06), and decoding and evaluation (06.07).

The whole lesson follows one pipeline:

~~~dg-flow Each stage is a choice the model is then locked into: the same text must go through the same stages at training and serving time.
Raw text -(NFKC, lowercase?)-> Normalised -(regex split)-> Pre-tokens -(BPE / WordPiece)-> *Subwords -(vocab lookup)-> Ids + offsets
~~~

## The idea in one picture

Take one sentence: "The tokeniser splits unbelievably long words." It has 45 characters and 6 words. Here are three ways to cut it into tokens. The subword row is the real output of a byte-level BPE tokeniser with 8,256 entries, trained for this lesson on about 530,000 characters of English (the docstrings of Python's standard library).

| Granularity | Tokens produced | Count
| --- | --- | --- |
| Whole words | `The` `tokeniser` `splits` `unbelievably` `long` `words` `.` | 7
| Characters | `T` `h` `e` `␣` `t` `o` `k` ... `s` `.` | 45
| Subwords (BPE) | `The` `␣token` `is` `er` `␣splits` `␣un` `be` `li` `e` `v` `ably` `␣long` `␣words` `.` | 14

(`␣` marks a space kept inside the token.) Common words stay whole; the rarer "unbelievably" breaks into pieces seen often, such as `un` and `ably`. Nothing is ever unknown: at worst a word falls apart into single bytes.

~~~dg-fig 06.01-tradeoff.svg
Tokens per word on held-out text against vocabulary size. Characters (left) need a tiny vocabulary but 6.6 tokens per word; whole words need one token each but 4.3% of new words are missing from the vocabulary. BPE slides between the two as it learns more merges.
~~~

A **vocabulary** is the fixed list of tokens a model knows, each with an integer id. A small vocabulary makes sequences long. A word vocabulary makes them short but leaves holes: any word not in the list becomes the **unknown token** `[UNK]` and its meaning is gone. **Subword tokenisation** picks a point in between: here, 8,256 entries and 1.70 tokens per word (counting all tokens, including spaces and punctuation, divided by the number of words).

## Step 1: normalise the characters

Computers store text as **code points** (Unicode's numbered characters), and the same visible text can have several code-point spellings. "café" can end in one code point, `é` (U+00E9), or in two, `e` followed by a combining accent (U+0301). The two strings look identical on screen but compare as different, so without care they would get different token ids.

**Unicode normalisation** rewrites text into one canonical spelling:

- **NFC** composes letter-plus-accent pairs into single code points where possible. Both spellings of "café" become the 4-code-point form.
- **NFKC** does that and also folds *compatibility* characters: the ligature "ﬁ" becomes "fi", full-width "ＡＢＣ" becomes "ABC", superscript "x²" becomes "x2". It is the usual choice for model input. Note it is lossy: x² and x2 now look the same.

Text from PDFs and web pages is full of these variants, plus invisible characters such as the zero-width space (U+200B). In Python, `unicodedata.normalize("NFKC", s)`, then deleting zero-width characters and collapsing whitespace with `re.sub(r"\s+", " ", s)`, covers most of it.

Other normalisation steps are real modelling choices, each with a cost:

- **Lowercasing** merges "Apple" and "apple". It helps small-data classifiers and hurts named-entity recognition, where capitals are a strong clue. BERT is published in cased and uncased versions for this reason.
- **Accent stripping** helps English search ("resume" finds "résumé") and destroys meaning in French or Vietnamese.
- **Whitespace collapsing** is safe for prose and wrong for code, where indentation matters.

## Step 2: split into sentences (when you need them)

Translation, summarisation and building bounded-length training examples work sentence by sentence. Splitting after `.`, `!` or `?` followed by a space turns "Dr. Smith paid $3.50 for it. Then he left! U.S. troops arrived." into five pieces, including the fragments "Dr." and "U.S.". The decimal survives only because no space follows its point.

Practical splitters combine a candidate rule (full stop, whitespace, capital letter) with a list of exceptions (Dr., e.g., U.S.). The **Punkt** algorithm learns the abbreviations from unlabelled text: they are short and almost always followed by a full stop. Legal citations, chat messages and clinical notes break general-purpose splitters in different ways, so test on your own data.

## Step 3: pre-tokenise with a regular expression

Before subword splitting, most tokenisers cut the text into rough chunks: runs of letters, runs of digits, runs of punctuation. This **pre-tokenisation** guarantees that later merges never glue the end of one word to the start of the next. GPT-2's pattern also splits off English contractions (`'s`, `'re`) and keeps a leading space attached to each word, so " the" and "the" are different tokens and the original spacing can be rebuilt exactly. A simplified version:

```python
PAT = re.compile(r''''s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+''')
PAT.findall("We're testing 3.14 tokenisers!")
# ['We', "'re", ' testing', ' 3', '.', '14', ' tokenisers', '!']
```

Read it as alternatives: a contraction, or optional space plus letters (`[^\W\d_]` is "a word character that is not a digit or underscore"), digits, symbols, or whitespace. Notice "3.14" became three chunks.

Regexes also extract URLs, e-mail addresses, @mentions and product codes before later steps chop them apart.

## Step 4: why words and characters both fail

Why not just use whole words as tokens? Because word frequencies are extremely uneven.

~~~dg-fig 06.01-zipf.svg
Left: on a log-log plot, word counts fall almost in a straight line with rank, and 1,794 of the 5,229 distinct words appear only once. Right: even a vocabulary holding every training word misses 4.3% of the words in new text.
~~~

The left panel shows **Zipf's law**: a word's count is roughly proportional to one over its rank. A few words ("the", "of") are everywhere; a long tail of names, typos and inflections appears once or twice. On the right, keeping the 1,000 most common words leaves 16.9% of new words unknown, and keeping all 5,229 still leaves 4.3%. On web-scale text the tail never ends.

Characters fix the unknown-word problem (our corpus has only 103 distinct characters), but every word costs about 6.6 tokens. Self-attention cost grows with the square of sequence length, and the model must learn to spell before it learns meaning.

Subwords give frequent words one token and build rare words from frequent pieces. Production vocabularies hold 30,000 to 100,000 entries; as published values, BERT-base uses 30,522 WordPiece tokens and GPT-2 uses 50,257 byte-level BPE tokens. A common rule of thumb for English is about 4 characters, or 0.75 words, per token.

## Step 5: train BPE by counting pairs

**Byte-pair encoding (BPE)**, adapted to NLP by Sennrich et al. (2016), builds the vocabulary bottom up. Each **merge** glues one frequent pair of adjacent symbols into a new symbol:

1. Count the words in the training corpus. Write each word as a sequence of characters (or bytes), with an end-of-word marker `_`.
2. Count every pair of adjacent symbols, weighting by how often the word occurs.
3. Merge the most frequent pair everywhere into one new symbol, and record the merge.
4. Repeat until the vocabulary reaches the target size (base symbols plus number of merges).

The output is an ordered **merge list**.

<div class="ex">
<div class="lab">Worked example</div>

**the first BPE merges, by hand**

Corpus: "low" ×5, "lower" ×2, "newest" ×6, "widest" ×3. As symbols: `l o w _`, `l o w e r _`, `n e w e s t _`, `w i d e s t _`. Total: $4 \times 5 + 6 \times 2 + 7 \times 6 + 7 \times 3 = 20 + 12 + 42 + 21 = 95$ symbols.

Pair counts: `e s` appears in newest (6) and widest (3), so $6 + 3 = 9$. Likewise `s t` $= 9$ and `t _` $= 9$. Next, `w e` appears in lower (2) and newest (6), $= 8$. Then `l o` $= 5 + 2 = 7$ and `o w` $= 7$.

Merge 1: three pairs tie at 9. With "first seen wins" tie-breaking, `e s` becomes `es`. Newest and widest each lose one symbol: $95 - 6 - 3 = 86$.

Merge 2: `es t` now counts 9, so `est` (86 → 77). Merge 3: `est _` counts 9, so `est_` (77 → 68). "newest" and "widest" now share one suffix token.

Merges 4 and 5: `l o` (7) and then `lo w` (7) give the stem `low` (68 → 61 → 54).

Encode the unseen word "lowest" by replaying the merges in order: `l o w e s t _` → `l o w es t _` → `l o w est _` → `l o w est_` → `lo w est_` → `low est_`. Two known tokens, no `[UNK]`.

</div>

~~~dg-fig 06.01-bpe.svg
Left: pair counts on the toy corpus at the start; three pairs tie at 9 and the first one seen is merged. Right: the number of symbols needed to write the corpus falls from 95 to 36 over eight merges. Compression is what BPE optimises, one greedy step at a time.
~~~

Encoding new text never re-counts; it replays the merges in learned order, so the merge list is part of the tokeniser.

## Step 6: WordPiece, byte-level BPE and unigram

Three variants dominate in practice.

**WordPiece** (BERT) changes both training and encoding. Training picks the pair that most increases the likelihood of the corpus, which works out to roughly $\text{score}(a, b) = \frac{\text{count}(ab)}{\text{count}(a)\,\text{count}(b)}$, where $\text{count}(ab)$ is how often the pair occurs and the denominator is how often each part occurs at all. The score favours pairs whose parts rarely appear apart. Encoding is **greedy longest-match-first**: take the longest vocabulary entry that starts the word, then the longest `##`-marked continuation that starts the rest, and so on. "unaffable" becomes `un ##aff ##able`. If some remainder matches nothing, the *whole word* becomes `[UNK]`.

**Byte-level BPE** (GPT-2 and most later generative models) runs BPE over UTF-8 bytes instead of characters. The base vocabulary is the 256 possible byte values, so every string in every script, including emoji, has an encoding and there is no unknown token at all. Rare characters simply cost several tokens.

**Unigram** (Kudo, 2018), in the SentencePiece library, starts from a large candidate vocabulary and prunes the pieces whose removal hurts corpus likelihood least. SentencePiece treats whitespace as an ordinary symbol (`▁`), so it suits languages written without spaces, such as Chinese and Japanese.

The final step is lookup: each token maps to its row number in the vocabulary, the input to the embedding layer. Also keep **offsets**, each token's start and end character in the original text; token-level tasks (06.05) need them to map predictions back to what the user wrote.

## Tokenisation in code

BPE and WordPiece in plain Python. Production libraries (Hugging Face `tokenizers`, `tiktoken`, SentencePiece) do the same in fast native code.

```python
from collections import Counter

def train_bpe(word_counts, n_merges):
    words = {tuple(w) + ("_",): c for w, c in word_counts.items()}
    merges = []
    for _ in range(n_merges):
        pairs = Counter()
        for w, c in words.items():
            for p in zip(w, w[1:]): pairs[p] += c
        if not pairs: break
        best = max(pairs, key=pairs.get)              # ties: first seen wins
        merges.append(best)
        new = {}
        for w, c in words.items():
            out, i = [], 0
            while i < len(w):
                if i + 1 < len(w) and (w[i], w[i + 1]) == best:
                    out.append(w[i] + w[i + 1]); i += 2
                else:
                    out.append(w[i]); i += 1
            new[tuple(out)] = c
        words = new
    return merges

def encode(word, merges):                             # replay merges in learned order
    sym = list(word) + ["_"]
    for a, b in merges:
        i = 0
        while i < len(sym) - 1:
            if sym[i] == a and sym[i + 1] == b: sym[i:i + 2] = [a + b]
            else: i += 1
    return sym

merges = train_bpe({"low": 5, "lower": 2, "newest": 6, "widest": 3}, 10)
print(merges[:5])  # [('e','s'), ('es','t'), ('est','_'), ('l','o'), ('lo','w')]
print(encode("lowest", merges))   # ['low', 'est_']
print(encode("slowly", merges))   # ['s', 'low', 'l', 'y', '_']

def wordpiece(word, vocab):                           # greedy longest match
    pieces, start = [], 0
    while start < len(word):
        end = len(word)
        while end > start:
            piece = word[start:end] if start == 0 else "##" + word[start:end]
            if piece in vocab: break
            end -= 1
        if end == start: return ["[UNK]"]             # nothing matched: whole word lost
        pieces.append(piece); start = end
    return pieces

vocab = {"un", "##aff", "##able", "##a", "##ble", "play", "##ing", "[UNK]"}
print(wordpiece("unaffable", vocab))  # ['un', '##aff', '##able']
print(wordpiece("player", vocab))     # ['[UNK]']  (no piece for "er")

print(list("🙂".encode("utf-8")))   # [240, 159, 153, 130]: 4 byte tokens, never [UNK]
```

WordPiece loses "player" entirely although "play" is known; BPE's "slowly" degrades gracefully into known pieces.

## Checking a tokeniser

A tokeniser can be "working" and still be quietly wrong. Run these checks on a sample of your own data:

- **Round trip.** `decode(encode(text)) == text` (after normalisation). A failure points to a lossy normaliser or a broken pre-tokeniser.
- **Unknown rate.** The share of `[UNK]` ids should be near zero; a few percent means characters the vocabulary never saw.
- **Fertility.** Tokens per word. Much higher than on the training domain means your effective context window shrinks.
- **Eyeball the splits** of the 50 most frequent domain terms. If each shatters into five pieces, consider a domain tokeniser.

Fertility differs most sharply across languages. Our tokeniser was trained only on English, and it shows:

~~~dg-fig 06.01-scripts.svg
The same sentence in seven languages. Our English-trained byte-level BPE needs 19 tokens for English and 26 to 27 for French and German, but 85 to 112 for Russian, Greek and Hindi: it has no merges for those scripts and falls back to one token per UTF-8 byte.
~~~

A Cyrillic or Greek letter takes 2 bytes in UTF-8 and a Devanagari or Chinese character 3, so with no learned merges tokens equal bytes. Multilingual tokenisers narrow this gap but rarely close it, so users of some languages pay more per sentence.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| Same word, different ids; accuracy drops on text pasted from PDFs | No Unicode normalisation; ligatures, full-width or combining characters | Apply NFKC (or the model's own normaliser) in both training and serving
| Model works in the notebook, fails in production | Serving pipeline differs (lowercased, stripped, different tokeniser version) | Ship the tokeniser files with the model; test that ids match
| Many `[UNK]` for emoji, rare scripts or symbols | WordPiece or word vocabulary lacks those characters | Byte-level BPE, or add characters to the base vocabulary
| Sentences cut at "Dr." or "U.S." | Naive full-stop splitting | Exception lists or a trained splitter such as Punkt; test on your domain
| Server hangs on one odd input | Regex with nested quantifiers backtracking | Rewrite the pattern; set input length limits or timeouts
| Predicted spans land on the wrong characters | Token offsets lost or computed on normalised text | Keep offsets into the original string

The regex row deserves a picture, because it surprises people:

~~~dg-fig 06.01-regex.svg
Matching `(a+)+$` against "aaa...a!" (which almost matches) takes time that roughly doubles with each extra "a": under 1 ms at 14 characters and close to 2 seconds at 24, measured with Python's `re` on this machine. The equivalent `a+$` stays under a microsecond.
~~~

The nested pattern has exponentially many ways to divide the a's between the inner and outer `+`, and a backtracking engine tries them all. Avoid nested quantifiers over the same characters, and prefer explicit character classes or lazy `.+?` over greedy `.+`, which swallows too much.

## Going deeper

**What BPE optimises.** Each merge reduces the symbols needed to write the corpus by exactly the merged pair's count (right panel of the BPE figure), so BPE is greedy compression. It is not guaranteed to find the best-compressing vocabulary of a given size. Recounting all pairs after every merge costs (merges) × (corpus size); real implementations keep an index from each pair to the words containing it and update only those. With that trick, 8,000 merges over our corpus took about 5 seconds in pure Python.

**WordPiece's score is a likelihood gain.** If symbols are drawn independently, merging $a$ and $b$ changes corpus log-likelihood by roughly $\text{count}(ab)\,\big(\log p(ab) - \log p(a) - \log p(b)\big)$: pairs that co-occur more than chance predicts win. That is pointwise mutual information, which returns in 06.04.

**Unigram tokenisation is probabilistic.** Each segmentation gets the product of its pieces' probabilities, and encoding picks the most likely one by dynamic programming (Viterbi). Sampling other segmentations in training, **subword regularisation**, makes models robust to spelling noise; **BPE-dropout** (Provilkov et al., 2020) does the same for BPE by randomly skipping merges.

**The tokeniser shapes what the model can learn.** Irregular number chunks make arithmetic harder, which is why some models split every digit. "Hello" and " Hello" are different tokens. Spelling questions are hard because the model sees ids, not letters. And changing the tokeniser means retraining at least the embedding layer, since every id changes meaning.

## Key terms

| Term | Meaning
| --- | --- |
| Token | One unit of text the model sees, represented by an integer id
| Vocabulary | The fixed list of tokens a model knows; its size sets the embedding table's rows
| Unicode normalisation (NFC / NFKC) | Rewriting text into one canonical code-point spelling; NFKC also folds look-alike characters
| Pre-tokenisation | A regex split into words, numbers and punctuation before subword splitting
| Byte-pair encoding (BPE) | Learns an ordered list of merges of the most frequent adjacent pairs
| WordPiece | BERT's subword method: likelihood-based merges, greedy longest-match encoding, `##` continuations
| Byte-level BPE | BPE over UTF-8 bytes; 256 base tokens, so no unknown token is possible
| `[UNK]` | The unknown token: a placeholder that erases whatever it replaced
| Fertility | Average number of tokens per word; higher means longer sequences
| Offsets | Character start and end positions of each token in the original text

<div class="take">
<div class="lab">Key takeaway</div>

tokenisation is part of the model. Normalise deliberately, pre-split with a regex, and use a subword tokeniser so the vocabulary stays bounded and rare words decompose into known pieces; byte-level BPE removes unknown tokens entirely, and the exact same pipeline must run at training and serving time.

</div>
