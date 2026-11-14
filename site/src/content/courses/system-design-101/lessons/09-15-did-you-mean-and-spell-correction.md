---
id: "09.15"
course: "system-design-101"
module: "09"
slug: "did-you-mean-and-spell-correction"
title: "Did-you-mean and spell correction"
keyIdeas: ["Spell correction finds the query the user probably meant.", "Generate candidates within a small edit distance (insertions, deletions, substitutions, transpositions) of the typed word.", "Rank candidates by probability: P(intended word) × P(typed | intended). Word frequency from a corpus or query logs gives the first; an error model gives the second.", "Edit distance alone ties many candidates; frequency picks the likely one (\"speling\" → \"spelling\", not \"spieling\").", "At scale, use k-gram indexes or SymSpell to generate candidates fast, and use query logs to learn common rewrites."]
question: "Why do you need both edit distance and word frequency to correct \"speling\"?"
answer: "Edit distance alone finds several known words one edit from \"speling\" (for example \"spelling\" and \"spewing\") and can't say which the user meant, while word frequency alone would pick very common words like \"the\" that look nothing like what was typed. In the model correction = argmax P(c) × P(w | c), edit distance acts as the error model, limiting candidates to plausible typos and preferring fewer edits, and frequency acts as the language model, picking the likeliest among the nearest candidates. So \"spelling\" wins because it is both one edit away and far more common than the alternatives."
sources: [{"title": "Peter Norvig: How to Write a Spelling Corrector", "url": "https://norvig.com/spell-correct.html", "note": "A working corrector in 21 lines."}, {"title": "IR book ch.3.3: Spelling correction", "url": "https://nlp.stanford.edu/IR-book/pdf/irbookonlinereading.pdf", "note": "Edit distance and k-gram indexes."}]
tags: ["Design"]
---

## Why search needs spelling correction

Around 10 to 15% of search queries contain a spelling mistake, and on mobile keyboards the share is higher. A misspelled query usually matches nothing, or matches the wrong things: "iphnoe case" returns zero products, "speling" returns pages that contain the same typo. Each of those is a lost search. A spelling corrector turns them back into successful searches, and it's one of the highest-value pieces of query understanding (lesson 09.07).

The problem looks like it needs a dictionary and some clever string matching. The cleaner way to think about it is **probability**: of all the words the user could have meant, which is most likely, given what they typed?

## The model behind spelling correction

Given a typed word w, choose the candidate correction c that is most probable:

`correction(w) = argmax over c of P(c | w)`

By Bayes' rule, P(c | w) = P(c) × P(w | c) / P(w). P(w) is the same for every candidate, so it drops out:

`correction(w) = argmax over c of P(c) × P(w | c)`

This splits the problem into three parts:

- **Candidate model**: which corrections c to consider at all. Usually, known words within a small edit distance of w.
- **Language model**, P(c): how common the word c is, regardless of the typo. Estimate it by counting words in a large text corpus; around a million words is enough for a decent start. In English text, "the" makes up about 7% of all words.
- **Error model**, P(w | c): how likely it is that someone meaning c would type w. A real one is learned from data about actual typos; a simple one just prefers fewer edits.

A working corrector built on this idea fits in about 20 lines of Python, and its reasoning answers this lesson's question directly.

## Generating candidates

An **edit** is one of four operations on a string:

- **Deletion**: remove one letter ("spelling" to "speling").
- **Transposition**: swap two adjacent letters ("the" to "teh").
- **Replacement**: change one letter to another.
- **Insertion**: add one letter.

The **edit distance** between two strings is the smallest number of edits that turns one into the other. (Counting transpositions as one edit gives the Damerau-Levenshtein distance; plain Levenshtein counts a swap as two replacements.)

For a word of length n, there are n deletions, n − 1 transpositions, 26n replacements and 26(n + 1) insertions: **54n + 25** strings at edit distance 1. For "speling" (n = 7) that's 403 strings, a few of them duplicates, and almost none are real words. Applying edits twice gives edit distance 2: about 403² ≈ 160,000 strings. That sounds like a lot, but generating them takes milliseconds, and filtering to **known words** (those in the vocabulary) leaves only a handful.

~~~dg-tree Candidates for "speling", grouped by edit distance. Only known words survive, and frequency picks among the nearest group.
speling (not a known word)
  Edit distance 1
    *spelling | insert l, very common
    spewing | replace l with w, rare
    spieling | insert i, rare
  Edit distance 2
    peeling | delete s, insert e
    sapling | insert a, delete e
~~~

### A simple error model: priority by distance

The simplest error model doesn't compute real probabilities. It uses a priority order:

1. The original word, if it's a known word.
2. Otherwise, known words at edit distance 1.
3. Otherwise, known words at edit distance 2.
4. Otherwise, the original word unchanged (don't invent a correction).

Within the first non-empty group, pick the word with the highest P(c). In other words: **choose the nearest correction, and when several are equally near, choose the most common**.

```
def correction(word):
    return max(candidates(word), key=P)

def candidates(word):
    return (known([word]) or known(edits1(word))
            or known(edits2(word)) or [word])
```

~~~dg-flow Candidate order: prefer the word itself if known, then known words one edit away, then two, then give up.
"speling" -(known word?)-> No -(edit distance 1)-> spelling, spewing -(pick most frequent)-> *spelling
~~~

## Why you need both parts

Now the question: why both edit distance and frequency?

- **Edit distance alone gives ties.** "speling" is one edit from "spelling" (insert l) and also one edit from "spewing" (replace l with w). Edit distance can't choose between them. For short words it's worse: "teh" is one edit from "the", "ten", "tea", "tech", "eh" and more.
- **Frequency alone is absurd.** "the" is the most common word in English. A frequency-only corrector would "correct" every unknown word to "the". Frequency says nothing about whether c resembles what was typed.
- **Together** they work: edit distance limits the candidates to plausible typos, and frequency picks the likeliest among them. "spelling" wins because it's both one edit away and far more common than "spewing". Similarly, "grnt" could be "grant" or "grunt" (both one insertion away), and the more frequent "grant" wins.

<div class="ex">
<div class="lab">Worked example</div>

**scoring candidates for "speling"**

Suppose in a 1-million-word corpus, "spelling" occurs 130 times and "spewing" 4 times. P(spelling) = 0.00013, P(spewing) = 0.000004. Both are at distance 1, so the simple error model treats them equally, and the corrector picks "spelling", about 30 times more probable.

Now a real error model. From typo data, dropping a doubled letter ("ll" to "l") is a very common mistake, say probability 0.01 for that position; typing "l" when you meant "w" (keys far apart) is rare, say 0.0001. Then P(c) × P(w | c) gives spelling: 0.00013 × 0.01 = 1.3 × 10⁻⁶, spewing: 0.000004 × 0.0001 = 4 × 10⁻¹⁰. The real error model makes the decision even clearer.

</div>

## How well the simple version works

A corrector built exactly this way, with word counts from about a million words of text, scored **75% of 270** test words correct at 41 words per second on a development set, and **68% of 400** at 35 words per second on a held-out test set. Not bad for 20 lines, and the failures point to every improvement real systems make:

- **Better language model**: more and more representative text. And **context**: bigram or trigram probabilities, so "their" versus "there" depends on the neighbouring words.
- **A learned error model**: some mistakes (adjacent keys on the keyboard, doubled letters, vowel swaps, phonetic spellings like "fone") are much more likely than others. Learn P(w | c) per edit type from pairs of typo and correction.
- **Unknown words in the dictionary**: if the language model doesn't contain the intended word, no correction can find it.
- **Larger distances** in some cases, and words that should be **split** ("runningshoes") or **joined** ("ear buds").

~~~dg-bars Rough share of typos by edit distance in real query logs. Most errors are a single edit, so distance 1 and 2 cover nearly everything.
Distance 1 | 80 | ~80% of typos
Distance 2 | 15 | ~15%
*Distance 3 or more | 5 | ~5%
~~~

## Spelling correction in search

Search engines adapt the model in several ways.

### Correct against your own vocabulary and logs

Don't use a generic dictionary. Product names, brands, slang and model numbers aren't in it, and generic word frequency is wrong for your domain. Build the language model from the **index vocabulary** (terms in your documents, weighted by document frequency) and from **query logs** (what other searchers typed and then didn't correct). The right correction is what other users meant. Query logs are also where the best error model comes from: a query followed seconds later by a similar query that succeeds is a free (typo, correction) pair.

### Find candidates fast with a k-gram index

Generating 160,000 strings per word is fine for one word but wasteful at thousands of queries a second, and edits2 gets expensive for long words. A **k-gram index** narrows candidates first:

1. Break every vocabulary term into overlapping character k-grams (usually bigrams or trigrams, with boundary markers: "\$bo", "bor", "ord", "rd\$").
2. Build an inverted index from each k-gram to the terms containing it.
3. For a misspelled word, break it into k-grams, look up their postings, and find terms that share many k-grams (by Jaccard overlap).
4. Compute exact edit distance only for those few terms.

~~~dg-flow A k-gram index narrows candidates before computing edit distance.
"bord" -(bigrams: bo, or, rd)-> Postings for each bigram -(most shared)-> *board, border, lord -(edit distance)-> board
~~~

Another fast approach is **symmetric delete** (used by the SymSpell library): precompute all deletions of every vocabulary word up to distance 2 and index them. At query time, generate only deletions of the input (n strings, not 54n) and look them up. It's very fast at the cost of a large precomputed table.

### Context-sensitive correction

Some errors are real words in the wrong place: "flew form Heathrow" has no misspelled word, but "form" should be "from". Only phrase context reveals it. Use the query log and a bigram language model: P("flew from heathrow") is much higher than P("flew form heathrow"), so suggest the change. Treat each word as having candidate alternatives and score whole query variants.

### How to present it

- **Auto-correct silently** when you're very confident and the original query returns nothing ("iphnoe" to "iphone").
- **Show results for the correction with a link back**: "Showing results for *iphone*. Search instead for *iphnoe*."
- **Suggest only**: "Did you mean *carrot*?" when the original returns reasonable results and confidence is moderate.
- **Never correct** exact identifiers, quoted phrases, or known brand names that look like typos ("Asics" is not "basics").

~~~dg-matrix How to act on a correction, by confidence and whether the original query found results.
Situation | Auto-correct | Show both | Suggest only
High confidence, 0 results | yes | partial | no
High confidence, some results | partial | yes | partial
Low confidence, 0 results | no | partial | yes
Low confidence, good results | no | no | yes
~~~

<div class="ex">
<div class="lab">Worked example</div>

**"acress"**

Candidates at distance 1 include "actress" (insert t), "across" (replace e with o), "access" (replace r with c), "caress" (swap a and c), "cress" (delete a) and "acres" (delete s). A pure frequency choice picks "across", the most common of them. A learned error model says omitting a "t" after "c" is a much more common typo than replacing "o" with "e", which pushes "actress" up. Context settles it: "a famous acress" makes "actress" far more likely under a bigram model, while "acress the river" makes "across" likely. This is why production systems learn the language model, the error model and context from data together.

</div>

## Common mistakes

- Correcting with edit distance alone, giving arbitrary picks among ties.
- Using a generic dictionary, so brands and product names are "corrected" into common words.
- Auto-correcting confidently when the original query had good results.
- Ignoring context, so real-word errors ("form" for "from") are never caught.
- Running edits2 over a huge vocabulary per request without a k-gram or symmetric-delete index.

## In an interview

State the noisy-channel model: pick c maximising P(c) × P(w | c). Candidates come from edit distance 1 and 2 filtered to the vocabulary; frequency (from your corpus and query logs) is the language model; edit distance, or a learned per-edit model, is the error model. Then mention a k-gram index for speed, context for real-word errors, and confidence-based presentation.

<div class="take">
<div class="lab">Key takeaway</div>

pick the correction c that maximises P(c) × P(w | c): edit distance generates and orders candidates (the error model), and word frequency chooses among them (the language model). Search engines build both models from their own vocabulary and query logs, use k-gram or symmetric-delete indexes to find candidates fast, and use context and confidence to decide how to present the correction.

</div>
