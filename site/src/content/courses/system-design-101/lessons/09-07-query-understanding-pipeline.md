---
id: "09.07"
course: "system-design-101"
module: "09"
slug: "query-understanding-pipeline"
title: "Query understanding pipeline"
keyIdeas: ["Query understanding works out what the user means before retrieval.", "Steps: normalize (lowercase, Unicode, punctuation) → tokenize → spell-correct → stem or lemmatize → recognize entities and phrases (\"new york\") → expand with synonyms → classify intent and category → rewrite into an engine query with filters.", "Example: \"cheap red nike shoes size 10\" → category: shoes, brand: Nike, colour: red, size: 10, sort: price.", "Getting the query right often helps more than improving the ranker."]
question: "List the steps between a user's raw query and the index lookup."
answer: "Normalise characters (lower-case, fold accents, unify Unicode and punctuation), tokenise (handling hyphens, symbols and languages without spaces), apply stemming or lemmatisation, correct spelling, segment the query into phrases such as \"new york\", recognise entities like brands, categories and price constraints, and classify the query's intent and category. Then rewrite it: expand synonyms, restrict parts to fields such as brand, turn constraints like \"under \\$100\" into structured filters, and relax it by dropping the least important term if it would match nothing. The resulting structured query is what is looked up in the index."
sources: [{"title": "Daniel Tunkelang: Query Understanding", "url": "https://queryunderstanding.com/", "note": "Free series: tokenization, spelling, rewriting, intent."}]
---

## Why query understanding comes first

**Query understanding** is everything that happens to a query **before** retrieval and ranking: working out what the searcher means from what they typed. It's easy to underrate, because most engineering attention goes to ranking. But ranking can only order documents that retrieval found, and retrieval can only find what the interpreted query asks for. If the engine thinks "apple" means fruit when the user wants a phone charger, no ranking model can fix the result.

Think of search as a communication channel. The searcher has an **intent** in their head ("cheap Nike trainers for running, under \$100"). They **encode** it in a few words, quickly, often misspelled ("nike runing shoes under 100"). The engine must **decode** it. Queries are short (two to three words on average in web and e-commerce search), ambiguous, and full of noise. Query understanding is the decoder.

The payoff is large. In e-commerce, a meaningful share of queries (often 10 to 20%) return zero or poor results because of misspellings, vocabulary mismatch or over-specific queries. Fixing interpretation for those queries usually moves revenue more than any ranking change.

## The stack, from characters to meaning

Query understanding is a stack of layers. Each one works on the output of the one below, from raw characters up to the searcher's intent.

~~~dg-layers The query understanding stack, from raw characters up to the searcher's intent.
Characters | normalisation (case, accents, Unicode), tokenisation
Words | stemming and lemmatisation, spelling correction
Phrases | segmentation ("new york" is one unit), entity recognition
*Meaning | synonyms, taxonomies, classification of intent
Rewriting | relaxation, expansion, field restriction
Interaction | autocomplete, context, personalisation, conversation
~~~

### 1. Character filtering and normalisation

Make equivalent strings identical. Lower-case ("iPhone" to "iphone"). Fold accents ("café" to "cafe", "Zürich" to "zurich"). Apply **Unicode normalisation** so that "é" typed as one code point and "e" plus a combining accent compare equal. Unify punctuation and quotes, full-width and half-width characters (common in Japanese input), and strip invisible control characters.

### 2. Tokenisation

Split the string into **tokens**. It's harder than splitting on spaces:

- "wi-fi", "e-mail": one token or two? Index both forms.
- "C++", "C#", ".NET", "AT&T": naive tokenisers strip the symbols and lose the meaning.
- "iphone15" versus "iphone 15": split letters from digits.
- Chinese, Japanese and Thai don't use spaces between words, so tokenisation needs a dictionary or a statistical segmenter, or the engine indexes overlapping character pairs (bigrams).

### 3. Stemming and lemmatisation

Map word variants to a common form so they match.

- **Stemming** chops suffixes with rules. The Porter stemmer turns "running" and "runs" into "run", and "connection" and "connected" into "connect". Fast, language-specific, and sometimes wrong: "university" and "universe" both become "univers".
- **Lemmatisation** uses a dictionary and part of speech to find the true base form: "ran" to "run", "better" to "good", "mice" to "mouse". More accurate, slower.
Over-stemming merges words that shouldn't match ("news" to "new"); under-stemming misses words that should. E-commerce engines often prefer light stemming (just plurals) because "shoes" to "shoe" is safe but aggressive stemming causes odd matches.

### 4. Spelling correction

Fix typos such as "runing" or "iphnoe" using edit distance plus word frequency and context (lesson 09.15). Decide whether to correct silently, search both forms, or ask "Did you mean...?".

### 5. Query segmentation

Decide which words belong together as a unit. "new york times square" should be "new york" + "times square", not "new" + "york times" + "square". Segmentation uses statistics from documents and query logs: how often words appear together compared with how often you'd expect by chance (measures such as **pointwise mutual information**). Segments then become phrase queries or boosted proximity matches.

### 6. Entity recognition

Tag parts of the query with types: brands ("nike"), places ("leeds"), people, categories ("running shoes"), attributes ("red", "size 9"), and numeric constraints ("under \$100", "2 bedroom"). Usually done with a dictionary of known values from your catalogue (all brands, all colours) plus a trained sequence model for ambiguous cases.

### 7. Query classification

Map the whole query to a category or intent:

- **Navigational**: the user wants a specific page or product ("nike login", "iphone 15 pro 256gb blue").
- **Informational**: they want to learn ("how to clean suede").
- **Transactional or browse**: they want to shop a category ("running shoes").
Category classification ("this query belongs to Footwear > Running") lets the engine filter or boost the right department. It's usually a text classifier trained on which categories users clicked or bought from for each query.

### 8. Synonyms and taxonomies

Know that "sofa" and "couch" mean the same, that "tv" equals "television", that "sneakers" and "trainers" are regional synonyms, and that "sneakers" is a kind of "shoes" (a **hypernym** relation in a taxonomy). Synonyms can be applied at index time (store both forms) or at query time (expand the query); query time is easier to change without reindexing. Be careful with one-directional relations: "shoes" should not be expanded to every kind of shoe, but "sneakers" can safely include "trainers".

### 9. Query rewriting

Change the query to retrieve better results:

- **Expansion**: add synonyms or related terms (lesson 09.16) to increase recall.
- **Relaxation**: if nothing matches all the terms, drop the least important one. "nike air max 90 red size 9 waterproof" with zero results might drop "waterproof" and say so.
- **Field restriction**: search "nike" only in the brand field rather than everywhere, so a product description that mentions "unlike nike" doesn't match.
- **Filter extraction**: turn "under \$100" into a structured `price < 100` filter instead of text.

### 10. Autocomplete, context, personalisation and conversation

Guide users while they type (lesson 09.14), so many queries are well-formed before they're submitted. Use **context**: the page they're on, their location, the previous query in the session ("cheaper ones"). **Personalise** carefully: a user who always buys women's clothing may want "jeans" interpreted that way. And when the engine is unsure, **ask**: facets, "Did you mean Apple the brand or apples the fruit?", or a conversational follow-up.

## A worked pass

~~~dg-flow One query moving through the stack before it ever touches the index.
[Text] "Nike runing shoes under $100" -(normalise, tokenise)-> nike runing shoes under 100 -(spell, stem)-> nike run shoe
[Meaning] nike run shoe -(entities)-> brand nike, price under 100 -(rewrite)-> *Structured query
~~~

<div class="ex">
<div class="lab">Worked example</div>

**"Nike runing shoes under \$100"**

Normalise: "nike runing shoes under \$100". Tokenise: nike, runing, shoes, under, \$100. Spell-correct: "runing" is not in the vocabulary; the closest frequent word at edit distance 1 is "running". Stem lightly: running → run, shoes → shoe.

Segment: "running shoes" is a known category phrase. Recognise entities: "nike" matches the brand dictionary, "under \$100" matches a price pattern. Classify: category Footwear > Running, high confidence.

Rewrite into a structured query: `brand = nike AND category = running-shoes AND price < 100`, with free text "run shoe" scored by BM25 against title and description, and "trainers" added as a synonym. Only now does the engine look up postings and rank. Without the price and brand extraction, BM25 would have tried to match the literal tokens "under" and "100" in product text, and returned products whose descriptions mention "100% cotton".

</div>

## Latency budget and architecture

All of this must happen in a few milliseconds. A typical budget for an e-commerce search request is 100 to 200 ms total, of which query understanding gets perhaps 5 to 20 ms. That means:

- Dictionaries (brands, synonyms, categories) loaded in memory, refreshed periodically.
- Small, fast models (a classifier that runs in under a millisecond) on the hot path; larger models (an LLM rewriting the query) only for the long tail, or precomputed offline for frequent queries.
- **Caching by query string**: the most frequent 10,000 queries make up a large share of traffic, so their interpretations can be precomputed and cached.

~~~dg-seq The query understanding service sits in front of the search engine and returns a structured query within a small latency budget.
User > Search API : "nike runing shoes under $100"
Search API > QU service : interpret
QU service > QU service : cache miss: normalise, spell, tag
QU service -> Search API : brand, category, price filter, text
Search API > Search engine : structured query
Search engine -> Search API : top 48 products
Search API -> User : results + "showing results for running"
~~~

<div class="ex">
<div class="lab">Worked example</div>

**when relaxation saves a query**

"red nike air max 90 size 9 waterproof" returns zero results: no product has all those attributes. The rewriter ranks the constraints by importance learned from logs (brand and model matter most, "waterproof" least), drops "waterproof", and finds 14 products. The page says "No exact matches for waterproof. Showing red Nike Air Max 90 in size 9." A zero-result page becomes a sale. Logging which constraint was dropped also tells the merchandising team that customers want a waterproof version.

</div>

## How to judge it

The measure of success is **query performance**: whether queries lead to successful outcomes such as clicks, add-to-carts or purchases, and whether zero-result and reformulation rates go down. Evaluate each step by whether it improves those, not by whether it's linguistically correct. An aggressive stemmer that is "more correct" linguistically may hurt conversions. Run changes as A/B tests and track per-query-segment metrics (head, torso and tail queries behave differently).

## Common mistakes

- Different analysis at index and query time, so terms don't line up.
- Treating every token as text, so "under", "\$100" and "size 9" are matched as words instead of becoming filters.
- Over-expanding synonyms, so "apple" matches "fruit" for a phone accessories shop.
- Correcting valid brand names or product codes as typos ("Asics" to "basics").
- Shipping changes without measuring zero-result rate and conversion per query segment.

## In an interview

List the steps in order quickly: normalise, tokenise, stem or lemmatise, spell-correct, segment, recognise entities, classify intent, rewrite (expand, restrict to fields, relax). Then show judgement: keep the hot path cheap with in-memory dictionaries and caching, use a model only where it pays, and measure with zero-result rate and conversions.

<div class="take">
<div class="lab">Key takeaway</div>

before retrieval, a query is normalised, tokenised, stemmed, spell-checked, segmented into phrases, tagged with entities and intent, and rewritten (expanded, restricted to fields, relaxed). Retrieval and ranking can only work with the interpretation this produces, so it's judged by search outcomes, not linguistic purity.

</div>
