import math, re

META = dict(title='Capstone: a complete extractive RAG pipeline', tags=['NLP', 'Language Models'], diff='Hard', runtime='python',
            related=['chunk-text-with-overlap', 'tf-idf', 'rag-context-packing', 'validate-citations'])

STATEMENT = r"""
Wire the RAG primitives into one function: `rag_answer(docs, question, size, overlap, k, budget)`.

1. **Chunk.** `docs` maps doc id to text. Tokens are `text.split()`. Chunk each doc (in sorted doc-id order) into windows of `size` tokens whose starts advance by `size - overlap`, stopping once a window reaches the end. Chunk id is `f"{doc}#{n}"` with `n` counting from 0 per doc; chunk text is the window joined by spaces.
2. **Embed.** Terms are lowercase matches of `[a-z0-9]+`. Fit TF-IDF on the chunks: tf is the raw count, idf $= \ln\frac{1+N}{1+\mathrm{df}} + 1$, and each vector is L2-normalised. The question is embedded with the same idf (terms not in the vocabulary are ignored).
3. **Retrieve.** Score chunks by cosine similarity (dot product of the normalised vectors). Take the top `k` with score > 0, ties broken by chunk order.
4. **Pack.** Walk the retrieved chunks in rank order and keep each one whose token count fits in the remaining `budget`.
5. **Answer.** Split each packed chunk into sentences with `(?<=[.!?])\s+`. Question keywords are the distinct question terms that are in the vocabulary and not in `STOP` (given in the starter). Score a sentence by the sum of idf over the keywords it contains. Pick the highest-scoring sentence (ties: earlier packed chunk, then earlier sentence). The answer is that sentence plus `" [chunk_id]"`. If no sentence scores above 0, the answer is `"I don't know."`.

Return `{"retrieved": [ids], "packed": [ids], "answer": str}`.
"""

STARTER = '''STOP = {"a", "an", "the", "is", "was", "are", "were", "what", "when", "which", "who", "how", "many", "did", "does", "do", "of", "in", "on", "for", "to", "and", "it"}

def rag_answer(docs, question, size, overlap, k, budget):
    """Chunk, embed, retrieve, pack and answer with a citation."""
    return {"retrieved": [], "packed": [], "answer": "I don't know."}
'''

SOLUTION = '''import math, re
from collections import Counter

STOP = {"a", "an", "the", "is", "was", "are", "were", "what", "when", "which", "who", "how", "many", "did", "does", "do", "of", "in", "on", "for", "to", "and", "it"}

def _terms(t):
    return re.findall(r"[a-z0-9]+", t.lower())

def rag_answer(docs, question, size, overlap, k, budget):
    chunks = []
    for d in sorted(docs):
        toks, start, n = docs[d].split(), 0, 0
        while start < len(toks):
            chunks.append((f"{d}#{n}", toks[start:start + size]))
            n += 1
            if start + size >= len(toks):
                break
            start += size - overlap
    texts = [" ".join(w) for _, w in chunks]
    tfs = [Counter(_terms(t)) for t in texts]
    N = len(chunks)
    df = Counter(term for tf in tfs for term in tf)
    idf = {t: math.log((1 + N) / (1 + c)) + 1 for t, c in df.items()}
    def vec(tf):
        v = {t: c * idf[t] for t, c in tf.items() if t in idf}
        norm = math.sqrt(sum(x * x for x in v.values()))
        return {t: x / norm for t, x in v.items()} if norm else {}
    q = vec(Counter(_terms(question)))
    scores = [sum(q.get(t, 0.0) * x for t, x in vec(tf).items()) for tf in tfs]
    order = sorted((i for i in range(N) if scores[i] > 0), key=lambda i: (-scores[i], i))[:k]
    packed, left = [], budget
    for i in order:
        if len(chunks[i][1]) <= left:
            packed.append(i); left -= len(chunks[i][1])
    keys = {t for t in _terms(question) if t in idf and t not in STOP}
    best, best_n = None, 0.0
    for i in packed:
        for s in re.split(r"(?<=[.!?])\\s+", texts[i]):
            n = sum(idf[t] for t in keys & set(_terms(s)))
            if n > best_n:
                best, best_n = (s, chunks[i][0]), n
    answer = f"{best[0]} [{best[1]}]" if best else "I don't know."
    return {"retrieved": [chunks[i][0] for i in order], "packed": [chunks[i][0] for i in packed], "answer": answer}
'''

HINTS = [
    'Build each piece as its own helper (chunk, embed, score, pack, pick) and test them in that order.',
    'Fit idf on the chunks only; the question reuses those weights.',
    'Use a strict > when picking the best sentence so ties keep the earliest candidate.',
]

_D = {"eiffel": "The Eiffel Tower stands in Paris. The Eiffel Tower was completed in 1889 for the World's Fair. The Eiffel Tower is 330 metres tall. The company of Gustave Eiffel designed and built the tower.",
      "louvre": "The Louvre is the most visited museum in the world. The Louvre stands on the Seine in Paris. The Louvre holds the Mona Lisa.",
      "colosseum": "The Colosseum is an amphitheatre in Rome. The Colosseum was completed in 80 AD under Titus. The Colosseum could seat about 50,000 spectators."}
_P = {"eiffel": "The Eiffel Tower stands in Paris. It was completed in 1889. It is 330 metres tall."}
TESTS = [
    (_D, "When was the Eiffel Tower completed?", 16, 6, 3, 40),
    (_D, "When was the Eiffel Tower completed?", 12, 4, 3, 40),
    (_D, "Which museum holds the Mona Lisa?", 10, 3, 2, 30),
    (_D, "How many spectators could the Colosseum seat?", 10, 3, 4, 24),
    (_D, "Who designed the Eiffel Tower?", 12, 4, 2, 24),
    (_D, "Quantum chromodynamics", 10, 2, 3, 50),
    (_D, "Paris", 30, 5, 5, 10),
    (_P, "When was the Eiffel Tower completed?", 20, 0, 1, 20),
    ({"a": "Short doc."}, "doc", 5, 1, 1, 5),
]

EXPLANATION = r"""
## How the pieces fit
Every RAG system has the same skeleton: split documents so a retrieval unit holds one idea, map text and question into a common vector space, rank by similarity, fit the best evidence into a context budget, and produce an answer that points back to its source. This capstone uses TF-IDF and an extractive answer so the whole pipeline is deterministic and testable; swapping in a neural embedding model and an LLM changes two functions, not the shape.

## Where it breaks
- Chunk size: small chunks retrieve precisely but lose context; large chunks dilute the score.
- Vocabulary mismatch: TF-IDF fails on "When was it finished?" against "completed", and "hold" does not match "held". Dense embeddings fix that, and hybrid retrieval keeps the exact-match strength.
- Extractive answering: when sentences use pronouns ("It was completed in 1889"), the sentence that names the subject outscores the one with the answer, as in the test with the short pronoun-heavy document. Chunk boundaries do similar damage: with 12-token chunks the best-scoring "sentence" is the cut-off fragment "The Eiffel Tower was completed in". Retrieval was right and the answer step was wrong, which is exactly the failure an end-to-end metric hides. An LLM generator resolves the pronoun; a lexical one cannot.
- Budget: a strong but long chunk can be skipped while a shorter, weaker one fits, which is why packing order and chunk size interact.

## From primitives to a framework
In LangChain or LlamaIndex these steps are a text splitter, an embedding model plus a vector store such as Chroma, a retriever with `k`, a prompt template that stuffs the context, and an LLM call. Knowing each step lets you debug which one failed when an answer is wrong.

## Interview follow-ups
- How would you evaluate this pipeline end to end, separating retrieval failures from generation failures?
"""
