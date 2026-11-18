---
id: "10.04"
course: "ai-engineering-course"
module: "10"
slug: "evaluating-rag"
title: "Evaluating RAG"
keyIdeas: ["Score retrieval and generation separately: retrieval asks whether the evidence reached the prompt, generation asks whether the answer used it faithfully, answered the question and cited correctly.", "Label evidence at the claim level: split each reference answer into atomic claims and record which passages support each one. Claim recall catches gaps that chunk recall and MRR hide.", "Store gold evidence as document spans, not chunk ids, so the labels survive a change of chunking.", "Faithfulness (supported claims over all claims) and citation precision and recall are computed from a claim-by-chunk support matrix that a judge fills in; the arithmetic is trivial, the judge is the error source.", "Reference-free scores can be high while answers are wrong: faithful to an outdated passage, vague but relevant, or abstaining. Keep a reference-based correctness metric and calibrate the judge on human labels."]
question: "Your RAG assistant scores 0.93 average faithfulness with an off-the-shelf judge, yet support tickets say about a fifth of its answers are wrong. Explain how both can be true, and describe the evaluation you would build to find which component is failing."
answer: "Faithfulness only checks that answer claims are supported by the retrieved context. It is blind to wrong context (an outdated policy retrieved instead of the current one), to missing claims (a faithful but incomplete answer), and to judge error. Build a labelled set of a few hundred real questions, including unanswerable ones. For each, write a reference answer, split it into atomic claims, and mark the document spans that support each claim, pooling candidates from several retrievers. Then measure retrieval with claim-level context recall, context precision and MRR at the k you actually send; generation with faithfulness, citation precision and recall, answer correctness against the reference claims, and abstention accuracy. Cross retrieval hit against answer correctness: wrong answers with the evidence retrieved are generation bugs, wrong answers without it are retrieval bugs. Calibrate the judge against about a hundred human-labelled claims before trusting any of the numbers."
rubric: ["Explains that faithfulness is relative to retrieved context, so wrong or stale context can still score as faithful", "Separates retrieval metrics from generation metrics and names at least two of each", "Labels evidence per claim (atomic facts mapped to supporting passages), not just per question", "Includes unanswerable questions and measures abstention", "Uses a reference-based correctness metric alongside reference-free ones", "Calibrates the LLM judge against human labels before trusting scores"]
sources: [{"title": "Es et al.: Ragas, Automated Evaluation of Retrieval Augmented Generation (2023)", "url": "https://arxiv.org/abs/2309.15217", "note": "Reference-free faithfulness, answer relevance and context relevance."}, {"title": "Ragas docs: Context Precision", "url": "https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/", "note": "The rank-weighted context precision formula and its LLM and ID-based variants."}, {"title": "Gao et al.: Enabling Large Language Models to Generate Text with Citations (ALCE, 2023)", "url": "https://arxiv.org/abs/2305.14627", "note": "Citation recall and precision, scored with an entailment model."}, {"title": "Min et al.: FActScore (2023)", "url": "https://arxiv.org/abs/2305.14251", "note": "Splitting long answers into atomic facts and scoring each against a knowledge source."}]
practice: ["context-precision-recall", "faithfulness-claim-support", "validate-citations", "ndcg-at-k", "rag-pipeline-capstone"]
tools: ["Ragas"]
tags: ["RAG evaluation", "faithfulness"]
---

## Two systems, scored separately

A RAG answer can be wrong for two unrelated reasons: the evidence never reached the prompt, or it did and the model misused it. One end-to-end correctness score cannot tell these apart, and the fixes are completely different (re-chunking and re-ranking versus prompt and model changes). So a RAG eval scores each half on its own, then crosses them.

~~~dg-flow Where each metric attaches.
Question -(retriever, top k)-> Retrieved chunks -(context recall, context precision, MRR)-> *Retrieval score
Retrieved chunks -(generator)-> Answer with citations -(faithfulness, citation precision and recall)-> Generation score
Answer with citations -(compare with reference claims)-> Correctness + abstention
~~~

The ranking metrics themselves (recall@k, MRR, nDCG) are defined in [Retrieval and ranking metrics](lesson:mlsd:05.02), and the order in which to debug a bad answer is in [Retrieval-augmented generation](lesson:aieng:03.05). This lesson is about how to label a RAG set so those metrics can be computed, the RAG-specific metrics on top, and where the popular automatic versions mislead.

## Labelling a RAG eval set

Start from the general recipe in [Building an eval set](lesson:aieng:10.02): real questions, stratified by topic, with hard cases. RAG adds three fields per item.

| Field | What it holds | Why |
| --- | --- | --- |
| Reference claims | The reference answer split into atomic facts ("annual plans: 30-day refund", "EU buyers: 14-day withdrawal right") | Lets you score partial answers and compute claim recall |
| Evidence per claim | For each claim, the document spans (doc id, character range) that support it | Lets you score retrieval without a judge |
| Answerable flag | False when the corpus does not contain the answer | Measures abstention; 10 to 20 percent of items is a reasonable share |

Three implementation details decide whether this set stays useful.

**Pool the candidates.** Nobody can read a whole corpus per question. Run several retrievers (BM25, your dense model, a hybrid) to depth 20 or so, take the union, and have annotators mark which passages support which claim. This is the TREC pooling idea: passages none of your systems found go unlabelled, so add a new retriever to the pool before trusting its scores.

**Label spans, not chunk ids.** Chunk ids change every time you change chunk size, overlap or parser ([chunking](lesson:aieng:11.04) is one of the things you will tune with this set). Store evidence as character spans in the source document, and at eval time mark a chunk relevant to a claim if it covers, say, at least half of a supporting span. The same labels then score every chunking variant.

**Allow several supporting spans per claim.** The same fact often appears in a policy page, an FAQ and a changelog. Any one of them is enough. This matters for the metrics below.

## Retrieval metrics at the claim level

With labels as claim → supporting chunks, four numbers per question, all at the $k$ you actually put in the prompt:

- **Chunk recall@k**: relevant chunks retrieved over all relevant chunks.
- **Claim recall@k** (what Ragas calls context recall): claims with at least one supporting chunk retrieved, over all claims. This is the one that predicts whether a complete answer is possible.
- **MRR**: reciprocal rank of the first relevant chunk.
- **Context precision@k**: a rank-weighted precision, averaging precision@i over the positions $i$ that hold a relevant chunk ($v_i = 1$):

$$
\text{CP@}K = \frac{\sum_{i=1}^{K} \text{P@}i \cdot v_i}{\sum_{i=1}^{K} v_i}
$$

It is average precision truncated at $K$: relevant chunks near the top raise it, noise above them lowers it.

<div class="ex">
<div class="lab">Worked example</div>

**two rankings for one question**

The reference answer has two claims. Claim 1 is supported by chunks P7 and P8 (the policy page and the FAQ say the same thing); claim 2 only by E2. X, Y and Z are irrelevant.

| Ranking (top 5) | Chunk recall | Claim recall | MRR | Context precision |
| --- | --- | --- | --- | --- |
| A: P7, X, P8, Y, Z | 2/3 = 0.67 | 1/2 = 0.50 | 1.0 | (1 + 2/3) / 2 = 0.83 |
| B: X, P7, Y, Z, E2 | 2/3 = 0.67 | 2/2 = 1.00 | 0.5 | (1/2 + 2/5) / 2 = 0.45 |

Ranking A looks better on MRR and context precision, and chunk recall calls them equal. But A retrieved the same fact twice and missed claim 2 entirely, so no generator can produce a complete answer from it. B makes a complete answer possible. Report claim recall as the headline retrieval number and context precision as the "how much noise is above the evidence" number.

</div>

## Generation metrics from a support matrix

Split the generated answer into claims too (an LLM does this; FActScore showed atomic-fact splitting makes long answers scorable). Then a judge labels every (answer claim, retrieved chunk) pair as supports, contradicts or neutral. Everything else is arithmetic on that matrix:

- **Faithfulness**: answer claims supported by at least one retrieved chunk, over all answer claims. Contradicted and unsupported claims both count against it; report contradictions separately, because they are worse.
- **Citation precision**: of all citations the answer makes, the share that support the claim they are attached to.
- **Citation recall**: answer claims whose cited chunks include a supporting one, over all claims. ALCE's version is stricter (the cited passages together must entail the statement) and uses an entailment model as the judge.
- **Answer correctness**: reference claims covered by the answer, plus answer claims that contradict the reference. This is the only metric here that compares against ground truth.
- **Abstention**: on unanswerable items, did it say it could not find the answer; on answerable ones, how often did it refuse anyway.

<div class="ex">
<div class="lab">Worked example</div>

**scoring one answer with five claims**

| Claim | Cites | Judge: which retrieved chunks support it | Status |
| --- | --- | --- | --- |
| a1 | P7 | P7 | supported, citation correct |
| a2 | P7 | P8 only | supported, citation wrong |
| a3 | none | none | unsupported (from the model's memory) |
| a4 | Y | Y | supported, citation correct |
| a5 | P8 | P8 contradicts it | contradicted |

Faithfulness = 3/5 = 0.60 (a1, a2, a4). Citation precision = 2/4 = 0.50 (four citations, two support their claim). Citation recall = 2/5 = 0.40 (a1, a4). One contradiction, which in a policy assistant should fail the item outright whatever the averages say.

</div>

## Crossing the two halves

For each answerable item, record whether claim recall was 1 and whether the answer was correct. The 2×2 tells you where to work:

| | Answer correct | Answer wrong |
| --- | --- | --- |
| Evidence retrieved | working | generation bug: prompt, model, context order |
| Evidence missing | lucky or from memory (check faithfulness) | retrieval bug: chunking, hybrid search, reranking |

If most failures sit bottom right, no prompt change will help; [reranking](lesson:aieng:11.03) and chunking will.

## Where RAGAS-style metrics mislead

Automatic RAG metrics are judge calls with a formula around them. The formula is fine; the failure modes come from what the formula leaves out and from the judge.

1. **Faithful is not correct.** Faithfulness is relative to whatever was retrieved. An answer that faithfully repeats a 2023 policy when the 2025 one exists scores 1.0. Only reference-based correctness, or freshness metadata in the eval, catches it.
2. **Saying less scores better.** A one-claim hedge is easier to make fully faithful than a complete answer. Pair faithfulness with claim coverage of the reference, or you will optimise toward vague answers.
3. **Empty answers are undefined, not perfect.** An abstention has zero claims. Treat its faithfulness as missing and score it with the abstention metric; averaging it in as 1.0 (or 0.0) distorts the mean.
4. **Reference-free relevance rewards topicality.** Ragas' answer relevance generates questions from the answer and compares them with the original question by embedding similarity. A generic answer on the right topic scores well.
5. **Claim splitting is noisy.** Two runs of the same judge can split an answer into five or seven claims, which moves every ratio. Fix the judge model snapshot and temperature, and cache the splits so a re-run of the generator is compared on stable claims where possible.
6. **Scores do not transfer across judges.** A judge model upgrade can shift faithfulness by several points with no change to your system. Version the judge with the eval (see [Regression testing in CI](lesson:aieng:10.06)) and re-baseline when it changes.
7. **Synthetic questions leak wording.** Questions an LLM generates from a chunk reuse that chunk's phrasing, so lexical and dense retrieval both look better than on real user questions. Keep real queries as the core of the set.

The guard against all of these is the same one [Graders and LLM-as-judge](lesson:aieng:10.03) describes: label 100 or so claim-chunk pairs by hand, measure the judge's agreement, and re-check whenever the judge, the prompt or the domain changes.

## What interviewers probe

- "Your faithfulness is high but users say answers are wrong. Why?" Wrong or stale context, incomplete answers, judge error; the answer is the 2×2 above plus a reference-based metric.
- "How do you get relevance labels without reading the corpus?" Pooling from several retrievers, claim-level labels, spans that survive re-chunking.
- "What happens to your eval when you change chunk size?" With chunk-id labels it breaks; with span labels it does not.
- "How do you evaluate questions the corpus cannot answer?" An answerable flag and an abstention rate, reported both ways (missed abstentions and false refusals).

<div class="take">
<div class="lab">Key takeaway</div>

Label each question's reference answer as atomic claims with the document spans that support them. From that one structure you get claim-level retrieval recall and context precision without a judge, and faithfulness and citation scores from a judge-filled support matrix. Keep a reference-based correctness score next to the reference-free ones, because faithful to the wrong passage still scores as faithful.

</div>
