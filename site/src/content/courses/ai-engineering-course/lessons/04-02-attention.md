---
id: "04.02"
course: "ai-engineering-course"
module: "04"
slug: "attention"
title: "Attention"
keyIdeas: ["Attention lets each token build its new representation from a weighted mix of other tokens.", "Each token's vector is projected three ways: query (what I'm looking for), key (what I offer), value (what I pass on).", "Score = Q . K for every pair of tokens; divide by sqrt(d_k); softmax over each row; multiply by V.", "Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V.", "Dividing by sqrt(d_k) keeps dot products from growing with dimension, which would push softmax into near one-hot outputs with tiny gradients.", "Cost is quadratic in sequence length: n tokens give an n x n score matrix. This drives most long-context optimisations."]
question: "Write the scaled dot-product attention formula and explain what Q, K and V are and why the scores are divided by sqrt(d_k)."
answer: "Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V. Q, K and V are linear projections of the token vectors: queries say what a token looks for, keys what each token can be matched on, values the information passed on. Scores are query-key dot products, softmaxed into weights that mix the values. Dividing by sqrt(d_k) keeps score variance near 1; otherwise large dot products make softmax nearly one-hot and gradients vanish."
sources: [{"title": "Jay Alammar: The Illustrated Transformer", "url": "https://jalammar.github.io/illustrated-transformer/", "note": "Step-by-step pictures of Q, K, V and attention scores."}, {"title": "Vaswani et al.: Attention Is All You Need (2017)", "url": "https://arxiv.org/abs/1706.03762", "note": "Scaled dot-product attention, equation 1."}, {"title": "3Blue1Brown: Attention in transformers, visually explained", "url": "https://www.youtube.com/watch?v=eMlx5fFNoYc", "note": "Visual intuition for queries, keys and values."}]
steps: [{"kind": "explain", "title": "Scores to weights", "body": "Attention starts with one number for every pair of tokens: the dot product of a token's query with another token's key. These **scores** can be anything: 3.1, 0, −2.4.\n\nTo blend the value vectors we need **weights** instead: none of them negative, and all of them adding up to 1, so the result is an average. Softmax turns a row of scores into exactly that:\n\n$$w_i = \\frac{e^{s_i}}{\\sum_j e^{s_j}}$$\n\nRaise $e$ to each score, then divide by the total.", "question": {"q": "Three keys score 2, 0 and −1. After softmax, which is true?", "options": ["The weights are 2, 0 and −1, rescaled to add up to 1", "All three weights are positive and they add up to 1", "The key that scored −1 gets a weight of exactly 0"], "answer": 1, "why": "$e$ to any power is positive, so every key keeps some weight, even one with a negative score. Dividing by the total makes the three add up to 1 (here about 0.84, 0.11 and 0.04).", "hint": "An exponential is never negative and never exactly zero. What does dividing by the sum of them do?"}}, {"kind": "explain", "title": "Why exponentiate", "body": "Why not divide each score by the sum of the scores? Scores can be negative, and a sum close to zero would make the weights explode.\n\nExponentials fix both problems. $e^s$ is always positive, and it turns **differences** into **ratios**: since $e^{a}/e^{b} = e^{a-b}$, a key that scores 1 higher than another gets $e \\approx 2.7$ times its weight, and a gap of 2 gives about 7.4 times. Softmax only ever looks at how far apart the scores are.", "question": {"q": "Key A scores 3 and key B scores 1. How many times more weight does A get than B?", "options": ["3 times", "About 7.4 times", "2 times"], "answer": 1, "why": "Only the gap matters: $e^{3-1} = e^2 \\approx 7.4$. The same gap anywhere on the scale, say 10 and 8, gives the same ratio.", "hint": "Softmax compares two keys through $e$ raised to the difference of their scores."}}, {"kind": "explain", "title": "Subtract the max", "body": "Because only differences matter, subtracting the same number from every score leaves the weights unchanged. Real implementations use that: they subtract the row's largest score before exponentiating.\n\nThe top score becomes $e^0 = 1$ and every other exponential lands between 0 and 1, so nothing overflows. $e^{1000}$ is infinity in floating point; $e^{1000 - 1000}$ is just 1. The same trick, carried across blocks of keys, is what lets FlashAttention compute softmax one tile at a time.", "question": {"q": "Scores [1000, 999, 998] go through a stable softmax. Which numbers actually get exponentiated?", "options": ["[1000, 999, 998]", "[0, −1, −2]", "[1, 0.999, 0.998]"], "answer": 1, "why": "Subtracting the maximum, 1000, leaves [0, −1, −2]. The weights come out the same as before (about 0.67, 0.24 and 0.09), and every exponential stays at or below 1.", "hint": "The stable version subtracts the largest score from every score first."}}, {"kind": "interact", "title": "Where does “it” look?", "widget": "softmax-temperature", "body": "Here the query is **it**, and its scores against every word of the sentence are fixed. Each bar is one weight from $\\text{softmax}(s / T)$, where the **temperature** $T$ divides every score before the softmax. Drag the slider and watch the weights shift.", "question": {"q": "At temperature 1, which word gets the most weight?", "options": ["animal", "street", "tired"], "answer": 0, "why": "“it” refers back to the animal, so that key has the highest score and takes about 43% of the weight. Lower the temperature and the weight piles onto it; raise it and attention spreads out.", "hint": "Set the slider back to 1 and find the longest bar. Softmax always gives the most weight to the highest score."}}, {"kind": "interact", "title": "Temperature extremes", "widget": "softmax-temperature", "body": "Dividing by $T$ stretches or squeezes the gaps between scores, and gaps are all softmax sees.\n\n- **Small $T$** widens the gaps. The top score takes nearly everything; as $T \\to 0$ softmax becomes a hard pick of the maximum.\n- **Large $T$** shrinks the gaps. As $T$ grows, every weight drifts towards $1/n$.\n\nTry both ends of the slider: 0.25, then 3.", "question": {"q": "At T = 3, what happens to the weight on “animal”?", "options": ["It is still the largest, but its share falls", "It drops below “street”", "It rises above 90%"], "answer": 0, "why": "Temperature divides every score by the same number, so their order never changes. At $T = 3$ “animal” keeps the top spot but falls to about 17%, not far above the rest.", "hint": "Dividing every score by the same positive number can't change which one is largest."}}, {"kind": "explain", "title": "Scaling by √d", "body": "Attention has a temperature built in. A query–key dot product adds up $d_k$ products; if each entry has variance 1, the score has variance $d_k$, so its typical size grows like $\\sqrt{d_k}$. With $d_k = 128$ that is a standard deviation of about 11.\n\nScores that far apart make softmax nearly one-hot, and the keys that get almost no weight also get almost no gradient. Dividing by $\\sqrt{d_k}$ is a temperature of $T = \\sqrt{d_k}$: it brings the scores back to unit variance, so training starts with soft weights it can learn from.\n\n$$\\text{Attention}(Q, K, V) = \\text{softmax}\\!\\left(\\frac{QK^\\top}{\\sqrt{d_k}}\\right)V$$", "question": {"q": "A head has $d_k = 64$. What temperature does the $\\sqrt{d_k}$ scaling apply?", "options": ["T = 64", "T = 8", "T = 1/8"], "answer": 1, "why": "Dividing the scores by $\\sqrt{64} = 8$ is $\\text{softmax}(s / 8)$: a temperature of 8, which cancels the growth of dot products with dimension.", "hint": "Dividing every score by a number is the same as setting $T$ to that number. What is $\\sqrt{64}$?"}}, {"kind": "question", "title": "Recap and a check", "body": "Put together: score every key with $q \\cdot k$, divide by $\\sqrt{d_k}$, softmax each row, and use the weights to average the value vectors. Softmax keeps every weight positive, makes them add up to 1, depends only on differences between scores, and its sharpness is set by the temperature.", "question": {"q": "You remove the $\\sqrt{d_k}$ scaling from a model with $d_k = 256$. What is the most likely effect early in training?", "options": ["The attention weights become nearly uniform", "Attention becomes nearly one-hot and most keys get almost no gradient", "Nothing changes, because softmax already normalises the scores"], "answer": 1, "why": "Unscaled scores have a standard deviation of about $\\sqrt{256} = 16$, so softmax behaves as if the temperature were very low: one key takes almost all the weight and the others barely learn.", "hint": "Softmax does normalise, but it doesn't undo large gaps between scores. How far apart do unscaled scores get?"}}]
---

## The job attention does

Take "The animal didn't cross the street because it was too tired." To represent "it", the model needs information from "animal". Attention is the mechanism that lets each token pull in information from the other tokens it needs. Jay Alammar's Illustrated Transformer uses this exact sentence.

## Queries, keys and values

Each token's vector x is multiplied by three learned matrices:

- **Query** q = x W_Q: what this token is looking for.
- **Key** k = x W_K: what this token can be matched on.
- **Value** v = x W_V: the information it hands over if selected.

The analogy is a soft dictionary lookup. The query is compared against every key; instead of returning one match, attention returns a weighted average of all values, weighted by how well each key matched.

## The formula

```
Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V
```

~~~dg-flow Scaled dot-product attention for a sequence of n tokens.
Q, K -(Q K^T)-> n x n scores -(/ sqrt(d_k))-> Scaled -(softmax per row)-> *Attention weights -(x V)-> New token vectors
~~~

1. **Scores**: dot product of each query with every key, an n x n matrix.
2. **Scale**: divide by sqrt(d_k), the key dimension.
3. **Softmax** each row so the weights for one token add to 1.
4. **Mix**: multiply by V, so each output is a weighted sum of value vectors.

## Why divide by sqrt(d_k)

If entries of q and k have unit variance, their dot product has variance d_k. With d_k = 128, scores spread over a range of roughly plus or minus 30, and softmax becomes almost one-hot: one weight near 1, the rest near 0, with vanishing gradients. Dividing by sqrt(d_k) brings the variance back to about 1, as the original paper explains.

## The cost

The score matrix has n^2 entries per head per layer. Double the context and attention compute and memory quadruple. Prefill on a 100k-token prompt is dominated by this. FlashAttention (module 6) reduces the memory traffic; sparse attention (module 7) reduces the number of pairs.

<div class="ex">
<div class="lab">Worked example</div>

**three tokens, tiny numbers**

Query for "it" scores [animal: 4.0, street: 1.0, tired: 2.0] after scaling.

Softmax gives roughly [0.84, 0.04, 0.11].

The output for "it" is about 84% the value vector of "animal", plus small parts of the others.

</div>

## Click a token, see its attention

Each row of the grid is one query token, each column a key token, and the colour is the attention weight after softmax. Click "it" to see it attend mostly to "animal". Turn on the causal mask to see what a decoder is allowed to look at.

~~~dg-raw
%%widget attention Attention weights for one sentence, with an optional causal mask.
~~~

Scores in the figure are hand-set to show the pattern, but the mechanism is real: each output is a weighted average of value vectors, with weights from softmax(q · k / √d_k).

## A computation small enough to do by hand

Take three tokens with 2-dim queries, keys and values:

```
q(it)      = [1, 0]
k(animal)  = [2, 0]   v(animal) = [10, 0]
k(street)  = [0, 2]   v(street) = [0, 10]
k(it)      = [1, 1]   v(it)     = [5, 5]
```

1. Scores: q · k = 2, 0, 1. Divide by √2: 1.41, 0, 0.71.
2. Softmax: e^1.41 = 4.1, e^0 = 1, e^0.71 = 2.0, total 7.1, so weights 0.58, 0.14, 0.28.
3. Output: 0.58 × [10, 0] + 0.14 × [0, 10] + 0.28 × [5, 5] = [7.2, 2.8].

The output for "it" is now mostly "animal" information. In a real model, d_k is 64 or 128, there are dozens of heads, and the projections that produce q, k and v are learned.

## Why the √d_k matters

Dot products of random vectors grow with dimension: with unit-variance entries, the standard deviation of q · k is √d_k. Without scaling, a 128-dim head produces scores spread over a range of about ±30, and softmax turns that into a one-hot vector. Gradients for every other key then vanish. Toggle the scaling below to see it.

~~~dg-raw
%%widget scaling Softmax over random scores with and without dividing by √d_k.
~~~

<div class="take">
<div class="lab">Key takeaway</div>

attention is softmax(QK^T / sqrt(d_k))V, a learned soft lookup where each token mixes in values from the tokens whose keys match its query. It is powerful and quadratic in length.

</div>
