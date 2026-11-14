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
