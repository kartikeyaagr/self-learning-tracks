---
id: "01.25"
course: "ml-maths-course"
module: "01"
slug: "reading-argmax-and-argmin-from-data"
title: "Reading Argmax and Argmin from Data"
keyIdeas: ["`np.argmax` returns only the first index of the maximum. Use `np.flatnonzero(a == a.max())` to see ties.", "To pick the last best index, reverse the array: `len(a) - 1 - np.argmax(a[::-1])`.", "`np.argmax` treats NaN as the maximum; use `np.nanargmax`."]
question: "Validation accuracy per epoch is ⟦434⟧. Give the max, the full argmax (epochs counted from 1), what `np.argmax` returns, and the index to use for \"keep the latest best checkpoint\"."
answer: "Step 1, the max is 0.88. Step 2, it occurs at epochs 3 and 4, so ⟦438⟧, a tie. Step 3, `np.argmax` returns only the first index, counted from 0: 2 (epoch 3). Step 4, for the latest best, reverse the array: `len(acc) - 1 - np.argmax(acc[::-1])` gives 3, which is epoch 4."
sources: [{"title": "Hammack: Book of Proof", "url": "https://richardhammack.github.io/BookOfProof/", "note": "Read: ch. 1, 8"}, {"title": "Lebl: Basic Analysis I", "url": "https://www.jirka.org/ra/", "note": "Read: §1.2, §7.2"}]
tags: ["Properties of sets"]
tools: ["NumPy"]
topic: {"id": "01.03", "title": "Properties of sets"}
---

~~~dg-raw
%%viz f_bars v=0.92,0.41,0.37,0.55 l=1,2,3,4 hi=min yl=validation_loss | Validation loss by epoch. argmin is epoch 3, the checkpoint you keep.
~~~

## The big picture

In practice, argmax is something you compute from arrays: the best epoch, the best hyperparameter, the predicted class. NumPy's `argmax` makes choices for you (first index, NaN handling) that are easy to miss, and those choices can change which checkpoint you ship.

## Ties

```
acc = np.array([0.81, 0.86, 0.88, 0.88, 0.87])
print(acc.max(), np.argmax(acc))  # 0.88 2
print(np.flatnonzero(acc == acc.max()))  # [2 3]
print(len(acc) - 1 - np.argmax(acc[::-1]))  # 3
```

- Line 1: the max and the first index that reaches it.
- Line 2: the whole argmax set, both tied epochs.
- Line 3: the last index that reaches it.

<div class="ex">
<div class="lab">Worked example</div>

**which checkpoint to keep**

Epochs (counted from 1) give accuracy <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>0.81</mn><mo separator="true">,</mo><mn>0.86</mn><mo separator="true">,</mo><mn>0.88</mn><mo separator="true">,</mo><mn>0.88</mn><mo separator="true">,</mo><mn>0.87</mn><mo stretchy="false">)</mo></mrow></math></span>. The argmax is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">{</mo><mn>3</mn><mo separator="true">,</mo><mn>4</mn><mo stretchy="false">}</mo></mrow></math></span>.

"First best" keeps epoch 3, "latest best" keeps epoch 4. With noisy validation scores, neither is clearly better; some people average the tied checkpoints instead.

</div>

## Argmax along an axis

For a batch of predicted probabilities, take the argmax of each row:

```
P = np.array([[0.1, 0.7, 0.2], [0.5, 0.4, 0.1], [0.3, 0.3, 0.4]])
print(P.argmax(axis=1))  # [1 0 2]
```

Each prediction is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mover accent="true"><mi>y</mi><mo>^</mo></mover><mi>i</mi></msub><mo>=</mo><mi>arg</mi><mo>⁡</mo><msub><mrow><mi>max</mi><mo>⁡</mo></mrow><mi>k</mi></msub><msub><mi>P</mi><mrow><mi>i</mi><mi>k</mi></mrow></msub></mrow></math></span>.

## Argmin of a loss over a grid

```
lrs = np.array([1e-4, 3e-4, 1e-3, 3e-3])
val_loss = np.array([0.52, 0.47, 0.45, 0.61])
print(lrs[np.argmin(val_loss)])  # 0.001
```

The argmin of validation loss picks the learning rate <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><msup><mn>0</mn><mrow><mo>−</mo><mn>3</mn></mrow></msup></mrow></math></span>.

## Where this breaks in practice

- **NaN.** `np.argmax` on an array with NaN returns the first NaN's position, because NaN beats every comparison check. Use `np.nanargmax`.

```
a = np.array([0.2, np.nan, 0.9])
print(np.argmax(a), np.nanargmax(a))  # 1 2
```

- **Greedy decoding.** Picking the most likely word at each step is an argmax per step. The sequence of step-wise argmaxes is not always the most likely whole sentence, which is why beam search exists.

<div class="take">
<div class="lab">Key takeaway</div>

from data, argmax is an index. Check for ties, decide first versus last, and guard against NaN.

</div>
