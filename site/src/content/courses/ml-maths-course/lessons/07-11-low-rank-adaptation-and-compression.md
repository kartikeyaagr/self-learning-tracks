---
id: "07.11"
course: "ml-maths-course"
module: "07"
slug: "low-rank-adaptation-and-compression"
title: "Low-Rank Adaptation and Compression"
keyIdeas: ["A rank-⟦1199⟧ factorisation of an ⟦961⟧ matrix stores ⟦2296⟧ numbers instead of ⟦1558⟧.", "LoRA freezes ⟦1259⟧ and learns a low-rank update ⟦2297⟧, with ⟦41⟧ of size ⟦1556⟧ and ⟦8⟧ of size ⟦2298⟧.", "A small matrix error does not guarantee a small change in model accuracy; always measure the model."]
question: "A ⟦2294⟧ layer is replaced by its rank-64 truncation, which keeps 90% of ⟦2295⟧. What fraction of the parameters is stored, and what is the relative Frobenius error?"
answer: "Step 1, the factors hold ⟦2302⟧ numbers against ⟦2303⟧: 12.5%. Step 2, 10% of the squared norm is dropped, so the relative error is ⟦2304⟧. Step 3, that error is in the weights; whether the model still works depends on which input directions it hits, so check validation loss and usually fine-tune briefly."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §4.5-4.6"}, {"title": "MIT 18.065 Matrix Methods in Data Analysis (Strang)", "url": "https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/", "note": "Read: lectures 6-9"}]
tags: ["Singular value decomposition"]
tools: ["NumPy"]
topic: {"id": "07.02", "title": "Singular value decomposition"}
---

## The big picture

Large models have huge weight matrices. Low rank lets you compress them after training, or fine-tune them by learning only a small low-rank change. Both are direct uses of the previous lesson.

## Storage arithmetic

A rank-<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>k</mi></mrow></math></span> matrix is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi><msup><mi>Q</mi><mi mathvariant="normal">⊤</mi></msup></mrow></math></span> with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi></mrow></math></span> of size <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>m</mi><mo>×</mo><mi>k</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>Q</mi></mrow></math></span> of size <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi><mo>×</mo><mi>k</mi></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>k</mi><mo stretchy="false">(</mo><mi>m</mi><mo>+</mo><mi>n</mi><mo stretchy="false">)</mo></mrow></math></span> numbers. It saves memory when <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>k</mi><mo>&lt;</mo><mi>m</mi><mi>n</mi><mi mathvariant="normal">/</mi><mo stretchy="false">(</mo><mi>m</mi><mo>+</mo><mi>n</mi><mo stretchy="false">)</mo></mrow></math></span>.

| Layer | Full | Rank 8 | Share |
| --- | --- | --- | --- |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1024</mn><mo>×</mo><mn>1024</mn></mrow></math></span> | 1,048,576 | 16,384 | 1.6% |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>4096</mn><mo>×</mo><mn>4096</mn></mrow></math></span> | 16,777,216 | 65,536 | 0.39% |

## LoRA

~~~dg-flow Low-rank adaptation (LoRA): learn a rank-r update instead of a full one.
x -(frozen W)-> W x -(+)-> *W x + B A x
x -(A: r x n)-> A x -(B: m x r)-> B A x
~~~

LoRA bets that the **change** needed for fine-tuning has low rank. Only <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>B</mi></mrow></math></span> are trained, so optimiser state and checkpoints are tiny, and the update can be merged into <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>W</mi></mrow></math></span> after training.

```
rng = np.random.default_rng(0)
m, n, r = 4096, 4096, 8
B = np.zeros((m, r)); A = rng.normal(size=(r, n)) * 0.01
print(B.size + A.size, round(100 * (B.size + A.size) / (m * n), 2))  # 65536 0.39
x = rng.normal(size=n)
print(np.allclose(B @ (A @ x), 0))  # True
```

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>B</mi></mrow></math></span> starts at zero, so the adapted model starts identical to the original.

## Compressing after training

Truncating trained weights with the SVD works when their singular values decay fast. Errors in one layer feed into the next, so the safe recipe is: truncate, measure validation loss, fine-tune briefly.

<div class="take">
<div class="lab">Key takeaway</div>

rank <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>k</mi></mrow></math></span> costs <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>k</mi><mo stretchy="false">(</mo><mi>m</mi><mo>+</mo><mi>n</mi><mo stretchy="false">)</mo></mrow></math></span> numbers; LoRA learns a low-rank change, and SVD truncation compresses existing weights.

</div>
