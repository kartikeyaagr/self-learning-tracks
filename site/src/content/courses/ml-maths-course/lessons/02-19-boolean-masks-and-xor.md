---
id: "02.19"
course: "ml-maths-course"
module: "02"
slug: "boolean-masks-and-xor"
title: "Boolean Masks and XOR"
keyIdeas: ["In NumPy, `&`, `|` and `~` are elementwise AND, OR and NOT on boolean arrays.", "Put brackets around comparisons: `&` binds tighter than `>`.", "A single threshold unit draws one straight line, so it cannot compute XOR. A hidden layer fixes this."]
question: "Per-token losses are ⟦649⟧ and the last two tokens are padding. Compute the correct masked mean, and the wrong one you get from `(loss * mask).mean()`."
answer: "Step 1, the real tokens are the first three, so the correct mean is ⟦653⟧. Step 2, `(loss * mask).mean()` divides the same sum by all 5 positions: ⟦654⟧. Step 3, the fix is to divide by the number of real tokens: `(loss * mask).sum() / mask.sum()`, which gives 2.0."
sources: [{"title": "Hammack: Book of Proof", "url": "https://richardhammack.github.io/BookOfProof/", "note": "Read: ch. 2, 4-6"}, {"title": "Levin: Discrete Mathematics, an Open Introduction", "url": "https://discrete.openmathbooks.org/", "note": "Read: §0.2, 3.1"}]
tags: ["Boolean algebra"]
tools: ["NumPy"]
topic: {"id": "02.03", "title": "Boolean algebra"}
---

## The big picture

Every padding mask, attention mask and "ignore this label" flag is a boolean array, and Boolean algebra combines them. This lesson covers the array version, its traps, and the XOR problem, the classic reason neural networks need hidden layers.

## Masks in NumPy

A comparison on an array gives a boolean array, a mask:

- `a & b` is elementwise AND.
- `a | b` is elementwise OR.
- `~a` is elementwise NOT.

```
y = np.array([3, -100, 1, 0, -100])
valid = (y != -100)
print(valid)  # [ True False  True  True False]
print(valid.sum())  # 3
```

The value -100 is PyTorch's default "ignore this label" marker, so `valid` marks the positions that count toward the loss.

A causal mask combined with a padding mask, at size 4:

```
T = 4
causal = np.tril(np.ones((T, T), bool))
key_not_pad = np.array([True, True, True, False])
attend = causal & key_not_pad[None, :]
print(attend.astype(int))
```

Row <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>i</mi></mrow></math></span> says which positions token <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>i</mi></mrow></math></span> may look at: itself and earlier tokens, never the padding at position 3.

## Two traps

- Python's `and` / `or` raise "The truth value of an array with more than one element is ambiguous". Use `&`, `|`, `~`.
- `x > 0 & y > 0` is parsed as `x > (0 & y) > 0`. Write `(x > 0) & (y > 0)`.

```
x = np.array([1, -1, 2])
z = np.array([1, 1, -1])
print((x > 0) & (z > 0))  # [ True False False]
```

## Masked averages

```
loss = np.array([2.0, 1.0, 3.0, 0.0, 0.0])
mask = np.array([1, 1, 1, 0, 0], bool)
print(loss[mask].mean())  # 2.0
print((loss * mask).mean())  # 1.2
print((loss * mask).sum() / mask.sum())  # 2.0
```

The second line is the bug: padding drags the average down, more so as padding grows.

## XOR and one straight line

A **threshold unit** outputs <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn mathvariant="normal">𝟏</mn><mo stretchy="false">[</mo><msub><mi>w</mi><mn>1</mn></msub><msub><mi>x</mi><mn>1</mn></msub><mo>+</mo><msub><mi>w</mi><mn>2</mn></msub><msub><mi>x</mi><mn>2</mn></msub><mo>+</mo><mi>b</mi><mo>&gt;</mo><mn>0</mn><mo stretchy="false">]</mo></mrow></math></span>: 1 on one side of a straight line, 0 on the other. AND and OR can each be drawn with one line (<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>w</mi><mo>=</mo><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span> with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>b</mi><mo>=</mo><mo>−</mo><mn>1.5</mn></mrow></math></span> or <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>b</mi><mo>=</mo><mo>−</mo><mn>0.5</mn></mrow></math></span>). XOR cannot: its 1s sit on opposite corners of the square.

~~~dg-raw
%%viz xor live | Try to split the 1s from the 0s with one straight line. It works for AND, OR and NAND, never for XOR.
~~~

The fix is a hidden layer: compute OR and AND first, then "OR and not AND", which is XOR.

~~~dg-flow XOR from two threshold units: a hidden layer creates the features that make the problem linear.
x1, x2 -(x1+x2 > 0.5)-> h1 = OR
x1, x2 -(x1+x2 > 1.5)-> h2 = AND
h1, h2 -(h1 - h2 > 0.5)-> *XOR
~~~

```
step = lambda z: (z > 0).astype(int)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
h1 = step(X.sum(axis=1) - 0.5)
h2 = step(X.sum(axis=1) - 1.5)
print(step(h1 - h2 - 0.5))  # [0 1 1 0]
```

## Where this breaks in practice

- `~` on an integer array is bitwise NOT: `~1 == -2`. Make sure the mask is `bool` first.
- Some libraries use "True means keep" and others "True means block" for attention masks. A flipped mask often trains without errors and just performs badly.

<div class="take">
<div class="lab">Key takeaway</div>

masks are Boolean algebra on arrays. Bracket every comparison, and divide by the mask's sum. One line cannot separate XOR; a hidden layer can.

</div>
