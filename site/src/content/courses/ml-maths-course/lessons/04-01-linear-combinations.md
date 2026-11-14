---
id: "04.01"
course: "ml-maths-course"
module: "04"
slug: "linear-combinations"
title: "Linear Combinations"
keyIdeas: ["A linear combination of ⟦1299⟧ is ⟦1300⟧: scale each vector, then add.", "Asking \"can I reach ⟦682⟧?\" means solving for the numbers ⟦1301⟧.", "Embedding averages, weighted sums of features and attention outputs are all linear combinations."]
question: "With ⟦1294⟧ and ⟦1295⟧, compute ⟦1296⟧, and find ⟦1297⟧ with ⟦1298⟧."
answer: "Step 1, ⟦1311⟧. Step 2, for ⟦1306⟧ we need ⟦1307⟧ (first entries) and ⟦1308⟧ (second entries). Step 3, so ⟦1309⟧ and ⟦1310⟧. Check: ⟦1312⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 2"}, {"title": "Axler: Linear Algebra Done Right", "url": "https://linear.axler.net/", "note": "Read: ch. 1-3"}, {"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 5-10"}]
tags: ["Vectors, spans and dependence"]
tools: ["NumPy"]
topic: {"id": "04.01", "title": "Vectors, spans and dependence"}
---

## The big picture

Almost everything a linear model or a layer does is "scale some vectors and add them up". That operation has a name, the linear combination, and the rest of this module is about what you can and can't build with it.

## The definition

Take vectors <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>v</mi><mn>1</mn></msub><mo separator="true">,</mo><mo>…</mo><mo separator="true">,</mo><msub><mi>v</mi><mi>k</mi></msub></mrow></math></span> in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">ℝ</mi><mi>d</mi></msup></mrow></math></span> and numbers <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>c</mi><mn>1</mn></msub><mo separator="true">,</mo><mo>…</mo><mo separator="true">,</mo><msub><mi>c</mi><mi>k</mi></msub></mrow></math></span>. The vector
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msub><mi>c</mi><mn>1</mn></msub><msub><mi>v</mi><mn>1</mn></msub><mo>+</mo><msub><mi>c</mi><mn>2</mn></msub><msub><mi>v</mi><mn>2</mn></msub><mo>+</mo><mo>⋯</mo><mo>+</mo><msub><mi>c</mi><mi>k</mi></msub><msub><mi>v</mi><mi>k</mi></msub></mrow></math></span></span>
is a **linear combination** of them. The numbers <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>c</mi><mi>i</mi></msub></mrow></math></span> are called coefficients or weights.

~~~dg-raw
%%viz f_plane x=-1,5 y=-1,4 v=1,0,v₁;1,2,v₂,f va=0,0,3,0,,m;3,0,4,2,3v₁+v₂,a | 3v₁ + v₂: walk three copies of v₁, then one copy of v₂. You end at (4, 2).
~~~

<div class="ex">
<div class="lab">Worked example</div>

**mixing two vectors**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>v</mi><mn>1</mn></msub><mo>=</mo><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>v</mi><mn>2</mn></msub><mo>=</mo><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo stretchy="false">)</mo></mrow></math></span>.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>3</mn><msub><mi>v</mi><mn>1</mn></msub><mo>+</mo><mn>1</mn><msub><mi>v</mi><mn>2</mn></msub><mo>=</mo><mo stretchy="false">(</mo><mn>3</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo><mo>+</mo><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo stretchy="false">)</mo><mo>=</mo><mo stretchy="false">(</mo><mn>4</mn><mo separator="true">,</mo><mn>2</mn><mo stretchy="false">)</mo></mrow></math></span>.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>−</mo><mn>1</mn><msub><mi>v</mi><mn>1</mn></msub><mo>+</mo><mn>0.5</mn><msub><mi>v</mi><mn>2</mn></msub><mo>=</mo><mo stretchy="false">(</mo><mo>−</mo><mn>1</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo><mo>+</mo><mo stretchy="false">(</mo><mn>0.5</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo><mo>=</mo><mo stretchy="false">(</mo><mo>−</mo><mn>0.5</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>.

</div>

## Can I reach a target?

To ask whether <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>b</mi></mrow></math></span> is a linear combination, solve for the coefficients. That is a linear system.

<div class="ex">
<div class="lab">Worked example</div>

**solving for the weights**

Reach <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>4</mn><mo stretchy="false">)</mo></mrow></math></span> from <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>v</mi><mn>1</mn></msub><mo>=</mo><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>v</mi><mn>2</mn></msub><mo>=</mo><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo stretchy="false">)</mo></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>c</mi><mn>1</mn></msub><mo>+</mo><msub><mi>c</mi><mn>2</mn></msub><mo>=</mo><mn>0</mn></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>2</mn><msub><mi>c</mi><mn>2</mn></msub><mo>=</mo><mn>4</mn></mrow></math></span>.

So <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>c</mi><mn>2</mn></msub><mo>=</mo><mn>2</mn></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>c</mi><mn>1</mn></msub><mo>=</mo><mo>−</mo><mn>2</mn></mrow></math></span>.

</div>

```
v1 = np.array([1, 0]); v2 = np.array([1, 2])
print(3 * v1 + v2)  # [4 2]
print(-2 * v1 + 2 * v2)  # [0 4]
```

## In ML

- A sentence embedding made by averaging word vectors is a linear combination with all weights <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><mi mathvariant="normal">/</mi><mi>n</mi></mrow></math></span>.
- An attention output is a combination of value vectors, weighted by attention scores.
- A linear model's prediction is a combination of feature values, weighted by the coefficients.

<div class="take">
<div class="lab">Key takeaway</div>

a linear combination scales each vector and adds the results. Finding the weights to hit a target is solving a linear system.

</div>
