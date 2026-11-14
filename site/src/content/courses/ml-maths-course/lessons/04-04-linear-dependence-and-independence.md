---
id: "04.04"
course: "ml-maths-course"
module: "04"
slug: "linear-dependence-and-independence"
title: "Linear Dependence and Independence"
keyIdeas: ["⟦1299⟧ are linearly independent if ⟦1341⟧ forces every ⟦1342⟧.", "If some non-zero mix gives 0, they are dependent: at least one vector is a combination of the others and adds nothing to the span.", "Test: stack them as columns of ⟦1204⟧; they are independent exactly when ⟦1343⟧."]
question: "You one-hot encode a feature with 3 levels and also keep an intercept column of ones. Show the columns are dependent and say what that does to least squares. Give two fixes."
answer: "Step 1, each row has exactly one 1 among the one-hot columns ⟦1362⟧, so ⟦1363⟧. Step 2, so ⟦1364⟧ with coefficients ⟦1365⟧, not all zero: dependent. This is the \"dummy variable trap\". Step 3, least squares has infinitely many equally good coefficient vectors: add ⟦166⟧ to each level and subtract ⟦166⟧ from the intercept and no prediction changes. Step 4, fixes: drop one level (reference coding), drop the intercept, or add ridge regularisation."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 2"}, {"title": "Axler: Linear Algebra Done Right", "url": "https://linear.axler.net/", "note": "Read: ch. 1-3"}, {"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 5-10"}]
tags: ["Vectors, spans and dependence"]
tools: ["NumPy"]
topic: {"id": "04.01", "title": "Vectors, spans and dependence"}
---

## The big picture

A dataset with 50 features is 50 columns. Are all 50 telling you something new, or are some just copies or mixtures of others? "Height in cm" and "height in inches" are the same information twice. **Linear independence** is the precise version of "telling you something new".

## The definition

Vectors <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>v</mi><mn>1</mn></msub><mo separator="true">,</mo><mo>…</mo><mo separator="true">,</mo><msub><mi>v</mi><mi>k</mi></msub></mrow></math></span> are **linearly independent** if
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msub><mi>c</mi><mn>1</mn></msub><msub><mi>v</mi><mn>1</mn></msub><mo>+</mo><mo>⋯</mo><mo>+</mo><msub><mi>c</mi><mi>k</mi></msub><msub><mi>v</mi><mi>k</mi></msub><mo>=</mo><mn>0</mn><mtext>  </mtext><mo>⟹</mo><mtext>  </mtext><msub><mi>c</mi><mn>1</mn></msub><mo>=</mo><mo>⋯</mo><mo>=</mo><msub><mi>c</mi><mi>k</mi></msub><mo>=</mo><mn>0</mn></mrow></math></span></span>
Read it: "the only way to mix them into the zero vector is to use zero of each".

If some mix with not-all-zero numbers gives 0, they are **dependent**. Then one vector can be written using the others, so removing it does not shrink the span.

~~~dg-raw
%%viz f_plane x=-1,5 y=-1,4 v=1,0.5,v₁;2,1,v₂_=_2v₁,f;0.5,2,v₃,b | v₁ and v₂ point the same way (v₂ = 2v₁), so they are dependent. v₁ and v₃ point different ways and are independent.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**spotting dependence**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>v</mi><mn>1</mn></msub><mo>=</mo><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo separator="true">,</mo><mn>3</mn><mo stretchy="false">)</mo></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>v</mi><mn>2</mn></msub><mo>=</mo><mo stretchy="false">(</mo><mn>2</mn><mo separator="true">,</mo><mn>4</mn><mo separator="true">,</mo><mn>6</mn><mo stretchy="false">)</mo></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>v</mi><mn>3</mn></msub><mo>=</mo><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>v</mi><mn>2</mn></msub><mo>=</mo><mn>2</mn><msub><mi>v</mi><mn>1</mn></msub></mrow></math></span>, so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>2</mn><msub><mi>v</mi><mn>1</mn></msub><mo>−</mo><msub><mi>v</mi><mn>2</mn></msub><mo>+</mo><mn>0</mn><msub><mi>v</mi><mn>3</mn></msub><mo>=</mo><mn>0</mn></mrow></math></span> with coefficients <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>2</mn><mo separator="true">,</mo><mo>−</mo><mn>1</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo></mrow></math></span>. Dependent.

Drop <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>v</mi><mn>2</mn></msub></mrow></math></span> and the span doesn't change.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**checking independence by solving**

Are <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mo>−</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span> independent? Solve <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>c</mi><mn>1</mn></msub><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo><mo>+</mo><msub><mi>c</mi><mn>2</mn></msub><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mo>−</mo><mn>1</mn><mo stretchy="false">)</mo><mo>=</mo><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo></mrow></math></span>.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>c</mi><mn>1</mn></msub><mo>+</mo><msub><mi>c</mi><mn>2</mn></msub><mo>=</mo><mn>0</mn></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>c</mi><mn>1</mn></msub><mo>−</mo><msub><mi>c</mi><mn>2</mn></msub><mo>=</mo><mn>0</mn></mrow></math></span> give <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>c</mi><mn>1</mn></msub><mo>=</mo><msub><mi>c</mi><mn>2</mn></msub><mo>=</mo><mn>0</mn></mrow></math></span>. Only the zero solution: independent.

</div>

## Ways to check

- Columns of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>V</mi></mrow></math></span> independent <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mtext>  </mtext><mo>⟺</mo><mtext>  </mtext></mrow></math></span> <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>V</mi><mi>c</mi><mo>=</mo><mn>0</mn></mrow></math></span> only for <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>c</mi><mo>=</mo><mn>0</mn></mrow></math></span> <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mtext>  </mtext><mo>⟺</mo><mtext>  </mtext></mrow></math></span> <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">rank</mi><mo>⁡</mo><mo stretchy="false">(</mo><mi>V</mi><mo stretchy="false">)</mo><mo>=</mo><mi>k</mi></mrow></math></span>.
- More than <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>d</mi></mrow></math></span> vectors in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">ℝ</mi><mi>d</mi></msup></mrow></math></span> are always dependent.
- Any set containing the zero vector is dependent.

```
V = np.array([[1, 2, 0], [2, 4, 1], [3, 6, 1]])
print(np.linalg.matrix_rank(V))  # 2
print(V @ np.array([2, -1, 0]))  # [0 0 0]
```

## Dependence in data

If feature columns are dependent, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>X</mi><mi>v</mi><mo>=</mo><mn>0</mn></mrow></math></span> for some non-zero <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>v</mi></mrow></math></span>, so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>X</mi><mo stretchy="false">(</mo><mi>β</mi><mo>+</mo><mi>t</mi><mi>v</mi><mo stretchy="false">)</mo><mo>=</mo><mi>X</mi><mi>β</mi></mrow></math></span> for every <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>t</mi></mrow></math></span>. The coefficients are not unique and can't be interpreted.

```
h_in = np.array([60.0, 65, 70, 72]); X = np.column_stack([2.54 * h_in, h_in])
for beta in ([1, 0], [0, 2.54], [2, -2.54]):
    print(X @ np.array(beta))
```

All three coefficient vectors give identical predictions.

<div class="take">
<div class="lab">Key takeaway</div>

independent means no vector is redundant. Dependent feature columns make regression coefficients non-unique.

</div>
