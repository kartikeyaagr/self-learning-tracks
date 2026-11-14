---
id: "08.01"
course: "ml-maths-course"
module: "08"
slug: "the-least-squares-problem"
title: "The Least-Squares Problem"
keyIdeas: ["Least squares picks ⟦2319⟧ when ⟦816⟧ has no exact solution.", "The residual ⟦2320⟧ is perpendicular to every column of ⟦8⟧, which gives the normal equations ⟦2030⟧.", "Any model linear in its parameters (lines, parabolas, polynomial features) is fitted this way."]
question: "Fit ⟦2317⟧ to ⟦2318⟧. Give the coefficients, residuals, and check the residuals are perpendicular to both columns."
answer: "Step 1, ⟦2330⟧, ⟦2331⟧, ⟦2332⟧, ⟦2333⟧. Step 2, slope ⟦2334⟧, intercept ⟦2335⟧. Step 3, predictions ⟦2336⟧; residuals ⟦2337⟧. Step 4, they sum to 0 (perpendicular to the ones column), and ⟦2338⟧ (perpendicular to the ⟦7⟧ column)."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 9-10"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 12-13"}]
tags: ["Residual minimization"]
tools: ["NumPy"]
topic: {"id": "08.01", "title": "Residual minimization"}
---

## The big picture

Ten noisy measurements and a straight line with two parameters: no line passes through all ten points. **Least squares** picks the line that makes the total squared miss as small as possible. It is the oldest fitting method in statistics and the starting point for linear regression. Module 06 already has the answer: project onto the column space.

## The problem

A tall system <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mi>x</mi><mo>=</mo><mi>b</mi></mrow></math></span> usually has no solution because <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>b</mi></mrow></math></span> is not in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>C</mi><mo stretchy="false">(</mo><mi>A</mi><mo stretchy="false">)</mo></mrow></math></span>. So ask for the closest thing:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mover accent="true"><mi>x</mi><mo>^</mo></mover><mo>=</mo><mi>arg</mi><mo>⁡</mo><munder><mrow><mi>min</mi><mo>⁡</mo></mrow><mi>x</mi></munder><mi mathvariant="normal">∥</mi><mi>A</mi><mi>x</mi><mo>−</mo><mi>b</mi><msup><mi mathvariant="normal">∥</mi><mn>2</mn></msup></mrow></math></span></span>
"<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>arg</mi><mo>⁡</mo><mi>min</mi><mo>⁡</mo></mrow></math></span>" means the input that achieves the minimum. The **residual** <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>r</mi><mo>=</mo><mi>b</mi><mo>−</mo><mi>A</mi><mover accent="true"><mi>x</mi><mo>^</mo></mover></mrow></math></span> is what the model cannot explain.

By lesson 06.14, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mover accent="true"><mi>x</mi><mo>^</mo></mover></mrow></math></span> is the projection of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>b</mi></mrow></math></span> onto <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>C</mi><mo stretchy="false">(</mo><mi>A</mi><mo stretchy="false">)</mo></mrow></math></span>, so the residual is perpendicular to every column:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msup><mi>A</mi><mi mathvariant="normal">⊤</mi></msup><mo stretchy="false">(</mo><mi>b</mi><mo>−</mo><mi>A</mi><mover accent="true"><mi>x</mi><mo>^</mo></mover><mo stretchy="false">)</mo><mo>=</mo><mn>0</mn><mspace width="1em"/><mo>⟺</mo><mspace width="1em"/><msup><mi>A</mi><mi mathvariant="normal">⊤</mi></msup><mi>A</mi><mover accent="true"><mi>x</mi><mo>^</mo></mover><mo>=</mo><msup><mi>A</mi><mi mathvariant="normal">⊤</mi></msup><mi>b</mi></mrow></math></span></span>
Calculus agrees: the gradient of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∥</mi><mi>A</mi><mi>x</mi><mo>−</mo><mi>b</mi><msup><mi mathvariant="normal">∥</mi><mn>2</mn></msup></mrow></math></span> is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>2</mn><msup><mi>A</mi><mi mathvariant="normal">⊤</mi></msup><mo stretchy="false">(</mo><mi>A</mi><mi>x</mi><mo>−</mo><mi>b</mi><mo stretchy="false">)</mo></mrow></math></span>, and setting it to zero gives the same equations.

## A line through four points

~~~dg-raw
%%viz f_plane x=-0.5,3.5 y=0,4.5 tk=1 ln=0.9,-1,-0.9,ŷ_=_0.9_+_0.9x,a,0.85 sg=0,1,0,0.9,f;1,2,1,1.8,f;2,2,2,2.7,f;3,4,3,3.6,f pt=0,1,,k;1,2,,k;2,2,,k;3,4,,k | The least-squares line through four points. The orange segments are the residuals 0.1, 0.2, −0.7, 0.4; their squares add to 0.70, smaller than for any other line.
~~~

```
x = np.array([0.0, 1, 2, 3]); y = np.array([1.0, 2, 2, 4])
A = np.column_stack([np.ones(4), x])
beta, sse, rank, sv = np.linalg.lstsq(A, y, rcond=None)
print(np.round(beta, 6), np.round(sse, 6))  # [0.9 0.9] [0.7]
r = y - A @ beta
print(np.allclose(A.T @ r, 0))  # True
```

## Curved models, linear parameters

A parabola <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi><mo>=</mo><msub><mi>β</mi><mn>0</mn></msub><mo>+</mo><msub><mi>β</mi><mn>1</mn></msub><mi>x</mi><mo>+</mo><msub><mi>β</mi><mn>2</mn></msub><msup><mi>x</mi><mn>2</mn></msup></mrow></math></span> is still linear in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>β</mi></mrow></math></span>: just add a column <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>x</mi><mn>2</mn></msup></mrow></math></span>.

```
x = np.array([-2.0, -1, 0, 1, 2]); y = np.array([4.1, 0.9, 0.1, 1.2, 3.9])
A = np.column_stack([np.ones(5), x, x ** 2])
print(np.round(np.linalg.lstsq(A, y, rcond=None)[0], 3))  # [ 0.083 -0.01   0.979]
```

The fit recovers roughly <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi><mo>≈</mo><msup><mi>x</mi><mn>2</mn></msup></mrow></math></span>, which is how the data was made.

<div class="take">
<div class="lab">Key takeaway</div>

least squares finds the point of the column space nearest to <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>b</mi></mrow></math></span>; the residual is perpendicular to every column.

</div>
