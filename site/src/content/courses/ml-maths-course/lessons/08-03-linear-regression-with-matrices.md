---
id: "08.03"
course: "ml-maths-course"
module: "08"
slug: "linear-regression-with-matrices"
title: "Linear Regression With Matrices"
keyIdeas: ["Stack examples as rows of ⟦513⟧ (with a ones column for the intercept): ⟦2349⟧, and ⟦1170⟧.", "The hat matrix ⟦2350⟧ projects ⟦509⟧ onto the column space: ⟦2351⟧.", "Its diagonal entries are leverages: how strongly each example pulls its own prediction. They sum to the number of parameters."]
question: "Houses of 50, 80 and 120 m² sold for 150, 210 and 320 thousand. Write ⟦513⟧ and ⟦509⟧, and say which house has the most leverage."
answer: "Step 1, ⟦2359⟧, ⟦2360⟧. Step 2, for a line, ⟦2361⟧, with ⟦2362⟧. Step 3, the 120 m² house is farthest from the mean, so it has the most leverage (about 0.878)."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 9-10"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 12-13"}]
tags: ["Linear regression"]
tools: ["NumPy"]
topic: {"id": "08.02", "title": "Linear regression"}
---

## The big picture

Linear regression is least squares with features as columns and examples as rows. Writing it with matrices makes it one line of code for any number of features, and the hat matrix shows which examples control the fit.

## The design matrix

Each row of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>X</mi></mrow></math></span> (<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi><mo>×</mo><mi>p</mi></mrow></math></span>) is one example; each column is one feature; a column of ones gives the intercept. The model is
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>y</mi><mo>=</mo><mi>X</mi><mi>β</mi><mo>+</mo><mi>ε</mi></mrow></math></span></span>
and least squares gives
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mover accent="true"><mi>β</mi><mo>^</mo></mover><mo>=</mo><mo stretchy="false">(</mo><msup><mi>X</mi><mi mathvariant="normal">⊤</mi></msup><mi>X</mi><msup><mo stretchy="false">)</mo><mrow><mo>−</mo><mn>1</mn></mrow></msup><msup><mi>X</mi><mi mathvariant="normal">⊤</mi></msup><mi>y</mi><mo separator="true">,</mo><mspace width="2em"/><mover accent="true"><mi>y</mi><mo>^</mo></mover><mo>=</mo><mi>X</mi><mover accent="true"><mi>β</mi><mo>^</mo></mover><mo>=</mo><mi>H</mi><mi>y</mi></mrow></math></span></span>

```
X = np.array([[1, 50.0], [1, 80], [1, 120]]); y = np.array([150.0, 210, 320])
beta = np.linalg.lstsq(X, y, rcond=None)[0]
print(np.round(beta, 3))  # [22.838  2.446]
H = X @ np.linalg.inv(X.T @ X) @ X.T
print(np.round(np.diag(H), 3), round(np.trace(H), 6))  # [0.784 0.338 0.878] 2.0
```

About 23 thousand base price plus 2.4 thousand per square metre.

## The hat matrix and leverage

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>H</mi></mrow></math></span> "puts the hat on <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi></mrow></math></span>". It is a projection (<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>H</mi><mn>2</mn></msup><mo>=</mo><mi>H</mi></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>H</mi><mi mathvariant="normal">⊤</mi></msup><mo>=</mo><mi>H</mi></mrow></math></span>) with trace <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi></mrow></math></span>. Its diagonal entries <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>h</mi><mrow><mi>i</mi><mi>i</mi></mrow></msub></mrow></math></span> are **leverages**.

~~~dg-raw
%%viz f_bars v=0.784,0.338,0.878 l=50_m²,80_m²,120_m² hi=none yl=leverage_hᵢᵢ | Leverage of each house. The 80 m² house sits near the mean size and barely pulls the line; the 120 m² house is far out and pulls the line almost onto itself. The three values add to 2, the number of parameters.
~~~

For a straight line, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>h</mi><mrow><mi>i</mi><mi>i</mi></mrow></msub><mo>=</mo><mfrac><mn>1</mn><mi>n</mi></mfrac><mo>+</mo><mstyle displaystyle="true" scriptlevel="0"><mfrac><mrow><mo stretchy="false">(</mo><msub><mi>x</mi><mi>i</mi></msub><mo>−</mo><mover accent="true"><mi>x</mi><mo>ˉ</mo></mover><msup><mo stretchy="false">)</mo><mn>2</mn></msup></mrow><msub><mi>S</mi><mrow><mi>x</mi><mi>x</mi></mrow></msub></mfrac></mstyle></mrow></math></span>. Points far from the others in feature space get high leverage; one bad label at a high-leverage point can move the whole fit.

<div class="take">
<div class="lab">Key takeaway</div>

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mover accent="true"><mi>β</mi><mo>^</mo></mover><mo>=</mo><mo stretchy="false">(</mo><msup><mi>X</mi><mi mathvariant="normal">⊤</mi></msup><mi>X</mi><msup><mo stretchy="false">)</mo><mrow><mo>−</mo><mn>1</mn></mrow></msup><msup><mi>X</mi><mi mathvariant="normal">⊤</mi></msup><mi>y</mi></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mover accent="true"><mi>y</mi><mo>^</mo></mover><mo>=</mo><mi>H</mi><mi>y</mi></mrow></math></span>, and the hat matrix's diagonal tells you which examples control the fit.

</div>
