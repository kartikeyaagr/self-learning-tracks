---
id: "07.01"
course: "ml-maths-course"
module: "07"
slug: "quadratic-forms"
title: "Quadratic Forms"
keyIdeas: ["A quadratic form is ⟦2121⟧ with ⟦8⟧ symmetric; every term has degree 2.", "The diagonal entries give the squared terms; each off-diagonal pair ⟦2122⟧ gives the cross term ⟦2123⟧.", "Any quadratic polynomial without linear or constant terms can be written this way, with a unique symmetric ⟦8⟧."]
question: "Expand ⟦1765⟧ for ⟦2118⟧ and evaluate it at ⟦1774⟧. Then write ⟦2119⟧ as ⟦2120⟧."
answer: "Step 1, ⟦2138⟧. Step 2, at ⟦1352⟧: ⟦2128⟧. Step 3, for ⟦404⟧: the diagonal is ⟦2139⟧ and the cross coefficient ⟦2140⟧ is split evenly, so ⟦2141⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §4.5-4.6"}, {"title": "MIT 18.065 Matrix Methods in Data Analysis (Strang)", "url": "https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/", "note": "Read: lectures 6-9"}]
tags: ["Curvature and quadratic objectives"]
tools: ["NumPy"]
topic: {"id": "07.01", "title": "Curvature and quadratic objectives"}
---

## The big picture

Losses near a minimum, variances of linear combinations, Mahalanobis distances and the exponent of a Gaussian are all quadratic forms. This lesson sets up the notation; the next one reads off their shape.

## Definition

For a symmetric <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi><mo>×</mo><mi>n</mi></mrow></math></span> matrix <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span>,
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>q</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><msup><mi>x</mi><mi mathvariant="normal">⊤</mi></msup><mi>A</mi><mi>x</mi><mo>=</mo><munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover><munderover><mo>∑</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover><msub><mi>A</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><msub><mi>x</mi><mi>i</mi></msub><msub><mi>x</mi><mi>j</mi></msub></mrow></math></span></span>
In 2D with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>=</mo><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mi>a</mi></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mi>b</mi></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mi>b</mi></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mi>c</mi></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></mrow></math></span>:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msup><mi>x</mi><mi mathvariant="normal">⊤</mi></msup><mi>A</mi><mi>x</mi><mo>=</mo><mi>a</mi><msubsup><mi>x</mi><mn>1</mn><mn>2</mn></msubsup><mo>+</mo><mn>2</mn><mi>b</mi><msub><mi>x</mi><mn>1</mn></msub><msub><mi>x</mi><mn>2</mn></msub><mo>+</mo><mi>c</mi><msubsup><mi>x</mi><mn>2</mn><mn>2</mn></msubsup></mrow></math></span></span>

<div class="ex">
<div class="lab">Worked example</div>

**expanding**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>=</mo><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>3</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>x</mi><mi mathvariant="normal">⊤</mi></msup><mi>A</mi><mi>x</mi><mo>=</mo><mn>2</mn><msubsup><mi>x</mi><mn>1</mn><mn>2</mn></msubsup><mo>+</mo><mn>2</mn><msub><mi>x</mi><mn>1</mn></msub><msub><mi>x</mi><mn>2</mn></msub><mo>+</mo><mn>3</mn><msubsup><mi>x</mi><mn>2</mn><mn>2</mn></msubsup></mrow></math></span>.

At <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mo>−</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>2</mn><mo>−</mo><mn>2</mn><mo>+</mo><mn>3</mn><mo>=</mo><mn>3</mn></mrow></math></span>. At <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>2</mn><mo>+</mo><mn>2</mn><mo>+</mo><mn>3</mn><mo>=</mo><mn>7</mn></mrow></math></span>.

</div>

```
A = np.array([[2, 1], [1, 3]])
for x in ([1, -1], [1, 1]):
    x = np.array(x); print(x @ A @ x)
```

## Why symmetric

A non-symmetric <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>M</mi></mrow></math></span> gives the same values as its symmetric part <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mfrac><mn>1</mn><mn>2</mn></mfrac><mo stretchy="false">(</mo><mi>M</mi><mo>+</mo><msup><mi>M</mi><mi mathvariant="normal">⊤</mi></msup><mo stretchy="false">)</mo></mrow></math></span>, because <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>x</mi><mi mathvariant="normal">⊤</mi></msup><mi>M</mi><mi>x</mi><mo>=</mo><msup><mi>x</mi><mi mathvariant="normal">⊤</mi></msup><msup><mi>M</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi></mrow></math></span>. So we always use the symmetric version; that is also what makes the eigen-tools of module 05 apply.

## Where you meet them

| Expression | Quadratic form |
| --- | --- |
| Variance of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi>X</mi></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi mathvariant="normal">Σ</mi><mi>w</mi></mrow></math></span> |
| Squared Mahalanobis distance | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mi>x</mi><mo>−</mo><mi>μ</mi><msup><mo stretchy="false">)</mo><mi mathvariant="normal">⊤</mi></msup><msup><mi mathvariant="normal">Σ</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup><mo stretchy="false">(</mo><mi>x</mi><mo>−</mo><mi>μ</mi><mo stretchy="false">)</mo></mrow></math></span> |
| Least-squares loss (expanded) | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>β</mi><mi mathvariant="normal">⊤</mi></msup><msup><mi>X</mi><mi mathvariant="normal">⊤</mi></msup><mi>X</mi><mi>β</mi><mo>−</mo><mn>2</mn><msup><mi>y</mi><mi mathvariant="normal">⊤</mi></msup><mi>X</mi><mi>β</mi><mo>+</mo><msup><mi>y</mi><mi mathvariant="normal">⊤</mi></msup><mi>y</mi></mrow></math></span> |
| Second-order Taylor term | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mfrac><mn>1</mn><mn>2</mn></mfrac><msup><mi>d</mi><mi mathvariant="normal">⊤</mi></msup><mi>H</mi><mi>d</mi></mrow></math></span> |

<div class="take">
<div class="lab">Key takeaway</div>

a quadratic form <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>x</mi><mi mathvariant="normal">⊤</mi></msup><mi>A</mi><mi>x</mi></mrow></math></span> is a degree-2 polynomial packed into a symmetric matrix; diagonal entries give squares, off-diagonal ones give cross terms.

</div>
